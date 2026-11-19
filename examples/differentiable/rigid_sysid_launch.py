"""Recover the unknown launch velocity and spin of a box sliding on the floor from its observed trajectory.

A box launched with a velocity and a spin that the optimizer does not know slides to rest under friction, and its
trajectory is recorded. Differentiable rollouts start from a wrong guess, and torch autograd carries the trajectory error
back through every step of the rigid solver, contacts included, to the launch, which Adam refines. A kinematic ghost
replays the recorded trajectory beside each rollout. Rigid gradients reach the state and the controls of a scene while
the model parameters, such as the mass or the friction coefficient, carry none, so the launch is what gets identified.

With the viewer (-v), the final rollout plays again once the optimization ends, and the box can then be dragged with the
mouse.
"""

import argparse
import os

import numpy as np
import torch

import genesis as gs
from genesis.utils.misc import tensor_to_array


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--steps", type=int, default=80, help="Number of optimization iterations")
    parser.add_argument("-v", "--vis", action="store_true", help="Show the interactive viewer")
    parser.add_argument("-g", "--gpu", action="store_true", help="Run on GPU instead of CPU")
    parser.add_argument(
        "-r", "--record", action="store_true", help="Record the first and last rollouts to 'out/rigid_sysid_launch.mp4'"
    )
    args = parser.parse_args()

    gs.init(backend=gs.gpu if args.gpu else gs.cpu)

    camera_pos = (0.7, -1.5, 1.9)
    camera_lookat = (0.7, 0.4, 0.0)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1e-2,
            requires_grad=True,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=camera_pos,
            camera_lookat=camera_lookat,
            camera_fov=40,
        ),
        show_viewer=args.vis,
    )

    # Low friction on both surfaces lets the box slide and spin for a couple of metres before it comes to rest.
    scene.add_entity(
        morph=gs.morphs.Plane(),
        material=gs.materials.Rigid(
            friction=0.1,
        ),
    )
    box = scene.add_entity(
        morph=gs.morphs.Box(
            pos=(0.0, 0.0, 0.05),
            size=(0.2, 0.12, 0.1),
        ),
        material=gs.materials.Rigid(
            friction=0.1,
        ),
        surface=gs.surfaces.Default(
            color=(0.85, 0.35, 0.2),
        ),
    )
    ghost = scene.add_entity(
        morph=gs.morphs.Box(
            pos=(0.0, 0.0, 0.05),
            size=(0.2, 0.12, 0.1),
        ),
        material=gs.materials.Kinematic(),
        surface=gs.surfaces.Default(
            color=(0.3, 0.6, 0.9, 0.4),
        ),
    )

    camera = None
    if args.record:
        camera = scene.add_camera(
            res=(1280, 720),
            pos=camera_pos,
            lookat=camera_lookat,
            fov=40,
            debug=True,
        )

    if args.vis:
        scene.viewer.add_plugin(
            gs.vis.viewer_plugins.MouseInteractionPlugin(),
        )

    scene.build()

    n_rollout_steps = 150 if "PYTEST_VERSION" not in os.environ else 20
    n_iterations = args.steps if "PYTEST_VERSION" not in os.environ else 2

    # The launch is a planar velocity and a spin about the vertical, which this matrix maps to the velocity of the free
    # joint of the box, its other components staying zero.
    dofs_vel_from_launch = torch.zeros((3, 6), dtype=gs.tc_float, device=gs.device)
    dofs_vel_from_launch[(0, 1, 2), (0, 1, 5)] = 1.0

    launch_true = torch.tensor((1.6, 0.5, 1.5), dtype=gs.tc_float, device=gs.device)
    box.set_dofs_velocity(launch_true @ dofs_vel_from_launch)
    target_pos, target_quat = [], []
    for _ in range(n_rollout_steps):
        scene.step()
        target_pos.append(box.get_pos().detach())
        target_quat.append(box.get_quat().detach())
    target_pos, target_quat = torch.stack(target_pos), torch.stack(target_quat)
    scene.draw_debug_trajectory(target_pos, radius=0.004, color=(0.3, 0.6, 0.9, 0.8))

    def rollout(launch):
        """Launch the box and step the scene, the ghost replaying the recorded trajectory, and return the trajectory."""
        scene.reset()
        box.set_dofs_velocity(launch @ dofs_vel_from_launch)
        rollout_pos, rollout_quat = [], []
        for i_step in range(n_rollout_steps):
            ghost.set_qpos(torch.cat((target_pos[i_step], target_quat[i_step])))
            scene.step()
            rollout_pos.append(box.get_pos())
            rollout_quat.append(box.get_quat())
        return torch.stack(rollout_pos), torch.stack(rollout_quat)

    launch = torch.tensor((0.8, -0.2, 0.0), dtype=gs.tc_float, device=gs.device, requires_grad=True)
    optimizer = torch.optim.Adam([launch], lr=0.2)
    # The step size decays so that Adam settles on the minimum instead of circling it.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_iterations, eta_min=1e-3)
    for i_iteration in range(n_iterations):
        # The first and the last rollouts are recorded, to show the trajectory before and after the optimization.
        is_recorded = camera is not None and i_iteration in (0, n_iterations - 1)
        if is_recorded:
            if i_iteration == 0:
                camera.start_recording(save_to_filename="out/rigid_sysid_launch.mp4", fps=50)
            else:
                camera.start_recording()
        optimizer.zero_grad()
        rollout_pos, rollout_quat = rollout(launch)
        if is_recorded:
            camera.pause_recording()
        # The orientation error is measured through the dot product of the quaternions, which is blind to their sign.
        pos_err = ((rollout_pos - target_pos) ** 2).sum(dim=-1)
        quat_err = 1.0 - ((rollout_quat * target_quat).sum(dim=-1)) ** 2
        loss = (pos_err + quat_err).mean()
        loss.backward()
        optimizer.step()
        scheduler.step()
        gs.logger.info(
            f"iteration {i_iteration:3d}: loss {loss.item():.2e}, launch (vx, vy, wz) "
            f"{np.round(tensor_to_array(launch), 3)} (true {tensor_to_array(launch_true)})"
        )
    if camera is not None:
        camera.stop_recording()

    if args.vis:
        with torch.no_grad():
            rollout(launch)
    while args.vis and scene.viewer.is_alive() and "PYTEST_VERSION" not in os.environ:
        scene.step()


if __name__ == "__main__":
    main()
