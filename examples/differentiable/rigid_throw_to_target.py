"""A Franka learns to throw a ball into a bin by gradient descent through its own swing.

The throw is a short swing of the base, shoulder and elbow joints under velocity control, after which the hand lets the
ball go. The ball leaves with the velocity of the grasp point at release, so torch autograd carries the distance between
its landing point and the centre of the bin back through the flight and the controlled swing to the three joint
velocity targets, which Adam refines. The landing error of every throw is logged, and the recording shows a few throws
of the optimization, from the first miss to the final hit.

With the viewer (-v), the final throw plays again once the optimization ends.

Keyboard controls:
T - Throw again with the optimized swing
N - Throw with a random perturbation of the optimized swing

Plus all default viewer controls (press 'i' to see them).
"""

import argparse
import os

import numpy as np
import torch

import genesis as gs
import genesis.utils.geom as gu
from genesis.utils.misc import tensor_to_array
from genesis.vis.keybindings import Key, KeyAction, Keybind


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--steps", type=int, default=40, help="Number of optimization iterations")
    parser.add_argument("-v", "--vis", action="store_true", help="Show the interactive viewer")
    parser.add_argument("-g", "--gpu", action="store_true", help="Run on GPU instead of CPU")
    parser.add_argument(
        "-r", "--record", action="store_true", help="Record a few throws to 'out/rigid_throw_to_target.mp4'"
    )
    args = parser.parse_args()

    gs.init(backend=gs.gpu if args.gpu else gs.cpu)

    camera_pos = (0.7, -1.3, 0.9)
    camera_lookat = (0.5, 0.15, 0.35)

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

    # The ball collides through bit 16 alone, which the robot geoms lack, so that the hand lets it go without contact.
    # The plane and the bin take that bit on top of the default ones.
    ball_collision_mask = 1 << 16
    world_collision_mask = 0xFFFF | ball_collision_mask
    scene.add_entity(
        morph=gs.morphs.Plane(
            contype=world_collision_mask,
            conaffinity=world_collision_mask,
        ),
    )
    robot = scene.add_entity(
        morph=gs.morphs.MJCF(
            file="xml/franka_emika_panda/panda.xml",
        ),
        material=gs.materials.Rigid(
            gravity_compensation=1.0,
        ),
    )
    ball_radius = 0.025
    ball = scene.add_entity(
        morph=gs.morphs.Sphere(
            radius=ball_radius,
            pos=(0.5, 0.0, 0.5),
            contype=ball_collision_mask,
            conaffinity=ball_collision_mask,
        ),
        surface=gs.surfaces.Default(
            color=(0.9, 0.3, 0.2),
        ),
    )

    # The bin is a floor and four walls, its rim at 'rim_height'.
    bin_center = (0.9, 0.3)
    bin_half_width, rim_height, wall_thickness = 0.12, 0.12, 0.01
    for wall_pos, wall_size in (
        ((0.0, 0.0, 0.5 * wall_thickness), (2.0 * bin_half_width, 2.0 * bin_half_width, wall_thickness)),
        ((bin_half_width, 0.0, 0.5 * rim_height), (wall_thickness, 2.0 * bin_half_width, rim_height)),
        ((-bin_half_width, 0.0, 0.5 * rim_height), (wall_thickness, 2.0 * bin_half_width, rim_height)),
        ((0.0, bin_half_width, 0.5 * rim_height), (2.0 * bin_half_width, wall_thickness, rim_height)),
        ((0.0, -bin_half_width, 0.5 * rim_height), (2.0 * bin_half_width, wall_thickness, rim_height)),
    ):
        scene.add_entity(
            morph=gs.morphs.Box(
                pos=(bin_center[0] + wall_pos[0], bin_center[1] + wall_pos[1], wall_pos[2]),
                size=wall_size,
                fixed=True,
                contype=world_collision_mask,
                conaffinity=world_collision_mask,
            ),
            surface=gs.surfaces.Default(
                color=(0.3, 0.5, 0.8),
            ),
        )

    camera = None
    if args.record:
        camera = scene.add_camera(
            res=(1280, 720),
            pos=camera_pos,
            lookat=camera_lookat,
            fov=40,
        )

    scene.build()

    is_pytest = "PYTEST_VERSION" in os.environ
    n_iterations = args.steps if not is_pytest else 2
    n_swing_steps = 20
    n_flight_steps_max = 80 if not is_pytest else 30
    n_settle_steps = 60

    hand = robot.get_link("hand")
    swing_dofs_idx = [0, 1, 3]
    hold_dofs_idx = [2, 4, 5, 6]
    fingers_dofs_idx = [7, 8]
    # The arm starts folded, the elbow high, and the ball sits between the open fingers.
    qpos_init = torch.tensor((0.0, -0.6, 0.0, -2.4, 0.0, 2.0, 0.8, 0.04, 0.04), dtype=gs.tc_float, device=gs.device)
    grasp_offset = torch.tensor((0.0, 0.0, 0.11), dtype=gs.tc_float, device=gs.device)
    ball_quat = torch.tensor((1.0, 0.0, 0.0, 0.0), dtype=gs.tc_float, device=gs.device)
    target = torch.tensor(bin_center, dtype=gs.tc_float, device=gs.device)

    def throw(swing_vel):
        """Swing the arm with the ball in the hand, release it and return where it drops through the bin opening."""
        scene.reset()
        robot.set_qpos(qpos_init)
        robot.control_dofs_position(qpos_init[hold_dofs_idx], hold_dofs_idx)
        # Finger targets span [0, 255], from closed to open.
        robot.control_dofs_position((255.0, 255.0), fingers_dofs_idx)
        robot.control_dofs_velocity(swing_vel, swing_dofs_idx)
        for _ in range(n_swing_steps):
            ball_pos = (hand.get_pos() + gu.transform_by_quat(grasp_offset, hand.get_quat())).detach()
            ball.set_qpos(torch.cat((ball_pos, ball_quat)))
            ball.set_dofs_velocity(torch.zeros(6, dtype=gs.tc_float, device=gs.device))
            scene.step()

        # The ball leaves with the velocity of the grasp point, which carries the gradient back into the swing.
        grasp_offset_world = gu.transform_by_quat(grasp_offset, hand.get_quat())
        release_pos = hand.get_pos() + grasp_offset_world
        release_vel = hand.get_vel() + torch.linalg.cross(hand.get_ang(), grasp_offset_world)
        ball.set_qpos(torch.cat((release_pos, ball_quat)))
        ball.set_dofs_velocity(torch.cat((release_vel, torch.zeros(3, dtype=gs.tc_float, device=gs.device))))
        robot.control_dofs_velocity(torch.zeros(3, dtype=gs.tc_float, device=gs.device), swing_dofs_idx)

        # The flight stops as the ball drops through a plane above the rim, before any contact with the bin can bend the
        # gradient. The landing point is interpolated between the last two steps, so that it moves smoothly with the
        # swing instead of jumping from one step to the next.
        landing_height = rim_height + 2.0 * ball_radius
        ball_pos_prev = release_pos
        for _ in range(n_flight_steps_max):
            scene.step()
            ball_pos = ball.get_pos()
            if ball_pos[2] < landing_height:
                break
            ball_pos_prev = ball_pos
        ratio = (ball_pos_prev[2] - landing_height) / (ball_pos_prev[2] - ball_pos[2])
        return (ball_pos_prev + ratio * (ball_pos - ball_pos_prev))[:2]

    # The first guess swings straight ahead and too slowly, so the ball falls short of the bin and beside it.
    swing_vel = torch.tensor((0.0, 1.5, 1.5), dtype=gs.tc_float, device=gs.device, requires_grad=True)
    optimizer = torch.optim.Adam([swing_vel], lr=0.2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_iterations, eta_min=1e-3)
    recorded_iterations = sorted({0, n_iterations // 8, n_iterations // 4, n_iterations - 1})
    for i_iteration in range(n_iterations):
        is_recorded = camera is not None and i_iteration in recorded_iterations
        if is_recorded:
            if i_iteration == 0:
                camera.start_recording(save_to_filename="out/rigid_throw_to_target.mp4", fps=50)
            else:
                camera.start_recording()
        optimizer.zero_grad()
        landing_pos = throw(swing_vel)
        landing_err = torch.linalg.norm(landing_pos - target)
        # The recorded throws go on until the ball settles, before the backward pass that ends the rollout.
        if is_recorded:
            for _ in range(n_settle_steps):
                scene.step()
            camera.pause_recording()
        landing_err.backward()
        gs.logger.info(
            f"iteration {i_iteration:3d}: landing error {100.0 * landing_err.item():5.1f} cm, swing velocity targets "
            f"(base, shoulder, elbow) {np.round(tensor_to_array(swing_vel), 3)} rad/s"
        )
        optimizer.step()
        scheduler.step()
    if camera is not None:
        camera.stop_recording()

    swing_vel_pending = None

    def request_throw(noise_scale):
        nonlocal swing_vel_pending
        swing_vel_pending = swing_vel.detach() + noise_scale * torch.randn_like(swing_vel)

    if args.vis:
        scene.viewer.register_keybinds(
            Keybind("throw", Key.T, KeyAction.RELEASE, callback=request_throw, args=(0.0,)),
            Keybind("throw_perturbed", Key.N, KeyAction.RELEASE, callback=request_throw, args=(0.2,)),
        )
        gs.logger.info("Keys: T throws again with the optimized swing, N throws with a random perturbation of it.")
        request_throw(0.0)
    while args.vis and scene.viewer.is_alive() and not is_pytest:
        if swing_vel_pending is not None:
            landing_pos = throw(swing_vel_pending)
            swing_vel_pending = None
            gs.logger.info(f"Landing error: {100.0 * torch.linalg.norm(landing_pos - target).item():.1f} cm")
        scene.step()


if __name__ == "__main__":
    main()
