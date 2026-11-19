"""Throughput benchmark: ANYmal C quadrupeds holding a standing pose under PD control, in parallel environments.

With the viewer (-v), drag the robots with the mouse, during the run and after it.
"""

import argparse
import os

import genesis as gs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", help="Show visualization GUI")
    parser.add_argument("-b", "--num-envs", type=int, default=30000, help="Number of parallel environments")
    parser.add_argument("-s", "--steps", type=int, default=1000, help="Number of simulation steps")
    parser.add_argument("-r", "--record", action="store_true", help="Record a video of the run to 'out/anymal_c.mp4'")
    args = parser.parse_args()
    if "PYTEST_VERSION" in os.environ:
        args.num_envs, args.steps = 2, 5

    # The benchmark measures throughput at tens of thousands of parallel environments, which needs a GPU.
    gs.init(backend=gs.gpu, performance_mode=True)

    camera_pos = (2.5, -2.5, 1.5)
    camera_lookat = (0.0, 0.0, 0.5)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.01,
        ),
        rigid_options=gs.options.RigidOptions(
            constraint_solver=gs.constraint_solver.Newton,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=camera_pos,
            camera_lookat=camera_lookat,
        ),
        show_viewer=args.vis,
    )

    scene.add_entity(
        gs.morphs.Plane(),
    )
    robot = scene.add_entity(
        gs.morphs.URDF(
            file="urdf/anymal_c/urdf/anymal_c.urdf",
            pos=(0, 0, 0.8),
        ),
    )

    camera = None
    if args.record:
        camera = scene.add_camera(
            res=(1280, 720),
            pos=camera_pos,
            lookat=camera_lookat,
        )

    if args.vis:
        scene.viewer.add_plugin(
            gs.vis.viewer_plugins.MouseInteractionPlugin(),
        )

    scene.build(n_envs=args.num_envs)
    if camera is not None:
        camera.start_recording(save_to_filename="out/anymal_c.mp4", fps=round(1.0 / scene.dt))

    joints_name = (
        "RH_HAA",
        "LH_HAA",
        "RF_HAA",
        "LF_HAA",
        "RH_HFE",
        "LH_HFE",
        "RF_HFE",
        "LF_HFE",
        "RH_KFE",
        "LH_KFE",
        "RF_KFE",
        "LF_KFE",
    )
    motors_dof_idx = [robot.get_joint(name).dofs_idx_local[0] for name in joints_name]

    robot.set_dofs_kp(1000, motors_dof_idx)
    robot.control_dofs_position(0.0, motors_dof_idx)

    # Speed: 14.4M FPS on GPU with 30000 environments
    for _ in range(args.steps):
        scene.step()
    if camera is not None:
        camera.stop_recording()
    while args.vis and scene.viewer.is_alive() and "PYTEST_VERSION" not in os.environ:
        scene.step()


if __name__ == "__main__":
    main()
