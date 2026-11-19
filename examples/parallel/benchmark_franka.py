"""Throughput benchmark: Franka arms holding a pose under PD control, in parallel environments.

With the viewer (-v), drag the arms with the mouse, during the run and after it.
"""

import argparse
import os

import genesis as gs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", help="Show visualization GUI")
    parser.add_argument("-b", "--num-envs", type=int, default=30000, help="Number of parallel environments")
    parser.add_argument("-s", "--steps", type=int, default=1000, help="Number of simulation steps")
    parser.add_argument(
        "-r", "--record", action="store_true", help="Record a video of the run to 'out/benchmark_franka.mp4'"
    )
    args = parser.parse_args()
    if "PYTEST_VERSION" in os.environ:
        args.num_envs, args.steps = 2, 5

    # The benchmark measures throughput at tens of thousands of parallel environments, which needs a GPU.
    gs.init(backend=gs.gpu, performance_mode=True)

    camera_pos = (3.5, -1.0, 2.5)
    camera_lookat = (0.0, 0.0, 0.5)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.01,
        ),
        viewer_options=gs.options.ViewerOptions(
            res=(1920, 1080),
            camera_pos=camera_pos,
            camera_lookat=camera_lookat,
            camera_fov=40,
        ),
        show_viewer=args.vis,
    )

    plane = scene.add_entity(
        gs.morphs.Plane(),
    )

    franka = scene.add_entity(
        gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
    )

    camera = None
    if args.record:
        camera = scene.add_camera(
            res=(1280, 720),
            pos=camera_pos,
            lookat=camera_lookat,
            fov=40,
        )

    if args.vis:
        scene.viewer.add_plugin(
            gs.vis.viewer_plugins.MouseInteractionPlugin(),
        )

    scene.build(n_envs=args.num_envs, env_spacing=(1.0, 1.0))
    if camera is not None:
        camera.start_recording(save_to_filename="out/benchmark_franka.mp4", fps=round(1.0 / scene.dt))

    # control all the robots
    # with the following control: 43M FPS on GPU with 30000 environments
    # without the following control (arm in collision with the floor): 32M FPS
    franka.control_dofs_position([0, 0, 0, -1.0, 0, 1.0, 0, 0.02, 0.02])

    for _ in range(args.steps):
        scene.step()
    if camera is not None:
        camera.stop_recording()
    while args.vis and scene.viewer.is_alive() and "PYTEST_VERSION" not in os.environ:
        scene.step()


if __name__ == "__main__":
    main()
