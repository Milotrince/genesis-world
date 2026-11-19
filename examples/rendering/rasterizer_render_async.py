"""A Franka falls limp under gravity, simulated in a background thread while the viewer renders on the main thread.

The simulation thread steps the scene without refreshing the viewer, and the viewer (-v) runs its own render loop on
the main thread, at its own rate. Without the viewer, the simulation thread runs alone. With the viewer, drag the arm
with the mouse.
"""

import argparse
import os
import threading

import genesis as gs


def run_sim(scene, n_steps, camera, is_interactive):
    for _ in range(n_steps):
        scene.step(refresh_visualizer=False)
    if camera is not None:
        camera.stop_recording()
    while is_interactive and scene.viewer.is_alive():
        scene.step(refresh_visualizer=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", help="Show the interactive viewer")
    parser.add_argument("-s", "--steps", type=int, default=200, help="Number of simulation steps")
    parser.add_argument(
        "-r", "--record", action="store_true", help="Record a video of the run to 'out/rasterizer_render_async.mp4'"
    )
    args = parser.parse_args()
    if "PYTEST_VERSION" in os.environ:
        args.steps = 1

    gs.init(backend=gs.cpu)

    camera_pos = (3.5, 0.0, 2.5)
    camera_lookat = (0.0, 0.0, 0.5)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.01,
        ),
        rigid_options=gs.options.RigidOptions(
            gravity=(0.0, 0.0, -10.0),
        ),
        viewer_options=gs.options.ViewerOptions(
            run_in_thread=False,
            camera_pos=camera_pos,
            camera_lookat=camera_lookat,
            camera_fov=40,
        ),
        profiling_options=gs.options.ProfilingOptions(
            show_FPS=True,
        ),
        show_viewer=args.vis,
    )

    scene.add_entity(gs.morphs.Plane())
    scene.add_entity(
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

    scene.build()
    if camera is not None:
        camera.start_recording(save_to_filename="out/rasterizer_render_async.mp4", fps=round(1.0 / scene.dt))

    is_interactive = args.vis and "PYTEST_VERSION" not in os.environ
    sim_thread = threading.Thread(target=run_sim, args=(scene, args.steps, camera, is_interactive))
    sim_thread.start()
    if is_interactive:
        scene.viewer.run()
    sim_thread.join()


if __name__ == "__main__":
    main()
