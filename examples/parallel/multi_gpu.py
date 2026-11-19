"""One independent simulation per GPU, each in its own process pinned to its device.

With the viewer (-v), each process opens its own viewer, where the Franka can be dragged with the mouse during the run
and after it. With -r, the process of the first GPU records its scene.
"""

import argparse
import multiprocessing
import os
import sys

import torch

import genesis as gs


def simulate(gpu_id, args):
    # Pin the process to its GPU before Genesis initializes, then select that CUDA device.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["QD_VISIBLE_DEVICE"] = str(gpu_id)
    os.environ["EGL_DEVICE_ID"] = str(gpu_id)
    gs.init(backend=gs.gpu, logger_verbose_time=True)

    camera_pos = (3.5, 0.0, 2.5)
    camera_lookat = (0.0, 0.0, 0.5)

    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=camera_pos,
            camera_lookat=camera_lookat,
            camera_fov=40,
        ),
        profiling_options=gs.options.ProfilingOptions(
            show_FPS=False,
        ),
        show_viewer=args.vis,
    )

    plane = scene.add_entity(
        gs.morphs.Plane(),
    )
    franka = scene.add_entity(
        gs.morphs.MJCF(
            file="xml/franka_emika_panda/panda.xml",
        ),
        visualize_contact=True,
    )

    camera = None
    if args.record and gpu_id == 0:
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
        camera.start_recording(save_to_filename="out/multi_gpu.mp4", fps=round(1.0 / scene.dt))
    for i in range(1000 if "PYTEST_VERSION" not in os.environ else 5):
        scene.step()
    if camera is not None:
        camera.stop_recording()
    while args.vis and scene.viewer.is_alive() and "PYTEST_VERSION" not in os.environ:
        scene.step()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", help="Show visualization GUI")
    parser.add_argument("-r", "--record", action="store_true", help="Record a video of the run to 'out/multi_gpu.mp4'")
    args = parser.parse_args()

    # One simulation per GPU, a single one when there is no GPU
    num_gpus = max(torch.cuda.device_count(), 1)

    processes = []
    for i in range(num_gpus):
        p = multiprocessing.Process(target=simulate, args=(i, args))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()
    if any(p.exitcode != 0 for p in processes):
        sys.exit(1)


if __name__ == "__main__":
    main()
