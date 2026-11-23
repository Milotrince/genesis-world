"""A hundred passive humanoids, MuJoCo's own 100-humanoid demo, collapse into one pile, loaded as a single entity.

The scene is the MJCF file of the MuJoCo demo: ten spiral arms of ten humanoids each, every one attached from the
humanoid model file. Genesis loads the whole file as one entity with a hundred free root links, 1600 links and 2700
degrees of freedom in all. The humanoids receive no control, so they fold up and fall onto each other. The script logs
how long the scene takes to build and to step.

With the viewer (-v), drag any humanoid with the mouse, during the fall and after it.
"""

import argparse
import os
import time
import xml.etree.ElementTree as ET

import numpy as np

from huggingface_hub import snapshot_download

import genesis as gs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--steps", type=int, default=400, help="Number of simulation steps")
    parser.add_argument("-v", "--vis", action="store_true", help="Show the interactive viewer")
    parser.add_argument("-g", "--gpu", action="store_true", help="Run on GPU instead of CPU")
    parser.add_argument("-r", "--record", action="store_true", help="Record the run to 'out/humanoids_100.mp4'")
    args = parser.parse_args()
    horizon = 100 if "PYTEST_VERSION" in os.environ else args.steps

    gs.init(backend=gs.gpu if args.gpu else gs.cpu)

    models_dir = snapshot_download(
        repo_type="dataset",
        repo_id="Genesis-Intelligence/assets",
        revision="b86fb2c62cad1822b26add2fb355a694150ff1be",
        allow_patterns=["mujoco_models/humanoid/100_humanoids.xml", "mujoco_models/humanoid/humanoid.xml"],
        max_workers=1,
    )
    model_path = os.path.join(models_dir, "mujoco_models", "humanoid", "100_humanoids.xml")
    # Genesis compiles the model from its text, so the MuJoCo compiler resolves the humanoid model file against the
    # working directory. An absolute path keeps the lookup next to the scene file.
    mjcf = ET.parse(model_path).getroot()
    for model in mjcf.findall("asset/model"):
        model.attrib["file"] = os.path.join(os.path.dirname(model_path), model.attrib["file"])

    camera_pos = (8.0, -9.0, 6.0)
    camera_lookat = (0.0, 0.0, 0.5)

    scene = gs.Scene(
        rigid_options=gs.options.RigidOptions(
            # Each humanoid touches the floor and its neighbours through many of its 19 geoms at once.
            max_collision_pairs=10000,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=camera_pos,
            camera_lookat=camera_lookat,
        ),
        show_viewer=args.vis,
    )

    humanoids = scene.add_entity(
        gs.morphs.MJCF(
            file=mjcf,
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

    time_start = time.perf_counter()
    scene.build()
    build_time = time.perf_counter() - time_start
    gs.logger.info(f"Built one entity of {humanoids.n_links} links and {humanoids.n_dofs} dofs in {build_time:.1f} s.")

    if camera is not None:
        camera.start_recording(save_to_filename="out/humanoids_100.mp4", fps=25)
    physics_times = np.empty(horizon)
    for i_step in range(horizon):
        scene.step()
        physics_times[i_step] = scene.timings["physics"]
    if camera is not None:
        camera.stop_recording()
    # The median leaves out the first steps, which compile the kernels the contacts need.
    gs.logger.info(f"Median physics step {1e3 * np.median(physics_times):.1f} ms over {horizon} steps.")
    while args.vis and scene.viewer.is_alive() and "PYTEST_VERSION" not in os.environ:
        scene.step()


if __name__ == "__main__":
    main()
