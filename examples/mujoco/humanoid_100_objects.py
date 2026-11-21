"""A hundred primitives rain on a passive humanoid in a walled arena, MuJoCo's own humanoid-and-100-objects demo.

The scene is the MJCF file of the MuJoCo demo: a humanoid attached from the humanoid model file, standing in a square
arena bordered by capsules, and twenty each of capsules, ellipsoids, boxes, cylinders and spheres, free and stacked in
a grid above it. The humanoid receives no control, so it collapses under the falling objects. The cylinders declare
torsional friction, which Genesis enables in the rigid options rather than from the file, so the script sets it there.

With the viewer (-v), drag the humanoid or any object with the mouse, during the fall and after it.
"""

import argparse
import os
import xml.etree.ElementTree as ET

from huggingface_hub import snapshot_download

import genesis as gs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--steps", type=int, default=300, help="Number of simulation steps")
    parser.add_argument("-v", "--vis", action="store_true", help="Show the interactive viewer")
    parser.add_argument("-g", "--gpu", action="store_true", help="Run on GPU instead of CPU")
    parser.add_argument("-r", "--record", action="store_true", help="Record the run to 'out/humanoid_100_objects.mp4'")
    args = parser.parse_args()
    horizon = 100 if "PYTEST_VERSION" in os.environ else args.steps

    gs.init(backend=gs.gpu if args.gpu else gs.cpu)

    models_dir = snapshot_download(
        repo_type="dataset",
        repo_id="Genesis-Intelligence/assets",
        revision="b86fb2c62cad1822b26add2fb355a694150ff1be",
        allow_patterns=["mujoco_models/humanoid/humanoid100.xml", "mujoco_models/humanoid/humanoid.xml"],
        max_workers=1,
    )
    model_path = os.path.join(models_dir, "mujoco_models", "humanoid", "humanoid100.xml")
    # Genesis compiles the model from its text, so the MuJoCo compiler resolves the humanoid model file against the
    # working directory. An absolute path keeps the lookup next to the scene file.
    mjcf = ET.parse(model_path).getroot()
    for model in mjcf.findall("asset/model"):
        model.attrib["file"] = os.path.join(os.path.dirname(model_path), model.attrib["file"])

    camera_pos = (5.5, -6.5, 4.5)
    camera_lookat = (0.0, 0.0, 0.3)

    scene = gs.Scene(
        rigid_options=gs.options.RigidOptions(
            enable_torsional_friction=True,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=camera_pos,
            camera_lookat=camera_lookat,
        ),
        show_viewer=args.vis,
    )

    scene.add_entity(
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

    scene.build()

    if camera is not None:
        camera.start_recording(save_to_filename="out/humanoid_100_objects.mp4", fps=25)
    for _ in range(horizon):
        scene.step()
    if camera is not None:
        camera.stop_recording()
    while args.vis and scene.viewer.is_alive() and "PYTEST_VERSION" not in os.environ:
        scene.step()


if __name__ == "__main__":
    main()
