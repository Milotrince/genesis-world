"""A deck of playing cards, MuJoCo's own cards demo, spills over a bar and scatters on the floor.

The scene is the MJCF file of the MuJoCo demo, loaded as it is: 54 free cards, each a textured mesh with a thin box for
collision, a capsule bar and a checkered floor. The model also sets the density and viscosity of air, so that the cards
flutter in MuJoCo. Genesis applies no aerodynamic forces to rigid bodies, so here the cards tumble without drag.

With the viewer (-v), drag any card with the mouse, during the fall and after it.
"""

import argparse
import os
import xml.etree.ElementTree as ET

from huggingface_hub import snapshot_download

import genesis as gs


# The timestep and playback speed of the model, which Genesis reads from the scene options rather than from the file
DT = 1e-3
REALTIME_FACTOR = 0.3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--steps", type=int, default=1500, help="Number of simulation steps")
    parser.add_argument("-v", "--vis", action="store_true", help="Show the interactive viewer")
    parser.add_argument("-g", "--gpu", action="store_true", help="Run on GPU instead of CPU")
    parser.add_argument("-r", "--record", action="store_true", help="Record the run to 'out/cards.mp4'")
    args = parser.parse_args()
    horizon = 100 if "PYTEST_VERSION" in os.environ else args.steps

    gs.init(backend=gs.gpu if args.gpu else gs.cpu)

    models_dir = snapshot_download(
        repo_type="dataset",
        repo_id="Genesis-Intelligence/assets",
        revision="b86fb2c62cad1822b26add2fb355a694150ff1be",
        allow_patterns="mujoco_models/cards/*",
        max_workers=1,
    )
    model_path = os.path.join(models_dir, "mujoco_models", "cards", "cards.xml")
    # The model finds its mesh and textures through 'assetdir', which Genesis overrides by setting the mesh and texture
    # directories to the directory of the file. Pointing both at the asset directory restores the lookup.
    mjcf = ET.parse(model_path).getroot()
    compiler = mjcf.find("compiler")
    assets_dir = os.path.join(os.path.dirname(model_path), compiler.attrib["assetdir"])
    compiler.attrib.update(meshdir=assets_dir, texturedir=assets_dir)

    camera_pos = (1.0, -1.6, 0.2)
    camera_lookat = (0.12, 0.0, -0.6)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=DT,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=camera_pos,
            camera_lookat=camera_lookat,
            realtime_factor=REALTIME_FACTOR,
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
        camera.start_recording(save_to_filename="out/cards.mp4", fps=30)
    for _ in range(horizon):
        scene.step()
    if camera is not None:
        camera.stop_recording()
    while args.vis and scene.viewer.is_alive() and "PYTEST_VERSION" not in os.environ:
        scene.step()


if __name__ == "__main__":
    main()
