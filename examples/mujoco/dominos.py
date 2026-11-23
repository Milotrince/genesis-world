"""A serpentine run of 96 dominoes, MuJoCo's own sleeping dominoes demo, falls from end to end while toppled dominoes
fall asleep.

The scene is the MJCF file of the MuJoCo demo, loaded as it is: five rows of dominoes joined by U-turns, and a first
domino tilted so that it starts the chain. The model asks for an elliptic friction cone, an impedance ratio of 10 and
sleeping bodies, which Genesis reads from the rigid options rather than from the file, so the script sets the same
three there. Hibernation takes the dominoes at rest out of the solve, so the step gets cheaper as the run settles.

With the viewer (-v), drag any domino with the mouse, during the run and after it.
"""

import argparse
import os

from huggingface_hub import snapshot_download

import genesis as gs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--steps", type=int, default=1300, help="Number of simulation steps")
    parser.add_argument("-v", "--vis", action="store_true", help="Show the interactive viewer")
    parser.add_argument("-g", "--gpu", action="store_true", help="Run on GPU instead of CPU")
    parser.add_argument("-r", "--record", action="store_true", help="Record the run to 'out/dominos.mp4'")
    args = parser.parse_args()
    horizon = 100 if "PYTEST_VERSION" in os.environ else args.steps

    gs.init(backend=gs.gpu if args.gpu else gs.cpu)

    models_dir = snapshot_download(
        repo_type="dataset",
        repo_id="Genesis-Intelligence/assets",
        revision="b86fb2c62cad1822b26add2fb355a694150ff1be",
        allow_patterns="mujoco_models/sleep/dominos.xml",
        max_workers=1,
    )

    camera_pos = (0.5, -3.1, 2.6)
    camera_lookat = (0.5, 0.15, 0.0)

    scene = gs.Scene(
        rigid_options=gs.options.RigidOptions(
            friction_cone=gs.friction_cone.elliptic,
            impratio=10.0,
            use_hibernation=True,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=camera_pos,
            camera_lookat=camera_lookat,
        ),
        show_viewer=args.vis,
    )

    scene.add_entity(
        gs.morphs.MJCF(
            file=os.path.join(models_dir, "mujoco_models", "sleep", "dominos.xml"),
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
        camera.start_recording(save_to_filename="out/dominos.mp4", fps=25)
    for _ in range(horizon):
        scene.step()
    if camera is not None:
        camera.stop_recording()
    while args.vis and scene.viewer.is_alive() and "PYTEST_VERSION" not in os.environ:
        scene.step()


if __name__ == "__main__":
    main()
