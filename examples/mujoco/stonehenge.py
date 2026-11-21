"""A wrecking ball rolls down a ramp into a ring of trilithons, MuJoCo's own Stonehenge demo.

The scene is the MJCF file of the MuJoCo demo, loaded as it is: an outer ring of 30 standing stones capped by 30
lintels, an inner horseshoe of 5 trilithons, and a free sphere 11 m across resting at the top of a ramp tilted 30
degrees. Gravity alone sends the ball down the ramp and through the ring. The model builds both rings with the MJCF
'replicate' element, which the MuJoCo compiler expands before Genesis reads the model.

With the viewer (-v), drag any stone or the ball with the mouse, during the run and after it.
"""

import argparse
import os

from huggingface_hub import snapshot_download

import genesis as gs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--steps", type=int, default=1000, help="Number of simulation steps")
    parser.add_argument("-v", "--vis", action="store_true", help="Show the interactive viewer")
    parser.add_argument("-g", "--gpu", action="store_true", help="Run on GPU instead of CPU")
    parser.add_argument("-r", "--record", action="store_true", help="Record the run to 'out/stonehenge.mp4'")
    args = parser.parse_args()
    horizon = 100 if "PYTEST_VERSION" in os.environ else args.steps

    gs.init(backend=gs.gpu if args.gpu else gs.cpu)

    models_dir = snapshot_download(
        repo_type="dataset",
        repo_id="Genesis-Intelligence/assets",
        revision="b86fb2c62cad1822b26add2fb355a694150ff1be",
        allow_patterns="mujoco_models/replicate/stonehenge.xml",
        max_workers=1,
    )

    camera_pos = (-50.0, -70.0, 42.0)
    camera_lookat = (10.0, 0.0, 5.0)

    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=camera_pos,
            camera_lookat=camera_lookat,
        ),
        show_viewer=args.vis,
    )

    scene.add_entity(
        gs.morphs.MJCF(
            file=os.path.join(models_dir, "mujoco_models", "replicate", "stonehenge.xml"),
        ),
    )

    camera = None
    if args.record:
        camera = scene.add_camera(
            res=(1280, 720),
            pos=camera_pos,
            lookat=camera_lookat,
            # The scene spans about 100 m, past the default far plane of a camera.
            far=500.0,
        )

    if args.vis:
        scene.viewer.add_plugin(
            gs.vis.viewer_plugins.MouseInteractionPlugin(),
        )

    scene.build()

    if camera is not None:
        camera.start_recording(save_to_filename="out/stonehenge.mp4", fps=25)
    for _ in range(horizon):
        scene.step()
    if camera is not None:
        camera.stop_recording()
    while args.vis and scene.viewer.is_alive() and "PYTEST_VERSION" not in os.environ:
        scene.step()


if __name__ == "__main__":
    main()
