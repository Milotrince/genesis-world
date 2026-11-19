"""
Load the Lightwheel Kitchen USD assets in Genesis.

Usage
-----
    python examples/rigid/usd_kitchen.py                    # sample assets, headless
    python examples/rigid/usd_kitchen.py --asset dishwasher
    python examples/rigid/usd_kitchen.py --vis              # interactive viewer (drag with the mouse)
    python examples/rigid/usd_kitchen.py --full --vis       # the whole kitchen scene
    python examples/rigid/usd_kitchen.py -r                 # record the settle to out/usd_kitchen.mp4

Headless, the scene runs a short bounded settle and exits. With ``-v``/``--vis`` the same settle runs in the
interactive viewer, which then stays open until it is closed or ``Esc`` is pressed. Drag the entities with the mouse,
during the settle and after it.
"""

import argparse
import os

from huggingface_hub import snapshot_download

import genesis as gs
from genesis.utils.misc import tensor_to_array
from genesis.vis.keybindings import Key, KeyAction, Keybind

SAMPLE_ASSETS = {
    "dishwasher": ("Lightwheel_Kitchen/Dishwasher054/Dishwasher054.usd", ["Lightwheel_Kitchen/Dishwasher054/*"]),
    "bottle": (
        "Lightwheel_Kitchen/Kitchen_Other/Kitchen_Bottle006.usd",
        ["Lightwheel_Kitchen/Kitchen_Other/Kitchen_Bottle006.usd", "Lightwheel_Kitchen/Kitchen_Other/texture/*"],
    ),
}
FULL_ROOM_ASSETS = ("Lightwheel_Kitchen/KitchenRoom.usd", ["Lightwheel_Kitchen/*"])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--asset",
        default="bottle" if "PYTEST_VERSION" in os.environ else "all",
        choices=[*SAMPLE_ASSETS, "all"],
        help="Which sample asset(s) to load",
    )
    parser.add_argument("--full", action="store_true", help="Load the entire kitchen scene instead of the samples")
    parser.add_argument("-v", "--vis", action="store_true", help="Show the interactive viewer")
    parser.add_argument(
        "--collision", action="store_true", help="Visualize collision geometry instead of the visual meshes"
    )
    parser.add_argument(
        "-r", "--record", action="store_true", help="Record a video of the run to 'out/usd_kitchen.mp4'"
    )
    args = parser.parse_args()

    gs.init(backend=gs.cpu)

    if args.full:
        camera_pos, camera_lookat = (6.0, -6.0, 4.0), (0.0, 0.0, 1.0)
    else:
        camera_pos, camera_lookat = (1.5, -2.5, 1.5), (0.6, 0.0, 0.4)

    scene = gs.Scene(
        rigid_options=gs.options.RigidOptions(
            enable_neutral_collision=True,  # Enable so articulated parts (e.g. dishwasher) don't clip
        ),
        vis_options=gs.options.VisOptions(
            debug=args.collision,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=camera_pos,
            camera_lookat=camera_lookat,
        ),
        profiling_options=gs.options.ProfilingOptions(
            show_FPS=False,
        ),
        show_viewer=args.vis,
    )
    scene.add_entity(gs.morphs.Plane())

    if args.full:
        patterns = FULL_ROOM_ASSETS[1]
        rel_paths = [FULL_ROOM_ASSETS[0]]
        fixed = None  # Keep the scene's authored fixed/free states.
    else:
        selected = list(SAMPLE_ASSETS) if args.asset == "all" else [args.asset]
        patterns = [pattern for key in selected for pattern in SAMPLE_ASSETS[key][1]]
        rel_paths = [SAMPLE_ASSETS[key][0] for key in selected]
        fixed = False  # Free base so every sample (incl. the authored-fixed dishwasher) drops and is draggable.

    asset_path = snapshot_download(
        repo_type="dataset",
        repo_id="Genesis-Intelligence/assets",
        revision="c3d4f971ac7da1ca2257adc7663b3aaea82c9a32",
        allow_patterns=patterns,
    )

    all_entities = []
    for rel_path in rel_paths:
        usd_file = os.path.join(asset_path, rel_path)
        if not os.path.isfile(usd_file):
            gs.raise_exception(f"USD file not found: {usd_file}")
        gs.logger.info(f"Loading {rel_path} ...")
        entities = scene.add_stage(
            morph=gs.morphs.USD(
                file=usd_file,
                fixed=fixed,
            ),
        )
        for entity in entities:
            gs.logger.info(
                f"     {entity.__class__.__name__}: "
                f"n_links={entity.n_links} n_joints={entity.n_joints} n_geoms={entity.n_geoms}"
            )
        all_entities += entities

    camera = None
    if args.record:
        camera = scene.add_camera(
            res=(1280, 720),
            pos=camera_pos,
            lookat=camera_lookat,
        )

    if args.vis:
        # Drag entities around with the mouse; the plugin must be attached before build.
        scene.viewer.add_plugin(
            gs.vis.viewer_plugins.MouseInteractionPlugin(
                use_force=True,
                color=(0.1, 0.6, 0.8, 0.6),
            )
        )

    scene.build()
    if not args.full:
        # Lay the sample entities out in a row on the ground plane.
        x, gap = 0.0, 0.05
        for entity in all_entities:
            lo, hi = tensor_to_array(entity.get_AABB())
            size = hi - lo
            target_min = (x, -0.5 * size[1], gap)
            entity.set_pos(tensor_to_array(entity.get_pos()) + (target_min - lo))
            x += size[0] + gap
    gs.logger.info(f"Scene built successfully with {len(all_entities)} entities.")

    is_running = True
    if args.vis:

        def stop():
            nonlocal is_running
            is_running = False

        scene.viewer.register_keybinds(
            Keybind("quit", Key.ESCAPE, KeyAction.RELEASE, callback=stop),
        )
        gs.logger.info("Drag the entities with the mouse. Esc quits.")

    if camera is not None:
        camera.start_recording(save_to_filename="out/usd_kitchen.mp4", fps=round(1.0 / scene.dt))
    for _ in range(200 if "PYTEST_VERSION" not in os.environ else 5):
        scene.step()
    if camera is not None:
        camera.stop_recording()

    while args.vis and is_running and scene.viewer.is_alive() and "PYTEST_VERSION" not in os.environ:
        scene.step()


if __name__ == "__main__":
    main()
