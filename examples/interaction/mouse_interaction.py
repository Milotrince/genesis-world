"""Drag a sphere, a duck and a ring of boxes around with the mouse.

The objects drop onto the ground, and the mouse grabs and drags them from the first frame until the viewer closes. Esc
quits.
"""

import argparse
import math
import os

import torch

import genesis as gs
from genesis.vis.keybindings import Key, KeyAction, Keybind


def main():
    parser = argparse.ArgumentParser(description="Mouse interaction viewer plugin example.")
    parser.add_argument("--use-force", action="store_true", help="Apply spring forces instead of setting position")
    parser.add_argument(
        "--use-visual-geom",
        action="store_true",
        help="Grab entities by their visual mesh instead of their collision one",
    )
    parser.add_argument("-b", "--num-envs", type=int, default=1, help="Number of parallel environments")
    parser.add_argument(
        "-r", "--record", action="store_true", help="Record a video of the run to 'out/mouse_interaction.mp4'"
    )
    args = parser.parse_args()

    gs.init(backend=gs.cpu)

    camera_pos = (3.5, 0.0, 2.5)
    camera_lookat = (0.0, 0.0, 0.5)

    scene = gs.Scene(
        vis_options=gs.options.VisOptions(
            debug=not args.use_visual_geom,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=camera_pos,
            camera_lookat=camera_lookat,
            camera_fov=40,
        ),
        show_viewer=True,
    )

    scene.add_entity(
        gs.morphs.Plane(),
    )

    # Only entities opting into visual raycasting can be grabbed with --use-visual-geom.
    sphere = scene.add_entity(
        spec=gs.entities.Rigid(
            morph=gs.morphs.Sphere(
                pos=(-0.3, -0.3, 0),
                radius=0.1,
            ),
            surface=gs.surfaces.Default(),
            use_visual_raycasting=True,
        ),
    )
    duck = scene.add_entity(
        spec=gs.entities.Rigid(
            morph=gs.morphs.Mesh(
                file="meshes/duck/duck.obj",
                pos=(0.0, 0.0, 0.5),
                scale=0.001,
            ),
            surface=gs.surfaces.Default(),
            use_visual_raycasting=True,
        ),
    )
    boxes = []
    for i in range(6):
        angle = i * (2 * math.pi / 6)
        radius = 0.5 + i * 0.1
        box = scene.add_entity(
            spec=gs.entities.Rigid(
                morph=gs.morphs.Box(
                    pos=(radius * math.cos(angle), radius * math.sin(angle), 0.1 + i * 0.1),
                    size=(0.2, 0.2, 0.2),
                ),
                surface=gs.surfaces.Default(),
                use_visual_raycasting=True,
            ),
        )
        boxes.append(box)

    camera = None
    if args.record:
        camera = scene.add_camera(
            res=(1280, 720),
            pos=camera_pos,
            lookat=camera_lookat,
            fov=40,
        )

    scene.viewer.add_plugin(
        gs.vis.viewer_plugins.MouseInteractionPlugin(
            use_force=args.use_force,
            color=(0.1, 0.6, 0.8, 0.6),
            use_visual_geom=args.use_visual_geom,
        )
    )

    scene.build(n_envs=args.num_envs)

    is_running = True

    def stop():
        nonlocal is_running
        is_running = False

    scene.viewer.register_keybinds(
        Keybind("quit", Key.ESCAPE, KeyAction.RELEASE, callback=stop),
    )
    gs.logger.info("Drag any object with the mouse, Esc quits.")

    if camera is not None:
        camera.start_recording(save_to_filename="out/mouse_interaction.mp4", fps=30)
    # The viewer is always open, so the scripted toss runs under the test suite alone, which cannot drag anything.
    if "PYTEST_VERSION" in os.environ:
        # Toss every object up and towards the centre, spinning about a horizontal axis.
        vel_up_spin = torch.tensor((4.0, 3.0, -2.0, 0.0), dtype=gs.tc_float, device=gs.device)
        for entity in (sphere, duck, *boxes):
            vel_inward = -1.5 * entity.get_pos()[..., :2]
            entity.set_dofs_velocity(torch.cat((vel_inward, vel_up_spin.expand(*vel_inward.shape[:-1], -1)), dim=-1))
        for _ in range(20):
            scene.step()
    else:
        while is_running and scene.viewer.is_alive():
            scene.step()
    if camera is not None:
        camera.stop_recording()


if __name__ == "__main__":
    main()
