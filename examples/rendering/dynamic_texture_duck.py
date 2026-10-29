"""A duck whose texture is swapped at runtime through a dynamic texture.

Without the viewer, the textures cycle on their own. With it, the keyboard swaps them:
Right   - Next texture
Left    - Previous texture
Esc     - Quit
"""

import argparse
import math
import os

import numpy as np
from PIL import Image

import genesis as gs
from genesis.utils.misc import get_assets_dir
from genesis.vis.keybindings import Key, KeyAction, Keybind

TEXTURE_RES = (256, 256)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", help="Show visualization GUI")
    args = parser.parse_args()

    gs.init(backend=gs.cpu)

    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.35, -0.35, 0.25),
            camera_lookat=(0.0, 0.0, 0.07),
            camera_fov=40,
        ),
        show_viewer=args.vis,
    )
    scene.add_entity(
        gs.morphs.Plane(),
    )
    duck = scene.add_entity(
        morph=gs.morphs.Mesh(
            file="meshes/duck/duck.obj",
            scale=0.0005,
            pos=(0.0, 0.0, 0.04),
            euler=(90.0, 0.0, 90.0),
            fixed=True,
        ),
        surface=gs.surfaces.Default(
            diffuse_texture=gs.textures.DynamicTexture(
                res=TEXTURE_RES,
                initial=gs.textures.ImageTexture(
                    image_path="meshes/duck/material_0.png",
                ),
            ),
        ),
    )
    scene.build()

    # The images to swap between, each the size of the dynamic texture: the texture of the asset, a checkerboard,
    # vertical stripes and a flat color
    width, height = TEXTURE_RES
    image_path = os.path.join(get_assets_dir(), "meshes/duck/material_0.png")
    cols, rows = np.arange(width) // 32, np.arange(height)[:, None] // 32
    is_checker_dark = (rows + cols) % 2 == 0
    is_stripe = np.broadcast_to(cols % 2 == 0, (height, width))
    images = [
        np.asarray(Image.open(image_path).convert("RGB").resize(TEXTURE_RES)),
        np.where(is_checker_dark[..., None], [240, 140, 30], [250, 240, 230]).astype(np.uint8),
        np.where(is_stripe[..., None], [230, 60, 60], [250, 240, 230]).astype(np.uint8),
        np.full((height, width, 3), [60, 120, 220], dtype=np.uint8),
    ]
    texture = duck.get_dynamic_texture(gs.TEXTURE_SLOT.DIFFUSE)

    i_image = 0
    i_image_shown = 0
    is_running = True

    # Keybind callbacks run on the viewer thread, so they only pick the image that the main loop writes between steps
    def swap(offset):
        nonlocal i_image
        i_image = (i_image + offset) % len(images)

    def stop():
        nonlocal is_running
        is_running = False

    if args.vis:
        scene.viewer.register_keybinds(
            Keybind("next_texture", Key.RIGHT, KeyAction.PRESS, callback=swap, args=(1,)),
            Keybind("previous_texture", Key.LEFT, KeyAction.PRESS, callback=swap, args=(-1,)),
            Keybind("quit", Key.ESCAPE, KeyAction.RELEASE, callback=stop),
        )
        horizon = math.inf
    else:
        horizon = 400 if "PYTEST_VERSION" not in os.environ else 5

    step = 0
    while is_running and step < horizon:
        if not args.vis and step % 100 == 99:
            swap(1)
        if i_image != i_image_shown:
            texture.set_image(images[i_image])
            i_image_shown = i_image
        scene.step()
        step += 1


if __name__ == "__main__":
    main()
