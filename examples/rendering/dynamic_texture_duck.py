"""A duck switching between the images of a dynamic texture, and drawn on, at runtime.

In parallel environments, the duck of environment i starts on image i. Without the viewer, the textures cycle on their
own. With it, the keyboard sets them in every environment:
Right   - Next texture
Left    - Previous texture
Space   - Draw a random color per environment
Esc     - Quit
"""

import argparse
import math
import os

import numpy as np

import genesis as gs
from genesis.vis.keybindings import Key, KeyAction, Keybind

TEXTURE_RES = (256, 256)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", help="Show visualization GUI")
    parser.add_argument("-n", "--n_envs", type=int, default=4, help="Number of parallel environments")
    args = parser.parse_args()

    gs.init(backend=gs.cpu)

    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.95, -0.65, 0.6),
            camera_lookat=(0.15, 0.15, 0.05),
            camera_fov=40,
        ),
        show_viewer=args.vis,
    )
    scene.add_entity(
        gs.morphs.Plane(),
    )
    # The images to switch between, named and each the size of the dynamic texture: the texture of the asset, a
    # checkerboard, vertical stripes and a flat color
    width, height = TEXTURE_RES
    cols, rows = np.arange(width) // 32, np.arange(height)[:, None] // 32
    is_checker_dark = (rows + cols) % 2 == 0
    is_stripe = np.broadcast_to(cols % 2 == 0, (height, width))
    images = {
        "duck": gs.textures.AssetTexture(),
        "checker": gs.textures.ImageTexture(
            image_array=np.where(is_checker_dark[..., None], [240, 140, 30], [250, 240, 230]).astype(np.uint8),
        ),
        "stripes": gs.textures.ImageTexture(
            image_array=np.where(is_stripe[..., None], [230, 60, 60], [250, 240, 230]).astype(np.uint8),
        ),
        "blue": gs.textures.ColorTexture(
            color=(0.24, 0.47, 0.86),
        ),
    }
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
                images=images,
                canvas="per_env",
            ),
        ),
    )
    scene.build(n_envs=args.n_envs, env_spacing=(0.3, 0.3))

    images_name = tuple(images)

    # The image each environment shows is offset by the index of the environment
    envs_offset = np.arange(max(args.n_envs, 1))
    i_image = 0
    i_image_shown = None
    is_drawing = False
    is_running = True

    # Keybind callbacks run on the viewer thread, so they only pick what the main loop sets between steps
    def swap(offset):
        nonlocal i_image, i_image_shown
        i_image = (i_image + offset) % len(images_name)
        i_image_shown = None

    def draw():
        nonlocal is_drawing
        is_drawing = True

    def stop():
        nonlocal is_running
        is_running = False

    if args.vis:
        scene.viewer.register_keybinds(
            Keybind("next_texture", Key.RIGHT, KeyAction.PRESS, callback=swap, args=(1,)),
            Keybind("previous_texture", Key.LEFT, KeyAction.PRESS, callback=swap, args=(-1,)),
            Keybind("draw_texture", Key.SPACE, KeyAction.PRESS, callback=draw),
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
            duck.set_texture((i_image + envs_offset) % len(images_name))
            i_image_shown = i_image
        if is_drawing:
            duck.draw_texture(np.random.rand(len(envs_offset), 1, 1, 3))
            is_drawing = False
        scene.step()
        step += 1


if __name__ == "__main__":
    main()
