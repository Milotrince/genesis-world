import argparse
import os

import numpy as np

import genesis as gs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", help="Show visualization GUI")
    args = parser.parse_args()

    gs.init(backend=gs.cpu)

    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(2.5, 0.0, 1.0),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=40,
        ),
        vis_options=gs.options.VisOptions(
            split_envs=True,
        ),
        show_viewer=args.vis,
    )
    scene.add_entity(
        gs.morphs.Plane(),
    )
    # A screen facing the camera, whose emissive texture shows a progress bar of its own in each environment. A plane
    # repeats its texture every 'tile_size', which the size of the plane spans once.
    screen = scene.add_entity(
        morph=gs.morphs.Plane(
            pos=(0.0, 0.0, 0.5),
            normal=(1.0, 0.0, 0.0),
            plane_size=(0.8, 0.5),
            tile_size=(0.8, 0.5),
            fixed=True,
        ),
        surface=gs.surfaces.Emission(
            emissive_texture=gs.textures.DynamicTexture(
                res=(64, 16),
                initial=gs.textures.ColorTexture(
                    color=(0.1, 0.1, 0.1),
                ),
            ),
        ),
    )
    cam = scene.add_camera(
        res=(640, 480),
        pos=(2.5, 0.0, 1.0),
        lookat=(0.0, 0.0, 0.5),
        fov=40,
        GUI=args.vis,
    )
    scene.build(n_envs=2)

    texture = screen.get_dynamic_texture(gs.TEXTURE_SLOT.EMISSIVE)
    width, height = texture.res
    envs_speed = np.array([0.5, 1.0])
    horizon = 200 if "PYTEST_VERSION" not in os.environ else 5
    for i in range(horizon):
        scene.step()
        # The bar of each environment fills at its own speed, one image per environment
        envs_progress = (i * envs_speed / horizon) % 1.0
        is_filled = np.arange(width) < envs_progress[:, None] * width
        image = np.full((len(envs_speed), height, width, 3), 25, dtype=np.uint8)
        image[..., 1] = np.where(is_filled[:, None, :], 230, 25)
        texture.set_image(image)
        cam.render(rgb=True)


if __name__ == "__main__":
    main()
