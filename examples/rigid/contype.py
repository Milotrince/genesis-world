"""
NOTE: contype and conaffinity are 32-bit integer bitmasks used for contact filtering of contact pairs.
When the contype of one geom and the conaffinity of the other geom share a common bit set to 1, two geoms can collide.
Plane:      contype=0xFFFF, conaffinity=0xFFFF (1111 1111 1111 1111)
Red Cube:   contype=1, conaffinity=1 (0001) -> collide with Plane and Blue Cube
Green Cube: contype=2, conaffinity=2 (0010) -> collide with Plane and Blue Cube
Blue Cube:  contype=3, conaffinity=3 (0011) -> collide with Plane, Red Cube, and Green Cube
Dragon:     contype=4, conaffinity=4 (0100) -> collide with Plane only

With the viewer (-v), drag the cubes and the dragon into each other with the mouse to see which pairs collide, during
the drop and after it.
"""

import argparse
import os

import genesis as gs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", help="Show visualization GUI")
    parser.add_argument("-r", "--record", action="store_true", help="Record a video of the run to 'out/contype.mp4'")
    args = parser.parse_args()

    gs.init(backend=gs.cpu)

    camera_pos = (0.5, -1.2, 0.8)
    camera_lookat = (0.0, 0.0, 0.25)

    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=camera_pos,
            camera_lookat=camera_lookat,
            camera_fov=40,
        ),
        show_viewer=args.vis,
    )

    scene.add_entity(gs.morphs.Plane())

    scene.add_entity(
        gs.morphs.Box(
            pos=(0.025, 0, 0.5),
            quat=(0, 0, 0, 1),
            size=(0.1, 0.1, 0.1),
            contype=0b001,
            conaffinity=0b001,
        ),
        surface=gs.surfaces.Default(
            color=(1.0, 0.0, 0.0, 1.0),
        ),
    )
    scene.add_entity(
        gs.morphs.Box(
            pos=(-0.025, 0, 1.0),
            quat=(0, 0, 0, 1),
            size=(0.1, 0.1, 0.1),
            contype=0b010,
            conaffinity=0b010,
        ),
        surface=gs.surfaces.Default(
            color=(0.0, 1.0, 0.0, 1.0),
        ),
    )
    scene.add_entity(
        gs.morphs.Box(
            pos=(0.0, 0, 1.5),
            quat=(0, 0, 0, 1),
            size=(0.1, 0.1, 0.1),
            contype=0b011,
            conaffinity=0b011,
        ),
        surface=gs.surfaces.Default(
            color=(0.0, 0.0, 1.0, 1.0),
        ),
    )
    scene.add_entity(
        morph=gs.morphs.Mesh(
            file="meshes/dragon/dragon.obj",
            scale=0.004,
            euler=(0, 0, 90),
            pos=(-0.1, 0.0, 1.0),
            contype=0b100,
            conaffinity=0b100,
        ),
    )

    camera = None
    if args.record:
        camera = scene.add_camera(
            res=(1280, 720),
            pos=camera_pos,
            lookat=camera_lookat,
            fov=40,
        )

    if args.vis:
        scene.viewer.add_plugin(
            gs.vis.viewer_plugins.MouseInteractionPlugin(),
        )

    scene.build()
    if camera is not None:
        camera.start_recording(save_to_filename="out/contype.mp4", fps=round(1.0 / scene.dt))

    for _ in range(200 if "PYTEST_VERSION" not in os.environ else 90):
        scene.step()
    if camera is not None:
        camera.stop_recording()
    while args.vis and scene.viewer.is_alive() and "PYTEST_VERSION" not in os.environ:
        scene.step()


if __name__ == "__main__":
    main()
