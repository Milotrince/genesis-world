"""Three PBD cloth flags pinned to their poles flap in wind and turbulence, then a point attractor pulls them in.

Force fields act on every particle of the scene as an acceleration. The scripted run turns them on one after the other:
a steady wind first, then Perlin-noise turbulence on top of it, and finally the wind gives way to a point attractor in
front of the poles, which gathers the three flags towards it.

Keyboard controls (with -v, once the scripted run is over):
1 - Toggle the wind
2 - Toggle the turbulence
3 - Toggle the point attractor

Plus all default viewer controls (press 'i' to see them).
"""

import argparse
import os

import numpy as np

import genesis as gs
from genesis.vis.keybindings import Key, KeyAction, Keybind


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", help="Show the interactive viewer")
    parser.add_argument("-g", "--gpu", action="store_true", help="Run on GPU instead of CPU")
    parser.add_argument("-r", "--record", action="store_true", help="Record the run to 'out/pbd_flags_in_wind.mp4'")
    args = parser.parse_args()

    gs.init(backend=gs.gpu if args.gpu else gs.cpu)

    particle_size = 2e-2
    pole_height = 1.8
    pole_radius = 0.02
    flag_size = (0.6, 0.4)
    flags_y = (-0.8, 0.0, 0.8)

    camera_pos = (1.2, -2.6, 2.2)
    camera_lookat = (0.5, 0.0, 1.3)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=4e-3,
            substeps=10,
        ),
        pbd_options=gs.options.PBDOptions(
            particle_size=particle_size,
            # The neighbour hash grid repeats in space, so a one-metre cube covers a flag. The default grid, a few metres
            # wide, costs more to sweep at every substep than the cloth itself.
            hash_grid_res=(1.0, 1.0, 1.0),
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=camera_pos,
            camera_lookat=camera_lookat,
            camera_fov=40,
        ),
        show_viewer=args.vis,
    )

    scene.add_entity(
        gs.morphs.Plane(),
    )
    flags = []
    for flag_y, color in zip(flags_y, ((0.8, 0.2, 0.2), (0.95, 0.85, 0.2), (0.2, 0.4, 0.8))):
        scene.add_entity(
            gs.morphs.Cylinder(
                pos=(0.0, flag_y, 0.5 * pole_height),
                radius=pole_radius,
                height=pole_height,
                fixed=True,
            ),
            surface=gs.surfaces.Default(
                color=(0.6, 0.6, 0.6),
            ),
        )
        # The cloth mesh is a half-metre square in the x-y plane, stretched to the flag size and turned upright in the
        # x-z plane. Its pinned edge stays one particle clear of the pole, which would otherwise push it out.
        flags.append(
            scene.add_entity(
                morph=gs.morphs.Mesh(
                    file="meshes/cloth.obj",
                    scale=(2.0 * flag_size[0], 2.0 * flag_size[1], 1.0),
                    pos=(
                        pole_radius + particle_size + 0.5 * flag_size[0],
                        flag_y,
                        pole_height - 0.5 * flag_size[1] - 0.05,
                    ),
                    euler=(90.0, 0.0, 0.0),
                ),
                material=gs.materials.PBD.Cloth(),
                surface=gs.surfaces.Default(
                    color=color,
                ),
            )
        )

    wind = scene.add_force_field(
        gs.force_fields.Wind(
            direction=(1.0, 0.0, 0.0),
            strength=8.0,
            radius=3.0,
            center=(0.0, 0.0, pole_height),
        )
    )
    turbulence = scene.add_force_field(
        gs.force_fields.Turbulence(
            strength=30.0,
            frequency=5,
            seed=0,
        )
    )
    # A point field pushes the particles away from its position along the radius, so a negative strength pulls them in.
    attractor = scene.add_force_field(
        gs.force_fields.Point(
            strength=-15.0,
            position=(1.0, 0.0, 1.2),
        )
    )
    turbulence.deactivate()
    attractor.deactivate()

    camera = None
    if args.record:
        camera = scene.add_camera(
            res=(1280, 720),
            pos=camera_pos,
            lookat=camera_lookat,
            fov=40,
        )

    scene.build()

    # Pin the column of particles along the pole side of each flag.
    for flag in flags:
        flag_particles_x = flag.init_particles[:, 0]
        flag.fix_particles(np.flatnonzero(flag_particles_x < flag_particles_x.min() + 0.5 * flag.particle_size))

    def toggle(force_field):
        if force_field.active:
            force_field.deactivate()
        else:
            force_field.activate()

    if camera is not None:
        camera.start_recording(save_to_filename="out/pbd_flags_in_wind.mp4", fps=30)
    n_wind, n_turbulence, n_attractor = (500, 750, 750) if "PYTEST_VERSION" not in os.environ else (10, 10, 10)
    for _ in range(n_wind):
        scene.step()
    toggle(turbulence)
    for _ in range(n_turbulence):
        scene.step()
    toggle(wind)
    toggle(attractor)
    for _ in range(n_attractor):
        scene.step()
    if camera is not None:
        camera.stop_recording()

    if args.vis:
        scene.viewer.register_keybinds(
            Keybind("toggle_wind", Key._1, KeyAction.PRESS, callback=toggle, args=(wind,)),
            Keybind("toggle_turbulence", Key._2, KeyAction.PRESS, callback=toggle, args=(turbulence,)),
            Keybind("toggle_attractor", Key._3, KeyAction.PRESS, callback=toggle, args=(attractor,)),
        )
        gs.logger.info("Keys: 1 toggles the wind, 2 the turbulence, 3 the point attractor.")
    while args.vis and scene.viewer.is_alive() and "PYTEST_VERSION" not in os.environ:
        scene.step()


if __name__ == "__main__":
    main()
