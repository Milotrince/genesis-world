"""An MPM rubber duck floats in a bathtub of SPH water, rides the stream of the faucet, then circles the drain.

The water is a smoothed particle hydrodynamics (SPH) liquid and the duck a light elastic material point method (MPM)
solid, so the water carries the duck through the coupling of the two solvers. The tub starts part full and the faucet
tops it up. Pulling the plug deactivates the water particles that reach the drain, at a set flow rate, and turns on a
vortex force field that swirls the water, the duck and a ring of foam balls around it. The water level and the
position of the duck are logged along the way. Particle scenes are slow on CPU, so '-g' speeds the run up a lot.

Keyboard controls (with -v, once the scripted run is over):
1 - Toggle the faucet
2 - Pull or replace the plug

Plus all default viewer controls (press 'i' to see them).
"""

import argparse
import os
import xml.etree.ElementTree as ET

import numpy as np
import torch

import genesis as gs
from genesis.vis.keybindings import Key, KeyAction, Keybind


# Inner size of the tub (x, y, z) and the thickness of its walls. The tub stands on the ground, its floor on top.
TUB_SIZE = (0.4, 0.25, 0.12)
WALL = 0.01
FILL_HEIGHT = 0.07
# The faucet pours at one end of the tub and the drain sits in the floor at the other end.
FAUCET_POS = (-0.15, 0.0, 0.17)
DRAIN_POS = (0.14, 0.0, WALL)
DRAIN_RADIUS = 0.035
# The drain swallows at most this many particles per step, which sets its flow rate. The water carries the duck along
# its velocity, so a faster drain drags the duck under as it passes over the drain.
DRAIN_RATE = 3


def tub_mjcf():
    """MJCF model of a fixed open tub with a rolled rim, and a drain grate marking the drain on its floor."""
    size_x, size_y, size_z = TUB_SIZE
    mjcf = ET.Element("mujoco", model="tub")
    body = ET.SubElement(ET.SubElement(mjcf, "worldbody"), "body", name="tub")
    outer_x, outer_y = 0.5 * size_x + WALL, 0.5 * size_y + WALL
    wall_z = 0.5 * (size_z + WALL)
    for pos, half_size, rgba in (
        ((0.0, 0.0, 0.5 * WALL), (outer_x, outer_y, 0.5 * WALL), "0.78 0.86 0.9 1"),
        ((0.5 * (size_x + WALL), 0.0, wall_z), (0.5 * WALL, outer_y, wall_z), "0.97 0.97 0.95 1"),
        ((-0.5 * (size_x + WALL), 0.0, wall_z), (0.5 * WALL, outer_y, wall_z), "0.97 0.97 0.95 1"),
        ((0.0, 0.5 * (size_y + WALL), wall_z), (0.5 * size_x, 0.5 * WALL, wall_z), "0.97 0.97 0.95 1"),
        ((0.0, -0.5 * (size_y + WALL), wall_z), (0.5 * size_x, 0.5 * WALL, wall_z), "0.97 0.97 0.95 1"),
    ):
        ET.SubElement(
            body,
            "geom",
            type="box",
            pos=" ".join(f"{coord:.6g}" for coord in pos),
            size=" ".join(f"{coord:.6g}" for coord in half_size),
            rgba=rgba,
        )
    # The rim, the grate and the hole of the drain are drawn alone, and take no part in the contacts.
    rim_x, rim_y, rim_z = 0.5 * (size_x + WALL), 0.5 * (size_y + WALL), size_z + WALL
    for from_point, to_point in (
        ((-rim_x, -rim_y), (rim_x, -rim_y)),
        ((rim_x, -rim_y), (rim_x, rim_y)),
        ((rim_x, rim_y), (-rim_x, rim_y)),
        ((-rim_x, rim_y), (-rim_x, -rim_y)),
    ):
        ET.SubElement(
            body,
            "geom",
            type="capsule",
            fromto=f"{from_point[0]} {from_point[1]} {rim_z} {to_point[0]} {to_point[1]} {rim_z}",
            size=f"{WALL}",
            rgba="0.97 0.97 0.95 1",
            contype="0",
            conaffinity="0",
        )
    for radius, rgba in ((DRAIN_RADIUS, "0.75 0.75 0.78 1"), (0.7 * DRAIN_RADIUS, "0.05 0.05 0.06 1")):
        ET.SubElement(
            body,
            "geom",
            type="cylinder",
            pos=f"{DRAIN_POS[0]} {DRAIN_POS[1]} {WALL}",
            size=f"{radius} {0.001 + 0.2 * (DRAIN_RADIUS - radius)}",
            rgba=rgba,
            contype="0",
            conaffinity="0",
        )
    return ET.tostring(mjcf, encoding="unicode")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", help="Show the interactive viewer")
    parser.add_argument("-g", "--gpu", action="store_true", help="Run on GPU instead of CPU")
    parser.add_argument("-r", "--record", action="store_true", help="Record the run to 'out/sph_mpm_bathtub.mp4'")
    args = parser.parse_args()

    gs.init(backend=gs.gpu if args.gpu else gs.cpu, precision="32", seed=0)

    camera_pos = (0.2, -0.38, 0.62)
    camera_lookat = (0.0, 0.0, 0.04)
    particle_size = 0.014
    size_x, size_y, _ = TUB_SIZE

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1e-2,
            substeps=32,
        ),
        sph_options=gs.options.SPHOptions(
            particle_size=particle_size,
            lower_bound=(-0.5 * size_x - WALL, -0.5 * size_y - WALL, 0.0),
            upper_bound=(0.5 * size_x + WALL, 0.5 * size_y + WALL, 0.4),
        ),
        # The MPM domain keeps a few cells of padding inside its bounds, which reach past the tub so that its walls
        # alone stop the duck.
        mpm_options=gs.options.MPMOptions(
            lower_bound=(-0.5 * size_x - 0.06, -0.5 * size_y - 0.06, -0.05),
            upper_bound=(0.5 * size_x + 0.06, 0.5 * size_y + 0.06, 0.4),
        ),
        vis_options=gs.options.VisOptions(
            # A light shining down from the side of the camera keeps the walls from shading the water and the duck.
            lights=[
                {"type": "directional", "dir": (0.3, 0.6, -1.5), "color": (1.0, 1.0, 1.0), "intensity": 5.0},
            ],
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
    scene.add_entity(
        gs.morphs.MJCF(
            file=tub_mjcf(),
        ),
    )
    # The spout is for show: the liquid leaves it through the emitter below.
    scene.add_entity(
        gs.morphs.Cylinder(
            pos=(FAUCET_POS[0] - 0.06, FAUCET_POS[1], FAUCET_POS[2] + 0.015),
            euler=(0.0, 90.0, 0.0),
            radius=0.015,
            height=0.14,
            fixed=True,
        ),
        material=gs.materials.Rigid(
            needs_coup=False,
        ),
        surface=gs.surfaces.Metal(
            color=(0.8, 0.8, 0.82),
        ),
    )
    water = scene.add_entity(
        material=gs.materials.SPH.Liquid(),
        # The water starts a few millimetres clear of the walls and the floor.
        morph=gs.morphs.Box(
            pos=(0.0, 0.0, WALL + 0.5 * FILL_HEIGHT + 0.005),
            size=(size_x - 0.01, size_y - 0.01, FILL_HEIGHT),
        ),
        surface=gs.surfaces.Default(
            color=(0.3, 0.6, 0.95, 0.55),
        ),
    )
    faucet = scene.add_emitter(
        material=gs.materials.SPH.Liquid(),
        max_particles=2000,
        surface=gs.surfaces.Default(
            color=(0.3, 0.6, 0.95, 0.55),
        ),
    )
    duck = scene.add_entity(
        morph=gs.morphs.Mesh(
            file="meshes/duck.obj",
            pos=(0.0, 0.04, WALL + FILL_HEIGHT + 0.06),
            scale=0.025,
            euler=(90.0, 0.0, 90.0),
        ),
        # A rubber duck is hollow, so it floats as a solid much lighter than water.
        material=gs.materials.MPM.Elastic(
            E=5e4,
            rho=200.0,
        ),
        surface=gs.surfaces.Default(
            color=(1.0, 0.85, 0.1, 1.0),
        ),
    )
    # Foam balls float in a ring around the drain, so that the swirl of the water shows.
    for angle in np.linspace(0.0, 2.0 * np.pi, num=6, endpoint=False):
        scene.add_entity(
            morph=gs.morphs.Sphere(
                pos=(DRAIN_POS[0] - 0.02 + 0.07 * np.cos(angle), 0.07 * np.sin(angle), WALL + FILL_HEIGHT + 0.015),
                radius=0.015,
            ),
            material=gs.materials.MPM.Elastic(
                E=5e4,
                rho=200.0,
            ),
            surface=gs.surfaces.Default(
                color=(1.0, 1.0, 1.0, 1.0),
            ),
        )
    # A vortex about the vertical axis through the drain, swirling counterclockwise and pulling inwards.
    vortex = scene.add_force_field(
        gs.force_fields.Vortex(
            direction=(0.0, 0.0, 1.0),
            center=DRAIN_POS,
            strength_perpendicular=6.0,
            strength_radial=3.0,
            falloff_min=DRAIN_RADIUS,
            falloff_max=0.2,
        )
    )
    vortex.deactivate()

    camera = None
    if args.record:
        camera = scene.add_camera(
            res=(1280, 720),
            pos=camera_pos,
            lookat=camera_lookat,
            fov=40,
        )

    scene.build()

    is_faucet_on = False
    drain_pos = torch.tensor(DRAIN_POS, dtype=gs.tc_float, device=gs.device)

    def toggle_faucet():
        nonlocal is_faucet_on
        is_faucet_on = not is_faucet_on

    def toggle_plug():
        if vortex.active:
            vortex.deactivate()
        else:
            vortex.activate()

    def step():
        """Step the scene, pouring from the faucet when it runs and draining the water at the drain when unplugged."""
        if is_faucet_on:
            faucet.emit(
                droplet_shape="circle",
                droplet_size=0.025,
                pos=FAUCET_POS,
                direction=(0.0, 0.0, -1.0),
                speed=3.0,
            )
        if vortex.active:
            for entity in (water, faucet.entity):
                particles_offset = entity.get_particles_pos() - drain_pos
                is_drained = entity.get_particles_active()
                is_drained &= particles_offset[:, :2].norm(dim=-1) < DRAIN_RADIUS
                is_drained &= particles_offset[:, 2] < 1.5 * particle_size
                entity.set_particles_active(gs.INACTIVE, is_drained.nonzero()[:DRAIN_RATE, 0])
        scene.step()

    if camera is not None:
        camera.start_recording(save_to_filename="out/sph_mpm_bathtub.mp4", fps=25)
    # The duck drops in and settles, the faucet tops the tub up and pushes the duck along, then the plug is pulled.
    n_settle, n_fill, n_drain = (60, 150, 300) if "PYTEST_VERSION" not in os.environ else (10, 10, 10)
    for i_step in range(n_settle + n_fill + n_drain):
        if i_step == n_settle:
            toggle_faucet()
        if i_step == n_settle + n_fill:
            toggle_faucet()
            toggle_plug()
        step()
        if i_step % 20 == 19:
            # The particles sample the water on a lattice of the particle size, which gives its volume and depth.
            n_water_particles = sum(entity.get_particles_active().sum().item() for entity in (water, faucet.entity))
            water_level = n_water_particles * particle_size**3 / (size_x * size_y)
            duck_pos = duck.get_particles_pos().mean(dim=0).tolist()
            gs.logger.info(
                f"t = {scene.get_time():.2f} s: water level {100.0 * water_level:.1f} cm ({n_water_particles} "
                f"particles), duck at ({duck_pos[0]:+.3f}, {duck_pos[1]:+.3f}, {duck_pos[2]:.3f}) m."
            )
    if camera is not None:
        camera.stop_recording()

    if args.vis:
        scene.viewer.register_keybinds(
            Keybind("toggle_faucet", Key._1, KeyAction.PRESS, callback=toggle_faucet),
            Keybind("toggle_plug", Key._2, KeyAction.PRESS, callback=toggle_plug),
        )
        gs.logger.info("Keys: 1 toggles the faucet, 2 pulls or replaces the plug.")
    while args.vis and scene.viewer.is_alive() and "PYTEST_VERSION" not in os.environ:
        step()


if __name__ == "__main__":
    main()
