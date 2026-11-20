"""A Franka slashes MPM fruit falling in front of it with a blade, cutting each fruit in two.

The fruit is an elasto-plastic MPM material, and the blade is a flat rigid ellipsoid held in the hand. The blade is
much thinner than an MPM grid cell, so the solver runs with CPIC (compatible particle-in-cell), which keeps the
particles on either side of a thin rigid geom from interacting through the cells it crosses. The arm whips its base
joint, so the blade crosses the fruit at about 3 m/s: a slower stroke pushes the fruit away whole. The fruits drop one
or two at a time onto the circle the blade sweeps, and the run logs how many pieces each fruit is cut into.

Keyboard controls (-v), once the scripted rounds are over:
Left/Right - Whip the blade to the left/right
Up/Down    - Raise/lower the blade
T          - Drop the next fruit, while any is left
Esc        - Quit
"""

import argparse
import os
import xml.etree.ElementTree as ET

import numpy as np
from scipy import ndimage

import genesis as gs
from genesis.utils.misc import tensor_to_array
from genesis.vis.keybindings import Key, KeyAction, Keybind


DT = 4e-3
GRAVITY = 9.81
GRID_DENSITY = 64
PARTICLE_SIZE = 0.008
# Half-width, half-thickness and length of the blade, and the distance from the hand down to its top.
BLADE_HALF_WIDTH = 0.035
BLADE_HALF_THICKNESS = 0.006
BLADE_LENGTH = 0.26
BLADE_DEPTH = 0.06
# Radius of the circle the hand sweeps around the base, and the angle of the base joint on either side of the middle.
SWING_RADIUS = 0.6
SWING_ANGLE = 0.5
# Steps from the swing command until the blade passes the middle of its stroke, and steps per radian of the base joint
# around the middle, measured on this arm.
SWING_MIDDLE_STEPS = 43
SWING_STEPS_PER_RAD = 47
# Height the fruits drop from, and height where they meet the blade.
DROP_HEIGHT = 0.68
CUT_HEIGHT = 0.42
# Steps after the crossing during which the pieces are counted, in flight, before they land against each other.
COUNT_WINDOW_STEPS = 30
# Name, radius, color and azimuth on the swing circle of each fruit. The scripted rounds drop the first four, each at
# its own azimuth, so that a fruit never falls onto the pieces of another one. Each of them meets the blade before the
# middle of the stroke or just after it, since the base joint slows down over the second half and pushes the fruit away.
FRUITS = (
    ("apple", 0.045, (0.8, 0.08, 0.08), -0.2),
    ("orange", 0.05, (1.0, 0.5, 0.05), 0.2),
    ("lemon", 0.042, (0.95, 0.85, 0.15), -0.05),
    ("lime", 0.04, (0.35, 0.75, 0.15), 0.1),
    ("plum", 0.04, (0.45, 0.1, 0.4), 0.0),
    ("peach", 0.045, (1.0, 0.65, 0.45), -0.12),
    ("apricot", 0.04, (1.0, 0.6, 0.2), 0.12),
    ("melon", 0.055, (0.6, 0.85, 0.4), 0.05),
)


def blade_mjcf():
    """MJCF model of the blade, in the frame of the hand that holds it: the blade hangs down the z axis of the hand, its
    flat faces normal to the y axis, between the fingers."""
    mjcf = ET.Element("mujoco", model="blade")
    blade = ET.SubElement(ET.SubElement(mjcf, "worldbody"), "body", name="blade")
    # A flat ellipsoid has a sharp edge on both sides and thickens behind it, so it wedges the halves apart.
    ET.SubElement(
        blade,
        "geom",
        type="ellipsoid",
        pos=f"0 0 {BLADE_DEPTH + 0.5 * BLADE_LENGTH}",
        size=f"{BLADE_HALF_WIDTH} {BLADE_HALF_THICKNESS} {0.5 * BLADE_LENGTH}",
        density="7800",
        rgba="0.8 0.82 0.85 1",
    )
    return ET.tostring(mjcf, encoding="unicode")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", help="Show the interactive viewer")
    parser.add_argument("-g", "--gpu", action="store_true", help="Run on GPU instead of CPU")
    parser.add_argument("-r", "--record", action="store_true", help="Record the run to 'out/rigid_mpm_fruit_ninja.mp4'")
    args = parser.parse_args()

    gs.init(backend=gs.gpu if args.gpu else gs.cpu)

    camera_pos = (1.45, -1.1, 0.8)
    camera_lookat = (0.6, 0.0, 0.3)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=DT,
            substeps=13,
            gravity=(0.0, 0.0, -GRAVITY),
        ),
        mpm_options=gs.options.MPMOptions(
            grid_density=GRID_DENSITY,
            particle_size=PARTICLE_SIZE,
            enable_CPIC=True,
            lower_bound=(0.3, -0.45, -0.06),
            upper_bound=(1.0, 0.45, 0.8),
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
    fruits = []
    for _, radius, color, azimuth in FRUITS:
        fruit = scene.add_entity(
            spec=gs.entities.MPM(
                morph=gs.morphs.Sphere(
                    pos=(SWING_RADIUS * np.cos(azimuth), SWING_RADIUS * np.sin(azimuth), DROP_HEIGHT),
                    radius=radius,
                ),
                material=gs.materials.MPM.ElastoPlastic(
                    E=1e5,
                    nu=0.3,
                    rho=900.0,
                    von_mises_yield_stress=6000.0,
                ),
                surface=gs.surfaces.Rough(
                    color=color,
                ),
                vis=gs.options.EntityVisOptions(
                    debug=True,
                ),
            ),
        )
        fruits.append(fruit)
    # CPIC tests every particle against every coupled rigid geom at each substep, and the arm stays clear of the fruit,
    # so only the blade and the floor are coupled to the MPM solver.
    franka = scene.add_entity(
        morph=gs.morphs.MJCF(
            file="xml/franka_emika_panda/panda.xml",
        ),
        material=gs.materials.Rigid(
            needs_coup=False,
            gravity_compensation=1.0,
        ),
    )
    # The blade has no joint of its own, so it moves with the hand only once its vertices follow the hand.
    blade = scene.add_entity(
        morph=gs.morphs.MJCF(
            file=blade_mjcf(),
            batch_fixed_verts=True,
        ),
        # CPIC tells the sides of the blade apart from its signed distance field, which resolves the sharp edge only
        # on a grid finer than the default.
        material=gs.materials.Rigid(
            sdf_min_res=96,
        ),
    )
    blade.attach(franka, "hand", pos=(0.0, 0.0, 0.0))

    camera = None
    if args.record:
        camera = scene.add_camera(
            res=(1280, 720),
            pos=camera_pos,
            lookat=camera_lookat,
            fov=40,
            debug=True,
        )

    scene.build()

    # The fruits wait inactive, hence hidden, at the place they drop from.
    for fruit in fruits:
        fruit.deactivate()

    hand = franka.get_link("hand")
    # The finger actuator of panda.xml takes its target in [0, 255]. Its own gains, set here, take it in metres.
    fingers_dof = (7, 8)
    franka.set_dofs_kp(100.0, fingers_dof)
    franka.set_dofs_kv(10.0, fingers_dof)
    # The hand points down with its x axis along world y, so the edge of the blade leads as the base joint turns.
    hand_quat = np.array([0.0, np.sqrt(0.5), np.sqrt(0.5), 0.0])
    # The middle of the blade sits below the height where the fruits meet it, since the blade narrows toward its tip,
    # and a fruit that the blade reaches late has fallen further.
    blade_height = CUT_HEIGHT - 0.06
    swing_side = -1.0
    fall_steps = round(np.sqrt(2.0 * (DROP_HEIGHT - CUT_HEIGHT) / GRAVITY) / DT)

    def aim(side):
        nonlocal swing_side
        swing_side = side
        z_hand = blade_height + BLADE_DEPTH + 0.5 * BLADE_LENGTH
        q = tensor_to_array(franka.inverse_kinematics(link=hand, pos=(SWING_RADIUS, 0.0, z_hand), quat=hand_quat))
        q[0] += side * SWING_ANGLE
        q[7:] = 0.02
        # The target jumps to the end of the stroke, so the joints accelerate at their torque limits.
        franka.control_dofs_position(q)
        return q

    def count_pieces(fruit):
        # Pieces are the connected components of the cells holding particles, ignoring crumbs. The cells are just
        # coarser than the particle spacing, so that a whole fruit leaves no empty cell inside.
        cells = np.floor(tensor_to_array(fruit.get_particles_pos()) / (1.25 * PARTICLE_SIZE)).astype(gs.np_int)
        cells -= cells.min(axis=0)
        is_occupied = np.zeros(cells.max(axis=0) + 1, dtype=bool)
        is_occupied[tuple(cells.T)] = True
        labels, _ = ndimage.label(is_occupied)
        pieces_size = np.bincount(labels[tuple(cells.T)])[1:]
        return np.count_nonzero(pieces_size > 0.1 * fruit.n_particles)

    franka.set_qpos(aim(swing_side))

    if camera is not None:
        camera.start_recording(save_to_filename="out/rigid_mpm_fruit_ninja.mp4", fps=round(1.0 / DT))
    # Each round whips the blade to the other side, and drops each of its fruits so that the fruit falls through the
    # cut height as the blade reaches its azimuth.
    n_dropped = 0
    n_cut = 0
    fruits_n_pieces = np.zeros(len(fruits), dtype=gs.np_int)
    for i_round, fruits_idx in enumerate(((0,), (1,), (2, 3))):
        side = -swing_side
        crossings_step = [round(SWING_MIDDLE_STEPS + side * FRUITS[i][3] * SWING_STEPS_PER_RAD) for i in fruits_idx]
        whip_step = fall_steps - min(crossings_step)
        n_steps = whip_step + max(crossings_step) + COUNT_WINDOW_STEPS + 80
        for i_step in range(3 if "PYTEST_VERSION" in os.environ else n_steps):
            if i_step == whip_step:
                aim(side)
            for i_fruit, crossing_step in zip(fruits_idx, crossings_step):
                if i_step == whip_step + crossing_step - fall_steps:
                    fruits[i_fruit].activate()
                    n_dropped += 1
                if 0 <= i_step - whip_step - crossing_step <= COUNT_WINDOW_STEPS:
                    fruits_n_pieces[i_fruit] = max(fruits_n_pieces[i_fruit], count_pieces(fruits[i_fruit]))
                if i_step == whip_step + crossing_step + COUNT_WINDOW_STEPS:
                    n_cut += fruits_n_pieces[i_fruit] >= 2
                    n_pieces = fruits_n_pieces[i_fruit]
                    gs.logger.info(f"Round {i_round + 1}: the {FRUITS[i_fruit][0]} is cut into {n_pieces} piece(s).")
            scene.step()
    gs.logger.info(f"Cut {n_cut} of {n_dropped} dropped fruits in two.")
    if camera is not None:
        camera.stop_recording()

    if args.vis:
        is_running = True

        def drop_next():
            nonlocal n_dropped
            if n_dropped == len(fruits):
                gs.logger.info("No fruit left.")
                return
            fruits[n_dropped].activate()
            n_dropped += 1

        def raise_blade(dz):
            nonlocal blade_height
            blade_height += dz
            aim(swing_side)

        def stop():
            nonlocal is_running
            is_running = False

        scene.viewer.register_keybinds(
            Keybind("whip_left", Key.LEFT, KeyAction.PRESS, callback=aim, args=(-1.0,)),
            Keybind("whip_right", Key.RIGHT, KeyAction.PRESS, callback=aim, args=(1.0,)),
            Keybind("raise_blade", Key.UP, KeyAction.PRESS, callback=raise_blade, args=(0.02,)),
            Keybind("lower_blade", Key.DOWN, KeyAction.PRESS, callback=raise_blade, args=(-0.02,)),
            Keybind("drop", Key.T, KeyAction.PRESS, callback=drop_next),
            Keybind("quit", Key.ESCAPE, KeyAction.RELEASE, callback=stop),
        )
        gs.logger.info("Keys: Left/Right whip the blade, Up/Down raise/lower it, T drops a fruit, Esc quits.")
        while is_running and scene.viewer.is_alive() and "PYTEST_VERSION" not in os.environ:
            scene.step()


if __name__ == "__main__":
    main()
