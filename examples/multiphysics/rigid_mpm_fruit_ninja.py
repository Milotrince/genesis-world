"""A Franka slashes MPM fruit falling in front of it with a blade, cutting each fruit in two.

The fruit is an elasto-plastic MPM material, and the blade is a flat rigid ellipsoid that the hand holds level, like a
sword. The blade is much thinner than an MPM grid cell, so the solver runs with CPIC (compatible particle-in-cell),
which keeps the particles on either side of a thin rigid geom from interacting through the cells it crosses. The arm
whips its base joint, so the blade sweeps a horizontal ring at about 3 m/s and slices each fruit falling through it
across: a slower stroke pushes the fruit away whole. The blade cuts on its strokes in both directions, which alternate,
and the fruits drop one or two at a time. The run logs the pieces of each fruit. With the viewer (-v), the scripted
rounds are skipped: the blade rests level beside the drop zone, no fruit has fallen yet, and the keys drive it.

Keyboard controls (-v):
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
import genesis.utils.geom as gu
from genesis.utils.misc import tensor_to_array
from genesis.vis.keybindings import Key, KeyAction, Keybind


DT = 4e-3
GRAVITY = 9.81
GRID_DENSITY = 64
PARTICLE_SIZE = 0.008
# Half-width, half-thickness and length of the blade, the distance from the hand down to the blade, below the
# fingertips, and the length of the blade behind the axis of the hand.
BLADE_HALF_WIDTH = 0.035
BLADE_HALF_THICKNESS = 0.006
BLADE_LENGTH = 0.3
BLADE_DEPTH = 0.125
BLADE_BACK = 0.03
# Radius of the circle the hand sweeps around the base, radius where the fruits fall through the widest part of the
# blade, and the angle of the base joint on either side of the middle.
HAND_RADIUS = 0.45
SWING_RADIUS = HAND_RADIUS + 0.5 * BLADE_LENGTH - BLADE_BACK
SWING_ANGLE = 0.5
# Steps from the swing command until the blade passes the middle of its stroke, and steps per radian of the base joint
# around the middle, measured on this arm.
SWING_MIDDLE_STEPS = 33
SWING_STEPS_PER_RAD = 42
# Height the fruits drop from, and height where they meet the blade.
DROP_HEIGHT = 0.56
# Height where the fruits wait below the floor, out of sight, since the mesh of a fruit renders even while its particles
# are inactive.
PARK_HEIGHT = -0.15
CUT_HEIGHT = 0.42
# The fruit falls while the blade crosses it, so the hand rolls the blade about its length, its leading edge up, until
# its plane holds the motion of the blade relative to the fruit. A level blade cuts a steep slope through the fruit and
# carries the top part on its face.
BLADE_ROLL = np.arctan(np.sqrt(2.0 * GRAVITY * (DROP_HEIGHT - CUT_HEIGHT)) * SWING_STEPS_PER_RAD * DT / SWING_RADIUS)
# Steps after the crossing during which the pieces are counted, in flight, before they land against each other, steps
# from a whip until the hand rolls the blade for the reverse stroke, once the stroke is over, and steps after the last
# crossing of a round until the next round.
COUNT_WINDOW_STEPS = 30
READY_STEPS = 70
ROUND_END_STEPS = 80
# Name, radius, color and azimuth on the swing circle of each fruit. The scripted rounds drop the first six, each at
# its own azimuth, so that a fruit never falls onto the pieces of another one. Each of them meets the blade before the
# middle of its stroke or just after it, since the base joint slows down over the second half, where the blade pushes
# the fruit away whole.
FRUITS = (
    ("apple", 0.045, (0.8, 0.08, 0.08), -0.1),
    ("orange", 0.05, (1.0, 0.5, 0.05), 0.2),
    ("lemon", 0.042, (0.95, 0.85, 0.15), -0.2),
    ("lime", 0.04, (0.35, 0.75, 0.15), 0.0),
    ("plum", 0.04, (0.45, 0.1, 0.4), 0.25),
    ("peach", 0.045, (1.0, 0.65, 0.45), 0.08),
    ("apricot", 0.04, (1.0, 0.6, 0.2), -0.15),
    ("melon", 0.055, (0.6, 0.85, 0.4), 0.12),
)
# Direction of the stroke of each scripted round, and the fruits it cuts.
ROUNDS = (
    (1.0, (0,)),
    (-1.0, (1,)),
    (1.0, (2, 3)),
    (-1.0, (4, 5)),
)


def blade_mjcf():
    """MJCF model of the blade, in the frame of the hand that holds it.

    The blade lies flat below the fingertips along the y axis of the hand.
    """
    mjcf = ET.Element("mujoco", model="blade")
    blade = ET.SubElement(ET.SubElement(mjcf, "worldbody"), "body", name="blade")
    # A flat ellipsoid has a sharp edge on both sides and thickens behind it, so it wedges the halves apart.
    ET.SubElement(
        blade,
        "geom",
        type="ellipsoid",
        pos=f"0 {0.5 * BLADE_LENGTH - BLADE_BACK} {BLADE_DEPTH}",
        size=f"{BLADE_HALF_WIDTH} {0.5 * BLADE_LENGTH} {BLADE_HALF_THICKNESS}",
        density="2700",
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
            lower_bound=(0.3, -0.45, -0.26),
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
                    pos=(SWING_RADIUS * np.cos(azimuth), SWING_RADIUS * np.sin(azimuth), PARK_HEIGHT),
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
        )

    scene.build()

    for fruit in fruits:
        fruit.deactivate()

    hand = franka.get_link("hand")
    # The finger actuator of panda.xml takes its target in [0, 255]. Its own gains, set here, take it in metres.
    fingers_dof = (7, 8)
    franka.set_dofs_kp(100.0, fingers_dof)
    franka.set_dofs_kv(10.0, fingers_dof)
    # The hand points down with its y axis pointing outward from the base, so the blade points outward and its edge
    # leads as the base joint turns.
    hand_quat = np.array([0.0, np.sqrt(0.5), np.sqrt(0.5), 0.0])
    blade_height = CUT_HEIGHT
    swing_side = -1.0
    fall_steps = round(np.sqrt(2.0 * (DROP_HEIGHT - CUT_HEIGHT) / GRAVITY) / DT)

    def pose(rest_side, stroke_side):
        # The roll of the blade turns about the root of the blade, which keeps its place on the swing circle.
        quat = gu.transform_quat_by_quat(hand_quat, gu.rotvec_to_quat(np.array([stroke_side * BLADE_ROLL, 0.0, 0.0])))
        pos = np.array([HAND_RADIUS, 0.0, blade_height]) - gu.transform_by_quat(np.array([0.0, 0.0, BLADE_DEPTH]), quat)
        q = tensor_to_array(franka.inverse_kinematics(link=hand, pos=pos, quat=quat))
        q[0] += rest_side * SWING_ANGLE
        q[7:] = 0.02
        franka.control_dofs_position(q)
        return q

    def aim(side):
        nonlocal swing_side
        swing_side = side
        # The target jumps to the end of the stroke, so the joints accelerate at their torque limits. The blade has
        # already rolled for this stroke, so the stroke turns the base joint alone.
        pose(side, side)

    def ready():
        # The hand rolls the blade for the reverse stroke while the arm rests, since the wrist lags behind a roll
        # commanded during a whip.
        pose(swing_side, -swing_side)

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

    franka.set_qpos(pose(swing_side, -swing_side))

    if camera is not None:
        camera.start_recording(save_to_filename="out/rigid_mpm_fruit_ninja.mp4", fps=round(1.0 / DT))
    n_dropped = 0

    def drop(fruit):
        nonlocal n_dropped
        fruit.activate()
        fruit.set_position((*fruit.morph.pos[:2], DROP_HEIGHT))
        n_dropped += 1

    if args.vis:
        is_running = True

        def drop_next():
            if n_dropped == len(fruits):
                gs.logger.info("No fruit left.")
                return
            drop(fruits[n_dropped])

        # Steps left until the hand rolls the blade for the reverse stroke, negative once it has.
        ready_countdown = -1

        def whip(side):
            nonlocal ready_countdown
            aim(side)
            ready_countdown = READY_STEPS

        def raise_blade(dz):
            nonlocal blade_height
            blade_height += dz
            ready()

        def stop():
            nonlocal is_running
            is_running = False

        scene.viewer.register_keybinds(
            Keybind("whip_left", Key.LEFT, KeyAction.PRESS, callback=whip, args=(-1.0,)),
            Keybind("whip_right", Key.RIGHT, KeyAction.PRESS, callback=whip, args=(1.0,)),
            Keybind("raise_blade", Key.UP, KeyAction.PRESS, callback=raise_blade, args=(0.02,)),
            Keybind("lower_blade", Key.DOWN, KeyAction.PRESS, callback=raise_blade, args=(-0.02,)),
            Keybind("drop", Key.T, KeyAction.PRESS, callback=drop_next),
            Keybind("quit", Key.ESCAPE, KeyAction.RELEASE, callback=stop),
        )
        gs.logger.info("Keys: Left/Right whip the blade, Up/Down raise/lower it, T drops a fruit, Esc quits.")
        while is_running and scene.viewer.is_alive():
            if ready_countdown == 0:
                ready()
            ready_countdown -= 1
            scene.step()
    else:
        # Each round whips the blade the other way, and drops each of its fruits so that the fruit falls through the cut
        # height as the blade reaches its azimuth.
        n_cut = 0
        fruits_n_pieces = np.zeros(len(fruits), dtype=gs.np_int)
        for i_round, (side, fruits_idx) in enumerate(ROUNDS):
            crossings_step = [round(SWING_MIDDLE_STEPS + side * FRUITS[i][3] * SWING_STEPS_PER_RAD) for i in fruits_idx]
            whip_step = fall_steps - min(crossings_step)
            n_steps = whip_step + max(crossings_step) + ROUND_END_STEPS
            for i_step in range(3 if "PYTEST_VERSION" in os.environ else n_steps):
                if i_step == whip_step:
                    aim(side)
                if i_step == whip_step + READY_STEPS:
                    ready()
                for i_fruit, crossing_step in zip(fruits_idx, crossings_step):
                    if i_step == whip_step + crossing_step - fall_steps:
                        drop(fruits[i_fruit])
                    if 0 <= i_step - whip_step - crossing_step <= COUNT_WINDOW_STEPS:
                        fruits_n_pieces[i_fruit] = max(fruits_n_pieces[i_fruit], count_pieces(fruits[i_fruit]))
                    if i_step == whip_step + crossing_step + COUNT_WINDOW_STEPS:
                        n_cut += fruits_n_pieces[i_fruit] >= 2
                        n_pieces = fruits_n_pieces[i_fruit]
                        gs.logger.info(
                            f"Round {i_round + 1}: the {FRUITS[i_fruit][0]} is cut into {n_pieces} piece(s)."
                        )
                scene.step()
        gs.logger.info(f"Cut {n_cut} of {n_dropped} dropped fruits in two.")
    if camera is not None:
        camera.stop_recording()


if __name__ == "__main__":
    main()
