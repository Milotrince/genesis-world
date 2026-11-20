"""A Franka finds a chamfered hole by touch and inserts the square peg it holds, with a spiral search under force control.

The peg starts 1.5 cm off the hole, which leaves 1 mm of clearance on each side under a 3 mm chamfer. A `ContactForce`
sensor on the peg, blind to the fingers, reads the force of the plate. The robot presses the peg down with a set force
and moves it along a spiral until that force drops, which means the peg is over the hole. It then stops the spiral,
follows the lateral push of the chamfer into the hole and inserts the peg. An `ElastomerTaxel` gel on each fingertip
reads the grip on the peg. The force arrow and the gel markers are drawn in the scene, and a live plot shows the
readings against time.

Keyboard controls (-v), once the scripted insertion ends:
Up/Down    - Move the peg along -x/+x
Left/Right - Move the peg along -y/+y
K/J        - Move the peg up/down
Space      - Start a spiral search from the current position, or stop it
Backslash  - Put the peg back in its start pose

Plus all default viewer controls (press 'i' to see them).
"""

import argparse
import os
import xml.etree.ElementTree as ET

import numpy as np
import trimesh

import genesis as gs
import genesis.utils.geom as gu
from genesis.utils.misc import tensor_to_array
from genesis.vis.keybindings import Key, KeyAction, Keybind


DT = 1e-2
RECORDING_FPS = 30

# Square peg, and the square hole through the plate: clearance over the whole width, and the depth of a 45 degree chamfer
# around its rim.
PEG_WIDTH = 0.02
PEG_LENGTH = 0.08
HOLE_CLEARANCE = 0.002
HOLE_CHAMFER = 0.003
PLATE_THICKNESS = 0.025
PLATE_WIDTH = 0.16
HOLE_POS = np.array((0.55, 0.0))
# The peg starts off the hole, as if the hole had been located by a camera with a centimetre of error.
PEG_START_POS = HOLE_POS + np.array((-0.012, 0.009))

# Force control: the pressing force, and the admittance gains mapping a force error to a velocity of the hand target
PRESS_FORCE = 5.0
PRESS_ADMITTANCE = 2e-3
LATERAL_ADMITTANCE = 5e-2
# Archimedean spiral: the radial gap between two turns, the speed along it, and the radius below which the angular rate
# stops growing.
SPIRAL_PITCH = 0.005
SPIRAL_SPEED = 0.02
SPIRAL_RADIUS_MIN = 0.003
INSERT_SPEED = 0.02


def plate_mjcf():
    """MJCF model of the plate standing on the ground, with a chamfered square hole through it.

    The plate is a picture frame of four convex pieces mitred along the diagonals, so that the chamfer of each piece
    meets its neighbours without a gap.
    """
    half_hole = 0.5 * (PEG_WIDTH + HOLE_CLEARANCE)
    half_plate = 0.5 * PLATE_WIDTH
    top = PLATE_THICKNESS
    piece_verts = np.array(
        (
            (half_hole, -half_hole, 0.0),
            (half_hole, half_hole, 0.0),
            (half_plate, half_plate, 0.0),
            (half_plate, -half_plate, 0.0),
            (half_plate, half_plate, top),
            (half_plate, -half_plate, top),
            (half_hole + HOLE_CHAMFER, -half_hole - HOLE_CHAMFER, top),
            (half_hole + HOLE_CHAMFER, half_hole + HOLE_CHAMFER, top),
            (half_hole, -half_hole, top - HOLE_CHAMFER),
            (half_hole, half_hole, top - HOLE_CHAMFER),
        )
    )
    piece = trimesh.convex.convex_hull(piece_verts)

    mjcf = ET.Element("mujoco", model="plate")
    asset = ET.SubElement(mjcf, "asset")
    ET.SubElement(
        asset,
        "mesh",
        name="piece",
        vertex=" ".join(f"{coord:.6g}" for coord in piece.vertices.ravel()),
        face=" ".join(str(idx) for idx in piece.faces.ravel()),
    )
    plate = ET.SubElement(ET.SubElement(mjcf, "worldbody"), "body", name="plate", pos=f"{HOLE_POS[0]} {HOLE_POS[1]} 0")
    for i_piece in range(4):
        ET.SubElement(plate, "geom", type="mesh", mesh="piece", euler=f"0 0 {90 * i_piece}")
    return mjcf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", help="Show the interactive viewer")
    parser.add_argument("-g", "--gpu", action="store_true", help="Run on GPU instead of CPU")
    parser.add_argument("-t", "--seconds", type=float, default=12.0, help="Time allowed for the scripted insertion")
    parser.add_argument(
        "-r",
        "--record",
        action="store_true",
        help="Record the scene and the plot to 'out/tactile_peg_insertion*.mp4'",
    )
    args = parser.parse_args()

    gs.init(backend=gs.gpu if args.gpu else gs.cpu, precision="32", logging_level="info")

    camera_pos = (0.78, -0.08, 0.3)
    camera_lookat = (0.55, 0.0, 0.02)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=DT,
            substeps=4,
        ),
        rigid_options=gs.options.RigidOptions(
            box_box_detection=True,
            # The grip holds the peg against the press force and the drag of the plate without creeping.
            friction_cone=gs.friction_cone.elliptic,
            # Two finger contacts alone would let the peg pivot about the axis between them.
            enable_torsional_friction=True,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=camera_pos,
            camera_lookat=camera_lookat,
            camera_fov=40,
        ),
        profiling_options=gs.options.ProfilingOptions(
            show_FPS=False,
        ),
        show_viewer=args.vis,
    )

    scene.add_entity(
        gs.morphs.Plane(),
    )
    scene.add_entity(
        gs.morphs.MJCF(
            file=plate_mjcf(),
        ),
        material=gs.materials.Rigid(
            friction=0.2,
        ),
        # Flat shading keeps the faces of the hull pieces flat, with sharp edges at the chamfer.
        surface=gs.surfaces.Default(
            color=(0.85, 0.85, 0.8),
            smooth=False,
        ),
    )
    franka = scene.add_entity(
        gs.morphs.MJCF(
            file="xml/franka_emika_panda/panda.xml",
        ),
    )
    # The peg is held by its upper end, its lower end hanging 3 cm above the plate.
    peg_start_z = PLATE_THICKNESS + 0.03 + 0.5 * PEG_LENGTH
    peg = scene.add_entity(
        gs.morphs.Box(
            pos=(*PEG_START_POS, peg_start_z),
            size=(PEG_WIDTH, PEG_WIDTH, PEG_LENGTH),
        ),
        material=gs.materials.Rigid(
            friction=0.2,
            friction_torsional=0.01,
        ),
        surface=gs.surfaces.Default(
            color=(0.9, 0.55, 0.2),
        ),
    )

    fingers_link = (franka.get_link("left_finger"), franka.get_link("right_finger"))
    force_sensor = scene.add_sensor(
        gs.sensors.ContactForce(
            entity_idx=peg.idx,
            filter_link_idx=[link.idx for link in fingers_link],
            draw_debug=True,
        )
    )
    # A grid of markers over the inner face of each fingertip, which faces -y in the frame of its finger
    gel_normal = (0.0, -1.0, 0.0)
    gel_markers_pos = gu.generate_grid_points_on_plane(
        lo=(-0.007, 0.0, 0.036),
        hi=(0.007, 0.0, 0.052),
        normal=gel_normal,
        nx=6,
        ny=6,
    )
    gel_sensors = [
        scene.add_sensor(
            gs.sensors.ElastomerTaxel(
                entity_idx=franka.idx,
                link_idx_local=link.idx_local,
                probe_local_pos=gel_markers_pos,
                probe_local_normal=gel_normal,
                probe_radius=0.002,
                track_link_idx=(peg.base_link_idx,),
                n_sample_points=300,
                lambda_d=5000.0,
                lambda_s=4000.0,
                dilate_scale=0.1,
                normal_exponent=1.0,
                compressibility=0.8,
                draw_debug=True,
                # The points the gels sample on the peg stay small, so that the drawn markers leave its faces readable.
                debug_point_cloud_radius=0.0002,
            )
        )
        for link in fingers_link
    ]

    camera = None
    if args.record:
        camera = scene.add_camera(
            res=(1280, 720),
            pos=camera_pos,
            lookat=camera_lookat,
            fov=40,
            debug=True,
        )

    # The force of the plate on the peg in the world frame, and the mean marker displacement of each gel in the frame
    # of its finger
    plate_force = np.zeros(3)
    gels_disp = np.zeros((2, 3))

    def plot_data():
        return {
            "plate force (N)": (plate_force[2], np.linalg.norm(plate_force[:2])),
            "gel indentation (mm)": tuple(-1e3 * gels_disp[:, 1]),
        }

    scene.add_recorder(
        plot_data,
        gs.recorders.MPLLinePlot(
            labels={
                "plate force (N)": ("vertical", "lateral"),
                "gel indentation (mm)": ("left", "right"),
            },
            x_label="time (s)",
            history_length=1500,
            hz=RECORDING_FPS,
            title="Peg insertion",
            save_to_filename="out/tactile_peg_insertion_plot.mp4" if args.record else None,
        ),
    )

    scene.build()

    end_effector = franka.get_link("hand")
    motors_dof = np.arange(7)
    fingers_dof = np.arange(7, 9)
    hand_quat = np.array((0.0, 1.0, 0.0, 0.0))
    # The pads of the fingers sit 10 cm below the hand frame. They grip the peg 1.5 cm below its top.
    hand_start_pos = np.array((*PEG_START_POS, peg_start_z + 0.5 * PEG_LENGTH - 0.015 + 0.103))
    hand_target_pos = hand_start_pos.copy()

    qpos = franka.inverse_kinematics(link=end_effector, pos=hand_start_pos, quat=hand_quat)
    qpos[fingers_dof] = 0.5 * PEG_WIDTH
    franka.set_qpos(qpos)
    franka.control_dofs_position(qpos[motors_dof], motors_dof)
    franka.control_dofs_force(-40.0, fingers_dof)

    # The search runs as a state machine stepped once per simulation step, by the script and by the keys alike.
    phase = "hold"
    spiral_center = PEG_START_POS.copy()
    spiral_angle = 0.0
    spiral_radius = 0.0
    contact_hand_z = 0.0
    n_search_steps = 0
    n_probe_steps = 0
    found_time = None
    found_radius = None
    # The path of the peg drawn on the plate, one segment every few steps
    path_markers = []
    path_last_pos = np.zeros(3)

    def start_search():
        nonlocal phase, spiral_angle
        if phase == "hold":
            spiral_center[:] = hand_target_pos[:2]
            spiral_angle = 0.0
            for marker in path_markers:
                scene.clear_debug_object(marker)
            path_markers.clear()
            phase = "descend"
        else:
            phase = "hold"

    def move(dpos: tuple[float, float, float]):
        hand_target_pos[:] += dpos

    def control_step():
        nonlocal phase, spiral_angle, spiral_radius, contact_hand_z, n_search_steps, n_probe_steps
        nonlocal found_time, found_radius
        plate_force[:] = gu.transform_by_quat(tensor_to_array(force_sensor.read()), tensor_to_array(peg.get_quat()))
        gels_disp[:] = [tensor_to_array(sensor.read()).mean(axis=(0, 1)) for sensor in gel_sensors]
        force_z = plate_force[2]
        hand_z = tensor_to_array(end_effector.get_pos())[2]
        peg_pos = tensor_to_array(peg.get_pos())
        peg_pos[2] = PLATE_THICKNESS + 0.001
        if phase in ("search", "probe") and np.linalg.norm(peg_pos - path_last_pos) > 0.001:
            path_markers.append(scene.draw_debug_line(path_last_pos, peg_pos, radius=0.001, color=(0.1, 0.4, 1.0, 1.0)))
            path_last_pos[:] = peg_pos

        if phase == "descend":
            hand_target_pos[2] -= INSERT_SPEED * DT
            if force_z > 0.5 * PRESS_FORCE:
                contact_hand_z = hand_z
                path_last_pos[:] = peg_pos
                n_search_steps = 0
                phase = "search"
        elif phase == "search":
            n_search_steps += 1
            hand_target_pos[2] -= PRESS_ADMITTANCE * (PRESS_FORCE - force_z) * DT
            spiral_angle += SPIRAL_SPEED * DT / max(spiral_radius, SPIRAL_RADIUS_MIN)
            spiral_radius = SPIRAL_PITCH * spiral_angle / (2.0 * np.pi)
            hand_target_pos[:2] = spiral_center + spiral_radius * np.array((np.cos(spiral_angle), np.sin(spiral_angle)))
            # The peg has lost the support of the plate, or a transient of the contact did. The probe tells them apart.
            if force_z < 0.4 * PRESS_FORCE:
                n_probe_steps = 0
                phase = "probe"
        elif phase == "probe":
            # The spiral holds still while the peg keeps pressing. The chamfer pushes it towards the axis of the hole.
            hand_target_pos[:2] += LATERAL_ADMITTANCE * plate_force[:2] * DT
            hand_target_pos[2] -= PRESS_ADMITTANCE * (PRESS_FORCE - force_z) * DT
            n_search_steps += 1
            n_probe_steps += 1
            if contact_hand_z - hand_z > 0.002:
                found_time = n_search_steps * DT
                found_radius = spiral_radius
                gs.logger.info(
                    f"Hole found after {found_time:.2f} s of search, at a spiral radius of {1e3 * found_radius:.1f} mm."
                )
                phase = "insert"
            elif n_probe_steps > 40:
                phase = "search"
        elif phase == "insert":
            hand_target_pos[:2] += LATERAL_ADMITTANCE * plate_force[:2] * DT
            hand_target_pos[2] -= INSERT_SPEED * DT
            if contact_hand_z - hand_z > PLATE_THICKNESS - 0.004:
                phase = "hold"

        qpos = franka.inverse_kinematics(link=end_effector, pos=hand_target_pos, quat=hand_quat)
        franka.control_dofs_position(qpos[motors_dof], motors_dof)
        scene.step()

    # The fingers close on the peg before the search starts.
    for _ in range(30):
        control_step()
    start_state = scene.get_state()

    if camera is not None:
        camera.start_recording(save_to_filename="out/tactile_peg_insertion.mp4", fps=RECORDING_FPS)
    n_steps = 100 if "PYTEST_VERSION" in os.environ else round(args.seconds / DT)
    try:
        start_search()
        for _ in range(n_steps):
            control_step()
            if phase == "hold":
                break
        if camera is not None:
            camera.stop_recording()

        peg_bottom_z = tensor_to_array(peg.get_pos())[2] - 0.5 * PEG_LENGTH
        if found_time is None:
            gs.logger.info("The search ended before the hole was found.")
        else:
            gs.logger.info(
                f"Search time {found_time:.2f} s, spiral radius {1e3 * found_radius:.1f} mm, insertion depth "
                f"{1e3 * (PLATE_THICKNESS - peg_bottom_z):.1f} mm."
            )

        def reset_scene():
            nonlocal phase
            scene.reset(start_state)
            hand_target_pos[:] = hand_start_pos
            phase = "hold"

        if args.vis:
            dpos = 0.002
            scene.viewer.register_keybinds(
                Keybind("move_forward", Key.UP, KeyAction.HOLD, callback=move, args=((-dpos, 0, 0),)),
                Keybind("move_back", Key.DOWN, KeyAction.HOLD, callback=move, args=((dpos, 0, 0),)),
                Keybind("move_left", Key.LEFT, KeyAction.HOLD, callback=move, args=((0, -dpos, 0),)),
                Keybind("move_right", Key.RIGHT, KeyAction.HOLD, callback=move, args=((0, dpos, 0),)),
                Keybind("move_up", Key.K, KeyAction.HOLD, callback=move, args=((0, 0, dpos),)),
                Keybind("move_down", Key.J, KeyAction.HOLD, callback=move, args=((0, 0, -dpos),)),
                Keybind("toggle_search", Key.SPACE, KeyAction.PRESS, callback=start_search),
                Keybind("reset_scene", Key.BACKSLASH, KeyAction.RELEASE, callback=reset_scene),
            )
            gs.logger.info(
                "Keys: arrows move the peg in x/y, K/J move it up/down, Space starts or stops a spiral search, Backslash "
                "puts the peg back."
            )
        while args.vis and scene.viewer.is_alive() and "PYTEST_VERSION" not in os.environ:
            control_step()
    finally:
        # The plot is written out here, since a recorder still running when the script exits fails at teardown.
        scene.stop_recording()


if __name__ == "__main__":
    main()
