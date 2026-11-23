"""A delta robot sorts boxes off a conveyor into two bins by color.

The robot is one MJCF model built inline. Three motors swing three arms, and each arm carries a parallelogram forearm
of two rods whose lower ends are pinned to the moving platform by `connect` equality constraints, which close the
kinematic loops a tree of joints cannot express. The parallelograms keep the platform level, so its position alone is
controlled. The joint targets come from the closed-form inverse kinematics of the delta robot, and the three motors
track them under PD control. A suction cup under the platform picks each box as it rides past on the conveyor (a slab
on a prismatic joint) through a weld constraint added at runtime, and drops it into the bin of its color.

Keyboard controls (-v):
Up/Down    - Move the platform along -x/+x
Left/Right - Move the platform along -y/+y
K/J        - Move the platform up/down
Space      - Hold to grip the box under the suction cup, release to drop it
B          - Hold to run the belt
Esc        - Quit
"""

import argparse
import math
import os
import xml.etree.ElementTree as ET

import numpy as np

import genesis as gs
from genesis.utils.misc import tensor_to_array
from genesis.vis.keybindings import Key, KeyAction, Keybind


DT = 5e-3

# Delta robot: height of the motor axes, radius of the motors around the center, length of the arms and the forearms,
# radius of the forearm joints around the platform center, half-width of the parallelograms, and the distance from the
# platform center down to the face of the suction cup.
BASE_HEIGHT = 0.9
BASE_RADIUS = 0.15
ARM_LENGTH = 0.25
FOREARM_LENGTH = 0.55
PLATFORM_RADIUS = 0.05
FOREARM_HALF_WIDTH = 0.04
TOOL_LENGTH = 0.08
MOTORS_ANGLE = np.array([0.0, 2.0 * np.pi / 3.0, 4.0 * np.pi / 3.0])
# Platform position the model is authored at, relative to the motors' plane.
HOME_POS = np.array([0.0, 0.0, -0.5])

# Conveyor: height of the belt surface, its length, and the speed it runs at. The boxes ride on it towards +x.
BELT_TOP = 0.3
BELT_LENGTH = 1.4
BELT_SPEED = 0.15
BOX_SIZE = 0.05
N_BOXES = 4
BOX_PITCH = 0.2
# Bins, one per color, on either side of the conveyor under the robot.
BIN_Y = 0.32
BIN_SIZE = (0.24, 0.2, 0.12)
BOX_COLORS = ((0.85, 0.25, 0.2), (0.2, 0.45, 0.85))


def delta_ik(pos):
    """Angles of the three motors below the horizontal that put the platform center at a position relative to the
    center of the motors' plane."""
    x = pos[0] * np.cos(MOTORS_ANGLE) + pos[1] * np.sin(MOTORS_ANGLE)
    y = -pos[0] * np.sin(MOTORS_ANGLE) + pos[1] * np.cos(MOTORS_ANGLE)
    z = pos[2]
    # In the vertical plane of each arm, the elbow at distance FOREARM_LENGTH from the platform joint reads
    # a * cos(theta) + z * sin(theta) = c, solved for the elbow pointing outwards.
    a = BASE_RADIUS - PLATFORM_RADIUS - x
    c = (FOREARM_LENGTH**2 - ARM_LENGTH**2 - a**2 - y**2 - z**2) / (2.0 * ARM_LENGTH)
    return np.arctan2(z, a) + np.arccos(c / np.hypot(a, z))


def delta_cell_mjcf():
    """MJCF model of the work cell: the delta robot at its home pose, its frame, the conveyor and the two bins."""
    mjcf = ET.Element("mujoco", model="delta_cell")
    default = ET.SubElement(mjcf, "default")
    # The rods are light, so their joints carry little armature. Collisions are left to the conveyor and the bins.
    ET.SubElement(default, "joint", armature="1e-4", damping="1e-3")
    ET.SubElement(default, "geom", contype="0", conaffinity="0", density="1000", rgba="0.75 0.76 0.78 1")
    worldbody = ET.SubElement(mjcf, "worldbody")

    # Frame: four posts and a top plate holding the motors.
    for x_post, y_post in ((0.5, 0.5), (-0.5, 0.5), (-0.5, -0.5), (0.5, -0.5)):
        ET.SubElement(
            worldbody,
            "geom",
            type="capsule",
            fromto=f"{x_post} {y_post} 0 {x_post} {y_post} {BASE_HEIGHT + 0.06}",
            size="0.025",
            rgba="0.3 0.3 0.33 1",
        )
        ET.SubElement(
            worldbody,
            "geom",
            type="capsule",
            fromto=f"{x_post} {y_post} {BASE_HEIGHT + 0.06} 0 0 {BASE_HEIGHT + 0.06}",
            size="0.02",
            rgba="0.3 0.3 0.33 1",
        )
    ET.SubElement(
        worldbody,
        "geom",
        type="cylinder",
        pos=f"0 0 {BASE_HEIGHT + 0.06}",
        size=f"{BASE_RADIUS + 0.06} 0.02",
        rgba="0.3 0.3 0.33 1",
    )

    platform_pos = np.array([0.0, 0.0, BASE_HEIGHT]) + HOME_POS
    equality = ET.SubElement(mjcf, "equality")
    for i_arm, (motor_angle, arm_angle) in enumerate(zip(MOTORS_ANGLE, delta_ik(HOME_POS))):
        radial = np.array([math.cos(motor_angle), math.sin(motor_angle), 0.0])
        tangent = np.array([-math.sin(motor_angle), math.cos(motor_angle), 0.0])
        pivot = np.array([0.0, 0.0, BASE_HEIGHT]) + BASE_RADIUS * radial
        elbow = ARM_LENGTH * (math.cos(arm_angle) * radial - math.sin(arm_angle) * np.array([0.0, 0.0, 1.0]))
        ET.SubElement(
            worldbody,
            "geom",
            type="cylinder",
            pos=" ".join(map(str, pivot)),
            zaxis=" ".join(map(str, tangent)),
            size="0.04 0.04",
            rgba="0.2 0.2 0.22 1",
        )
        # Every body keeps the world orientation, so the positions and axes below are world vectors.
        arm = ET.SubElement(worldbody, "body", name=f"arm_{i_arm}", pos=" ".join(map(str, pivot)))
        ET.SubElement(
            arm, "joint", name=f"motor_{i_arm}", type="hinge", axis=" ".join(map(str, tangent)), armature="0.01"
        )
        ET.SubElement(arm, "geom", type="capsule", fromto=f"0 0 0 {' '.join(map(str, elbow))}", size="0.018")
        ET.SubElement(
            arm,
            "geom",
            type="capsule",
            fromto=" ".join(
                map(str, (*(elbow - FOREARM_HALF_WIDTH * tangent), *(elbow + FOREARM_HALF_WIDTH * tangent)))
            ),
            size="0.012",
        )
        # Each rod of the parallelogram hangs from a universal joint at the elbow, two hinges whose axes are both
        # normal to the rod, so the rod cannot spin about itself.
        for side in (-1.0, 1.0):
            rod_top = pivot + elbow + side * FOREARM_HALF_WIDTH * tangent
            rod_bottom = platform_pos + PLATFORM_RADIUS * radial + side * FOREARM_HALF_WIDTH * tangent
            rod_dir = (rod_bottom - rod_top) / FOREARM_LENGTH
            rod_name = f"rod_{i_arm}_{'ab'[side > 0.0]}"
            rod = ET.SubElement(arm, "body", name=rod_name, pos=" ".join(map(str, rod_top - pivot)))
            ET.SubElement(rod, "joint", type="hinge", axis=" ".join(map(str, tangent)))
            ET.SubElement(rod, "joint", type="hinge", axis=" ".join(map(str, np.cross(tangent, rod_dir))))
            ET.SubElement(
                rod, "geom", type="capsule", fromto=f"0 0 0 {' '.join(map(str, rod_bottom - rod_top))}", size="0.007"
            )
            ET.SubElement(
                equality, "connect", body1=rod_name, body2="platform", anchor=" ".join(map(str, rod_bottom - rod_top))
            )

    platform = ET.SubElement(worldbody, "body", name="platform", pos=" ".join(map(str, platform_pos)))
    ET.SubElement(platform, "freejoint")
    ET.SubElement(platform, "geom", type="cylinder", size=f"{PLATFORM_RADIUS + 0.01} 0.012", rgba="0.2 0.2 0.22 1")
    ET.SubElement(platform, "geom", type="cylinder", pos=f"0 0 {-0.5 * TOOL_LENGTH}", size=f"0.01 {0.5 * TOOL_LENGTH}")
    ET.SubElement(
        platform, "geom", type="cylinder", pos=f"0 0 {-TOOL_LENGTH + 0.008}", size="0.02 0.008", rgba="0.1 0.1 0.1 1"
    )

    # Conveyor: a slab sliding on a prismatic joint over a fixed bed, starting with its downstream end under the robot.
    belt_x = 0.1 - 0.5 * BELT_LENGTH
    ET.SubElement(
        worldbody,
        "geom",
        type="box",
        pos=f"0 0 {0.5 * (BELT_TOP - 0.03)}",
        size=f"1.0 0.13 {0.5 * (BELT_TOP - 0.03)}",
        rgba="0.35 0.35 0.38 1",
    )
    belt = ET.SubElement(worldbody, "body", name="belt", pos=f"{belt_x} 0 {BELT_TOP - 0.015}")
    ET.SubElement(belt, "joint", name="belt_slide", type="slide", axis="1 0 0", armature="0", damping="0")
    ET.SubElement(
        belt,
        "geom",
        type="box",
        size=f"{0.5 * BELT_LENGTH} 0.12 0.015",
        contype="1",
        conaffinity="1",
        rgba="0.12 0.12 0.12 1",
    )

    # The bins stand on the floor, one on each side of the conveyor.
    bin_hx, bin_hy, bin_hz = (0.5 * size for size in BIN_SIZE)
    for i_bin, y_bin in enumerate((-BIN_Y, BIN_Y)):
        rgba = " ".join(map(str, (*(0.5 * np.array(BOX_COLORS[i_bin]) + 0.2), 1.0)))
        for size, pos in (
            ((bin_hx, bin_hy, 0.01), (0.0, 0.0, 0.01)),
            ((bin_hx, 0.01, bin_hz), (0.0, -bin_hy, bin_hz)),
            ((bin_hx, 0.01, bin_hz), (0.0, bin_hy, bin_hz)),
            ((0.01, bin_hy, bin_hz), (-bin_hx, 0.0, bin_hz)),
            ((0.01, bin_hy, bin_hz), (bin_hx, 0.0, bin_hz)),
        ):
            ET.SubElement(
                worldbody,
                "geom",
                type="box",
                size=" ".join(map(str, size)),
                pos=f"{pos[0]} {y_bin + pos[1]} {pos[2]}",
                contype="1",
                conaffinity="1",
                rgba=rgba,
            )
    return mjcf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", help="Show the interactive viewer")
    parser.add_argument("-g", "--gpu", action="store_true", help="Run on GPU instead of CPU")
    parser.add_argument("-r", "--record", action="store_true", help="Record the run to 'out/delta_robot.mp4'")
    args = parser.parse_args()

    gs.init(backend=gs.gpu if args.gpu else gs.cpu)

    camera_pos = (1.7, -0.9, 1.3)
    camera_lookat = (-0.1, 0.0, 0.4)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=DT,
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
    cell = scene.add_entity(
        gs.morphs.MJCF(
            file=ET.tostring(delta_cell_mjcf(), encoding="unicode"),
        ),
    )
    # The boxes wait upstream of the robot, alternating colors.
    boxes = []
    for i_box in range(N_BOXES):
        boxes.append(
            scene.add_entity(
                gs.morphs.Box(
                    pos=(-0.2 - i_box * BOX_PITCH, 0.0, BELT_TOP + 0.5 * BOX_SIZE),
                    size=(BOX_SIZE, BOX_SIZE, BOX_SIZE),
                ),
                surface=gs.surfaces.Default(
                    color=BOX_COLORS[i_box % 2],
                ),
            )
        )

    camera = None
    if args.record:
        camera = scene.add_camera(
            res=(1280, 720),
            pos=camera_pos,
            lookat=camera_lookat,
            fov=40,
        )

    scene.build()

    motors_dof = [cell.get_joint(f"motor_{i_arm}").dofs_idx_local[0] for i_arm in range(3)]
    belt_dof = cell.get_joint("belt_slide").dofs_idx_local
    platform_link = cell.get_link("platform")
    cell.set_dofs_kp([1000.0, 1000.0, 1000.0], motors_dof)
    cell.set_dofs_kv([40.0, 40.0, 40.0], motors_dof)
    cell.set_dofs_kv([200.0], belt_dof)

    base_pos = np.array([0.0, 0.0, BASE_HEIGHT])
    # The joint angles count from the pose the model is authored at.
    motors_home = delta_ik(HOME_POS)
    target_pos = base_pos + HOME_POS
    gripped_box = None
    # Distance per step the platform target moves at most, fast enough to catch up with a box on the belt.
    dpos = 1.0 * DT

    def move(delta):
        # The target stays inside the workspace, where the inverse kinematics has a solution.
        if np.isfinite(delta_ik(target_pos + delta - base_pos)).all():
            target_pos[:] += delta

    def track_target():
        cell.control_dofs_position(delta_ik(target_pos - base_pos) - motors_home, motors_dof)
        scene.step()

    def step_towards(waypoint):
        move(np.clip(np.asarray(waypoint) - target_pos, -dpos, dpos))
        track_target()

    def grip(is_on):
        nonlocal gripped_box
        if is_on and gripped_box is None:
            # The suction cup grips the box whose top face it touches.
            cup_pos = tensor_to_array(platform_link.get_pos()) - np.array([0.0, 0.0, TOOL_LENGTH])
            boxes_top = np.stack([tensor_to_array(box.get_pos()) for box in boxes]) + np.array(
                [0.0, 0.0, 0.5 * BOX_SIZE]
            )
            i_box = int(np.argmin(np.linalg.norm(boxes_top - cup_pos, axis=-1)))
            if np.linalg.norm(boxes_top[i_box] - cup_pos) < 0.02:
                gripped_box = boxes[i_box]
                scene.rigid_solver.add_weld_constraint(gripped_box.base_link.idx, platform_link.idx)
        elif not is_on and gripped_box is not None:
            scene.rigid_solver.delete_weld_constraint(gripped_box.base_link.idx, platform_link.idx)
            gripped_box = None

    if camera is not None:
        camera.start_recording(save_to_filename="out/delta_robot.mp4", fps=50)

    if args.vis:
        # The belt stands still until B brings the next boxes within reach of the robot.
        cell.control_dofs_velocity([0.0], belt_dof)
        is_running = True

        def stop():
            nonlocal is_running
            is_running = False

        scene.viewer.register_keybinds(
            Keybind("move_forward", Key.UP, KeyAction.HOLD, callback=move, args=((-0.002, 0.0, 0.0),)),
            Keybind("move_back", Key.DOWN, KeyAction.HOLD, callback=move, args=((0.002, 0.0, 0.0),)),
            Keybind("move_left", Key.LEFT, KeyAction.HOLD, callback=move, args=((0.0, -0.002, 0.0),)),
            Keybind("move_right", Key.RIGHT, KeyAction.HOLD, callback=move, args=((0.0, 0.002, 0.0),)),
            Keybind("move_up", Key.K, KeyAction.HOLD, callback=move, args=((0.0, 0.0, 0.002),)),
            Keybind("move_down", Key.J, KeyAction.HOLD, callback=move, args=((0.0, 0.0, -0.002),)),
            Keybind("grip", Key.SPACE, KeyAction.PRESS, callback=grip, args=(True,)),
            Keybind("release", Key.SPACE, KeyAction.RELEASE, callback=grip, args=(False,)),
            Keybind(
                "run_belt", Key.B, KeyAction.PRESS, callback=cell.control_dofs_velocity, args=([BELT_SPEED], belt_dof)
            ),
            Keybind("stop_belt", Key.B, KeyAction.RELEASE, callback=cell.control_dofs_velocity, args=([0.0], belt_dof)),
            Keybind("quit", Key.ESCAPE, KeyAction.RELEASE, callback=stop),
        )
        gs.logger.info(
            "Keys: arrows move the platform in x/y, K/J move it up/down, hold Space to grip, hold B to run the belt, "
            "Esc quits."
        )
        while is_running and scene.viewer.is_alive():
            track_target()
    else:
        cell.control_dofs_velocity([BELT_SPEED], belt_dof)
        # Platform heights above the center of a box on the belt: hovering, then with the suction cup on its top face.
        hover = np.array([0.0, 0.0, 0.5 * BOX_SIZE + TOOL_LENGTH + 0.08])
        grasp = np.array([0.0, 0.0, 0.5 * BOX_SIZE + TOOL_LENGTH - 0.004])
        if "PYTEST_VERSION" not in os.environ:
            n_boxes, n_approach, n_descend, n_lift, n_carry, n_home = N_BOXES, 100, 30, 30, 80, 60
        else:
            n_boxes, n_approach, n_descend, n_lift, n_carry, n_home = 1, 20, 20, 20, 20, 0
        for i_box, box in enumerate(boxes[:n_boxes]):
            # Follow the box down as it rides on, grip it, lift it, and carry it over its bin beside the previous one.
            for n_steps, offset in ((n_approach, hover), (n_descend, grasp)):
                for _ in range(n_steps):
                    step_towards(tensor_to_array(box.get_pos()) + offset)
            grip(True)
            for _ in range(n_lift):
                step_towards(tensor_to_array(box.get_pos()) + hover)
            bin_drop = (
                (-0.05, 0.05)[i_box // 2 % 2],
                (-BIN_Y, BIN_Y)[i_box % 2],
                BIN_SIZE[2] + 0.08 + BOX_SIZE + TOOL_LENGTH,
            )
            for _ in range(n_carry):
                step_towards(bin_drop)
            grip(False)
        for _ in range(n_home):
            step_towards(base_pos + HOME_POS)

        boxes_pos = np.stack([tensor_to_array(box.get_pos()) for box in boxes])
        n_sorted = sum(
            abs(boxes_pos[i_box, 0]) < 0.5 * BIN_SIZE[0]
            and abs(boxes_pos[i_box, 1] - (-BIN_Y, BIN_Y)[i_box % 2]) < 0.5 * BIN_SIZE[1]
            and boxes_pos[i_box, 2] < BIN_SIZE[2]
            for i_box in range(N_BOXES)
        )
        gs.logger.info(f"{n_sorted} of {N_BOXES} boxes sorted into the bin of their color.")

    if camera is not None:
        camera.stop_recording()


if __name__ == "__main__":
    main()
