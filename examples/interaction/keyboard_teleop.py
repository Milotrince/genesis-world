"""Drive a Franka by end-effector teleoperation from the keyboard, to grasp a cube, lift it and carry it aside.

The keyboard drives the robot from its start pose above the table, for as long as the viewer stays open.

Keyboard controls:
Up        - Move forward (-x)
Down      - Move backward (+x)
Left      - Move left (-y)
Right     - Move right (+y)
K         - Move up
J         - Move down
N         - Rotate counterclockwise
M         - Rotate clockwise
Backslash - Reset the robot and randomize the cube
Space     - Hold to close the gripper, release to open it
Esc       - Quit

Plus all default viewer controls (press 'i' to see them).
"""

import argparse
import os
import random
import xml.etree.ElementTree as ET

import numpy as np

import genesis as gs
import genesis.utils.geom as gu
from genesis.vis.keybindings import Key, KeyAction, Keybind


# The Franka stands on the floor beside a work table.
TABLE_HEIGHT = 0.3


def table_mjcf():
    """MJCF model of a white work table of boxes, its top face TABLE_HEIGHT above the floor."""
    mjcf = ET.Element("mujoco", model="table")
    ET.SubElement(ET.SubElement(mjcf, "default"), "geom", rgba="0.95 0.95 0.95 1")
    body = ET.SubElement(ET.SubElement(mjcf, "worldbody"), "body", name="table", pos=f"0.55 0 {TABLE_HEIGHT}")
    ET.SubElement(body, "geom", type="box", size="0.35 0.7 0.015", pos="0 0 -0.015")
    for x in (-0.31, 0.31):
        for y in (-0.66, 0.66):
            ET.SubElement(body, "geom", type="box", size="0.02 0.02 0.135", pos=f"{x} {y} -0.165")
    return ET.tostring(mjcf, encoding="unicode")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r", "--record", action="store_true", help="Record a video of the run to 'out/keyboard_teleop.mp4'"
    )
    args = parser.parse_args()

    gs.init(precision="32", logging_level="info", backend=gs.cpu)

    camera_pos = (1.5, 0.0, TABLE_HEIGHT + 0.7)
    camera_lookat = (0.2, 0.0, TABLE_HEIGHT + 0.1)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            substeps=4,
        ),
        rigid_options=gs.options.RigidOptions(
            constraint_timeconst=0.01,
            box_box_detection=True,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=camera_pos,
            camera_lookat=camera_lookat,
            camera_fov=50,
        ),
        profiling_options=gs.options.ProfilingOptions(
            show_FPS=False,
        ),
        show_viewer=True,
    )

    scene.add_entity(
        gs.morphs.Plane(),
    )
    scene.add_entity(
        gs.morphs.MJCF(
            file=table_mjcf(),
        ),
    )
    robot = scene.add_entity(
        morph=gs.morphs.MJCF(
            file="xml/franka_emika_panda/panda.xml",
        ),
        material=gs.materials.Rigid(
            gravity_compensation=1.0,
        ),
    )
    cube = scene.add_entity(
        morph=gs.morphs.Box(
            pos=(0.5, 0.0, TABLE_HEIGHT + 0.02),
            size=(0.04, 0.04, 0.04),
        ),
        material=gs.materials.Rigid(
            rho=300,
        ),
        surface=gs.surfaces.Default(
            color=(0.5, 1, 0.5),
        ),
    )
    target = scene.add_entity(
        gs.morphs.Mesh(
            file="meshes/axis.obj",
            scale=0.15,
            collision=False,
        ),
        surface=gs.surfaces.Default(
            color=(1, 0.5, 0.5, 1),
        ),
    )

    camera = None
    if args.record:
        camera = scene.add_camera(
            res=(1280, 720),
            pos=camera_pos,
            lookat=camera_lookat,
            fov=50,
        )

    scene.build()

    robot_init_pos = np.array([0.5, 0.0, TABLE_HEIGHT + 0.5])
    robot_init_quat = gu.xyz_to_quat(np.array([0.0, np.pi, 0.0]))
    motors_dof = np.arange(robot.n_dofs - 2)
    fingers_dof = np.arange(robot.n_dofs - 2, robot.n_dofs)
    ee_link = robot.get_link("hand")

    target_pos = robot_init_pos.copy()
    target_quat = robot_init_quat.copy()
    dpos = 0.002
    drot = 0.01

    def reset_robot():
        target_pos[:] = robot_init_pos
        target_quat[:] = robot_init_quat
        target.set_qpos(np.concatenate([target_pos, target_quat]))
        q = robot.inverse_kinematics(link=ee_link, pos=target_pos, quat=target_quat)
        robot.set_qpos(q[:-2], motors_dof)

    def reset_scene():
        reset_robot()
        cube.set_pos((random.uniform(0.35, 0.55), random.uniform(-0.2, 0.2), TABLE_HEIGHT + 0.05))
        cube.set_quat(gu.xyz_to_quat(np.array([0.0, 0.0, random.uniform(0.0, 2.0 * np.pi)])))

    def move(dpos: tuple[float, float, float]):
        target_pos[:] += dpos

    def rotate(drot: float):
        target_quat[:] = gu.transform_quat_by_quat(target_quat, gu.xyz_to_quat(np.array([0.0, 0.0, drot])))

    def toggle_gripper(close: bool):
        force = -1.0 if close else 1.0
        robot.control_dofs_force([force, force], fingers_dof)

    def track_target():
        target.set_qpos(np.concatenate([target_pos, target_quat]))
        q = robot.inverse_kinematics(link=ee_link, pos=target_pos, quat=target_quat)
        robot.control_dofs_position(q[:-2], motors_dof)
        scene.step()

    def step_towards(waypoint: tuple[float, float, float]):
        move(np.clip(np.array(waypoint) - target_pos, -dpos, dpos))
        track_target()

    reset_robot()

    # The fingers start closed, so open them before the approach.
    toggle_gripper(close=False)

    if camera is not None:
        camera.start_recording(save_to_filename="out/keyboard_teleop.mp4", fps=30)
    # The viewer is always open, so the scripted grasp runs under the test suite alone, which cannot press keys.
    # It calls the same functions as the keys, and each step towards a waypoint is one key press.
    if "PYTEST_VERSION" in os.environ:
        for _ in range(10):
            step_towards((0.5, 0.0, TABLE_HEIGHT + 0.135))
        toggle_gripper(close=True)
        for _ in range(10):
            track_target()
        for _ in range(10):
            step_towards((0.5, 0.0, TABLE_HEIGHT + 0.3))
        for _ in range(10):
            step_towards((0.5, 0.2, TABLE_HEIGHT + 0.3))
        toggle_gripper(close=False)
        for _ in range(10):
            track_target()
        gs.logger.info(f"Cube position after the scripted grasp: {cube.get_pos()}")
    else:
        is_running = True

        def stop():
            nonlocal is_running
            is_running = False

        scene.viewer.register_keybinds(
            Keybind("move_forward", Key.UP, KeyAction.HOLD, callback=move, args=((-dpos, 0, 0),)),
            Keybind("move_back", Key.DOWN, KeyAction.HOLD, callback=move, args=((dpos, 0, 0),)),
            Keybind("move_left", Key.LEFT, KeyAction.HOLD, callback=move, args=((0, -dpos, 0),)),
            Keybind("move_right", Key.RIGHT, KeyAction.HOLD, callback=move, args=((0, dpos, 0),)),
            Keybind("move_up", Key.K, KeyAction.HOLD, callback=move, args=((0, 0, dpos),)),
            Keybind("move_down", Key.J, KeyAction.HOLD, callback=move, args=((0, 0, -dpos),)),
            Keybind("rotate_ccw", Key.N, KeyAction.HOLD, callback=rotate, args=(drot,)),
            Keybind("rotate_cw", Key.M, KeyAction.HOLD, callback=rotate, args=(-drot,)),
            Keybind("reset_scene", Key.BACKSLASH, KeyAction.RELEASE, callback=reset_scene),
            Keybind("close_gripper", Key.SPACE, KeyAction.PRESS, callback=toggle_gripper, args=(True,)),
            Keybind("open_gripper", Key.SPACE, KeyAction.RELEASE, callback=toggle_gripper, args=(False,)),
            Keybind("quit", Key.ESCAPE, KeyAction.RELEASE, callback=stop),
        )
        gs.logger.info(
            "Keys: arrows move in x/y, K/J move up/down, N/M rotate, Backslash resets, hold Space to close the "
            "gripper, Esc quits."
        )
        while is_running and scene.viewer.is_alive():
            track_target()
    if camera is not None:
        camera.stop_recording()


if __name__ == "__main__":
    main()
