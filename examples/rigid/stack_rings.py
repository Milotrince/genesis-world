"""A Franka stacks the rings of a ring-stacking toy onto its peg, widest first.

Each ring is a hollow, non-convex part. Its collision geometry is an authored convex decomposition shipped with the
asset (sixteen convex wedges around the hole), so the rings slide over the peg exactly, with no automatic decomposition
at build time. The toy is built at 55% of its size, so that the widest ring fits between the open fingers of the Franka.
The rings start scattered on the table. The robot grasps each one by its outer rim, lifts it above the peg, centers its
hole over the peg from the measured position of the ring, lowers it onto the peg and lets it drop.

Keyboard controls (-v):
Up/Down    - Move the hand along -x/+x
Left/Right - Move the hand along -y/+y
K/J        - Move the hand up/down
N/M        - Rotate the hand counterclockwise/clockwise
Space      - Hold to close the gripper, release to open it
Esc        - Quit
"""

import argparse
import os
import xml.etree.ElementTree as ET

import numpy as np

import genesis as gs
import genesis.utils.geom as gu
from genesis.utils.misc import tensor_to_array
from genesis.vis.keybindings import Key, KeyAction, Keybind


# Height of the top of the table.
TABLE_HEIGHT = 0.3
SCALE = 0.55
# Dimensions of the toy at full size: height of the base, of a ring and of the peg above the base, and the diameter of
# each ring, widest first.
BASE_HEIGHT = 0.020
RING_HEIGHT = 0.020
POLE_HEIGHT = 0.145
RINGS_DIAMETER = (0.095, 0.085, 0.075, 0.065, 0.0575, 0.050)
RINGS_COLOR = (
    (0.95, 0.95, 0.95),
    (0.60, 0.80, 0.70),
    (0.78, 0.88, 0.80),
    (0.90, 0.55, 0.60),
    (0.85, 0.72, 0.35),
    (0.95, 0.95, 0.95),
)
PEG_POS = (0.5, 0.0)
# Where the rings lie on the table, around the peg.
RINGS_POS = ((0.4, -0.15), (0.55, -0.15), (0.4, 0.15), (0.55, 0.15), (0.65, -0.06), (0.65, 0.06))
# Height of the hand above the bottom of the fingertip pads.
FINGERTIP_OFFSET = 0.112


def table_mjcf():
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
    parser.add_argument("-v", "--vis", action="store_true", help="Show the interactive viewer")
    parser.add_argument("-g", "--gpu", action="store_true", help="Run on GPU instead of CPU")
    parser.add_argument("-r", "--record", action="store_true", help="Record the run to 'out/stack_rings.mp4'")
    args = parser.parse_args()

    gs.init(backend=gs.gpu if args.gpu else gs.cpu)

    camera_pos = (0.95, -0.4, TABLE_HEIGHT + 0.35)
    camera_lookat = (0.5, 0.0, TABLE_HEIGHT + 0.06)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.01,
            substeps=2,
        ),
        rigid_options=gs.options.RigidOptions(
            # The convex pieces of the stacked rings put many geom pairs in contact at once.
            max_collision_pairs=500,
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
        morph=gs.morphs.MJCF(
            file=table_mjcf(),
        ),
    )
    scene.add_entity(
        morph=gs.morphs.URDF(
            file="tower/base_pole.urdf",
            pos=(*PEG_POS, TABLE_HEIGHT + 0.5 * SCALE * BASE_HEIGHT),
            scale=SCALE,
            fixed=True,
            file_meshes_are_zup=True,
        ),
    )
    rings = []
    for i_ring, ring_pos in enumerate(RINGS_POS):
        rings.append(
            scene.add_entity(
                morph=gs.morphs.URDF(
                    file=f"tower/ring_{i_ring + 1:02d}.urdf",
                    pos=(*ring_pos, TABLE_HEIGHT + 0.5 * SCALE * RING_HEIGHT),
                    scale=SCALE,
                    file_meshes_are_zup=True,
                ),
                material=gs.materials.Rigid(
                    rho=600.0,
                ),
                surface=gs.surfaces.Default(
                    color=RINGS_COLOR[i_ring],
                ),
            )
        )
    franka = scene.add_entity(
        morph=gs.morphs.MJCF(
            file="xml/franka_emika_panda/panda.xml",
        ),
        material=gs.materials.Rigid(
            gravity_compensation=1.0,
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

    scene.build()

    motors_dof = np.arange(7)
    fingers_dof = np.arange(7, 9)
    hand = franka.get_link("hand")
    # The MJCF drives the fingers through a tendon actuator, so their position control needs PD gains of its own.
    franka.set_dofs_kp([100.0, 100.0], fingers_dof)
    franka.set_dofs_kv([10.0, 10.0], fingers_dof)

    target_pos = np.array([0.45, 0.0, TABLE_HEIGHT + 0.4])
    target_quat = gu.xyz_to_quat(np.array([0.0, np.pi, 0.0]))
    # Seed the first inverse kinematics solve with a ready pose, so that the arm starts on its elbow-up branch.
    qpos_ready = (0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.04, 0.04)
    franka.set_qpos(
        franka.inverse_kinematics(link=hand, pos=target_pos, quat=target_quat, init_qpos=qpos_ready)[:7], motors_dof
    )
    # Distance per step the hand target moves at most.
    dpos = 0.003

    def move(delta):
        target_pos[:] += delta

    def rotate(angle):
        target_quat[:] = gu.transform_quat_by_quat(target_quat, gu.xyz_to_quat(np.array([0.0, 0.0, angle])))

    def toggle_gripper(close):
        if close:
            franka.control_dofs_force([-5.0, -5.0], fingers_dof)
        else:
            franka.control_dofs_position([0.04, 0.04], fingers_dof)

    def track_target():
        q = franka.inverse_kinematics(link=hand, pos=target_pos, quat=target_quat)
        franka.control_dofs_position(q[:7], motors_dof)
        scene.step()

    def step_towards(waypoint):
        move(np.clip(waypoint - target_pos, -dpos, dpos))
        track_target()

    if camera is not None:
        camera.start_recording(save_to_filename="out/stack_rings.mp4", fps=50)
    if args.vis:
        toggle_gripper(close=False)
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
            Keybind("rotate_ccw", Key.N, KeyAction.HOLD, callback=rotate, args=(0.01,)),
            Keybind("rotate_cw", Key.M, KeyAction.HOLD, callback=rotate, args=(-0.01,)),
            Keybind("close_gripper", Key.SPACE, KeyAction.PRESS, callback=toggle_gripper, args=(True,)),
            Keybind("open_gripper", Key.SPACE, KeyAction.RELEASE, callback=toggle_gripper, args=(False,)),
            Keybind("quit", Key.ESCAPE, KeyAction.RELEASE, callback=stop),
        )
        gs.logger.info(
            "Keys: arrows move the hand in x/y, K/J move it up/down, N/M rotate it, hold Space to close the gripper, "
            "Esc quits."
        )
        while is_running and scene.viewer.is_alive():
            track_target()
    else:
        if "PYTEST_VERSION" not in os.environ:
            n_rings, n_reach, n_grasp, n_settle, n_carry, n_insert = len(rings), 150, 40, 30, 150, 60
        else:
            n_rings, n_reach, n_grasp, n_settle, n_carry, n_insert = 1, 20, 10, 10, 20, 10
        peg_top = TABLE_HEIGHT + SCALE * (BASE_HEIGHT + POLE_HEIGHT)
        for i_ring, ring in enumerate(rings[:n_rings]):
            # Reach over the ring from above, the fingertips just clear of the table, and let the arm settle before the
            # fingers close on the rim across a diameter.
            ring_pos = tensor_to_array(ring.get_pos())
            grasp_pos = np.array([*ring_pos[:2], ring_pos[2] - 0.5 * SCALE * RING_HEIGHT + FINGERTIP_OFFSET + 0.002])
            toggle_gripper(close=False)
            for waypoint, n_steps in ((grasp_pos + (0.0, 0.0, 0.12), n_reach), (grasp_pos, n_grasp)):
                for _ in range(n_steps):
                    step_towards(waypoint)
            for is_closed in (False, True):
                toggle_gripper(close=is_closed)
                for _ in range(n_settle):
                    track_target()
            # Carry the ring until its bottom clears the peg, then lower it onto the peg down to just above the stack.
            # The hand is steered so that the ring, measured where it actually sits in the hand, reaches each waypoint.
            ring_top_z = peg_top + 0.5 * SCALE * RING_HEIGHT + 0.02
            ring_insert_z = max(
                peg_top - 0.03, TABLE_HEIGHT + SCALE * (BASE_HEIGHT + (i_ring + 0.5) * RING_HEIGHT) + 0.01
            )
            for ring_target_z, n_steps in ((ring_top_z, n_carry), (ring_insert_z, n_insert)):
                for _ in range(n_steps):
                    ring_offset = tensor_to_array(hand.get_pos()) - tensor_to_array(ring.get_pos())
                    step_towards(np.array([*PEG_POS, ring_target_z]) + ring_offset)
            toggle_gripper(close=False)
            for _ in range(n_settle):
                track_target()
            for _ in range(n_settle):
                step_towards(np.array([*target_pos[:2], TABLE_HEIGHT + 0.4]))

        rings_pos = np.stack([tensor_to_array(ring.get_pos()) for ring in rings])
        n_stacked = np.count_nonzero(
            (np.linalg.norm(rings_pos[:, :2] - PEG_POS, axis=-1) < 0.005) & (rings_pos[:, 2] < peg_top)
        )
        gs.logger.info(f"{n_stacked} of {len(rings)} rings on the peg.")

    if camera is not None:
        camera.stop_recording()


if __name__ == "__main__":
    main()
