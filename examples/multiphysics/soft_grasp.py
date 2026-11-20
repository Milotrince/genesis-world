"""A Franka picks up a tomato, a jelly cube and a plush worm, and places the tomato in a bowl and the others in a box.

The three objects are finite element method (FEM) bodies of different character: the tomato is firm, the jelly is very
soft and nearly incompressible, the plush worm is soft, light and compressible. The gripper closes on each the same way,
so each squashes by its own amount. '--coupler' selects how the rigid robot and the soft objects interact: the
semi-analytic primal (SAP) coupler with hydroelastic contact, or the incremental potential contact (IPC) coupler, which
needs 'pyuipc' and a CUDA GPU. The run logs whether each object ended up placed, and how much it squashed in the grip.

Keyboard controls (-v), once the scripted sequence is over:
Up/Down    - Move the hand away from/towards the camera (+y/-y)
Left/Right - Move the hand left/right (-x/+x)
K/J        - Move the hand up/down
Space      - Hold to close the gripper, release to open it

Plus all default viewer controls (press 'i' to see them).
"""

import argparse
import os
import xml.etree.ElementTree as ET

import numpy as np

import genesis as gs
from genesis.utils.misc import tensor_to_array
from genesis.vis.keybindings import Key, KeyAction, Keybind


DT = 1e-2
# The hand points down, its fingers closing along y.
HAND_QUAT = (0.0, 1.0, 0.0, 0.0)
# Height of the hand above the middle of the fingertip pads.
TIP_DEPTH = 0.105
# Distance the hand moves per step along a scripted path or while a key is held.
HAND_SPEED = 0.004
# Outer size (x, y, z) and position of the bowl and of the box, and the thickness of their walls.
BOWL_SIZE = (0.16, 0.16, 0.06)
BOWL_POS = (0.35, 0.35, 0.0)
BOX_SIZE = (0.22, 0.24, 0.1)
BOX_POS = (0.6, 0.32, 0.0)
WALL = 0.01


def container_mjcf(size):
    """MJCF model of a fixed open container, its bottom centred on the origin of its body."""
    size_x, size_y, size_z = size
    mjcf = ET.Element("mujoco", model="container")
    body = ET.SubElement(ET.SubElement(mjcf, "worldbody"), "body", name="container")
    for pos, half_size in (
        ((0.0, 0.0, 0.5 * WALL), (0.5 * size_x, 0.5 * size_y, 0.5 * WALL)),
        ((0.5 * (size_x - WALL), 0.0, 0.5 * size_z), (0.5 * WALL, 0.5 * size_y, 0.5 * size_z)),
        ((-0.5 * (size_x - WALL), 0.0, 0.5 * size_z), (0.5 * WALL, 0.5 * size_y, 0.5 * size_z)),
        ((0.0, 0.5 * (size_y - WALL), 0.5 * size_z), (0.5 * size_x - WALL, 0.5 * WALL, 0.5 * size_z)),
        ((0.0, -0.5 * (size_y - WALL), 0.5 * size_z), (0.5 * size_x - WALL, 0.5 * WALL, 0.5 * size_z)),
    ):
        ET.SubElement(
            body,
            "geom",
            type="box",
            pos=" ".join(f"{coord:.6g}" for coord in pos),
            size=" ".join(f"{coord:.6g}" for coord in half_size),
        )
    return ET.tostring(mjcf, encoding="unicode")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--coupler", choices=("sap", "ipc"), default="sap", help="Coupling of the robot and the soft objects"
    )
    parser.add_argument("-v", "--vis", action="store_true", help="Show the interactive viewer")
    parser.add_argument("-g", "--gpu", action="store_true", help="Run on GPU instead of CPU")
    parser.add_argument("-r", "--record", action="store_true", help="Record the run to 'out/soft_grasp.mp4'")
    args = parser.parse_args()

    gs.init(backend=gs.gpu if args.gpu else gs.cpu, precision="32")

    camera_pos = (0.45, -1.2, 1.0)
    camera_lookat = (0.45, 0.05, 0.05)

    if args.coupler == "sap":
        coupler_options = gs.options.SAPCouplerOptions()
    else:
        coupler_options = gs.options.IPCCouplerOptions(
            newton_translation_tolerance=10.0,
            contact_d_hat=2e-3,
            constraint_strength_translation=10.0,
            constraint_strength_rotation=10.0,
            enable_rigid_ground_contact=False,
            enable_rigid_rigid_contact=False,
        )

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=DT,
        ),
        rigid_options=gs.options.RigidOptions(
            enable_self_collision=False,
        ),
        fem_options=gs.options.FEMOptions(
            use_implicit_solver=True,
        ),
        coupler_options=coupler_options,
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
    for size, pos, color in ((BOWL_SIZE, BOWL_POS, (0.9, 0.9, 0.92)), (BOX_SIZE, BOX_POS, (0.6, 0.45, 0.3))):
        scene.add_entity(
            morph=gs.morphs.MJCF(
                file=container_mjcf(size),
                pos=pos,
            ),
            surface=gs.surfaces.Rough(
                color=color,
            ),
        )
    franka = scene.add_entity(
        morph=gs.morphs.MJCF(
            file="xml/franka_emika_panda/panda_non_overlap.xml",
        ),
        # Only the fingers touch the soft objects. The IPC coupler drives them through soft constraints, which leave the
        # rest of the arm to the rigid solver.
        material=gs.materials.Rigid(
            friction=1.0,
            coup_friction=1.0,
            gravity_compensation=1.0,
            coup_type="two_way_soft_constraint",
            coup_links=("left_finger", "right_finger"),
        ),
    )

    # The objects start a few millimetres above the floor, clear of the contact distance of the IPC coupler. The tomato is
    # firm, the jelly very soft and nearly incompressible, the plush worm soft, light and compressible.
    # The jelly is refined into a few hundred tetrahedra, since its eight corners alone make five that barely bend. The
    # worm mesh is 5 units long along z, laid along x and scaled to a toy.
    soft_entities = []
    for morph, (young_modulus, poisson_ratio, density), color in (
        (gs.morphs.Sphere(pos=(0.4, -0.35, 0.035), radius=0.03), (2e5, 0.4, 1000.0), (0.85, 0.15, 0.1, 1.0)),
        (
            gs.morphs.Box(pos=(0.55, -0.3, 0.03), size=(0.05, 0.05, 0.05), nobisect=False, maxvolume=2e-7),
            (1e4, 0.45, 1000.0),
            (0.3, 0.8, 0.3, 0.8),
        ),
        (
            gs.morphs.Mesh(file="meshes/worm/worm.obj", pos=(0.6, -0.15, 0.032), euler=(0.0, 90.0, 0.0), scale=0.03),
            (2e4, 0.2, 200.0),
            (0.95, 0.65, 0.2, 1.0),
        ),
    ):
        soft_entities.append(
            scene.add_entity(
                morph=morph,
                material=gs.materials.FEM.Elastic(
                    E=young_modulus,
                    nu=poisson_ratio,
                    rho=density,
                    friction_mu=1.0,
                    model="linear_corotated",
                ),
                surface=gs.surfaces.Default(
                    color=color,
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

    motors_dof = np.arange(7)
    fingers_dof = np.arange(7, 9)
    hand = franka.get_link("hand")
    franka.set_qpos((0.0, -0.4, 0.0, -2.4, 0.0, 2.0, 0.785, 0.04, 0.04))
    # The MJCF drives the fingers through a tendon actuator, so position control needs PD gains of its own.
    franka.set_dofs_kp((300.0, 300.0), fingers_dof)
    franka.set_dofs_kv((20.0, 20.0), fingers_dof)
    franka.control_dofs_position((0.04, 0.04), fingers_dof)
    target_pos = tensor_to_array(hand.get_pos())

    def track_target():
        qpos = franka.inverse_kinematics(link=hand, pos=target_pos, quat=HAND_QUAT, init_qpos=franka.get_qpos())
        franka.control_dofs_position(qpos[:7], motors_dof)
        scene.step()

    def move(delta):
        target_pos[:] += delta

    def set_gripper(close):
        finger_pos = 0.0 if close else 0.04
        franka.control_dofs_position((finger_pos, finger_pos), fingers_dof)

    def move_to(waypoint):
        while np.linalg.norm(np.subtract(waypoint, target_pos)) > 1e-6:
            move(np.clip(np.subtract(waypoint, target_pos), -path_step, path_step))
            track_target()
        for _ in range(n_settle):
            track_target()

    is_ci = "PYTEST_VERSION" in os.environ
    path_step, n_settle, n_grip = (HAND_SPEED, 20, 40) if not is_ci else (0.05, 1, 1)
    if camera is not None:
        camera.start_recording(save_to_filename="out/soft_grasp.mp4", fps=50)

    objects = tuple(zip(soft_entities, (BOWL_POS, BOX_POS, BOX_POS), (BOWL_SIZE, BOX_SIZE, BOX_SIZE)))
    drop_offsets_x = (0.0, -0.05, 0.05)
    squash_ratios = []
    for (entity, container_pos, container_size), drop_offset_x in zip(objects[: 1 if is_ci else None], drop_offsets_x):
        verts_pos = entity.get_verts_pos()
        center = tensor_to_array(verts_pos.mean(dim=0))
        # The squash is measured across the fingers, on the vertices between the fingertip pads.
        is_gripped = ((verts_pos - verts_pos.mean(dim=0))[:, [0, 2]].abs() < 0.01).all(dim=-1)
        rest_width = verts_pos[is_gripped, 1].max() - verts_pos[is_gripped, 1].min()
        grasp_z = center[2] + TIP_DEPTH
        move_to((center[0], center[1], grasp_z + 0.1))
        move_to((center[0], center[1], grasp_z))
        set_gripper(close=True)
        for _ in range(n_grip):
            track_target()
        move_to((center[0], center[1], grasp_z + 0.15))
        verts_pos = entity.get_verts_pos()
        squash_ratios.append(1.0 - (verts_pos[is_gripped, 1].max() - verts_pos[is_gripped, 1].min()) / rest_width)
        move_to((container_pos[0] + drop_offset_x, container_pos[1], container_size[2] + TIP_DEPTH + 0.06))
        set_gripper(close=False)
        for _ in range(n_grip):
            track_target()
    if camera is not None:
        camera.stop_recording()

    for (entity, container_pos, container_size), name, squash_ratio in zip(
        objects, ("tomato", "jelly", "plush worm"), squash_ratios
    ):
        center = tensor_to_array(entity.get_verts_pos().mean(dim=0))
        is_placed = np.all(np.abs(center[:2] - container_pos[:2]) < 0.5 * np.array(container_size[:2]) - WALL)
        is_placed &= center[2] < container_size[2]
        gs.logger.info(
            f"The {name} is {'placed' if is_placed else 'not placed'}, squashed by {100.0 * squash_ratio:.0f}% across "
            "the grip."
        )

    if args.vis:
        scene.viewer.register_keybinds(
            Keybind("move_away", Key.UP, KeyAction.HOLD, callback=move, args=((0.0, HAND_SPEED, 0.0),)),
            Keybind("move_closer", Key.DOWN, KeyAction.HOLD, callback=move, args=((0.0, -HAND_SPEED, 0.0),)),
            Keybind("move_left", Key.LEFT, KeyAction.HOLD, callback=move, args=((-HAND_SPEED, 0.0, 0.0),)),
            Keybind("move_right", Key.RIGHT, KeyAction.HOLD, callback=move, args=((HAND_SPEED, 0.0, 0.0),)),
            Keybind("move_up", Key.K, KeyAction.HOLD, callback=move, args=((0.0, 0.0, HAND_SPEED),)),
            Keybind("move_down", Key.J, KeyAction.HOLD, callback=move, args=((0.0, 0.0, -HAND_SPEED),)),
            Keybind("close_gripper", Key.SPACE, KeyAction.PRESS, callback=set_gripper, args=(True,)),
            Keybind("open_gripper", Key.SPACE, KeyAction.RELEASE, callback=set_gripper, args=(False,)),
        )
        gs.logger.info(
            "Keys: Up/Down move the hand away from/towards the camera, Left/Right move it left/right, K/J move it "
            "up/down, hold Space to close the gripper."
        )
    while args.vis and scene.viewer.is_alive() and not is_ci:
        track_target()


if __name__ == "__main__":
    main()
