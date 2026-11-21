"""A Franka pokes blocks out of a Jenga-like tower with one fingertip, until it takes one block too many and the tower
falls.

The tower stands on a block on a table, three blocks per layer and each layer turned by a quarter turn. The hand points
forward with its gripper open and its left finger pushes the middle block of three low layers until it sticks out of the
far side, the outer blocks carrying the tower meanwhile. It then pushes both outer blocks of the lowest of those layers,
which shifts the support of the tower past its centre of mass, and the tower topples. Contacts use the elliptic friction
cone. With '--contact-patch' the contact points between the faces of the blocks come from the contact patch instead of
the default multi-contact detection, and either run logs how much the tower settles before the first poke.

Keyboard controls (-v):
Up        - Push forward (+x)
Down      - Pull back (-x)
Left      - Move left (+y)
Right     - Move right (-y)
K         - Move up
J         - Move down
Esc       - Quit
"""

import argparse
import os
import xml.etree.ElementTree as ET

import torch

import genesis as gs
from genesis.vis.keybindings import Key, KeyAction, Keybind


# Size of a block (length, width, height), and the gaps that keep neighbouring blocks from starting in contact. The
# blocks are taller and wider than the 21 x 24 mm cross-section of a finger, so that a poke touches one block only.
BLOCK_SIZE = (0.085, 0.028, 0.025)
SIDE_GAP = 1e-3
LAYER_GAP = 5e-4
# The middle block of a layer is thinner, so that the layer above rests on the outer blocks alone, as on the loose
# blocks a player looks for in a real tower. Pushing it out then drags nothing above it.
MIDDLE_THINNING = 2e-3
N_LAYERS = 12
# Height of the top of the table.
TABLE_HEIGHT = 0.3
# The tower stands on a block on the table, which raises the poked layers to a height the forward-pointing hand
# reaches away from its joint limits.
PEDESTAL_SIZE = (0.1, 0.1, 0.17)
TOWER_POS = (0.55, 0.0, TABLE_HEIGHT + PEDESTAL_SIZE[2])
# Distance from the origin of the hand to its fingertips, along its axis, and from its axis to the middle of the left
# finger of the open gripper. Both closed fingers side by side are wider than a block, so the left finger of the open
# gripper pokes alone, the right one passing beside the tower.
TIP_DEPTH = 0.11
FINGER_OFFSET = 0.054
# The hand points along +x, with its fingers closing along y.
HAND_QUAT = (0.7071068, 0.0, 0.7071068, 0.0)
# The fingers reach about 5 cm in front of the hand, which bounds how deep a poke goes into the tower.
# A deeper poke frees the first outer block far enough for the tower to fall before the second one is pushed.
PUSH_DEPTH = 0.045


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
    parser.add_argument("--contact-patch", action="store_true", help="Detect the contacts between faces as a patch")
    parser.add_argument("-v", "--vis", action="store_true", help="Show the interactive viewer")
    parser.add_argument("-g", "--gpu", action="store_true", help="Run on GPU instead of CPU")
    parser.add_argument("-r", "--record", action="store_true", help="Record the run to 'out/jenga.mp4'")
    args = parser.parse_args()

    gs.init(backend=gs.gpu if args.gpu else gs.cpu, precision="32", seed=0)

    camera_pos = (0.9, -0.95, TOWER_POS[2] + 0.45)
    camera_lookat = (0.48, 0.0, TOWER_POS[2] + 0.1)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.01,
            # A tall stack of light blocks needs stiff contacts, which need a short substep.
            substeps=4,
        ),
        rigid_options=gs.options.RigidOptions(
            friction_cone=gs.friction_cone.elliptic,
            # The contact patch is computed inside GJK.
            use_gjk_collision=args.contact_patch,
            enable_contact_patch=args.contact_patch,
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
            file=table_mjcf(),
        ),
    )
    scene.add_entity(
        gs.morphs.Box(
            pos=(TOWER_POS[0], TOWER_POS[1], TABLE_HEIGHT + 0.5 * PEDESTAL_SIZE[2]),
            size=PEDESTAL_SIZE,
            fixed=True,
        ),
        surface=gs.surfaces.Default(
            color=(0.95, 0.95, 0.95),
        ),
    )
    # Even layers run along x and odd ones along y. blocks[i_layer][i_block] counts the blocks of a layer from -y or -x.
    length, width, height = BLOCK_SIZE
    blocks = []
    for i_layer in range(N_LAYERS):
        z = TOWER_POS[2] + (i_layer + 0.5) * (height + LAYER_GAP)
        layer_blocks = []
        for i_block in range(3):
            offset = (i_block - 1) * (width + SIDE_GAP)
            is_along_x = i_layer % 2 == 0
            block_height = height - (MIDDLE_THINNING if i_block == 1 else 0.0)
            layer_blocks.append(
                scene.add_entity(
                    gs.morphs.Box(
                        pos=(
                            TOWER_POS[0] + (0.0 if is_along_x else offset),
                            TOWER_POS[1] + (offset if is_along_x else 0.0),
                            z - 0.5 * (height - block_height),
                        ),
                        size=(length, width, block_height) if is_along_x else (width, length, block_height),
                    ),
                    material=gs.materials.Rigid(
                        rho=600.0,
                        friction=0.5,
                    ),
                    surface=gs.surfaces.Rough(
                        color=(0.85, 0.7, 0.5) if (i_layer + i_block) % 2 == 0 else (0.8, 0.62, 0.42),
                    ),
                )
            )
        blocks.append(layer_blocks)

    franka = scene.add_entity(
        gs.morphs.MJCF(
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

    hand = franka.get_link("hand")
    arm_dofs = range(7)
    fingers_dofs = range(7, 9)
    step_scale = 1.0 if "PYTEST_VERSION" not in os.environ else 0.05
    # Start from a ready pose, so that every inverse kinematics solve stays on its elbow-up branch.
    franka.set_qpos((0.0, -0.4, 0.0, -2.4, 0.0, 2.0, 0.785, 0.04, 0.04))
    # The MJCF drives the fingers through a tendon actuator, so position control needs PD gains of its own.
    franka.set_dofs_kp((300, 300), fingers_dofs)
    franka.set_dofs_kv((20, 20), fingers_dofs)
    franka.control_dofs_position((0.04, 0.04), fingers_dofs)
    all_blocks = [block for layer_blocks in blocks for block in layer_blocks]

    def move_tips(pos, steps, is_joint_space=False):
        """Move the left fingertip to a position, along a straight line of the fingertip or of the joints."""
        hand_pos = (pos[0] - TIP_DEPTH, pos[1] - FINGER_OFFSET, pos[2])
        steps = max(int(steps * step_scale), 2)
        if is_joint_space:
            qpos_start = franka.get_qpos()
            qpos_goal = franka.inverse_kinematics(link=hand, pos=hand_pos, quat=HAND_QUAT)
            path = [qpos_start.lerp(qpos_goal, i_step / steps) for i_step in range(1, steps + 1)]
        else:
            # One inverse kinematics solve per step keeps the hand on the line, where the joints would arc between ends
            start = hand.get_pos()
            points = [start.lerp(start.new_tensor(hand_pos), i_step / steps) for i_step in range(1, steps + 1)]
            path = [franka.inverse_kinematics(link=hand, pos=point, quat=HAND_QUAT) for point in points]
        for waypoint in path:
            franka.control_dofs_position(waypoint[:7], arm_dofs)
            scene.step()
        # The position controller lags behind the path, so give the hand time to reach its target.
        for _ in range(int(30 * step_scale)):
            scene.step()

    if camera is not None:
        camera.start_recording(save_to_filename="out/jenga.mp4", fps=25)

    x_near = TOWER_POS[0] - 0.5 * length
    if args.vis:
        # The hand starts turned forward, its fingertip in front of the middle block of layer 2.
        _, y, z = blocks[2][1].get_pos().tolist()
        tips_target = hand.get_pos().new_tensor((x_near - 0.08, y, z))
        tips_offset = tips_target.new_tensor((TIP_DEPTH, FINGER_OFFSET, 0.0))
        franka.set_qpos(franka.inverse_kinematics(link=hand, pos=tips_target - tips_offset, quat=HAND_QUAT))
        is_running = True

        # Keybind callbacks run on the viewer thread, so they only move the target that the main loop tracks.
        def move(delta):
            tips_target[:] += tips_target.new_tensor(delta)

        def stop():
            nonlocal is_running
            is_running = False

        step = 0.002
        scene.viewer.register_keybinds(
            Keybind("push", Key.UP, KeyAction.HOLD, callback=move, args=((step, 0.0, 0.0),)),
            Keybind("pull", Key.DOWN, KeyAction.HOLD, callback=move, args=((-step, 0.0, 0.0),)),
            Keybind("move_left", Key.LEFT, KeyAction.HOLD, callback=move, args=((0.0, step, 0.0),)),
            Keybind("move_right", Key.RIGHT, KeyAction.HOLD, callback=move, args=((0.0, -step, 0.0),)),
            Keybind("move_up", Key.K, KeyAction.HOLD, callback=move, args=((0.0, 0.0, step),)),
            Keybind("move_down", Key.J, KeyAction.HOLD, callback=move, args=((0.0, 0.0, -step),)),
            Keybind("quit", Key.ESCAPE, KeyAction.RELEASE, callback=stop),
        )
        gs.logger.info("Keys: Up/Down push and pull, Left/Right move sideways, K/J move up/down, Esc quits.")
        while scene.viewer.is_alive() and is_running:
            qpos = franka.inverse_kinematics(link=hand, pos=tips_target - tips_offset, quat=HAND_QUAT)
            franka.control_dofs_position(qpos[:7], arm_dofs)
            scene.step()
    else:
        # Let the tower settle before the first poke.
        blocks_pos_start = torch.stack([block.get_pos() for block in all_blocks])
        for _ in range(int(100 * step_scale)):
            scene.step()
        blocks_drift = torch.stack([block.get_pos() for block in all_blocks]) - blocks_pos_start
        gs.logger.info(
            f"Contact {'patch' if args.contact_patch else 'multi-contact detection'}: the tower settled by "
            f"{1e3 * blocks_drift[:, 2].abs().max().item():.2f} mm vertically and "
            f"{1e3 * blocks_drift[:, :2].norm(dim=-1).max().item():.3f} mm horizontally."
        )
        top_z = blocks[-1][1].get_pos()[2].item()

        # Push the middle block of layers 2, 4 and 6, then both outer blocks of layer 2, the second of which leaves the
        # centre of mass of the tower above the far half of its support.
        pokes = ((2, 1), (4, 1), (6, 1), (2, 2), (2, 0))
        if "PYTEST_VERSION" in os.environ:
            pokes = pokes[:1]
        # The hand turns forward in front of the tower, then moves between pokes along straight lines clear of its face.
        _, y, z = blocks[pokes[0][0]][pokes[0][1]].get_pos().tolist()
        move_tips((x_near - 0.08, y, z + 0.05), steps=200, is_joint_space=True)
        for i_layer, i_block in pokes:
            _, y, z = blocks[i_layer][i_block].get_pos().tolist()
            move_tips((x_near - 0.08, y, z), steps=80)
            move_tips((x_near - 0.005, y, z), steps=60)
            move_tips((x_near + PUSH_DEPTH, y, z), steps=150)
            move_tips((x_near - 0.08, y, z), steps=60)
            for _ in range(int(60 * step_scale)):
                scene.step()
            tower_drop = top_z - max(block.get_pos()[2].item() for block in blocks[-1])
            block_shift = blocks[i_layer][i_block].get_pos()[0].item() - TOWER_POS[0]
            gs.logger.info(
                f"Pushed block {i_block} of layer {i_layer} by {1e3 * block_shift:.0f} mm: the top of the tower "
                f"dropped by {1e3 * tower_drop:.1f} mm."
            )
            if tower_drop > 5 * height:
                gs.logger.info("The tower toppled.")
                for _ in range(int(100 * step_scale)):
                    scene.step()
                break

    if camera is not None:
        camera.stop_recording()


if __name__ == "__main__":
    main()
