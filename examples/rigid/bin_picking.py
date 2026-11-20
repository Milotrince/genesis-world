"""A Franka picks household objects out of a bin one after the other and sets each down in a tray.

The bin is a non-convex mesh, and the donut, mug, cup and apple are meshes convex-decomposed at build time. Each pick
reads the position of the object, plans a collision-free path to a point above it, descends straight down, grasps and
lifts. A missed grasp or an object dropped back into the bin is retried twice from its new position, then skipped, and
the run logs how many objects reached the tray. Objects at rest hibernate, so the solver skips them until the hand or a
neighbour wakes them. With '--suction' the fingers stay closed and a weld constraint, added and removed at runtime,
holds the object as a suction cup would.

With the viewer (-v), the keyboard drives the hand once the bin is empty:
Up        - Move forward (-x)
Down      - Move backward (+x)
Left      - Move left (-y)
Right     - Move right (+y)
K         - Move up
J         - Move down
Space     - Hold to grasp (the suction cup takes the nearest object), release to let go
Esc       - Quit
"""

import argparse
import os
import xml.etree.ElementTree as ET

from huggingface_hub import snapshot_download

import genesis as gs
from genesis.vis.keybindings import Key, KeyAction, Keybind


# Height of the fingertip pads below the origin of the hand, along its axis.
PAD_DEPTH = 0.103
# The tank mesh is 15 cm long, 11 cm wide and 6 cm tall in its y-up frame, scaled into a wide and shallow bin.
BIN_SCALE = (3.5, 1.5, 3.5)
# Height of the top of the table.
TABLE_HEIGHT = 0.3
BIN_POS = (0.5, -0.25, TABLE_HEIGHT)
BIN_HALF_SIZE = (0.074 * BIN_SCALE[0], 0.057 * BIN_SCALE[2])
TRAY_POS = (0.45, 0.3, TABLE_HEIGHT)
TRAY_SIZE = 0.24
OBJECT_SCALE = 0.5
# The hand points down with its fingers closing along x, or along y.
HAND_QUATS = ((0.0, 0.7071068, 0.7071068, 0.0), (0.0, 1.0, 0.0, 0.0))


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
    parser.add_argument("--suction", action="store_true", help="Hold the objects with a suction cup instead of fingers")
    parser.add_argument("-v", "--vis", action="store_true", help="Show the interactive viewer")
    parser.add_argument("-g", "--gpu", action="store_true", help="Run on GPU instead of CPU")
    parser.add_argument("-r", "--record", action="store_true", help="Record the run to 'out/bin_picking.mp4'")
    args = parser.parse_args()

    gs.init(backend=gs.gpu if args.gpu else gs.cpu, precision="32", seed=0)

    camera_pos = (1.5, -0.35, TABLE_HEIGHT + 0.8)
    camera_lookat = (0.35, 0.0, TABLE_HEIGHT + 0.05)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.01,
        ),
        rigid_options=gs.options.RigidOptions(
            use_hibernation=True,
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
        gs.morphs.Mesh(
            file="meshes/tank.obj",
            scale=BIN_SCALE,
            pos=BIN_POS,
            euler=(90.0, 0.0, 0.0),
            fixed=True,
        ),
    )
    # The tray is a floor and four low walls.
    for size, offset in (
        ((TRAY_SIZE, TRAY_SIZE, 0.01), (0.0, 0.0, 0.005)),
        ((TRAY_SIZE, 0.01, 0.04), (0.0, 0.5 * TRAY_SIZE, 0.02)),
        ((TRAY_SIZE, 0.01, 0.04), (0.0, -0.5 * TRAY_SIZE, 0.02)),
        ((0.01, TRAY_SIZE, 0.04), (0.5 * TRAY_SIZE, 0.0, 0.02)),
        ((0.01, TRAY_SIZE, 0.04), (-0.5 * TRAY_SIZE, 0.0, 0.02)),
    ):
        scene.add_entity(
            gs.morphs.Box(
                pos=tuple(tray + delta for tray, delta in zip(TRAY_POS, offset)),
                size=size,
                fixed=True,
            ),
            surface=gs.surfaces.Rough(
                color=(0.25, 0.35, 0.55),
            ),
        )

    asset_names = ("donut_0", "mug_1", "cup_2", "apple_15")
    n_attempts = 3
    if "PYTEST_VERSION" in os.environ:
        asset_names = asset_names[:1]
        n_attempts = 1
    asset_path = snapshot_download(
        repo_type="dataset",
        repo_id="Genesis-Intelligence/assets",
        revision="4d96c3512df4421d4dd3d626055d0d1ebdfdd7cc",
        allow_patterns=[f"{asset_name}/*" for asset_name in asset_names],
        max_workers=1,
    )
    # The objects fall into the bin from a loose column, and settle before the robot starts.
    objects = [
        scene.add_entity(
            gs.morphs.MJCF(
                file=f"{asset_path}/{asset_name}/output.xml",
                scale=OBJECT_SCALE,
                pos=(
                    BIN_POS[0] + 0.04 * (i_object % 2 - 0.5),
                    BIN_POS[1] + 0.06 * (i_object - 1.5),
                    BIN_POS[2] + 0.15 + 0.08 * i_object,
                ),
            ),
        )
        for i_object, asset_name in enumerate(asset_names)
    ]

    franka_spec = gs.entities.Rigid(
        morph=gs.morphs.MJCF(
            file="xml/franka_emika_panda/panda.xml",
        ),
        material=gs.materials.Rigid(
            gravity_compensation=1.0,
        ),
    )
    # Raise the friction of the fingertip pads (geoms 1 to 5 of each finger link, geom 0 being the finger mesh).
    fingers = franka_spec.select(gs.indices.Link.named("left_finger", "right_finger"))
    fingers.select(gs.indices.Geom.at(slice(1, 6))).set_friction(1.5)
    franka = scene.add_entity(
        spec=franka_spec,
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
    # Start from a ready pose above the table, so that every inverse kinematics solve stays on its elbow-up branch.
    franka.set_qpos((0.0, -0.4, 0.0, -2.4, 0.0, 2.0, 0.785, 0.04, 0.04))
    # The MJCF drives the fingers through a tendon actuator, so position control needs PD gains of its own.
    franka.set_dofs_kp((300, 300), fingers_dofs)
    franka.set_dofs_kv((20, 20), fingers_dofs)
    step_scale = 1.0 if "PYTEST_VERSION" not in os.environ else 0.05
    hand_quat = HAND_QUATS[0]
    held_object = None

    def move_hand(pos, steps, plan=False):
        """Move the hand, pointing down, to a position along a collision-free planned path or a straight line."""
        steps = max(int(steps * step_scale), 2)
        if plan:
            qpos_goal = franka.inverse_kinematics(link=hand, pos=pos, quat=hand_quat)
            path = franka.plan_path(qpos_goal=qpos_goal, num_waypoints=steps)
        else:
            # One inverse kinematics solve per step keeps the hand on the line, where the joints would arc between ends
            start = hand.get_pos()
            points = [start.lerp(start.new_tensor(pos), i_step / steps) for i_step in range(1, steps + 1)]
            path = [franka.inverse_kinematics(link=hand, pos=point, quat=hand_quat) for point in points]
        for waypoint in path:
            franka.control_dofs_position(waypoint[:7], arm_dofs)
            scene.step()
        # The position controller lags behind the path, so give the hand time to reach its target.
        for _ in range(int(30 * step_scale)):
            scene.step()

    def grasp(obj):
        """Close the fingers, or weld the object to the hand when it has a suction cup."""
        nonlocal held_object
        if args.suction:
            scene.rigid_solver.add_weld_constraint(obj.base_link.idx, hand.idx)
            held_object = obj
        else:
            # The objects weigh a few grams, so the finger controller grips them by its stiffness alone. A force
            # command closing on nothing would drive the fingers past each other.
            franka.control_dofs_position((0.0, 0.0), fingers_dofs)

    def release():
        nonlocal held_object
        if args.suction:
            if held_object is not None:
                scene.rigid_solver.delete_weld_constraint(held_object.base_link.idx, hand.idx)
                held_object = None
        else:
            franka.control_dofs_position((0.04, 0.04), fingers_dofs)

    if args.suction:
        franka.control_dofs_position((0.0, 0.0), fingers_dofs)
    else:
        release()
    # Let the objects fall into the bin and come to rest.
    for _ in range(int(150 * step_scale)):
        scene.step()

    if camera is not None:
        camera.start_recording(save_to_filename="out/bin_picking.mp4", fps=30)

    def is_inside(obj, center, half_size):
        x, y, _ = obj.get_pos().tolist()
        return abs(x - center[0]) < half_size[0] and abs(y - center[1]) < half_size[1]

    # Pick the objects from the top of the pile down. A missed grasp or an object dropped on the way back into the bin
    # is retried from the new position of the object, twice at most.
    for asset_name, obj in sorted(zip(asset_names, objects), key=lambda item: -item[1].get_pos()[2].item()):
        for i_attempt in range(n_attempts):
            if not is_inside(obj, BIN_POS, BIN_HALF_SIZE):
                break
            x, y, _ = obj.get_pos().tolist()
            top_z = obj.get_AABB()[1, 2].item()
            # The hand is long along its fingers, so it lies parallel to the bin wall nearest to the object.
            is_near_x_wall = BIN_HALF_SIZE[0] - abs(x - BIN_POS[0]) < BIN_HALF_SIZE[1] - abs(y - BIN_POS[1])
            hand_quat = HAND_QUATS[int(is_near_x_wall)]
            move_hand((x, y, top_z + 0.25), steps=150, plan=True)
            # The fingers close 2 cm below the top of the object, and the suction cup presses on it.
            move_hand((x, y, top_z + PAD_DEPTH - (0.0 if args.suction else 0.02)), steps=120)
            grasp(obj)
            for _ in range(int(30 * step_scale)):
                scene.step()
            move_hand((x, y, top_z + 0.25), steps=80)
            if obj.get_pos()[2].item() > BIN_POS[2] + 0.12:
                hand_quat = HAND_QUATS[0]
                move_hand((TRAY_POS[0], TRAY_POS[1], TRAY_POS[2] + 0.35), steps=250, plan=True)
                # Set the object down on the tray floor before letting go, so that it neither bounces out nor hangs
                # on a finger.
                move_hand((TRAY_POS[0], TRAY_POS[1], TRAY_POS[2] + 0.17), steps=60)
            release()
            for _ in range(int(30 * step_scale)):
                scene.step()
            if is_inside(obj, TRAY_POS, (0.5 * TRAY_SIZE, 0.5 * TRAY_SIZE)):
                gs.logger.info(f"The {asset_name} is in the tray.")
                break
            gs.logger.info(f"Attempt {i_attempt + 1} at the {asset_name} failed.")
        move_hand((TRAY_POS[0], TRAY_POS[1], TRAY_POS[2] + 0.35), steps=60)

    n_in_tray = sum(is_inside(obj, TRAY_POS, (0.5 * TRAY_SIZE, 0.5 * TRAY_SIZE)) for obj in objects)
    gs.logger.info(f"{n_in_tray} of {len(objects)} objects ended up in the tray.")

    if camera is not None:
        camera.stop_recording()

    # Keybind callbacks run on the viewer thread, so they only set the requests that the main loop carries out.
    target_pos = hand.get_pos()
    is_grasp_requested = False
    is_running = True

    def move(delta):
        target_pos[:] += target_pos.new_tensor(delta)

    def request_grasp(is_requested):
        nonlocal is_grasp_requested
        is_grasp_requested = is_requested

    def stop():
        nonlocal is_running
        is_running = False

    if args.vis:
        step = 0.003
        scene.viewer.register_keybinds(
            Keybind("move_forward", Key.UP, KeyAction.HOLD, callback=move, args=((-step, 0.0, 0.0),)),
            Keybind("move_back", Key.DOWN, KeyAction.HOLD, callback=move, args=((step, 0.0, 0.0),)),
            Keybind("move_left", Key.LEFT, KeyAction.HOLD, callback=move, args=((0.0, -step, 0.0),)),
            Keybind("move_right", Key.RIGHT, KeyAction.HOLD, callback=move, args=((0.0, step, 0.0),)),
            Keybind("move_up", Key.K, KeyAction.HOLD, callback=move, args=((0.0, 0.0, step),)),
            Keybind("move_down", Key.J, KeyAction.HOLD, callback=move, args=((0.0, 0.0, -step),)),
            Keybind("grasp", Key.SPACE, KeyAction.PRESS, callback=request_grasp, args=(True,)),
            Keybind("release", Key.SPACE, KeyAction.RELEASE, callback=request_grasp, args=(False,)),
            Keybind("quit", Key.ESCAPE, KeyAction.RELEASE, callback=stop),
        )
        gs.logger.info("Keys: arrows move in x/y, K/J move up/down, hold Space to grasp, Esc quits.")
    is_grasping = False
    while args.vis and scene.viewer.is_alive() and is_running and "PYTEST_VERSION" not in os.environ:
        if is_grasp_requested and not is_grasping:
            # The suction cup takes the object nearest to the fingertips.
            fingertips_pos = hand.get_pos() - hand.get_pos().new_tensor((0.0, 0.0, PAD_DEPTH))
            grasp(min(objects, key=lambda obj: (obj.get_pos() - fingertips_pos).norm().item()))
            is_grasping = True
        elif not is_grasp_requested and is_grasping:
            release()
            is_grasping = False
        qpos = franka.inverse_kinematics(link=hand, pos=target_pos, quat=hand_quat)
        franka.control_dofs_position(qpos[:7], arm_dofs)
        scene.step()


if __name__ == "__main__":
    main()
