"""Two Frankas fold a towel in half on a table, pinching its corners between their fingers.

The towel is an Incremental Potential Contact (IPC) cloth whose near edge overhangs the table. Each Franka slides its
open fingers around one near corner, closes them, swings the corner over the fold line onto the far corner, releases it
and lifts its hand away. The IPC coupler resolves the frictional contact between the fingers and the cloth, which alone
holds the corners, and the contact of the cloth with itself and with the table. The log reports the fold quality: the
share of the near-half vertices that end up over the far half, and how far the near edge lands from the far edge.

The IPC solver runs on a CUDA GPU whatever the backend, which only selects the device of the rigid solver. With the
viewer (-v), the keys below then drive the hand of the Franka on the right, so it can grab the towel and drag it.

Keyboard controls:
Up        - Move forward (-x)
Down      - Move backward (+x)
Left      - Move left (-y)
Right     - Move right (+y)
K         - Move up
J         - Move down
N/M       - Yaw left/right (rotate around z)
U/Y       - Pitch up/down (rotate around y)
T/G       - Roll left/right (rotate around x)
Backslash - Reset the hand target
Space     - Hold to close the gripper, release to open it
Esc       - Quit

Plus all default viewer controls (press 'i' to see them).
"""

import argparse
import os

import numpy as np
from huggingface_hub import snapshot_download

import genesis as gs
import genesis.utils.geom as gu
from genesis.utils.misc import tensor_to_array
from genesis.vis.keybindings import Key, KeyAction, Keybind


DT = 0.01
RECORDING_FPS = 50

# The towel is square, its near edge overhanging the near edge of the table so that the fingers can close around it.
TOWEL_SIZE = 0.3
TABLE_HEIGHT = 0.2
TABLE_OVERHANG = 0.035
# Distance from the hand link to the point between the fingertips where the cloth is pinched.
HAND_TO_PINCH = 0.105

DELTA_POS = 0.003
DELTA_ROT = 0.02


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", help="Show the interactive viewer")
    parser.add_argument("-g", "--gpu", action="store_true", help="Run the rigid solver on GPU instead of CPU")
    parser.add_argument("-r", "--record", action="store_true", help="Record the fold to 'out/ipc_fold_towel.mp4'")
    args = parser.parse_args()

    gs.init(backend=gs.gpu if args.gpu else gs.cpu, logging_level="info")

    camera_pos = (1.0, -0.4, 0.95)
    camera_lookat = (0.0, 0.0, 0.22)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=DT,
        ),
        coupler_options=gs.options.IPCCouplerOptions(
            contact_d_hat=0.002,
            constraint_strength_translation=100.0,
            constraint_strength_rotation=100.0,
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
    # The table hovers a few millimetres above the floor, since IPC refuses to start from geometries in contact.
    x_table_edge = -0.5 * TOWEL_SIZE + TABLE_OVERHANG
    scene.add_entity(
        morph=gs.morphs.Box(
            pos=(x_table_edge + 0.25, 0.0, 0.5 * TABLE_HEIGHT + 0.005),
            size=(0.5, TOWEL_SIZE + 0.1, TABLE_HEIGHT - 0.01),
            fixed=True,
        ),
        material=gs.materials.Rigid(
            coup_friction=0.5,
            coup_type="ipc_only",
        ),
        surface=gs.surfaces.Rough(
            color=(0.55, 0.4, 0.3),
        ),
    )
    # Each Franka stands beside the table, level with the fold line, and only its fingers take part in IPC contact.
    frankas = [
        scene.add_entity(
            morph=gs.morphs.MJCF(
                file="xml/franka_emika_panda/panda_non_overlap.xml",
                pos=(-0.3, y_sign * 0.45, 0.0),
                euler=(0.0, 0.0, -y_sign * 90.0),
            ),
            material=gs.materials.Rigid(
                coup_friction=0.8,
                coup_type="two_way_soft_constraint",
                coup_links=("left_finger", "right_finger"),
            ),
        )
        for y_sign in (1.0, -1.0)
    ]
    asset_path = snapshot_download(
        repo_type="dataset",
        repo_id="Genesis-Intelligence/assets",
        revision="8aa8fcd60500b9f3a36c356080224bdb1be9ee59",
        allow_patterns="IPC/grid20x20.obj",
        max_workers=1,
    )
    towel = scene.add_entity(
        morph=gs.morphs.Mesh(
            file=f"{asset_path}/IPC/grid20x20.obj",
            scale=TOWEL_SIZE,
            pos=(0.0, 0.0, TABLE_HEIGHT + 0.003),
            euler=(90.0, 0.0, 0.0),
        ),
        material=gs.materials.FEM.Cloth(
            E=5e4,
            nu=0.3,
            rho=200.0,
            thickness=0.001,
            bending_stiffness=100.0,
            friction_mu=0.8,
        ),
        surface=gs.surfaces.Plastic(
            color=(0.85, 0.3, 0.25, 1.0),
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

    motors_dof, fingers_dof = slice(0, 7), slice(7, 9)
    hands = [franka.get_link("hand") for franka in frankas]
    for franka in frankas:
        franka.set_dofs_kp(500.0, dofs_idx_local=fingers_dof)
        franka.set_dofs_kv(50.0, dofs_idx_local=fingers_dof)

    # The fingertips point along +x at the overhanging corners and open vertically, so that one finger slides under
    # the cloth and the other over it.
    pinch_quat = gu.R_to_quat(np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))
    approach_dir = np.array([1.0, 0.0, 0.0])
    pinch_pos = np.array(
        [
            [-0.5 * TOWEL_SIZE + 0.5 * TABLE_OVERHANG, y_sign * (0.5 * TOWEL_SIZE - 0.02), TABLE_HEIGHT]
            for y_sign in (1.0, -1.0)
        ]
    )
    # The target of each hand, written by the scripted fold and then by the keys.
    targets_pos = pinch_pos - (HAND_TO_PINCH + 0.06) * approach_dir
    targets_quat = np.tile(pinch_quat, (2, 1))
    is_gripper_closed = np.zeros(2, dtype=gs.np_bool)
    target_init_pos, target_init_quat = targets_pos[0].copy(), targets_quat[0].copy()

    for franka, hand, target_pos, target_quat in zip(frankas, hands, targets_pos, targets_quat):
        qpos = franka.inverse_kinematics(link=hand, pos=target_pos, quat=target_quat)
        qpos[fingers_dof] = 0.04
        franka.set_qpos(qpos)
        franka.control_dofs_position(qpos)

    def track_targets():
        for franka, hand, target_pos, target_quat, is_closed in zip(
            frankas, hands, targets_pos, targets_quat, is_gripper_closed
        ):
            qpos = franka.inverse_kinematics(link=hand, pos=target_pos, quat=target_quat, dofs_idx_local=motors_dof)
            franka.control_dofs_position(qpos[motors_dof], dofs_idx_local=motors_dof)
            # The closed target lies past the closed position, so that the fingers keep squeezing the cloth.
            franka.control_dofs_position(-0.02 if is_closed else 0.04, dofs_idx_local=fingers_dof)
        scene.step()

    def move_hands_to(goals_pos, goals_quat, n_steps):
        starts_pos, starts_quat = targets_pos.copy(), targets_quat.copy()
        for i_step in range(n_steps):
            ratio = 0.5 - 0.5 * np.cos(np.pi * (i_step + 1) / n_steps)
            targets_pos[:] = starts_pos + ratio * (goals_pos - starts_pos)
            targets_quat[:] = gu.slerp(starts_quat, goals_quat, np.full(2, ratio))
            track_targets()

    is_pytest = "PYTEST_VERSION" in os.environ
    n_approach, n_close, n_fold, n_settle, n_retreat = (10, 10, 30, 10, 10) if is_pytest else (100, 50, 300, 50, 100)

    towel_verts_pos_0 = tensor_to_array(towel.get_verts_pos())
    if camera is not None:
        camera.start_recording(save_to_filename="out/ipc_fold_towel.mp4", fps=RECORDING_FPS)

    # Slide the open fingers around the overhanging corners, then pinch.
    move_hands_to(pinch_pos - HAND_TO_PINCH * approach_dir, targets_quat.copy(), n_approach)
    is_gripper_closed[:] = True
    for _ in range(n_close):
        track_targets()

    # Swing the pinched corners about the fold line. The wrists turn at half the rate of the swing, so that the hands
    # end pointing down onto the far half instead of upside down. The swing stops short of a half turn and a little
    # high, so that the near half lands on the far half rather than in it.
    fold_center = np.array([0.0, 0.0, TABLE_HEIGHT])
    for i_step in range(n_fold):
        angle = np.pi * 0.96 * (0.5 - 0.5 * np.cos(np.pi * (i_step + 1) / n_fold))
        fold_R = gu.axis_angle_to_R(np.array([0.0, 1.0, 0.0]), angle)
        wrist_quat = gu.R_to_quat(gu.axis_angle_to_R(np.array([0.0, 1.0, 0.0]), 0.5 * angle))
        targets_quat[:] = gu.transform_quat_by_quat(np.tile(pinch_quat, (2, 1)), np.tile(wrist_quat, (2, 1)))
        pinch_dir = gu.transform_by_quat(approach_dir, wrist_quat)
        fold_pos = fold_center + (pinch_pos - fold_center) @ fold_R.T + [0.0, 0.0, 0.03 * angle / np.pi]
        targets_pos[:] = fold_pos - HAND_TO_PINCH * pinch_dir
        track_targets()

    # Release, let the cloth settle, then lift the hands away.
    is_gripper_closed[:] = False
    for _ in range(n_settle):
        track_targets()
    move_hands_to(targets_pos + [0.0, 0.0, 0.15], targets_quat.copy(), n_retreat)

    if camera is not None:
        camera.stop_recording()

    # Fold quality: a near-half vertex is folded when it lies over the footprint of the far half and above the cloth
    # of the far half. The edge error is the distance from each near-edge vertex to the far-edge vertex it should meet.
    towel_verts_pos = tensor_to_array(towel.get_verts_pos())
    x_0, y_0 = towel_verts_pos_0[..., 0], towel_verts_pos_0[..., 1]
    is_near_half = x_0 < -gs.EPS
    is_far_half = x_0 > gs.EPS
    half_width = 0.5 * TOWEL_SIZE + 0.01
    x, y, z = towel_verts_pos[..., 0], towel_verts_pos[..., 1], towel_verts_pos[..., 2]
    is_over_far_half = (x > -0.01) & (x < half_width) & (np.abs(y) < half_width) & (z > z[is_far_half].mean())
    folded_ratio = np.count_nonzero(is_over_far_half & is_near_half) / np.count_nonzero(is_near_half)
    is_near_edge = x_0 < x_0.min() + gs.EPS
    is_far_edge = x_0 > x_0.max() - gs.EPS
    near_edge_order, far_edge_order = np.argsort(y_0[is_near_edge]), np.argsort(y_0[is_far_edge])
    edge_gap = np.linalg.norm(
        towel_verts_pos[is_near_edge][near_edge_order, :2] - towel_verts_pos[is_far_edge][far_edge_order, :2], axis=-1
    )
    gs.logger.info(
        f"Fold quality: {100.0 * folded_ratio:.0f}% of the near half lies over the far half, near edge "
        f"{1e3 * edge_gap.mean():.1f} mm from the far edge on average ({1e3 * edge_gap.max():.1f} mm at worst)."
    )

    if not args.vis or is_pytest:
        return

    def move(dpos):
        targets_pos[0] += dpos

    def rotate(axis_idx, delta):
        delta_xyz = np.zeros(3)
        delta_xyz[axis_idx] = delta
        targets_quat[0] = gu.transform_quat_by_quat(targets_quat[0], gu.xyz_to_quat(delta_xyz))

    def reset_target():
        targets_pos[0], targets_quat[0] = target_init_pos, target_init_quat

    def set_gripper(close):
        is_gripper_closed[0] = close

    is_running = True

    def stop():
        nonlocal is_running
        is_running = False

    scene.viewer.register_keybinds(
        Keybind("move_forward", Key.UP, KeyAction.HOLD, callback=move, args=((-DELTA_POS, 0, 0),)),
        Keybind("move_back", Key.DOWN, KeyAction.HOLD, callback=move, args=((DELTA_POS, 0, 0),)),
        Keybind("move_left", Key.LEFT, KeyAction.HOLD, callback=move, args=((0, -DELTA_POS, 0),)),
        Keybind("move_right", Key.RIGHT, KeyAction.HOLD, callback=move, args=((0, DELTA_POS, 0),)),
        Keybind("move_up", Key.K, KeyAction.HOLD, callback=move, args=((0, 0, DELTA_POS),)),
        Keybind("move_down", Key.J, KeyAction.HOLD, callback=move, args=((0, 0, -DELTA_POS),)),
        Keybind("yaw_left", Key.N, KeyAction.HOLD, callback=rotate, args=(2, DELTA_ROT)),
        Keybind("yaw_right", Key.M, KeyAction.HOLD, callback=rotate, args=(2, -DELTA_ROT)),
        Keybind("pitch_up", Key.U, KeyAction.HOLD, callback=rotate, args=(1, DELTA_ROT)),
        Keybind("pitch_down", Key.Y, KeyAction.HOLD, callback=rotate, args=(1, -DELTA_ROT)),
        Keybind("roll_left", Key.T, KeyAction.HOLD, callback=rotate, args=(0, DELTA_ROT)),
        Keybind("roll_right", Key.G, KeyAction.HOLD, callback=rotate, args=(0, -DELTA_ROT)),
        Keybind("reset_target", Key.BACKSLASH, KeyAction.RELEASE, callback=reset_target),
        Keybind("close_gripper", Key.SPACE, KeyAction.PRESS, callback=set_gripper, args=(True,)),
        Keybind("open_gripper", Key.SPACE, KeyAction.RELEASE, callback=set_gripper, args=(False,)),
        Keybind("quit", Key.ESCAPE, KeyAction.RELEASE, callback=stop),
    )
    gs.logger.info(
        "Keys: arrows move in x/y, K/J move up/down, N/M yaw, U/Y pitch, T/G roll, Backslash resets the target, "
        "hold Space to close the gripper, Esc quits."
    )
    while is_running and scene.viewer.is_alive():
        track_targets()


if __name__ == "__main__":
    main()
