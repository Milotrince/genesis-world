"""
SurfaceDistanceProbe sensors on the palm and fingertips of a Shadow Hand measure the distance to a rubber duck (mesh)
and a box.

Without the viewer, the hand follows a scripted sweep past both objects and logs the distances. With the viewer (-v),
the keyboard drives the hand from its start pose.

Keyboard controls (-v):
    Up / Down       move the hand along +x / -x
    Left / Right    move the hand along +y / -y
    j / k           move the hand down / up
    \\               reset the hand and the objects
"""

import argparse
import os

import numpy as np
import torch

import genesis as gs
import genesis.utils.geom as gu
from genesis.utils.misc import tensor_to_array
from genesis.vis.keybindings import Key, KeyAction, Keybind

# Teleop
KEY_DPOS = 0.015
FORCE_SCALE = 10.0

# Surface distance probe
MAX_RANGE = 0.5

# Objects
DUCK_POS = (-0.2, 0.4, 0.6)
DUCK_QUAT = gu.euler_to_quat((90.0, 0.0, -90.0))
BOX_POS = (-0.3, 0.2, 0.4)
BOX_QUAT = (1, 0, 0, 0)


def main():
    parser = argparse.ArgumentParser(description="Interactive SurfaceDistanceProbe with Shadow Hand")
    parser.add_argument("-v", "--vis", action="store_true", help="Show visualization GUI")
    parser.add_argument("-g", "--gpu", action="store_true", help="Run on GPU instead of CPU")
    parser.add_argument(
        "-r", "--record", action="store_true", help="Record a video of the run to 'out/surface_distance_shadowhand.mp4'"
    )
    args = parser.parse_args()

    gs.init(
        backend=gs.gpu if args.gpu else gs.cpu,
        precision="32",
        logging_level="info",
    )

    camera_pos = (-1.2, 0.6, 1.0)
    camera_lookat = (0.0, 0.0, 0.5)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            gravity=(0.0, 0.0, 0.0),
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=camera_pos,
            camera_lookat=camera_lookat,
        ),
        show_viewer=args.vis,
    )

    hand_pos_init = np.array([0.0, 0.0, 0.35], dtype=np.float32)
    hand_quat_init = gu.euler_to_quat((0.0, 0.0, -90.0))
    robot = scene.add_entity(
        morph=gs.morphs.URDF(
            file="urdf/shadow_hand/shadow_hand.urdf",
            pos=hand_pos_init,
            quat=hand_quat_init,
        ),
    )
    duck = scene.add_entity(
        material=gs.materials.Rigid(),
        morph=gs.morphs.Mesh(
            file="meshes/duck.obj",
            scale=0.06,
            pos=DUCK_POS,
            quat=DUCK_QUAT,
        ),
        surface=gs.surfaces.Default(
            color=(0.95, 0.75, 0.2, 1.0),
        ),
    )
    box = scene.add_entity(
        material=gs.materials.Rigid(),
        morph=gs.morphs.Box(
            size=(0.06, 0.06, 0.08),
            pos=BOX_POS,
        ),
        surface=gs.surfaces.Default(
            color=(0.3, 0.7, 0.4, 1.0),
        ),
    )

    sensors = []
    for link_name in (
        "palm",
        "index_finger_distal",
        "middle_finger_distal",
        "ring_finger_distal",
        "little_finger_distal",
        "thumb_distal",
    ):
        sensor = scene.add_sensor(
            gs.sensors.SurfaceDistanceProbe(
                entity_idx=robot.idx,
                link_idx_local=robot.get_link(link_name).idx_local,
                probe_local_pos=((0.0, 0.0, 0.0),),
                probe_radius=MAX_RANGE,
                track_link_idx=(duck.base_link_idx, box.base_link_idx),
                draw_debug=args.vis,
            )
        )
        sensors.append(sensor)

    camera = None
    if args.record:
        camera = scene.add_camera(
            res=(1280, 720),
            pos=camera_pos,
            lookat=camera_lookat,
        )

    scene.build()

    hand_pos = hand_pos_init.copy()
    for obj in (duck, box):
        obj.set_dofs_kv((0.8, 0.8, 0.8, 0.02, 0.02, 0.02))
        obj.control_dofs_position(0.0)
    robot.set_dofs_kp(FORCE_SCALE / KEY_DPOS)
    robot.set_dofs_kv(0.1 * FORCE_SCALE / KEY_DPOS)
    robot.control_dofs_position(robot.get_dofs_position())

    def translate(index: int, is_negative: bool):
        hand_pos[index] += (-1 if is_negative else 1) * KEY_DPOS

    if args.vis:

        def reset_pose():
            hand_pos[:] = hand_pos_init
            robot.set_pos(hand_pos_init)
            duck.set_pos(DUCK_POS)
            duck.set_quat(DUCK_QUAT)
            box.set_pos(BOX_POS)
            box.set_quat(BOX_QUAT)

        scene.viewer.register_keybinds(
            Keybind("move_forward", Key.UP, KeyAction.HOLD, callback=translate, args=(0, False)),
            Keybind("move_back", Key.DOWN, KeyAction.HOLD, callback=translate, args=(0, True)),
            Keybind("move_right", Key.RIGHT, KeyAction.HOLD, callback=translate, args=(1, True)),
            Keybind("move_left", Key.LEFT, KeyAction.HOLD, callback=translate, args=(1, False)),
            Keybind("move_down", Key.J, KeyAction.HOLD, callback=translate, args=(2, True)),
            Keybind("move_up", Key.K, KeyAction.HOLD, callback=translate, args=(2, False)),
            Keybind("reset", Key.BACKSLASH, KeyAction.RELEASE, callback=reset_pose),
        )
        gs.logger.info("Keyboard: [Up/Down/Left/Right] move hand XY, [j/k] down/up, [\\] reset")

    if camera is not None:
        camera.start_recording(save_to_filename="out/surface_distance_shadowhand.mp4", fps=round(1.0 / scene.dt))

    if args.vis:
        while scene.viewer.is_alive():
            robot.control_dofs_position(hand_pos, dofs_idx_local=slice(0, 3))
            scene.step()
    else:
        # The same moves the arrow keys and j/k issue, one per step: the hand climbs, sweeps past the duck, then dips
        # towards the box.
        scripted_moves = ((2, False, 15), (1, False, 25), (0, True, 15), (2, True, 20), (1, True, 10))
        if "PYTEST_VERSION" in os.environ:
            scripted_moves = scripted_moves[:1]
        i_step = 0
        for index, is_negative, n_steps in scripted_moves:
            for _ in range(n_steps):
                translate(index, is_negative)
                robot.control_dofs_position(hand_pos, dofs_idx_local=slice(0, 3))
                scene.step()
                if i_step % 10 == 0:
                    distances = torch.cat([sensor.read() for sensor in sensors], dim=-1)
                    gs.logger.info(f"Step {i_step}: palm and fingertip distances {tensor_to_array(distances).round(3)}")
                i_step += 1
    if camera is not None:
        camera.stop_recording()


if __name__ == "__main__":
    main()
