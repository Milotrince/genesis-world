"""Multi-link inverse kinematics (IK) on a Shadow Hand, whose fingertips and wrist follow targets moving on circles.

By default the index fingertip, the middle fingertip and the wrist each follow a target, and the IK solves over every
DOF of the hand. With --index-chain, the index fingertip alone follows a target, and the IK solves over the DOFs of the
wrist and the index finger only, leaving the rest of the hand still.

With the viewer (-v), move the centre of the circles with the keyboard, during the scripted run and after it:
Up/Down    - Move the circles along -x/+x
Left/Right - Move the circles along -y/+y
K/J        - Move the circles up/down
"""

import argparse
import os

import numpy as np

import genesis as gs
from genesis.vis.keybindings import Key, KeyAction, Keybind


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", help="Show visualization GUI")
    parser.add_argument(
        "--index-chain",
        action="store_true",
        help="Move the index fingertip alone, solving the IK over the wrist and index finger DOFs only",
    )
    parser.add_argument(
        "-r", "--record", action="store_true", help="Record a video of the run to 'out/ik_shadow_hand.mp4'"
    )
    args = parser.parse_args()
    n_steps = (200 if args.index_chain else 100) if "PYTEST_VERSION" not in os.environ else 50

    gs.init(backend=gs.cpu, precision="32", logging_level="info")

    if args.index_chain:
        camera_pos, camera_lookat = (0.5, -0.5, 0.6), (0.0, 0.0, 0.35)
    else:
        camera_pos, camera_lookat = (1.3, -0.3, 0.8), (0.5, 0.5, 0.2)

    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=camera_pos,
            camera_lookat=camera_lookat,
            camera_fov=40,
        ),
        show_viewer=args.vis,
    )

    targets_color = ((1, 0.5, 0.5, 1), (0.5, 1.0, 0.5, 1), (0.5, 0.5, 1.0, 1))
    targets = [
        scene.add_entity(
            gs.morphs.Mesh(
                file="meshes/axis.obj",
                scale=0.05,
            ),
            material=gs.materials.Kinematic(),
            surface=gs.surfaces.Default(
                color=color,
            ),
        )
        for color in targets_color[: 1 if args.index_chain else 3]
    ]
    robot = scene.add_entity(
        morph=gs.morphs.URDF(
            scale=1.0,
            file="urdf/shadow_hand/shadow_hand.urdf",
        ),
        material=gs.materials.Kinematic(),
        surface=gs.surfaces.Reflective(
            color=(0.4, 0.4, 0.4),
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
    scene.reset()
    if camera is not None:
        camera.start_recording(save_to_filename="out/ik_shadow_hand.mp4", fps=round(1.0 / scene.dt))

    target_quat = np.array([1, 0, 0, 0])
    index_finger_distal = robot.get_link("index_finger_distal")
    middle_finger_distal = robot.get_link("middle_finger_distal")
    wrist = robot.get_link("wrist")

    if args.index_chain:
        center = np.array([0.033, -0.01, 0.42])
        dofs_idx_local = []
        for joint in robot.joints:
            if joint.name in ("wrist_joint", "index_finger_joint1", "index_finger_joint2", "index_finger_joint3"):
                dofs_idx_local += joint.dofs_idx_local
    else:
        center = np.array([0.5, 0.5, 0.2])
        dofs_idx_local = None

    if args.vis:
        dpos = 0.002

        def move(delta):
            center[:] += delta

        scene.viewer.register_keybinds(
            Keybind("move_forward", Key.UP, KeyAction.HOLD, callback=move, args=((-dpos, 0, 0),)),
            Keybind("move_back", Key.DOWN, KeyAction.HOLD, callback=move, args=((dpos, 0, 0),)),
            Keybind("move_left", Key.LEFT, KeyAction.HOLD, callback=move, args=((0, -dpos, 0),)),
            Keybind("move_right", Key.RIGHT, KeyAction.HOLD, callback=move, args=((0, dpos, 0),)),
            Keybind("move_up", Key.K, KeyAction.HOLD, callback=move, args=((0, 0, dpos),)),
            Keybind("move_down", Key.J, KeyAction.HOLD, callback=move, args=((0, 0, -dpos),)),
        )
        gs.logger.info("Keys: Up/Down/Left/Right move the circles in the ground plane, K/J move them up/down.")

    # The poses follow the targets along a continuous path, so they are shown at the pace of the simulation.
    rate = gs.tools.Rate(1.0 / scene.dt)

    def track_circles(i):
        angle = i / 90 * np.pi
        if args.index_chain:
            links = [index_finger_distal]
            poss = [center + np.array([0.0, np.cos(angle) - 1.0, np.sin(angle) - 1.0]) * 0.05]
        else:
            index_finger_pos = center + np.array([np.cos(angle), np.sin(angle), 0]) * 0.1
            middle_finger_pos = center + np.array([np.cos(angle), np.sin(angle), 0]) * 0.13
            links = [index_finger_distal, middle_finger_distal, wrist]
            poss = [index_finger_pos, middle_finger_pos, index_finger_pos - np.array([0, 0, 0.20])]

        for target, pos in zip(targets, poss):
            target.set_qpos(np.concatenate([pos, target_quat]))

        qpos = robot.inverse_kinematics_multilink(links=links, poss=poss, dofs_idx_local=dofs_idx_local)

        robot.set_qpos(qpos)
        scene.step()
        if args.vis:
            rate.sleep()

    for i in range(n_steps):
        track_circles(i)
    if camera is not None:
        camera.stop_recording()
    # The viewer keeps the IK running on the circles the keys move.
    i = n_steps
    while args.vis and scene.viewer.is_alive() and "PYTEST_VERSION" not in os.environ:
        track_circles(i)
        i += 1


if __name__ == "__main__":
    main()
