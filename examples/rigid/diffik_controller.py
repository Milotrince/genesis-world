"""A damped least-squares differential IK controller drives an arm along a circle traced by a target frame.

With the viewer (-v), move the centre of the circle with the keyboard, during the scripted run and after it:
Up/Down    - Move the circle along +y/-y (away from/towards the camera)
Left/Right - Move the circle along -x/+x
K/J        - Move the circle up/down
"""

import argparse
import os

import numpy as np

import genesis as gs
from genesis.utils.misc import tensor_to_array
from genesis.vis.keybindings import Key, KeyAction, Keybind

config = {
    "ur5e": {"mjcf_file": "xml/universal_robots_ur5e/ur5e.xml", "end_effector_link": "ee_virtual_link"},
    "panda": {"mjcf_file": "xml/franka_emika_panda/panda.xml", "end_effector_link": "left_finger"},
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", help="Show visualization GUI")
    parser.add_argument("-g", "--gpu", action="store_true", help="Run on GPU instead of CPU")
    parser.add_argument("--robot", choices=["panda", "ur5e"], default="ur5e", help="Select robot model (panda or ur5e)")
    parser.add_argument(
        "-r", "--record", action="store_true", help="Record a video of the run to 'out/diffik_controller.mp4'"
    )
    args = parser.parse_args()
    n_steps = 720 if "PYTEST_VERSION" not in os.environ else 20

    backend = gs.gpu if args.gpu else gs.cpu
    gs.init(backend=backend)

    camera_pos = (0.0, -2, 1.5)
    camera_lookat = (0.0, 0.0, 0.5)

    scene = gs.Scene(
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

    plane = scene.add_entity(
        gs.morphs.Plane(),
        material=gs.materials.Kinematic(),
    )

    robot_config = config[args.robot]
    mjcf_file = robot_config["mjcf_file"]
    robot = scene.add_entity(
        gs.morphs.MJCF(
            file=mjcf_file,
        ),
        material=gs.materials.Kinematic(),
    )

    print("links=", robot.links)

    target_entity = scene.add_entity(
        gs.morphs.Mesh(
            file="meshes/axis.obj",
            scale=0.10,
        ),
        material=gs.materials.Kinematic(),
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
            fov=40,
        )

    scene.build()
    if camera is not None:
        camera.start_recording(save_to_filename="out/diffik_controller.mp4", fps=round(1.0 / scene.dt))

    target_quat = np.array([0, 1, 0, 0])
    center = np.array([0.5, 0, 0.5])
    r = 0.1
    damping = 1e-4
    diag = damping * np.eye(6)

    end_effector_link = robot_config["end_effector_link"]
    ee_link = robot.get_link(end_effector_link)

    if args.vis:
        dpos = 0.002

        def move(delta):
            center[:] += delta

        scene.viewer.register_keybinds(
            Keybind("move_forward", Key.UP, KeyAction.HOLD, callback=move, args=((0, dpos, 0),)),
            Keybind("move_back", Key.DOWN, KeyAction.HOLD, callback=move, args=((0, -dpos, 0),)),
            Keybind("move_left", Key.LEFT, KeyAction.HOLD, callback=move, args=((-dpos, 0, 0),)),
            Keybind("move_right", Key.RIGHT, KeyAction.HOLD, callback=move, args=((dpos, 0, 0),)),
            Keybind("move_up", Key.K, KeyAction.HOLD, callback=move, args=((0, 0, dpos),)),
            Keybind("move_down", Key.J, KeyAction.HOLD, callback=move, args=((0, 0, -dpos),)),
        )
        gs.logger.info("Keys: Up/Down move the circle along +y/-y, Left/Right along -x/+x, K/J up/down.")

    # The poses follow the target along a continuous path, so they are shown at the pace of the simulation.
    rate = gs.tools.Rate(1.0 / scene.dt)

    def track_circle(i):
        target_pos = center + np.array([np.cos(i / 360 * np.pi), np.sin(i / 360 * np.pi), 0]) * r
        target_entity.set_qpos(np.concatenate([target_pos, target_quat]))

        # Position error.
        error_pos = target_pos - tensor_to_array(ee_link.get_pos())

        # Orientation error.
        ee_quat = tensor_to_array(ee_link.get_quat())
        error_quat = gs.transform_quat_by_quat(gs.inv_quat(ee_quat), target_quat)
        error_rotvec = gs.quat_to_rotvec(error_quat)

        error = np.concatenate([error_pos, error_rotvec])

        jac = tensor_to_array(robot.get_jacobian(link=ee_link))
        dq = jac.T @ np.linalg.solve(jac @ jac.T + diag, error)
        q = tensor_to_array(robot.get_qpos()) + dq

        robot.set_qpos(q)
        scene.step()
        if args.vis:
            rate.sleep()

    for i in range(n_steps):
        track_circle(i)
    if camera is not None:
        camera.stop_recording()
    # The viewer keeps the controller running on the circle the keys move.
    i = n_steps
    while args.vis and scene.viewer.is_alive() and "PYTEST_VERSION" not in os.environ:
        track_circle(i)
        i += 1


if __name__ == "__main__":
    main()
