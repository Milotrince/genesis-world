"""A cascaded PID controller flies a Crazyflie drone through a route of waypoints.

Three PID loops in cascade turn the target position into a desired velocity, the velocity error into thrust and tilt
commands, and the attitude error into corrections that hold the drone level. A mixer spreads the commands over the four
propeller RPMs.

Keyboard controls (-v):
Up/Down    - Move the target along +x/-x
Left/Right - Move the target along +y/-y
K/J        - Move the target up/down
"""

import argparse
import os

import numpy as np

import genesis as gs
from genesis.utils.geom import quat_to_xyz
from genesis.utils.misc import tensor_to_array
from genesis.vis.keybindings import Key, KeyAction, Keybind


DT = 0.01
# Propeller speed at which the Crazyflie hovers.
BASE_RPM = 14468.429183500699
# Gains (kp, ki, kd) of the position, velocity and attitude loops, one row per axis (x, y, z or roll, pitch, yaw).
# They are tuned for a stable flight, not optimised.
PID_GAINS = np.array(
    [
        [[2.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        [[20.0, 0.0, 20.0], [20.0, 0.0, 20.0], [25.0, 0.0, 20.0]],
        [[10.0, 0.0, 1.0], [10.0, 0.0, 1.0], [2.0, 0.0, 0.2]],
    ]
)
# Contribution of (thrust, roll, pitch, yaw, x velocity, y velocity) to the speed of each propeller.
MIXER = np.array(
    [
        [1.0, -1.0, -1.0, -1.0, -1.0, 1.0],
        [1.0, -1.0, 1.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, -1.0, 1.0, -1.0],
        [1.0, 1.0, -1.0, 1.0, -1.0, -1.0],
    ]
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", help="Show visualization GUI")
    parser.add_argument(
        "-r", "--record", action="store_true", help="Record a video of the run to 'out/drone_fly_route.mp4'"
    )
    args = parser.parse_args()
    is_test = "PYTEST_VERSION" in os.environ

    gs.init(backend=gs.cpu)

    camera_pos = (1.0, -1.0, 0.8)
    camera_lookat = (0.0, 0.0, 0.2)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=DT,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=camera_pos,
            camera_lookat=camera_lookat,
        ),
        show_viewer=args.vis,
    )

    plane = scene.add_entity(
        morph=gs.morphs.Plane(),
    )

    drone = scene.add_entity(
        morph=gs.morphs.Drone(
            file="urdf/drones/cf2x.urdf",
            pos=(0, 0, 0.2),
        )
    )

    cam = None
    if args.record:
        cam = scene.add_camera(
            res=(1280, 720),
            pos=camera_pos,
            lookat=camera_lookat,
        )
        cam.follow_entity(drone)
    if args.vis:
        scene.viewer.follow_entity(drone)

    scene.build()
    if cam is not None:
        cam.start_recording(save_to_filename="out/drone_fly_route.mp4", fps=round(1.0 / scene.dt))

    pid_integral = np.zeros((3, 3))
    pid_prev_error = np.zeros((3, 3))

    def pid(i_loop, error):
        pid_integral[i_loop] += error * DT
        derivative = (error - pid_prev_error[i_loop]) / DT
        pid_prev_error[i_loop] = error
        kp, ki, kd = PID_GAINS[i_loop].T
        return kp * error + ki * pid_integral[i_loop] + kd * derivative

    def fly_step(target):
        att = tensor_to_array(quat_to_xyz(drone.get_quat(), rpy=True, degrees=True))
        vel_des = pid(0, target - tensor_to_array(drone.get_pos()))
        vel_x, vel_y, thrust = pid(1, vel_des - tensor_to_array(drone.get_vel()))
        roll, pitch, yaw = pid(2, -att)
        rpm = BASE_RPM + MIXER @ (thrust, roll, pitch, yaw, vel_x, vel_y)
        drone.set_propellers_rpm(np.clip(rpm, 0.9 * BASE_RPM, 1.5 * BASE_RPM))
        scene.step()

    target = np.array([0.0, 0.0, 0.2])
    if args.vis:
        dpos = 0.01

        def move(delta):
            target[:] += delta

        scene.viewer.register_keybinds(
            Keybind("move_forward", Key.UP, KeyAction.HOLD, callback=move, args=((dpos, 0, 0),)),
            Keybind("move_back", Key.DOWN, KeyAction.HOLD, callback=move, args=((-dpos, 0, 0),)),
            Keybind("move_left", Key.LEFT, KeyAction.HOLD, callback=move, args=((0, dpos, 0),)),
            Keybind("move_right", Key.RIGHT, KeyAction.HOLD, callback=move, args=((0, -dpos, 0),)),
            Keybind("move_up", Key.K, KeyAction.HOLD, callback=move, args=((0, 0, dpos),)),
            Keybind("move_down", Key.J, KeyAction.HOLD, callback=move, args=((0, 0, -dpos),)),
        )
        gs.logger.info("Keys: Up/Down/Left/Right move the target in the ground plane, K/J move it up/down.")

        while scene.viewer.is_alive():
            fly_step(target)
    else:
        # Each waypoint is the target until the drone comes within 10 cm of it or its step budget runs out.
        points = [(1, 1, 2), (-1, 2, 1), (0, 0, 0.5)] if not is_test else [(1, 1, 2)]
        for point in points:
            target[:] = point
            for _ in range(1000 if not is_test else 90):
                if np.linalg.norm(target - tensor_to_array(drone.get_pos())) < 0.1:
                    break
                fly_step(target)

    if cam is not None:
        cam.stop_recording()


if __name__ == "__main__":
    main()
