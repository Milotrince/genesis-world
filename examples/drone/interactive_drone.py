"""A Crazyflie drone flown open-loop by differential propeller RPMs, from a scripted sequence of key commands.

Headless, the script flies a short pre-planned sequence of the commands the keys issue: climb, tilt forward and
back, then turn left and back. With the viewer (-v), the same sequence runs first, then the keyboard flies the drone,
the keys adding to the scripted commands while they run:
Up         - Tilt forward (North)
Down       - Tilt backward (South)
Left       - Turn left
Right      - Turn right
Space      - Increase RPM
Left Shift - Decrease RPM
Esc        - Quit
"""

import argparse
import os

import numpy as np

import genesis as gs
from genesis.vis.keybindings import Key, KeyAction, Keybind


class DroneController:
    def __init__(self):
        self.thrust = 14475.8  # Base RPM for constant hover
        self.rotation_delta = 100.0  # Differential RPM for rotation
        self.thrust_delta = 10.0  # Amount to change thrust by when accelerating/decelerating
        self.cur_dir = np.array([0.0, 0.0, 0.0, 0.0])  # rotor directions

    def update_rpms(self):
        """Compute RPMs based on current direction and thrust"""
        clipped_dir = np.clip(self.cur_dir, -1.0, 1.0)
        rpms = self.thrust + clipped_dir * self.rotation_delta
        return np.clip(rpms, 0, 25000)

    def add_direction(self, direction: np.ndarray):
        """Add direction vector (on key press)"""
        self.cur_dir += direction

    def accelerate(self):
        """Increase base thrust"""
        self.thrust = min(self.thrust + self.thrust_delta, 25000)

    def decelerate(self):
        """Decrease base thrust"""
        self.thrust = max(self.thrust - self.thrust_delta, 0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", help="Show visualization GUI and fly with the keyboard")
    parser.add_argument(
        "-r", "--record", action="store_true", help="Record a video of the run to 'out/interactive_drone.mp4'"
    )
    args = parser.parse_args()
    is_test = "PYTEST_VERSION" in os.environ

    gs.init(backend=gs.cpu)

    camera_pos = (0.0, -2.0, 1.0)
    camera_lookat = (0.0, 0.0, 0.3)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.01,
            gravity=(0, 0, -9.81),
        ),
        vis_options=gs.options.VisOptions(
            show_world_frame=False,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=camera_pos,
            camera_lookat=camera_lookat,
            camera_fov=45,
        ),
        profiling_options=gs.options.ProfilingOptions(
            show_FPS=False,
        ),
        show_viewer=args.vis,
    )

    # Add entities
    scene.add_entity(gs.morphs.Plane())
    drone = scene.add_entity(
        morph=gs.morphs.Drone(
            file="urdf/drones/cf2x.urdf",
            pos=(0.0, 0, 0.5),  # Start a bit higher
        ),
    )

    camera = None
    if args.record:
        camera = scene.add_camera(
            res=(1280, 720),
            pos=camera_pos,
            lookat=camera_lookat,
            fov=45,
        )
        camera.follow_entity(drone)
    if args.vis:
        scene.viewer.follow_entity(drone)

    # Initialize controller
    controller = DroneController()

    scene.build()
    if camera is not None:
        camera.start_recording(save_to_filename="out/interactive_drone.mp4", fps=round(1.0 / scene.dt))

    dir_forward = np.array([1.0, 1.0, -1.0, -1.0])
    dir_backward = -dir_forward
    dir_left = np.array([-1.0, 1.0, -1.0, 1.0])
    dir_right = -dir_left

    is_running = True
    if args.vis:

        def stop():
            nonlocal is_running
            is_running = False

        keybinds = [
            Keybind("accelerate", Key.SPACE, KeyAction.HOLD, callback=controller.accelerate),
            Keybind("decelerate", Key.LSHIFT, KeyAction.HOLD, callback=controller.decelerate),
            Keybind("quit", Key.ESCAPE, KeyAction.RELEASE, callback=stop),
        ]
        for name, key, direction in (
            ("forward", Key.UP, dir_forward),
            ("backward", Key.DOWN, dir_backward),
            ("left", Key.LEFT, dir_left),
            ("right", Key.RIGHT, dir_right),
        ):
            keybinds += [
                Keybind(f"{name}_press", key, KeyAction.PRESS, callback=controller.add_direction, args=(direction,)),
                Keybind(
                    f"{name}_release", key, KeyAction.RELEASE, callback=controller.add_direction, args=(-direction,)
                ),
            ]
        scene.viewer.register_keybinds(*keybinds)
        gs.logger.info("Keys: Up/Down tilt, Left/Right turn, Space/Left Shift raise/lower the RPM, Esc quits.")

    def fly(n_steps):
        for _ in range(n_steps):
            drone.set_propellers_rpm(controller.update_rpms())
            scene.step()

    # The scripted flight calls the same functions as the keys. A thrust key acts on every step it is held, while a
    # tilt or turn key adds its direction on press and removes it on release.
    n_tap = 20 if not is_test else 2
    # Climb: raise the thrust, lower it as far below the hover RPM, then bring it back, so the drone climbs and stops.
    for _ in range(2 * n_tap):
        controller.accelerate()
        fly(1)
    for _ in range(4 * n_tap):
        controller.decelerate()
        fly(1)
    for _ in range(2 * n_tap):
        controller.accelerate()
        fly(1)
    # Tilt forward, twice as long backward, then forward again, which brings the drone back level.
    controller.add_direction(dir_forward)
    fly(n_tap)
    controller.add_direction(dir_backward - dir_forward)
    fly(2 * n_tap)
    controller.add_direction(dir_forward - dir_backward)
    fly(n_tap)
    controller.add_direction(-dir_forward)
    fly(5 * n_tap // 2)
    # Turn left, twice as long right, then left again.
    controller.add_direction(dir_left)
    fly(n_tap)
    controller.add_direction(dir_right - dir_left)
    fly(2 * n_tap)
    controller.add_direction(dir_left - dir_right)
    fly(n_tap)
    controller.add_direction(-dir_left)
    fly(5 * n_tap // 2)
    if camera is not None:
        camera.stop_recording()

    while args.vis and is_running and scene.viewer.is_alive() and not is_test:
        fly(1)


if __name__ == "__main__":
    main()
