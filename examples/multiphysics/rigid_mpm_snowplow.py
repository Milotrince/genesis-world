"""A wheeled robot with an angled plow blade drives through a patch of snow and pushes it aside into a berm.

The robot is an MJCF model built inline: a box chassis on four wheels whose hinges are driven by velocity control, and a
blade fixed to the front of the chassis, yawed so that the snow it meets slides off its trailing end. The snow is an MPM
`Snow` layer a few centimetres deep, which compacts and hardens as the blade piles it up. The scripted run makes one
straight pass and logs the share of the snow moved out of the cleared lane and the height of the berm.

Keyboard controls (with -v, once the scripted pass is over):
Up    - Drive forward
Down  - Drive backward
Left  - Turn left
Right - Turn right

Plus all default viewer controls (press 'i' to see them).
"""

import argparse
import math
import os
import xml.etree.ElementTree as ET

import numpy as np

import genesis as gs
from genesis.utils.misc import tensor_to_array
from genesis.vis.keybindings import Key, KeyAction, Keybind


WHEEL_RADIUS = 0.05
# Chassis half-extents and the half-distances between the wheel axles along x and between the wheels along y.
CHASSIS_HALF_SIZE = (0.15, 0.09, 0.03)
WHEEL_BASE_HALF = 0.1
WHEEL_TRACK_HALF = 0.13
# Blade half-extents, its yaw from the y axis and the clearance between its lower edge and the ground.
BLADE_HALF_SIZE = (0.02, 0.25, 0.06)
BLADE_YAW_DEG = 30.0
BLADE_CLEARANCE = 0.005
# Wheel speeds in m/s along the ground: driving straight, and the difference between the two sides when turning.
DRIVE_SPEED = 0.4
TURN_SPEED = 0.2

# Opposite corners of the snow layer. The layer reaches past the lane on the +y side, where the yawed blade sheds the
# snow, so that the berm forms on top of it.
SNOW_LOWER = (0.0, -0.17, 0.0)
SNOW_UPPER = (0.6, 0.3, 0.04)


def snowplow_mjcf():
    """MJCF model of the snowplow, its chassis centred at the wheel axle height and facing +x."""
    mjcf = ET.Element("mujoco", model="snowplow")
    worldbody = ET.SubElement(mjcf, "worldbody")
    chassis = ET.SubElement(worldbody, "body", name="chassis")
    ET.SubElement(chassis, "freejoint")
    ET.SubElement(
        chassis,
        "geom",
        type="box",
        size=" ".join(map(str, CHASSIS_HALF_SIZE)),
        density="4000",
        rgba="0.85 0.45 0.1 1",
    )
    # The blade sits far enough ahead of the chassis that its trailing end clears the front wheels.
    blade_x = CHASSIS_HALF_SIZE[0] + 0.11
    blade_z = BLADE_HALF_SIZE[2] + BLADE_CLEARANCE - WHEEL_RADIUS
    ET.SubElement(
        chassis,
        "geom",
        name="blade",
        type="box",
        pos=f"{blade_x} 0 {blade_z}",
        euler=f"0 0 {BLADE_YAW_DEG}",
        size=" ".join(map(str, BLADE_HALF_SIZE)),
        density="2000",
        rgba="0.95 0.8 0.1 1",
    )
    for side, y in (("left", WHEEL_TRACK_HALF), ("right", -WHEEL_TRACK_HALF)):
        for end, x in (("front", WHEEL_BASE_HALF), ("rear", -WHEEL_BASE_HALF)):
            wheel = ET.SubElement(chassis, "body", name=f"{end}_{side}_wheel", pos=f"{x} {y} 0")
            ET.SubElement(wheel, "joint", name=f"{end}_{side}_wheel", type="hinge", axis="0 1 0")
            ET.SubElement(
                wheel,
                "geom",
                type="cylinder",
                euler="90 0 0",
                size=f"{WHEEL_RADIUS} 0.02",
                density="1000",
                rgba="0.15 0.15 0.15 1",
            )
    return ET.tostring(mjcf, encoding="unicode")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", help="Show the viewer and drive the robot with the keys")
    parser.add_argument("-g", "--gpu", action="store_true", help="Run on GPU instead of CPU")
    parser.add_argument(
        "-r", "--record", action="store_true", help="Record a video of the run to 'out/rigid_mpm_snowplow.mp4'"
    )
    args = parser.parse_args()

    gs.init(backend=gs.gpu if args.gpu else gs.cpu, precision="32", logging_level="info")

    camera_pos = (1.5, -1.0, 0.9)
    camera_lookat = (0.45, 0.1, 0.0)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=3e-3,
            substeps=10,
        ),
        mpm_options=gs.options.MPMOptions(
            lower_bound=(-0.1, -0.35, -0.06),
            upper_bound=(1.25, 0.6, 0.25),
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=camera_pos,
            camera_lookat=camera_lookat,
            camera_fov=40,
        ),
        show_viewer=args.vis,
    )

    # Snow grips the ground, so that the blade shears the layer rather than sliding it along as one slab.
    scene.add_entity(
        morph=gs.morphs.Plane(),
        material=gs.materials.Rigid(
            coup_friction=1.0,
        ),
    )
    robot = scene.add_entity(
        gs.morphs.MJCF(
            file=snowplow_mjcf(),
            pos=(-0.45, 0.0, WHEEL_RADIUS),
        ),
    )
    snow = scene.add_entity(
        spec=gs.entities.MPM(
            morph=gs.morphs.Box(
                lower=SNOW_LOWER,
                upper=SNOW_UPPER,
            ),
            material=gs.materials.MPM.Snow(
                E=5e4,
                rho=400.0,
            ),
            surface=gs.surfaces.Default(
                color=(0.95, 0.97, 1.0),
            ),
            vis=gs.options.EntityVisOptions(
                debug=True,
            ),
        ),
    )

    camera = None
    if args.record:
        camera = scene.add_camera(
            res=(1280, 720),
            pos=camera_pos,
            lookat=camera_lookat,
            fov=40,
            debug=True,
        )

    scene.build()

    # The left wheels come first, as the drive commands expect.
    wheels_dof = [
        robot.get_joint(f"{end}_{side}_wheel").dofs_idx_local[0]
        for side in ("left", "right")
        for end in ("front", "rear")
    ]
    robot.set_dofs_kv(50.0, wheels_dof)

    # Throttle and turn commands in [-1, 1], set by the scripted pass and by the arrow keys alike.
    command = np.zeros(2)

    def set_command(i_axis: int, value: float):
        command[i_axis] = value

    def drive():
        speed_left = DRIVE_SPEED * command[0] - TURN_SPEED * command[1]
        speed_right = DRIVE_SPEED * command[0] + TURN_SPEED * command[1]
        robot.control_dofs_velocity(np.repeat((speed_left, speed_right), 2) / WHEEL_RADIUS, wheels_dof)
        scene.step()

    snow_pos_init = tensor_to_array(snow.get_particles_pos())

    if camera is not None:
        camera.start_recording(save_to_filename="out/rigid_mpm_snowplow.mp4", fps=30)
    n_pass, n_settle = (1000, 100) if "PYTEST_VERSION" not in os.environ else (20, 5)
    set_command(0, 1.0)
    for _ in range(n_pass):
        drive()
    set_command(0, 0.0)
    for _ in range(n_settle):
        drive()
    if camera is not None:
        camera.stop_recording()

    # The lane is the swath of the yawed blade along the path of the robot, up to the end of the layer.
    lane_half_width = BLADE_HALF_SIZE[1] * math.cos(math.radians(BLADE_YAW_DEG))
    snow_pos = tensor_to_array(snow.get_particles_pos())
    is_in_lane_init = np.abs(snow_pos_init[:, 1]) < lane_half_width
    is_in_lane = (np.abs(snow_pos[:, 1]) < lane_half_width) & (snow_pos[:, 0] < SNOW_UPPER[0])
    share_moved = 1.0 - np.count_nonzero(is_in_lane & is_in_lane_init) / np.count_nonzero(is_in_lane_init)
    berm_height = np.percentile(snow_pos[~is_in_lane, 2], 99)
    gs.logger.info(
        f"Snow moved out of the lane: {100.0 * share_moved:.0f}%. Berm height: {100.0 * berm_height:.1f} cm, on a "
        f"{100.0 * SNOW_UPPER[2]:.0f} cm layer."
    )

    # The keys are registered once the scripted pass is over, so that a key press cannot interfere with it.
    if args.vis:
        scene.viewer.register_keybinds(
            Keybind("forward", Key.UP, KeyAction.PRESS, callback=set_command, args=(0, 1.0)),
            Keybind("stop_forward", Key.UP, KeyAction.RELEASE, callback=set_command, args=(0, 0.0)),
            Keybind("backward", Key.DOWN, KeyAction.PRESS, callback=set_command, args=(0, -1.0)),
            Keybind("stop_backward", Key.DOWN, KeyAction.RELEASE, callback=set_command, args=(0, 0.0)),
            Keybind("turn_left", Key.LEFT, KeyAction.PRESS, callback=set_command, args=(1, 1.0)),
            Keybind("stop_left", Key.LEFT, KeyAction.RELEASE, callback=set_command, args=(1, 0.0)),
            Keybind("turn_right", Key.RIGHT, KeyAction.PRESS, callback=set_command, args=(1, -1.0)),
            Keybind("stop_right", Key.RIGHT, KeyAction.RELEASE, callback=set_command, args=(1, 0.0)),
        )
        gs.logger.info("Keys: Up/Down drive forward/backward, Left/Right turn.")
    while args.vis and scene.viewer.is_alive() and "PYTEST_VERSION" not in os.environ:
        drive()


if __name__ == "__main__":
    main()
