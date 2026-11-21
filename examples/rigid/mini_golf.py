"""A Franka putts golf balls across three greens of different speeds until each ball stops in its hole.

The greens differ by their rolling friction alone, which the elliptic friction cone applies at the contact as a
resisting torque: a rolling ball decelerates at the constant rate (5/7) * friction_rolling * g / r, so it coasts over
v^2 / (2 * deceleration) before it stops. The robot holds a putter, a shaft and a head attached to its hand as an extra
link. It picks the stroke speed of each green from that rolling-resistance model, so that the ball stops at the hole,
watches where the ball stops, and corrects the speed of its next putt until the ball ends in the hole. The headless
run logs how far each ball rolled.

Keyboard controls (-v):
Left/Right - Select the green
Up/Down    - Raise/lower the stroke speed
Space      - Putt the ball of the selected green from its tee
Esc        - Quit
"""

import argparse
import math
import os
import xml.etree.ElementTree as ET

import numpy as np

import genesis as gs
from genesis.utils.misc import tensor_to_array
from genesis.vis.keybindings import Key, KeyAction, Keybind


DT = 5e-3
GRAVITY = 9.81

BALL_RADIUS = 0.0214
BALL_DENSITY = 1120.0
# Rolling friction of the fast, medium and slow greens, their lateral position, and the tee of every green.
GREENS_FRICTION_ROLLING = (3e-4, 6e-4, 1.2e-3)
GREENS_Y = (-0.3, 0.0, 0.3)
GREENS_COLOR = ((0.45, 0.75, 0.35), (0.3, 0.6, 0.25), (0.2, 0.45, 0.18))
GREEN_THICKNESS = 0.01
TEE_X = 0.65
HOLE_DISTANCE = 0.8
HOLE_RADIUS = 0.054

# Putter: length of the shaft from the hand down to the center of the head, and the half-size of the head.
PUTTER_LENGTH = 0.35
HEAD_HALF_SIZE = (0.012, 0.05, 0.0125)
# Stroke of the hand: its acceleration, the length of the stretch at constant speed before the ball, how far the face
# pushes the ball, and the deceleration that leaves the ball behind.
STROKE_ACCELERATION = 15.0
STROKE_GLIDE = 0.1
STROKE_PUSH = 0.02
STROKE_DECELERATION = 40.0


def putter_mjcf():
    """MJCF model of the putter, in the frame of the hand that holds it: the shaft runs down its z axis to the head,
    whose face looks along its x axis."""
    mjcf = ET.Element("mujoco", model="putter")
    putter = ET.SubElement(ET.SubElement(mjcf, "worldbody"), "body", name="putter")
    # The shaft passes between the fingers, so it is left out of collisions.
    ET.SubElement(
        putter,
        "geom",
        type="capsule",
        fromto=f"0 0 0.04 0 0 {PUTTER_LENGTH - HEAD_HALF_SIZE[2]}",
        size="0.006",
        contype="0",
        conaffinity="0",
        density="2000",
        rgba="0.75 0.75 0.78 1",
    )
    # A contact takes the larger friction of its two geoms, so the face and the ball are both slippery. The face then
    # pushes the ball without holding its spin back, the green spins it up, and the ball leaves the face rolling.
    ET.SubElement(
        putter,
        "geom",
        type="box",
        pos=f"0 0 {PUTTER_LENGTH}",
        size=" ".join(map(str, HEAD_HALF_SIZE)),
        friction="0.01",
        density="7800",
        rgba="0.2 0.2 0.22 1",
    )
    return ET.tostring(mjcf, encoding="unicode")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", help="Show the interactive viewer")
    parser.add_argument("-g", "--gpu", action="store_true", help="Run on GPU instead of CPU")
    parser.add_argument("-r", "--record", action="store_true", help="Record the run to 'out/mini_golf.mp4'")
    args = parser.parse_args()

    gs.init(backend=gs.gpu if args.gpu else gs.cpu)

    camera_pos = (2.1, -1.2, 0.9)
    camera_lookat = (0.85, 0.0, 0.15)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=DT,
            substeps=4,
            gravity=(0.0, 0.0, -GRAVITY),
        ),
        rigid_options=gs.options.RigidOptions(
            friction_cone=gs.friction_cone.elliptic,
            enable_torsional_friction=True,
            enable_rolling_friction=True,
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
    balls = []
    for friction_rolling, y_green, color in zip(GREENS_FRICTION_ROLLING, GREENS_Y, GREENS_COLOR):
        scene.add_entity(
            morph=gs.morphs.Box(
                pos=(TEE_X + 0.5 * HOLE_DISTANCE, y_green, 0.5 * GREEN_THICKNESS),
                size=(HOLE_DISTANCE + 0.8, 0.26, GREEN_THICKNESS),
                fixed=True,
            ),
            material=gs.materials.Rigid(
                friction_rolling=friction_rolling,
            ),
            surface=gs.surfaces.Rough(
                color=color,
            ),
        )
        # The hole and its flag are drawn only, so the ball stops on the hole rather than falling in.
        scene.add_entity(
            morph=gs.morphs.Cylinder(
                pos=(TEE_X + HOLE_DISTANCE, y_green, GREEN_THICKNESS),
                radius=HOLE_RADIUS,
                height=0.002,
                fixed=True,
                collision=False,
            ),
            surface=gs.surfaces.Default(
                color=(0.05, 0.05, 0.05),
            ),
        )
        # The flagstick stands on the far rim, clear of a ball resting in the hole.
        scene.add_entity(
            morph=gs.morphs.Cylinder(
                pos=(TEE_X + HOLE_DISTANCE + HOLE_RADIUS, y_green, GREEN_THICKNESS + 0.17),
                radius=0.003,
                height=0.34,
                fixed=True,
                collision=False,
            ),
            surface=gs.surfaces.Default(
                color=(0.9, 0.9, 0.9),
            ),
        )
        scene.add_entity(
            morph=gs.morphs.Box(
                pos=(TEE_X + HOLE_DISTANCE + HOLE_RADIUS + 0.05, y_green, GREEN_THICKNESS + 0.31),
                size=(0.1, 0.004, 0.06),
                fixed=True,
                collision=False,
            ),
            surface=gs.surfaces.Default(
                color=(0.9, 0.15, 0.1),
            ),
        )
        balls.append(
            scene.add_entity(
                morph=gs.morphs.Sphere(
                    pos=(TEE_X, y_green, GREEN_THICKNESS + BALL_RADIUS),
                    radius=BALL_RADIUS,
                ),
                material=gs.materials.Rigid(
                    rho=BALL_DENSITY,
                    friction=0.01,
                ),
                surface=gs.surfaces.Default(
                    color=(0.95, 0.95, 0.95),
                ),
            )
        )
    franka = scene.add_entity(
        morph=gs.morphs.MJCF(
            file="xml/franka_emika_panda/panda.xml",
        ),
        material=gs.materials.Rigid(
            gravity_compensation=1.0,
        ),
    )
    # The putter has no joint of its own, so it moves with the hand only once its vertices follow the hand.
    putter = scene.add_entity(
        morph=gs.morphs.MJCF(
            file=putter_mjcf(),
            batch_fixed_verts=True,
        ),
    )
    putter.attach(franka, "hand", pos=(0.0, 0.0, 0.0))

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
    # The hand points down with its x axis along the greens, so that the face of the putter looks along +x.
    hand_quat = np.array([0.0, 1.0, 0.0, 0.0])
    # Height of the hand that puts the center of the head at the height of the center of the ball.
    hand_z = GREEN_THICKNESS + BALL_RADIUS + PUTTER_LENGTH
    target_pos = np.array([TEE_X - 0.1, GREENS_Y[0], hand_z + 0.1])
    franka.set_qpos(franka.inverse_kinematics(link=hand, pos=target_pos, quat=hand_quat)[:7], motors_dof)
    # The MJCF drives the fingers through a tendon actuator, so their position control needs PD gains of its own.
    franka.set_dofs_kp([100.0, 100.0], fingers_dof)
    franka.set_dofs_kv([10.0, 10.0], fingers_dof)
    franka.control_dofs_position([0.006, 0.006], fingers_dof)

    # The joint velocity between successive targets feeds the PD controller forward, so that the hand keeps up with a
    # target moving at the stroke speed.
    motors_target = franka.get_qpos()[:7]

    def track_target():
        nonlocal motors_target
        q = franka.inverse_kinematics(link=hand, pos=target_pos, quat=hand_quat)[:7]
        franka.control_dofs_position_velocity(q, (q - motors_target) / DT, motors_dof)
        motors_target = q
        scene.step()

    def step_towards(waypoint):
        target_pos[:] += np.clip(np.array(waypoint) - target_pos, -0.006, 0.006)
        track_target()

    def putt(i_green, speed):
        ball = balls[i_green]
        ball.set_pos((TEE_X, GREENS_Y[i_green], GREEN_THICKNESS + BALL_RADIUS))
        ball.zero_all_dofs_velocity()
        # Back the head off, above the green, to where it has room to reach the stroke speed before the ball.
        backswing = 0.5 * speed**2 / STROKE_ACCELERATION + STROKE_GLIDE
        x_contact = TEE_X - BALL_RADIUS - HEAD_HALF_SIZE[0]
        for dz, n_steps in ((0.05, n_address), (0.0, 2 * n_address)):
            for _ in range(n_steps):
                step_towards((x_contact - backswing, GREENS_Y[i_green], hand_z + dz))
        # Accelerate, glide at the stroke speed into the ball, then brake and lift the head off the green.
        hand_speed = 0.0
        while target_pos[0] < x_contact + STROKE_PUSH:
            hand_speed = min(hand_speed + STROKE_ACCELERATION * DT, speed)
            target_pos[0] += hand_speed * DT
            track_target()
        while hand_speed > 0.0:
            hand_speed = max(hand_speed - STROKE_DECELERATION * DT, 0.0)
            target_pos[0] += hand_speed * DT
            track_target()
        lift_pos = target_pos + (0.0, 0.0, 0.05)
        for _ in range(n_address):
            step_towards(lift_pos)

    if "PYTEST_VERSION" not in os.environ:
        n_greens, n_attempts, n_address, n_roll = len(balls), 3, 60, 900
    else:
        n_greens, n_attempts, n_address, n_roll = 1, 1, 10, 10
    # The ball rolls away at the speed of the head.
    greens_deceleration = 5.0 / 7.0 * np.array(GREENS_FRICTION_ROLLING) * GRAVITY / BALL_RADIUS
    greens_speed = np.sqrt(2.0 * greens_deceleration * HOLE_DISTANCE)

    if camera is not None:
        camera.start_recording(save_to_filename="out/mini_golf.mp4", fps=50)
    if args.vis:
        i_green_selected = 0
        stroke_speed = float(greens_speed[0])
        is_putt_requested = False
        is_running = True

        def select_green(step):
            nonlocal i_green_selected
            i_green_selected = (i_green_selected + step) % len(balls)
            gs.logger.info(f"Green {i_green_selected} selected.")

        def change_speed(step):
            nonlocal stroke_speed
            stroke_speed = max(stroke_speed + step, 0.1)
            gs.logger.info(f"Stroke speed {stroke_speed:.2f} m/s.")

        def request_putt():
            nonlocal is_putt_requested
            is_putt_requested = True

        def stop():
            nonlocal is_running
            is_running = False

        scene.viewer.register_keybinds(
            Keybind("previous_green", Key.LEFT, KeyAction.PRESS, callback=select_green, args=(-1,)),
            Keybind("next_green", Key.RIGHT, KeyAction.PRESS, callback=select_green, args=(1,)),
            Keybind("faster", Key.UP, KeyAction.PRESS, callback=change_speed, args=(0.05,)),
            Keybind("slower", Key.DOWN, KeyAction.PRESS, callback=change_speed, args=(-0.05,)),
            Keybind("putt", Key.SPACE, KeyAction.PRESS, callback=request_putt),
            Keybind("quit", Key.ESCAPE, KeyAction.RELEASE, callback=stop),
        )
        gs.logger.info("Keys: Left/Right select the green, Up/Down change the stroke speed, Space putts, Esc quits.")
        while is_running and scene.viewer.is_alive():
            if is_putt_requested:
                is_putt_requested = False
                putt(i_green_selected, stroke_speed)
            track_target()
    else:
        for i_green in range(n_greens):
            # The rolling distance grows as the square of the stroke speed, so a putt that misses the hole corrects the
            # speed of the next one by the square root of the ratio of the distances.
            for i_attempt in range(n_attempts):
                putt(i_green, greens_speed[i_green])
                for _ in range(n_roll):
                    track_target()
                distance = tensor_to_array(balls[i_green].get_pos())[0] - TEE_X
                gs.logger.info(
                    f"Green {i_green} (friction_rolling {GREENS_FRICTION_ROLLING[i_green]}), putt {i_attempt + 1} at "
                    f"{greens_speed[i_green]:.2f} m/s: ball rolled {distance:.2f} m, hole at {HOLE_DISTANCE:.2f} m."
                )
                if abs(distance - HOLE_DISTANCE) < HOLE_RADIUS:
                    break
                greens_speed[i_green] *= math.sqrt(HOLE_DISTANCE / distance)
    if camera is not None:
        camera.stop_recording()


if __name__ == "__main__":
    main()
