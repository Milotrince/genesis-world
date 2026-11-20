"""A Franka picks up a tray of three soft jellies, swings it to a delivery stand and stops hard, then sets it down.

The jellies are position-based dynamics (PBD) elastic bodies standing on a free rigid tray. The robot grasps the fin on
the handle of the tray, lifts it off the pick-up stand and swings it around its base at a brisk pace, then stops dead
over the delivery stand: the jellies slide into the lip of the tray, squash and tip against it, and spring back. The lip
alone holds them, since the contact between particles and rigid geoms has no friction. Once they settle, the robot sets
the tray down, lets go and backs away. The run logs the peak deformation of each jelly after the stop, how many jellies
are still on the tray, and whether the tray rests on the delivery stand.

Keyboard controls (-v), once the scripted delivery is over:
Up/Down    - Move the hand away from/towards the base
Left/Right - Swing the hand left/right
K/J        - Move the hand up/down
Space      - Hold to grasp the tray, release to let go
Esc        - Quit

Plus all default viewer controls (press 'i' to see them).
"""

import argparse
import os
import xml.etree.ElementTree as ET

import numpy as np

import genesis as gs
import genesis.utils.geom as gu
from genesis.utils.misc import tensor_to_array
from genesis.vis.keybindings import Key, KeyAction, Keybind


DT = 1e-2
# Half-size of the tray plate, its thickness, and the height and thickness of its lip. A particle that sinks past the
# middle of a geom comes out on the far side, so the plate and the lip are several particles thick.
TRAY_HALF_SIZE = (0.16, 0.1)
TRAY_THICKNESS = 0.04
LIP_HEIGHT = 0.03
LIP_THICKNESS = 0.03
# The handle sticks out of the near side of the tray and ends in an upright fin for the fingers: the distance of the fin
# from the center of the tray, its half-size, and the height of the hand above the tray top when the fingers hold it.
FIN_X = 0.27
FIN_HALF_SIZE = (0.025, 0.01, 0.025)
GRASP_Z = 0.12
# The fingers hold the tray by friction alone, against the torque of the tray and the jellies about the fin. The
# torsional friction of the finger pads resists that torque, and needs the elliptic friction cone. Light tray and
# jellies keep the torque within the grip through the hard stop. The grip force stays below the force that drives the
# fingers past their joint limit when they close on nothing.
FINGERS_FORCE = 15.0
GRIP_FRICTION = 2.0
GRIP_FRICTION_TORSIONAL = 0.03
TRAY_DENSITY = 30.0
JELLY_RHO = 300.0
# The robot swings the tray around its base: the distance of the hand from the base axis, the height of the top of the
# tray resting on a stand, how high the robot lifts it, and the azimuths of the pick-up and delivery stands.
HAND_RADIUS = 0.4
TRAY_Z = 0.3
LIFT_Z = 0.06
PICKUP_ANGLE = -1.0
DELIVERY_ANGLE = 1.0
# Peak speed and acceleration of the center of the tray along the arc.
CARRY_SPEED = 0.8
CARRY_ACCELERATION = 2.0


def tray_mjcf():
    """MJCF model of the free tray, in a frame at the center of its top: a plate with a lip, and a handle towards -x.

    The two long walls of the lip run the full length of the plate and the two short walls fit between them.
    """
    mjcf = ET.Element("mujoco", model="tray")
    tray = ET.SubElement(ET.SubElement(mjcf, "worldbody"), "body", name="tray")
    ET.SubElement(tray, "freejoint")
    half_x, half_y = TRAY_HALF_SIZE
    wall = 0.5 * LIP_THICKNESS
    handle_end_x = FIN_X + FIN_HALF_SIZE[0]
    for pos, size in (
        ((0.0, 0.0, -0.5 * TRAY_THICKNESS), (half_x, half_y, 0.5 * TRAY_THICKNESS)),
        ((0.0, half_y - wall, 0.5 * LIP_HEIGHT), (half_x, wall, 0.5 * LIP_HEIGHT)),
        ((0.0, -half_y + wall, 0.5 * LIP_HEIGHT), (half_x, wall, 0.5 * LIP_HEIGHT)),
        ((half_x - wall, 0.0, 0.5 * LIP_HEIGHT), (wall, half_y - 2.0 * wall, 0.5 * LIP_HEIGHT)),
        ((-half_x + wall, 0.0, 0.5 * LIP_HEIGHT), (wall, half_y - 2.0 * wall, 0.5 * LIP_HEIGHT)),
        (
            (-0.5 * (half_x + handle_end_x), 0.0, -0.5 * TRAY_THICKNESS),
            (0.5 * (handle_end_x - half_x), 0.02, 0.5 * TRAY_THICKNESS),
        ),
        ((-FIN_X, 0.0, FIN_HALF_SIZE[2]), FIN_HALF_SIZE),
    ):
        ET.SubElement(
            tray,
            "geom",
            type="box",
            pos=" ".join(map(str, pos)),
            size=" ".join(map(str, size)),
            density=str(TRAY_DENSITY),
            rgba="0.85 0.85 0.88 1",
        )
    return ET.tostring(mjcf, encoding="unicode")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", help="Show the interactive viewer")
    parser.add_argument("-g", "--gpu", action="store_true", help="Run on GPU instead of CPU")
    parser.add_argument("-r", "--record", action="store_true", help="Record the run to 'out/pbd_jelly_delivery.mp4'")
    args = parser.parse_args()

    gs.init(backend=gs.gpu if args.gpu else gs.cpu, precision="32")

    camera_pos = (1.6, 0.0, 1.0)
    camera_lookat = (0.3, 0.0, 0.25)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=DT,
            substeps=10,
        ),
        rigid_options=gs.options.RigidOptions(
            friction_cone=gs.friction_cone.elliptic,
            enable_torsional_friction=True,
        ),
        pbd_options=gs.options.PBDOptions(
            particle_size=1e-2,
            # The neighbour hash grid repeats in space, so a half-metre cube covers the tray. The default grid costs
            # more to sweep at every substep than the jellies themselves.
            hash_grid_res=(0.5, 0.5, 0.5),
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
    tray_radius = HAND_RADIUS + FIN_X
    stand_height = TRAY_Z - TRAY_THICKNESS
    stands_pos = []
    for angle, color in ((PICKUP_ANGLE, (0.55, 0.42, 0.3)), (DELIVERY_ANGLE, (0.3, 0.55, 0.35))):
        stands_pos.append(np.array([tray_radius * np.cos(angle), tray_radius * np.sin(angle)]))
        scene.add_entity(
            morph=gs.morphs.Box(
                pos=(*stands_pos[-1], 0.5 * stand_height),
                euler=(0.0, 0.0, np.rad2deg(angle)),
                size=(0.36, 0.24, stand_height),
                fixed=True,
            ),
            surface=gs.surfaces.Rough(
                color=color,
            ),
        )
    franka = scene.add_entity(
        morph=gs.morphs.MJCF(
            file="xml/franka_emika_panda/panda.xml",
        ),
        # The jellies only touch the tray, so the arm stays out of the coupling.
        material=gs.materials.Rigid(
            needs_coup=False,
            gravity_compensation=1.0,
        ),
    )
    tray = scene.add_entity(
        morph=gs.morphs.MJCF(
            file=tray_mjcf(),
            pos=(*stands_pos[0], TRAY_Z),
            euler=(0.0, 0.0, np.rad2deg(PICKUP_ANGLE)),
        ),
        material=gs.materials.Rigid(
            friction=GRIP_FRICTION,
            friction_torsional=GRIP_FRICTION_TORSIONAL,
        ),
    )

    jellies = []
    jelly_width, jelly_height = 0.06, 0.05
    for i_jelly, (morph_type, color) in enumerate(
        (
            (gs.morphs.Box, (0.9, 0.2, 0.25, 1.0)),
            (gs.morphs.Cylinder, (0.95, 0.75, 0.15, 1.0)),
            (gs.morphs.Box, (0.25, 0.75, 0.3, 1.0)),
        )
    ):
        jelly_radius = tray_radius + (i_jelly - 1) * 0.085
        pos = (
            jelly_radius * np.cos(PICKUP_ANGLE),
            jelly_radius * np.sin(PICKUP_ANGLE),
            TRAY_Z + 0.5 * jelly_height + 0.002,
        )
        if morph_type is gs.morphs.Box:
            morph = gs.morphs.Box(
                pos=pos, euler=(0.0, 0.0, np.rad2deg(PICKUP_ANGLE)), size=(jelly_width, jelly_width, jelly_height)
            )
        else:
            morph = gs.morphs.Cylinder(pos=pos, radius=0.5 * jelly_width, height=jelly_height)
        jellies.append(
            scene.add_entity(
                morph=morph,
                # Soft enough to wobble visibly, stiff enough to keep its shape under its own weight.
                material=gs.materials.PBD.Elastic(
                    rho=JELLY_RHO,
                    stretch_compliance=1e-2,
                    volume_compliance=1e-2,
                ),
                surface=gs.surfaces.Default(
                    color=color,
                ),
            )
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

    motors_dof = np.arange(7)
    fingers_dof = np.arange(7, 9)
    hand = franka.get_link("hand")
    # The hand points down, a half-turn about x, its fingers closing across the fin.
    hand_down_quat = np.array([0.0, 1.0, 0.0, 0.0])
    # The target of the hand in cylindrical coordinates around the base axis: radius, azimuth and height.
    target = np.array([HAND_RADIUS, PICKUP_ANGLE, TRAY_Z + GRASP_Z + 0.1])

    def solve_target():
        radius, angle, z = target
        quat = gu.transform_quat_by_quat(
            hand_down_quat, gu.axis_angle_to_quat(np.array(angle), np.array([0.0, 0.0, 1.0]))
        )
        pos = (radius * np.cos(angle), radius * np.sin(angle), z)
        return franka.inverse_kinematics(link=hand, pos=pos, quat=quat, init_qpos=franka.get_qpos())[:7]

    # Every solve starts from the previous one, which keeps the arm on the elbow-up branch it starts on.
    franka.set_qpos([PICKUP_ANGLE, -0.3, 0.0, -2.4, 0.0, 2.1, 0.785, 0.04, 0.04])
    franka.set_qpos(solve_target(), motors_dof)
    for link_name in ("left_finger", "right_finger"):
        franka.get_link(link_name).set_friction(GRIP_FRICTION)
        franka.get_link(link_name).set_friction_torsional(GRIP_FRICTION_TORSIONAL)
    franka.set_dofs_kp([100.0, 100.0], fingers_dof)
    franka.set_dofs_kv([10.0, 10.0], fingers_dof)
    franka.control_dofs_position([0.04, 0.04], fingers_dof)
    motors_target = franka.get_qpos()[:7]

    # The joint velocity between successive targets feeds the PD controller forward, so that the hand keeps up with a
    # moving target and stops when the target stops.
    def track_target():
        nonlocal motors_target
        q = solve_target()
        franka.control_dofs_position_velocity(q, (q - motors_target) / DT, motors_dof)
        motors_target = q
        scene.step()

    def move_to(radius, z, steps):
        start = target.copy()
        for i_step in range(1, steps + 1):
            target[[0, 2]] = start[[0, 2]] + (np.array([radius, z]) - start[[0, 2]]) * i_step / steps
            track_target()

    def grasp():
        franka.control_dofs_force([-FINGERS_FORCE, -FINGERS_FORCE], fingers_dof)

    def release():
        franka.control_dofs_position([0.04, 0.04], fingers_dof)

    if "PYTEST_VERSION" not in os.environ:
        n_reach, n_grip, n_wobble = 60, 40, 200
        delivery_angle = DELIVERY_ANGLE
    else:
        n_reach, n_grip, n_wobble = 5, 5, 5
        delivery_angle = PICKUP_ANGLE + 0.05

    if camera is not None:
        camera.start_recording(save_to_filename="out/pbd_jelly_delivery.mp4", fps=50)
    move_to(HAND_RADIUS, TRAY_Z + GRASP_Z, steps=n_reach)
    grasp()
    for _ in range(n_grip):
        track_target()
    move_to(HAND_RADIUS, TRAY_Z + GRASP_Z + LIFT_Z, steps=n_reach)

    jellies_rest = []
    for jelly in jellies:
        particles_pos = tensor_to_array(jelly.get_particles_pos())
        jellies_rest.append(particles_pos - particles_pos.mean(axis=0))
    # Speed up gently, cruise, then stop dead over the delivery stand.
    angular_speed = 0.0
    while target[1] < delivery_angle:
        angular_speed = min(angular_speed + CARRY_ACCELERATION / tray_radius * DT, CARRY_SPEED / tray_radius)
        target[1] = min(target[1] + angular_speed * DT, delivery_angle)
        track_target()
    # The deformation of a jelly is the root-mean-square distance of its particles from its rest shape, once the rigid
    # motion that best fits them is removed.
    jellies_deformation_peak = np.zeros(len(jellies))
    for _ in range(n_wobble):
        track_target()
        for i_jelly, (jelly, rest) in enumerate(zip(jellies, jellies_rest)):
            particles_pos = tensor_to_array(jelly.get_particles_pos())
            particles_pos -= particles_pos.mean(axis=0)
            u, _, vt = np.linalg.svd(rest.T @ particles_pos)
            deformation = np.sqrt(np.mean(np.sum((particles_pos - rest @ u @ vt) ** 2, axis=-1)))
            jellies_deformation_peak[i_jelly] = max(jellies_deformation_peak[i_jelly], deformation)

    move_to(HAND_RADIUS, TRAY_Z + GRASP_Z, steps=n_reach)
    release()
    for _ in range(n_grip):
        track_target()
    move_to(HAND_RADIUS - 0.1, TRAY_Z + GRASP_Z + 0.15, steps=n_reach)
    if camera is not None:
        camera.stop_recording()

    for i_jelly, deformation in enumerate(jellies_deformation_peak):
        gs.logger.info(f"Jelly {i_jelly}: peak deformation {1000.0 * deformation:.1f} mm after the stop.")
    tray_pos, tray_quat = tensor_to_array(tray.get_pos()), tensor_to_array(tray.get_quat())
    jellies_center = np.stack([tensor_to_array(jelly.get_particles_pos()).mean(axis=0) for jelly in jellies])
    jellies_center_local = gu.inv_transform_by_trans_quat(jellies_center, tray_pos, tray_quat)
    is_on_tray = (np.abs(jellies_center_local[:, :2]) < TRAY_HALF_SIZE).all(axis=-1) & (
        jellies_center_local[:, 2] > 0.0
    )
    gs.logger.info(f"{np.count_nonzero(is_on_tray)} of {len(jellies)} jellies delivered on the tray.")
    is_tray_on_stand = np.linalg.norm(tray_pos[:2] - stands_pos[1]) < 0.05 and abs(tray_pos[2] - TRAY_Z) < 0.01
    gs.logger.info(f"Tray {'rests' if is_tray_on_stand else 'does not rest'} on the delivery stand.")

    if args.vis:
        is_running = True

        def move(delta):
            target[:] += delta

        def stop():
            nonlocal is_running
            is_running = False

        scene.viewer.register_keybinds(
            Keybind("move_out", Key.UP, KeyAction.HOLD, callback=move, args=((0.004, 0.0, 0.0),)),
            Keybind("move_in", Key.DOWN, KeyAction.HOLD, callback=move, args=((-0.004, 0.0, 0.0),)),
            Keybind("swing_left", Key.LEFT, KeyAction.HOLD, callback=move, args=((0.0, -0.01, 0.0),)),
            Keybind("swing_right", Key.RIGHT, KeyAction.HOLD, callback=move, args=((0.0, 0.01, 0.0),)),
            Keybind("move_up", Key.K, KeyAction.HOLD, callback=move, args=((0.0, 0.0, 0.004),)),
            Keybind("move_down", Key.J, KeyAction.HOLD, callback=move, args=((0.0, 0.0, -0.004),)),
            Keybind("grasp", Key.SPACE, KeyAction.PRESS, callback=grasp),
            Keybind("release", Key.SPACE, KeyAction.RELEASE, callback=release),
            Keybind("quit", Key.ESCAPE, KeyAction.RELEASE, callback=stop),
        )
        gs.logger.info(
            "Keys: Up/Down move the hand away from/towards the base, Left/Right swing it, K/J move it up/down, hold "
            "Space to grasp the tray, Esc quits."
        )
        while is_running and scene.viewer.is_alive() and "PYTEST_VERSION" not in os.environ:
            track_target()


if __name__ == "__main__":
    main()
