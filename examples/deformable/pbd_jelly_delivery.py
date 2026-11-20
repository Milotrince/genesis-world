"""A Franka swings a tray of three soft jellies to a delivery spot and stops hard, and the jellies wobble and settle.

The jellies are position-based dynamics (PBD) elastic bodies standing on a rigid tray, which a bracket between the
fingers holds beside the hand. The robot swings the tray around its base at a brisk pace, then stops dead over the
delivery spot: the jellies slide into the lip of the tray, squash and tip against it, and spring back. The lip alone
holds them, since the contact between particles and rigid geoms has no friction. The run logs how many jellies are
still on the tray and the peak deformation of each one after the stop.

Keyboard controls (-v), once the scripted delivery is over:
Up/Down    - Move the tray away from/towards the base
Left/Right - Swing the tray left/right
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
# Position of the center of the tray top relative to the hand pointing down: the tray sits beside the fingertips, a
# bracket between the fingers holding it.
TRAY_OFFSET = (0.27, 0.0, -0.115)
# The hand swings the tray around the base of the robot, the tray on the far side: the distance of the hand from the
# base axis, the height of the top of the tray, and the azimuths of the pick-up and delivery spots.
HAND_RADIUS = 0.4
TRAY_Z = 0.3
PICKUP_ANGLE = -1.0
DELIVERY_ANGLE = 1.0
# Peak speed and acceleration of the center of the tray along the arc.
CARRY_SPEED = 0.8
CARRY_ACCELERATION = 2.0


def tray_mjcf():
    """MJCF model of the tray, in a frame at the center of its top: a plate with a lip on its four sides."""
    mjcf = ET.Element("mujoco", model="tray")
    tray = ET.SubElement(ET.SubElement(mjcf, "worldbody"), "body", name="tray")
    half_x, half_y = TRAY_HALF_SIZE
    wall = 0.5 * LIP_THICKNESS
    for pos, size in (
        ((0.0, 0.0, -0.5 * TRAY_THICKNESS), (half_x, half_y, 0.5 * TRAY_THICKNESS)),
        ((half_x - wall, 0.0, 0.5 * LIP_HEIGHT), (wall, half_y, 0.5 * LIP_HEIGHT)),
        ((-half_x + wall, 0.0, 0.5 * LIP_HEIGHT), (wall, half_y, 0.5 * LIP_HEIGHT)),
        ((0.0, half_y - wall, 0.5 * LIP_HEIGHT), (half_x, wall, 0.5 * LIP_HEIGHT)),
        ((0.0, -half_y + wall, 0.5 * LIP_HEIGHT), (half_x, wall, 0.5 * LIP_HEIGHT)),
    ):
        ET.SubElement(
            tray,
            "geom",
            type="box",
            pos=" ".join(map(str, pos)),
            size=" ".join(map(str, size)),
            density="500",
            rgba="0.85 0.85 0.88 1",
        )
    # The bracket runs down between the fingers, then out to the near edge of the plate. It passes between the fingers,
    # so it is left out of collisions.
    hand_x, hand_z = -TRAY_OFFSET[0], -TRAY_OFFSET[2]
    bracket_z = -0.5 * TRAY_THICKNESS
    for pos, size in (
        ((hand_x, 0.0, 0.5 * (hand_z - 0.055 + bracket_z)), (0.006, 0.012, 0.5 * (hand_z - 0.055 - bracket_z))),
        ((0.5 * (hand_x - half_x), 0.0, bracket_z), (0.5 * (-half_x - hand_x), 0.012, 0.006)),
    ):
        ET.SubElement(
            tray,
            "geom",
            type="box",
            pos=" ".join(map(str, pos)),
            size=" ".join(map(str, size)),
            contype="0",
            conaffinity="0",
            density="500",
            rgba="0.6 0.6 0.63 1",
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
    # The pick-up and delivery stands mark the spots under the tray.
    tray_radius = HAND_RADIUS + TRAY_OFFSET[0]
    for angle, color in ((PICKUP_ANGLE, (0.55, 0.42, 0.3)), (DELIVERY_ANGLE, (0.3, 0.55, 0.35))):
        scene.add_entity(
            morph=gs.morphs.Box(
                pos=(tray_radius * np.cos(angle), tray_radius * np.sin(angle), 0.1),
                euler=(0.0, 0.0, np.rad2deg(angle)),
                size=(0.36, 0.24, 0.2),
                fixed=True,
                collision=False,
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
    # The tray has no joint of its own, so it moves with the hand only once its vertices follow the hand.
    tray = scene.add_entity(
        morph=gs.morphs.MJCF(
            file=tray_mjcf(),
            batch_fixed_verts=True,
        ),
    )
    # The hand points down, a half-turn about x. The mounting rotation is the same half-turn, which levels the tray.
    hand_down_quat = np.array([0.0, 1.0, 0.0, 0.0])
    tray.attach(franka, "hand", pos=(TRAY_OFFSET[0], -TRAY_OFFSET[1], -TRAY_OFFSET[2]), quat=hand_down_quat)

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
    # The target of the hand in cylindrical coordinates around the base axis: radius, azimuth and height.
    target = np.array([HAND_RADIUS, PICKUP_ANGLE, TRAY_Z - TRAY_OFFSET[2]])

    def solve_target():
        radius, angle, z = target
        quat = gu.transform_quat_by_quat(
            hand_down_quat, gu.axis_angle_to_quat(np.array(angle), np.array([0.0, 0.0, 1.0]))
        )
        pos = (radius * np.cos(angle), radius * np.sin(angle), z)
        return franka.inverse_kinematics(link=hand, pos=pos, quat=quat, init_qpos=franka.get_qpos())[:7]

    # Every solve starts from the previous one, which keeps the arm on the elbow-up branch it starts on.
    franka.set_qpos([PICKUP_ANGLE, -0.3, 0.0, -2.4, 0.0, 2.1, 0.785, 0.0, 0.0])
    franka.set_qpos(solve_target(), motors_dof)
    franka.control_dofs_position([0.0, 0.0], fingers_dof)
    motors_target = franka.get_qpos()[:7]

    # The joint velocity between successive targets feeds the PD controller forward, so that the hand keeps up with a
    # moving target and stops when the target stops.
    def track_target():
        nonlocal motors_target
        q = solve_target()
        franka.control_dofs_position_velocity(q, (q - motors_target) / DT, motors_dof)
        motors_target = q
        scene.step()

    jellies_rest = []
    for jelly in jellies:
        particles_pos = tensor_to_array(jelly.get_particles_pos())
        jellies_rest.append(particles_pos - particles_pos.mean(axis=0))

    if "PYTEST_VERSION" not in os.environ:
        n_settle, n_after = 50, 250
        delivery_angle = DELIVERY_ANGLE
    else:
        n_settle, n_after = 5, 5
        delivery_angle = PICKUP_ANGLE + 0.05

    if camera is not None:
        camera.start_recording(save_to_filename="out/pbd_jelly_delivery.mp4", fps=50)
    for _ in range(n_settle):
        track_target()
    # Speed up gently, cruise, then stop dead over the delivery spot.
    angular_speed = 0.0
    while target[1] < delivery_angle:
        angular_speed = min(angular_speed + CARRY_ACCELERATION / tray_radius * DT, CARRY_SPEED / tray_radius)
        target[1] = min(target[1] + angular_speed * DT, delivery_angle)
        track_target()
    # The deformation of a jelly is the root-mean-square distance of its particles from its rest shape, once the rigid
    # motion that best fits them is removed.
    jellies_deformation_peak = np.zeros(len(jellies))
    for _ in range(n_after):
        track_target()
        for i_jelly, (jelly, rest) in enumerate(zip(jellies, jellies_rest)):
            particles_pos = tensor_to_array(jelly.get_particles_pos())
            particles_pos -= particles_pos.mean(axis=0)
            u, _, vt = np.linalg.svd(rest.T @ particles_pos)
            deformation = np.sqrt(np.mean(np.sum((particles_pos - rest @ u @ vt) ** 2, axis=-1)))
            jellies_deformation_peak[i_jelly] = max(jellies_deformation_peak[i_jelly], deformation)
    if camera is not None:
        camera.stop_recording()

    tray_top_z = tensor_to_array(tray.get_pos())[2]
    jellies_center = np.stack([tensor_to_array(jelly.get_particles_pos()).mean(axis=0) for jelly in jellies])
    n_on_tray = np.count_nonzero(jellies_center[:, 2] > tray_top_z)
    gs.logger.info(f"{n_on_tray} of {len(jellies)} jellies delivered on the tray.")
    for i_jelly, deformation in enumerate(jellies_deformation_peak):
        gs.logger.info(f"Jelly {i_jelly}: peak deformation {1000.0 * deformation:.1f} mm after the stop.")

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
            Keybind("quit", Key.ESCAPE, KeyAction.RELEASE, callback=stop),
        )
        gs.logger.info("Keys: Up/Down move the tray away from/towards the base, Left/Right swing it, Esc quits.")
        while is_running and scene.viewer.is_alive() and "PYTEST_VERSION" not in os.environ:
            track_target()


if __name__ == "__main__":
    main()
