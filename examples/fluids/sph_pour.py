"""A Franka grasps a cup of water, carries it next to a bowl and pours it in, then reports how much landed in the bowl.

The water is a smoothed particle hydrodynamics (SPH) liquid. The cup and the bowl are open containers assembled from
boxes, which the liquid meets through their surfaces. The hand tilts the cup about its fingertips, so that the water
pours over the far rim. Particle scenes are slow on CPU, so '-g' speeds the run up a lot.

With the viewer (-v), drag the cup with the mouse once the pour is over.
"""

import argparse
import math
import os
import xml.etree.ElementTree as ET

import numpy as np

import genesis as gs
import genesis.utils.geom as gu


# The hand points down, with its fingers closing along x.
HAND_QUAT = (0.0, 0.7071068, 0.7071068, 0.0)
# Distance from the origin of the hand to the middle of the fingertip pads, below it.
TIP_DEPTH = 0.1
TABLE_HEIGHT = 0.2
# Outer size of the cup and of the bowl (x, y, z), and the thickness of their walls. The cup fits in the open gripper.
CUP_SIZE = (0.07, 0.07, 0.1)
BOWL_SIZE = (0.26, 0.26, 0.08)
WALL = 0.008
CUP_POS = (0.45, -0.2, TABLE_HEIGHT)
BOWL_POS = (0.5, 0.15, TABLE_HEIGHT)
FILL_HEIGHT = 0.06
# Position of the hand while it pours, and the angle it tilts the cup by, away from the robot.
POUR_POS = (0.4, 0.15, 0.5)
POUR_ANGLE = 0.55 * math.pi


def container_mjcf(size, wall, free):
    """MJCF model of an open box, its bottom centred on the origin of its body."""
    size_x, size_y, size_z = size
    mjcf = ET.Element("mujoco", model="container")
    body = ET.SubElement(ET.SubElement(mjcf, "worldbody"), "body", name="container")
    if free:
        ET.SubElement(body, "freejoint")
    for pos, half_size in (
        ((0.0, 0.0, 0.5 * wall), (0.5 * size_x, 0.5 * size_y, 0.5 * wall)),
        ((0.5 * (size_x - wall), 0.0, 0.5 * size_z), (0.5 * wall, 0.5 * size_y, 0.5 * size_z)),
        ((-0.5 * (size_x - wall), 0.0, 0.5 * size_z), (0.5 * wall, 0.5 * size_y, 0.5 * size_z)),
        ((0.0, 0.5 * (size_y - wall), 0.5 * size_z), (0.5 * size_x - wall, 0.5 * wall, 0.5 * size_z)),
        ((0.0, -0.5 * (size_y - wall), 0.5 * size_z), (0.5 * size_x - wall, 0.5 * wall, 0.5 * size_z)),
    ):
        ET.SubElement(
            body,
            "geom",
            type="box",
            pos=" ".join(f"{coord:.6g}" for coord in pos),
            size=" ".join(f"{coord:.6g}" for coord in half_size),
        )
    return ET.tostring(mjcf, encoding="unicode")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", help="Show the interactive viewer")
    parser.add_argument("-g", "--gpu", action="store_true", help="Run on GPU instead of CPU")
    parser.add_argument("-r", "--record", action="store_true", help="Record the run to 'out/sph_pour.mp4'")
    args = parser.parse_args()

    gs.init(backend=gs.gpu if args.gpu else gs.cpu, precision="32", seed=0)

    camera_pos = (0.55, 1.3, 0.75)
    camera_lookat = (0.4, 0.0, 0.32)
    # The liquid lives in a box around the table, whose walls bound it where the containers do not.
    domain_lower = (0.2, -0.4, TABLE_HEIGHT - 0.02)
    domain_upper = (0.7, 0.4, 0.7)
    particle_size = 0.006

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1e-2,
            substeps=10,
        ),
        sph_options=gs.options.SPHOptions(
            particle_size=particle_size,
            lower_bound=domain_lower,
            upper_bound=domain_upper,
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
    scene.add_entity(
        gs.morphs.Box(
            pos=(0.5, 0.0, 0.5 * TABLE_HEIGHT),
            size=(0.4, 0.8, TABLE_HEIGHT),
            fixed=True,
        ),
        surface=gs.surfaces.Rough(
            color=(0.55, 0.42, 0.3),
        ),
    )
    cup = scene.add_entity(
        gs.morphs.MJCF(
            file=container_mjcf(CUP_SIZE, WALL, free=True),
            pos=CUP_POS,
        ),
        material=gs.materials.Rigid(
            rho=500.0,
            friction=1.5,
        ),
        surface=gs.surfaces.Default(
            color=(0.85, 0.3, 0.25),
        ),
    )
    scene.add_entity(
        gs.morphs.MJCF(
            file=container_mjcf(BOWL_SIZE, WALL, free=False),
            pos=BOWL_POS,
        ),
        surface=gs.surfaces.Default(
            color=(0.9, 0.9, 0.92),
        ),
    )
    inner_x, inner_y = CUP_SIZE[0] - 2 * WALL, CUP_SIZE[1] - 2 * WALL
    liquid = scene.add_entity(
        material=gs.materials.SPH.Liquid(
            # The viscosity of water, which also keeps the substep within the stable timestep at this particle size.
            mu=1e-3,
        ),
        # The liquid starts a few millimetres clear of the walls and the bottom of the cup.
        morph=gs.morphs.Box(
            pos=(CUP_POS[0], CUP_POS[1], CUP_POS[2] + WALL + 0.5 * FILL_HEIGHT + 0.005),
            size=(inner_x - 0.008, inner_y - 0.008, FILL_HEIGHT),
        ),
        surface=gs.surfaces.Default(
            color=(0.35, 0.6, 0.95, 1.0),
        ),
    )
    franka = scene.add_entity(
        gs.morphs.MJCF(
            file="xml/franka_emika_panda/panda.xml",
        ),
        # The liquid only meets the cup, so the arm stays out of the coupling, which would cost many boundary segments.
        material=gs.materials.Rigid(
            needs_coup=False,
            gravity_compensation=1.0,
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

    if args.vis:
        scene.viewer.add_plugin(
            gs.vis.viewer_plugins.MouseInteractionPlugin(),
        )

    scene.build()

    hand = franka.get_link("hand")
    arm_dofs = range(7)
    fingers_dofs = range(7, 9)
    step_scale = 1.0 if "PYTEST_VERSION" not in os.environ else 0.05
    franka.set_qpos((0.0, -0.4, 0.0, -2.4, 0.0, 2.0, -0.785, 0.04, 0.04))
    # The MJCF drives the fingers through a tendon actuator, so position control needs PD gains of its own.
    franka.set_dofs_kp((300, 300), fingers_dofs)
    franka.set_dofs_kv((20, 20), fingers_dofs)
    franka.control_dofs_position((0.04, 0.04), fingers_dofs)

    def move_hand(pos, steps):
        """Move the pointing-down hand along a straight line."""
        steps = max(int(steps * step_scale), 2)
        pos_start = hand.get_pos()
        # Each solve starts from the previous waypoint, which keeps the arm on one branch of the inverse kinematics.
        qpos = franka.get_qpos()
        for i_step in range(1, steps + 1):
            point = pos_start.lerp(pos_start.new_tensor(pos), i_step / steps)
            qpos = franka.inverse_kinematics(link=hand, pos=point, quat=HAND_QUAT, init_qpos=qpos)
            franka.control_dofs_position(qpos[:7], arm_dofs)
            scene.step()
        # The position controller lags behind the path, so give the hand time to reach its target.
        for _ in range(int(30 * step_scale)):
            scene.step()

    if camera is not None:
        camera.start_recording(save_to_filename="out/sph_pour.mp4", fps=25)

    # Let the liquid settle in the cup, then grasp the cup by the top of its walls.
    for _ in range(int(30 * step_scale)):
        scene.step()
    grip_z = CUP_POS[2] + CUP_SIZE[2] + TIP_DEPTH - 0.025
    move_hand((CUP_POS[0], CUP_POS[1], grip_z + 0.15), steps=80)
    move_hand((CUP_POS[0], CUP_POS[1], grip_z), steps=60)
    franka.control_dofs_force((-10.0, -10.0), fingers_dofs)
    for _ in range(int(30 * step_scale)):
        scene.step()

    # Lift the cup, carry it next to the bowl, then tilt it about the fingertips so that it pours over its far rim.
    move_hand((CUP_POS[0], CUP_POS[1], POUR_POS[2]), steps=80)
    move_hand(POUR_POS, steps=120)
    n_tilt_steps = max(int(150 * step_scale), 2)
    pivot = np.array(POUR_POS) - (0.0, 0.0, TIP_DEPTH)
    qpos = franka.get_qpos()
    for i_step in range(1, n_tilt_steps + 1):
        tilt_quat = gu.axis_angle_to_quat(np.array(POUR_ANGLE * i_step / n_tilt_steps), np.array((0.0, 1.0, 0.0)))
        qpos = franka.inverse_kinematics(
            link=hand,
            pos=pivot + gu.transform_by_quat(np.array((0.0, 0.0, TIP_DEPTH)), tilt_quat),
            quat=gu.transform_quat_by_quat(np.array(HAND_QUAT), tilt_quat),
            init_qpos=qpos,
        )
        franka.control_dofs_position(qpos[:7], arm_dofs)
        scene.step()
    for _ in range(int(60 * step_scale)):
        scene.step()

    particles_pos = liquid.get_particles_pos()
    particles_offset = particles_pos - particles_pos.new_tensor(BOWL_POS)
    is_in_bowl = particles_offset[:, 2] < BOWL_SIZE[2]
    is_in_bowl &= (particles_offset[:, :2].abs() < 0.5 * BOWL_SIZE[0] - WALL).all(dim=-1)
    gs.logger.info(f"{100.0 * is_in_bowl.float().mean().item():.0f}% of the liquid landed in the bowl.")

    if camera is not None:
        camera.stop_recording()

    while args.vis and scene.viewer.is_alive() and "PYTEST_VERSION" not in os.environ:
        scene.step()


if __name__ == "__main__":
    main()
