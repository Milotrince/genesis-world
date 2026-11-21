"""
Heterogeneous Simulation Example
================================

This example demonstrates heterogeneous simulation, where different parallel
environments can have different geometry variants for the same entity.

Variant Assignment Rules:
    When passing a list of morphs to scene.add_entity(), variants are distributed
    across environments using the following rules:

    1. When n_envs >= n_variants:
       Balanced block assignment. Environments are divided into blocks, with each
       block assigned to one variant. For example, with 4 variants and 8 environments:
       - Environments 0-1 -> Variant 0
       - Environments 2-3 -> Variant 1
       - Environments 4-5 -> Variant 2
       - Environments 6-7 -> Variant 3

    2. When n_envs < n_variants:
       Each environment i gets variant i (0-indexed). Variants beyond n_envs are
       unused. For example, with 4 variants and 2 environments:
       - Environment 0 -> Variant 0 (first morph in list)
       - Environment 1 -> Variant 1 (second morph in list)
       - Variants 2 and 3 are unused

With the viewer (-v), the arms hold their start pose beside the objects, and the mouse drags either.

Usage:
    python heterogeneous_simulation.py -v -b 4  # 4 environments (matches 4 variants)
    python heterogeneous_simulation.py -v -b 8  # 8 environments (2 per variant)
    python heterogeneous_simulation.py -v -b 2  # 2 environments (only first 2 variants used)
"""

import argparse
import os
import xml.etree.ElementTree as ET

import numpy as np

import genesis as gs

TABLE_HEIGHT = 0.3


def table_mjcf():
    mjcf = ET.Element("mujoco", model="table")
    ET.SubElement(ET.SubElement(mjcf, "default"), "geom", rgba="0.95 0.95 0.95 1")
    body = ET.SubElement(ET.SubElement(mjcf, "worldbody"), "body", name="table", pos=f"0.55 0 {TABLE_HEIGHT}")
    ET.SubElement(body, "geom", type="box", size="0.35 0.7 0.015", pos="0 0 -0.015")
    for x in (-0.31, 0.31):
        for y in (-0.66, 0.66):
            ET.SubElement(body, "geom", type="box", size="0.02 0.02 0.135", pos=f"{x} {y} -0.165")
    return ET.tostring(mjcf, encoding="unicode")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", help="Show visualization GUI")
    parser.add_argument("-b", "--num-envs", type=int, default=4, help="Number of parallel environments")
    parser.add_argument(
        "-r", "--record", action="store_true", help="Record a video of the run to 'out/franka_heterogeneous_grasp.mp4'"
    )
    args = parser.parse_args()
    n_hold_steps = 100 if "PYTEST_VERSION" not in os.environ else 20

    gs.init(backend=gs.cpu, precision="32")

    camera_pos = (-1.5, -2.2, 1.0 + TABLE_HEIGHT)
    camera_lookat = (0.5, 0.2, TABLE_HEIGHT)

    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=camera_pos,
            camera_lookat=camera_lookat,
        ),
        show_viewer=args.vis,
    )

    plane = scene.add_entity(
        gs.morphs.Plane(),
    )
    scene.add_entity(
        gs.morphs.MJCF(file=table_mjcf()),
    )
    franka = scene.add_entity(
        gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
    )

    # Define 4 geometry variants - see module docstring for variant assignment rules
    morphs_heterogeneous = [
        gs.morphs.Box(size=(0.04, 0.04, 0.04), pos=(0.65, 0.0, 0.02 + TABLE_HEIGHT)),  # Variant 0
        gs.morphs.Box(size=(0.02, 0.02, 0.02), pos=(0.65, 0.0, 0.02 + TABLE_HEIGHT)),  # Variant 1
        gs.morphs.Sphere(radius=0.015, pos=(0.65, 0.0, 0.02 + TABLE_HEIGHT)),  # Variant 2
        gs.morphs.Sphere(radius=0.025, pos=(0.65, 0.0, 0.02 + TABLE_HEIGHT)),  # Variant 3
    ]
    grasping_object = scene.add_entity(
        morph=morphs_heterogeneous,
    )

    camera = None
    if args.record:
        camera = scene.add_camera(
            res=(1280, 720),
            pos=camera_pos,
            lookat=camera_lookat,
        )

    if args.vis:
        scene.viewer.add_plugin(
            gs.vis.viewer_plugins.MouseInteractionPlugin(),
        )

    scene.build(n_envs=args.num_envs, env_spacing=(1.0, 1.5))
    if camera is not None:
        camera.start_recording(save_to_filename="out/franka_heterogeneous_grasp.mp4", fps=round(1.0 / scene.dt))

    motors_dof = np.arange(7)
    fingers_dof = np.arange(7, 9)
    franka.set_dofs_kp([100.0, 100.0], fingers_dof)
    franka.set_dofs_kv([10.0, 10.0], fingers_dof)
    l_qpos = [0.0, 0.316, 0.0, -1.661, 0.0, 1.976, 0.785, 0.04, 0.04]
    if args.num_envs == 0:
        franka.set_qpos(np.array(l_qpos))
    else:
        franka.set_qpos(np.array([l_qpos] * args.num_envs))
    scene.step()

    AABB = grasping_object.get_AABB()
    mass = grasping_object.get_mass()
    print("heterogeneous AABB", AABB)
    print("heterogeneous mass", mass)

    if args.vis:
        franka.control_dofs_position(franka.get_dofs_position())
        while scene.viewer.is_alive():
            scene.step()
    else:
        end_effector = franka.get_link("hand")
        qpos = franka.inverse_kinematics(
            link=end_effector,
            pos=np.array([[0.65, 0.0, 0.125 + TABLE_HEIGHT]] * args.num_envs),
            quat=np.array([[0, 1, 0, 0]] * args.num_envs),
        )
        franka.control_dofs_position(qpos[..., :-2], motors_dof)

        # hold
        for _ in range(n_hold_steps):
            scene.step()

        # grasp
        finder_pos = 0.0
        for _ in range(n_hold_steps):
            franka.control_dofs_position(qpos[..., :-2], motors_dof)
            franka.control_dofs_position(np.array([[finder_pos, finder_pos]] * args.num_envs), fingers_dof)
            scene.step()

        # lift
        qpos = franka.inverse_kinematics(
            link=end_effector,
            pos=np.array([[0.65, 0.0, 0.3 + TABLE_HEIGHT]] * args.num_envs),
            quat=np.array([[0, 1, 0, 0]] * args.num_envs),
        )
        for _ in range(2 * n_hold_steps):
            franka.control_dofs_position(qpos[..., :-2], motors_dof)
            franka.control_dofs_position(np.array([[finder_pos, finder_pos]] * args.num_envs), fingers_dof)
            scene.step()
    if camera is not None:
        camera.stop_recording()


if __name__ == "__main__":
    main()
