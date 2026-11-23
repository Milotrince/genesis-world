"""A Franka arm and its hand, loaded from separate URDF or MJCF files, are merged into one articulation.

The hand is attached to the arm before build, then both run through a short PD schedule.

Mouse controls (-v): drag the arm or the hand, which the PD controllers hold upright.
"""

import argparse
import os

import numpy as np

import genesis as gs


COMB = {
    "urdf2urdf",
    "urdf2mjcf",
    "mjcf2urdf",
    "mjcf2mjcf",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", help="Show visualization GUI")
    parser.add_argument("--comb", type=str, default="urdf2urdf", choices=COMB, help="Entities to merge")
    parser.add_argument(
        "-r", "--record", action="store_true", help="Record a video of the run to 'out/merge_entities.mp4'"
    )
    args = parser.parse_args()
    n_phase_steps = 250 if "PYTEST_VERSION" not in os.environ else 30

    gs.init(backend=gs.cpu)

    camera_pos = (0, -3.5, 2.5)
    camera_lookat = (0.0, 0.0, 0.5)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.01,
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

    if args.comb == "urdf2urdf" or args.comb == "urdf2mjcf":
        gs.logger.info("loading URDF panda arm")
        franka = scene.add_entity(
            gs.morphs.URDF(
                file="urdf/panda_bullet/panda_nohand.urdf",
                merge_fixed_links=False,
                fixed=True,
            ),
        )
    else:
        gs.logger.info("loading MJCF panda arm")
        franka = scene.add_entity(
            gs.morphs.MJCF(
                file="xml/franka_emika_panda/panda_nohand.xml",
            ),
        )

    if args.comb == "urdf2urdf" or args.comb == "mjcf2urdf":
        gs.logger.info("loading URDF panda hand")
        hand = scene.add_entity(
            gs.morphs.URDF(
                file="urdf/panda_bullet/hand.urdf",
                merge_fixed_links=False,
            ),
        )
    else:
        gs.logger.info("loading MJCF panda hand")
        hand = scene.add_entity(
            gs.morphs.MJCF(
                file="xml/franka_emika_panda/hand.xml",
            ),
        )

    print([link.name for link in franka.links])
    print([link.name for link in hand.links])
    hand.attach(franka, "attachment")

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
    if camera is not None:
        camera.start_recording(save_to_filename="out/merge_entities.mp4", fps=round(1.0 / scene.dt))

    arm_joints_name = (
        "joint1",
        "joint2",
        "joint3",
        "joint4",
        "joint5",
        "joint6",
        "joint7",
    )
    arm_dofs_idx = [franka.get_joint(name).dofs_idx_local[0] for name in arm_joints_name]

    franka.set_dofs_kp(
        np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000]),
        arm_dofs_idx,
    )
    franka.set_dofs_kv(
        np.array([450, 450, 350, 350, 200, 200, 200]),
        arm_dofs_idx,
    )
    franka.set_dofs_force_range(
        np.array([-87, -87, -87, -87, -12, -12, -12]),
        np.array([87, 87, 87, 87, 12, 12, 12]),
        arm_dofs_idx,
    )

    gripper_joints_name = (
        "finger_joint1",
        "finger_joint2",
    )
    gripper_dofs_idx = [hand.get_joint(name).dofs_idx_local[0] for name in gripper_joints_name]

    hand.set_dofs_kp(
        np.array([100, 100]),
        gripper_dofs_idx,
    )
    hand.set_dofs_kv(
        np.array([10, 10]),
        gripper_dofs_idx,
    )
    hand.set_dofs_force_range(
        np.array([-100, -100]),
        np.array([100, 100]),
        gripper_dofs_idx,
    )

    if args.vis:
        # The PD controllers hold the arm upright with the gripper open while the mouse drags it.
        franka.control_dofs_position([0, 0, 0, 0, 0, 0, 0], arm_dofs_idx)
        hand.control_dofs_position([0.04, 0.04], gripper_dofs_idx)
        while scene.viewer.is_alive():
            scene.step()
    else:
        # PD control
        for arm_qpos, gripper_qpos in (
            ([1, 1, 0, 0, 0, 0, 0], [0.04, 0.04]),
            ([-1, 0.8, 1, -2, 1, 0.5, -0.5], [0.0, 0.0]),
            ([0, 0, 0, 0, 0, 0, 0], [0.04, 0.04]),
        ):
            franka.control_dofs_position(arm_qpos, arm_dofs_idx)
            hand.control_dofs_position(gripper_qpos, gripper_dofs_idx)
            for _ in range(n_phase_steps):
                scene.step()
    if camera is not None:
        camera.stop_recording()


if __name__ == "__main__":
    main()
