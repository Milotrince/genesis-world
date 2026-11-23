"""Two Frankas pick a cloth up off the floor by two corners, carry it across a table and lay it down to drape it.

The cloth is a position-based dynamics (PBD) cloth. Each hand pins the particles under its fingertips with
'fix_particles_to_link', which carries them along with the hand until 'release_particles' drops them. The hands lift the
edge of the cloth next to the table and carry it across the table top, which drags the rest of the cloth over it, then
lay the edge down past the far side. Particle scenes are slow on CPU, so '-g' speeds the run up.

With the viewer (-v), the scripted drape is skipped: the cloth lies flat on the floor beside the table, the arms wait
in their start pose, and the keyboard drives both hands together.

Keyboard controls (-v):
Up        - Move forward (+x)
Down      - Move backward (-x)
Left      - Move left (+y)
Right     - Move right (-y)
K         - Move up
J         - Move down
G         - Grab the cloth under the fingertips
U         - Release the cloth
Esc       - Quit
"""

import argparse
import os

import torch

import genesis as gs
from genesis.vis.keybindings import Key, KeyAction, Keybind


# Distance from the origin of the hand to its fingertips, the hand pointing down.
TIP_DEPTH = 0.105
TABLE_SIZE = (0.25, 0.4, 0.18)
TABLE_POS = (0.35, 0.0)
# The cloth slides on the table without friction, so it is long enough to reach the floor on both sides, which holds it.
CLOTH_SIZE = 0.8
# The cloth lies on the floor in front of the near side of the table, its edge along y facing the table.
CLOTH_NEAR_X = 0.05
# Radius around the fingertips within which the particles follow the hand.
GRIP_RADIUS = 0.025


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", help="Show the interactive viewer")
    parser.add_argument("-g", "--gpu", action="store_true", help="Run on GPU instead of CPU")
    parser.add_argument("-r", "--record", action="store_true", help="Record the run to 'out/pbd_tablecloth.mp4'")
    args = parser.parse_args()

    gs.init(backend=gs.gpu if args.gpu else gs.cpu, precision="32", seed=0)

    camera_pos = (1.9, -1.7, 1.3)
    camera_lookat = (0.1, 0.0, 0.2)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1e-2,
            substeps=10,
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
            pos=(*TABLE_POS, 0.5 * TABLE_SIZE[2]),
            size=TABLE_SIZE,
            fixed=True,
        ),
        surface=gs.surfaces.Rough(
            color=(0.55, 0.42, 0.3),
        ),
    )
    cloth = scene.add_entity(
        material=gs.materials.PBD.Cloth(),
        morph=gs.morphs.Mesh(
            file="meshes/cloth.obj",
            scale=CLOTH_SIZE / 0.5,
            pos=(CLOTH_NEAR_X - 0.5 * CLOTH_SIZE, 0.0, 0.01),
        ),
        surface=gs.surfaces.Default(
            color=(0.85, 0.25, 0.25, 1.0),
        ),
    )
    frankas = [
        scene.add_entity(
            gs.morphs.MJCF(
                file="xml/franka_emika_panda/panda.xml",
                pos=(0.15, y, 0.0),
            ),
            # The corners ride on the hands, so the arms stay out of the cloth coupling.
            material=gs.materials.Rigid(
                needs_coup=False,
                gravity_compensation=1.0,
            ),
        )
        for y in (-0.75, 0.75)
    ]

    camera = None
    if args.record:
        camera = scene.add_camera(
            res=(1280, 720),
            pos=camera_pos,
            lookat=camera_lookat,
            fov=40,
        )

    scene.build()

    hands = [franka.get_link("hand") for franka in frankas]
    arm_dofs = range(7)
    fingers_dofs = range(7, 9)
    step_scale = 1.0 if "PYTEST_VERSION" not in os.environ else 0.05
    # Each arm turns its base towards the cloth, so that the inverse kinematics solves stay on the elbow-up branch.
    for franka, joint1 in zip(frankas, (1.57, -1.57)):
        franka.set_qpos((joint1, -0.4, 0.0, -2.4, 0.0, 2.0, 0.785, 0.04, 0.04))
        franka.set_dofs_kp((300, 300), fingers_dofs)
        franka.set_dofs_kv((20, 20), fingers_dofs)
        franka.control_dofs_position((0.04, 0.04), fingers_dofs)
    # Each hand keeps the orientation it starts with, pointing down.
    hands_quat = [hand.get_quat() for hand in hands]

    grabbed_particles_idx = [None, None]

    def move_tips(targets, steps):
        """Move the fingertips of both hands to their targets, along straight lines."""
        steps = max(int(steps * step_scale), 2)
        hands_start = [hand.get_pos() for hand in hands]
        # Each solve starts from the previous waypoint, which keeps the arm on one branch of the inverse kinematics.
        arms_qpos = [franka.get_qpos() for franka in frankas]
        for i_step in range(1, steps + 1):
            for i_hand, (franka, hand, start, target) in enumerate(zip(frankas, hands, hands_start, targets)):
                point = start.lerp(start.new_tensor((target[0], target[1], target[2] + TIP_DEPTH)), i_step / steps)
                arms_qpos[i_hand] = franka.inverse_kinematics(
                    link=hand, pos=point, quat=hands_quat[i_hand], init_qpos=arms_qpos[i_hand]
                )
                franka.control_dofs_position(arms_qpos[i_hand][:7], arm_dofs)
            scene.step()
        # The position controller lags behind the path, so give the hands time to reach their targets.
        for _ in range(int(30 * step_scale)):
            scene.step()

    def grab():
        """Pin the particles under the fingertips of each hand to it, and close the gripper on them."""
        particles_pos = cloth.get_particles_pos()
        for i_hand, (franka, hand) in enumerate(zip(frankas, hands)):
            tips_pos = hand.get_pos() - hand.get_pos().new_tensor((0.0, 0.0, TIP_DEPTH))
            (particles_idx,) = torch.nonzero((particles_pos - tips_pos).norm(dim=-1) < GRIP_RADIUS, as_tuple=True)
            grabbed_particles_idx[i_hand] = particles_idx
            franka.control_dofs_position((0.0, 0.0), fingers_dofs)
            if len(particles_idx) > 0:
                cloth.fix_particles_to_link(hand.idx, particles_idx_local=particles_idx)

    def release():
        for franka, particles_idx in zip(frankas, grabbed_particles_idx):
            franka.control_dofs_position((0.04, 0.04), fingers_dofs)
            if particles_idx is not None and len(particles_idx) > 0:
                cloth.release_particles(particles_idx)

    if camera is not None:
        camera.start_recording(save_to_filename="out/pbd_tablecloth.mp4", fps=50)

    if args.vis:
        # Keybind callbacks run on the viewer thread, so they only move the targets and raise requests for this loop.
        tips_targets = [hand.get_pos() - hand.get_pos().new_tensor((0.0, 0.0, TIP_DEPTH)) for hand in hands]
        is_running = True
        is_grab_requested = False
        is_release_requested = False

        def move(delta):
            for target in tips_targets:
                target[:] += target.new_tensor(delta)

        def request_grab():
            nonlocal is_grab_requested
            is_grab_requested = True

        def request_release():
            nonlocal is_release_requested
            is_release_requested = True

        def stop():
            nonlocal is_running
            is_running = False

        step = 0.003
        scene.viewer.register_keybinds(
            Keybind("move_forward", Key.UP, KeyAction.HOLD, callback=move, args=((step, 0.0, 0.0),)),
            Keybind("move_backward", Key.DOWN, KeyAction.HOLD, callback=move, args=((-step, 0.0, 0.0),)),
            Keybind("move_left", Key.LEFT, KeyAction.HOLD, callback=move, args=((0.0, step, 0.0),)),
            Keybind("move_right", Key.RIGHT, KeyAction.HOLD, callback=move, args=((0.0, -step, 0.0),)),
            Keybind("move_up", Key.K, KeyAction.HOLD, callback=move, args=((0.0, 0.0, step),)),
            Keybind("move_down", Key.J, KeyAction.HOLD, callback=move, args=((0.0, 0.0, -step),)),
            Keybind("grab", Key.G, KeyAction.RELEASE, callback=request_grab),
            Keybind("release", Key.U, KeyAction.RELEASE, callback=request_release),
            Keybind("quit", Key.ESCAPE, KeyAction.RELEASE, callback=stop),
        )
        gs.logger.info(
            "Keys: Up/Down/Left/Right move the hands, K/J move them up/down, G grabs, U releases, Esc quits."
        )
        while scene.viewer.is_alive() and is_running:
            if is_grab_requested:
                is_grab_requested = False
                grab()
            if is_release_requested:
                is_release_requested = False
                release()
            for franka, hand, quat, target in zip(frankas, hands, hands_quat, tips_targets):
                qpos = franka.inverse_kinematics(
                    link=hand, pos=target + target.new_tensor((0.0, 0.0, TIP_DEPTH)), quat=quat
                )
                franka.control_dofs_position(qpos[:7], arm_dofs)
            scene.step()
    else:
        corners_y = (-0.5 * CLOTH_SIZE, 0.5 * CLOTH_SIZE)
        table_far_x = TABLE_POS[0] + 0.5 * TABLE_SIZE[0]
        carry_z = TABLE_SIZE[2] + 0.15
        move_tips([(CLOTH_NEAR_X, y, 0.08) for y in corners_y], steps=80)
        move_tips([(CLOTH_NEAR_X, y, 0.01) for y in corners_y], steps=40)
        grab()
        # Lift the edge above the table and carry it past the far side, which drags the rest of the cloth up over the
        # near side and across the table top, then lower the edge past the far side and let go.
        move_tips([(CLOTH_NEAR_X, y, carry_z) for y in corners_y], steps=80)
        move_tips([(table_far_x + 0.08, y, carry_z) for y in corners_y], steps=200)
        move_tips([(table_far_x + 0.08, y, 0.03) for y in corners_y], steps=100)
        release()
        move_tips([(table_far_x + 0.08, y, carry_z) for y in corners_y], steps=60)
        for _ in range(int(50 * step_scale)):
            scene.step()

        particles_pos = cloth.get_particles_pos()
        is_on_table = particles_pos[:, 2] > TABLE_SIZE[2] - 0.01
        gs.logger.info(f"{100.0 * is_on_table.float().mean().item():.0f}% of the cloth lies on the table top.")
    if camera is not None:
        camera.stop_recording()


if __name__ == "__main__":
    main()
