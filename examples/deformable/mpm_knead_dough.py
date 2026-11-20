"""A Franka carries a lump of dough onto a board, presses it flat, folds it over and presses it again.

The dough is an elasto-plastic MPM material: past its yield stress it flows, so every press leaves a lasting dent
instead of springing back. The robot holds a flat paddle attached to its hand. It carries the lump by attaching the
particles under the paddle to the paddle with particle constraints, and folds the flattened dough the same way, by
lifting one half over the other. The run logs the height of the dough under the paddle and after the paddle lifts,
read from the particle heights. MPM is slow on CPU, so pass -g to run on GPU.

Keyboard controls (-v), once the scripted kneading is over:
Up/Down    - Move the paddle along -x/+x
Left/Right - Move the paddle along -y/+y
K/J        - Move the paddle up/down
Esc        - Quit
"""

import argparse
import os
import xml.etree.ElementTree as ET

import numpy as np

import genesis as gs
from genesis.utils.misc import tensor_to_array
from genesis.vis.keybindings import Key, KeyAction, Keybind


DT = 4e-3

BOARD_POS = (0.55, 0.08)
BOARD_TOP = 0.03
DOUGH_START = (0.55, -0.2)
DOUGH_SIZE = 0.08
# Half-sizes of the blade of the paddle, and the distance from the hand down to the bottom face of the blade.
BLADE_HALF_SIZE = (0.07, 0.05, 0.012)
PADDLE_LENGTH = 0.16


def paddle_mjcf():
    """MJCF model of the paddle, in the frame of the hand that holds it: the handle runs down its z axis to the blade."""
    mjcf = ET.Element("mujoco", model="paddle")
    paddle = ET.SubElement(ET.SubElement(mjcf, "worldbody"), "body", name="paddle")
    # The handle passes between the fingers, so it is left out of collisions.
    ET.SubElement(
        paddle,
        "geom",
        type="cylinder",
        fromto=f"0 0 0.03 0 0 {PADDLE_LENGTH - 2 * BLADE_HALF_SIZE[2]}",
        size="0.012",
        contype="0",
        conaffinity="0",
        density="600",
        rgba="0.55 0.35 0.2 1",
    )
    ET.SubElement(
        paddle,
        "geom",
        type="box",
        pos=f"0 0 {PADDLE_LENGTH - BLADE_HALF_SIZE[2]}",
        size=" ".join(map(str, BLADE_HALF_SIZE)),
        density="600",
        rgba="0.65 0.45 0.28 1",
    )
    return ET.tostring(mjcf, encoding="unicode")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", help="Show the interactive viewer")
    parser.add_argument("-g", "--gpu", action="store_true", help="Run on GPU instead of CPU")
    parser.add_argument("-r", "--record", action="store_true", help="Record the run to 'out/mpm_knead_dough.mp4'")
    args = parser.parse_args()

    gs.init(backend=gs.gpu if args.gpu else gs.cpu)

    camera_pos = (0.95, -0.6, 0.28)
    camera_lookat = (0.5, -0.02, 0.08)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=DT,
            substeps=10,
        ),
        mpm_options=gs.options.MPMOptions(
            grid_density=96,
            lower_bound=(0.35, -0.32, -0.03),
            upper_bound=(0.75, 0.25, 0.3),
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
        morph=gs.morphs.Box(
            pos=(*BOARD_POS, 0.5 * BOARD_TOP),
            size=(0.24, 0.24, BOARD_TOP),
            fixed=True,
        ),
        material=gs.materials.Rigid(
            coup_friction=1.0,
        ),
        surface=gs.surfaces.Rough(
            color=(0.75, 0.6, 0.4),
        ),
    )
    dough = scene.add_entity(
        morph=gs.morphs.Box(
            pos=(*DOUGH_START, 0.5 * DOUGH_SIZE),
            size=(DOUGH_SIZE, DOUGH_SIZE, DOUGH_SIZE),
        ),
        material=gs.materials.MPM.ElastoPlastic(
            E=5e4,
            nu=0.3,
            rho=1100.0,
            von_mises_yield_stress=500.0,
        ),
        surface=gs.surfaces.Rough(
            color=(0.95, 0.87, 0.65),
            vis_mode="particle",
        ),
    )
    franka = scene.add_entity(
        morph=gs.morphs.MJCF(
            file="xml/franka_emika_panda/panda.xml",
        ),
        material=gs.materials.Rigid(
            gravity_compensation=1.0,
        ),
    )
    # The paddle has no joint of its own, so it moves with the hand only once its vertices follow the hand.
    paddle = scene.add_entity(
        morph=gs.morphs.MJCF(
            file=paddle_mjcf(),
            batch_fixed_verts=True,
        ),
        material=gs.materials.Rigid(
            coup_friction=0.5,
        ),
    )
    paddle.attach(franka, "hand", pos=(0.0, 0.0, 0.0))

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
    # The hand points down with its x axis along world x, so the blade lies flat under it.
    hand_quat = np.array([0.0, 1.0, 0.0, 0.0])
    target_pos = np.array([DOUGH_START[0], DOUGH_START[1], DOUGH_SIZE + PADDLE_LENGTH + 0.04])
    franka.set_qpos(franka.inverse_kinematics(link=hand, pos=target_pos, quat=hand_quat)[:7], motors_dof)
    franka.set_dofs_kp([100.0, 100.0], fingers_dof)
    franka.set_dofs_kv([10.0, 10.0], fingers_dof)
    franka.control_dofs_position([0.012, 0.012], fingers_dof)

    def track_target():
        q = franka.inverse_kinematics(link=hand, pos=target_pos, quat=hand_quat)[:7]
        franka.control_dofs_position(q, motors_dof)
        scene.step()

    def move_to(waypoint, speed):
        n_steps = int(np.abs(np.array(waypoint) - target_pos).max() / (speed * DT)) + 10
        if "PYTEST_VERSION" in os.environ:
            n_steps = 3
        for _ in range(n_steps):
            target_pos[:] += np.clip(np.array(waypoint) - target_pos, -speed * DT, speed * DT)
            track_target()

    def dough_extent():
        """Center of the dough in the horizontal plane, its top and its thickness."""
        dough_pos = tensor_to_array(dough.get_particles_pos())
        z_bottom, z_top = np.percentile(dough_pos[..., 2], (2, 98))
        return dough_pos[..., :2].mean(axis=0), z_top, z_top - z_bottom

    def press(depth):
        center, z_top, _ = dough_extent()
        move_to((*center, z_top + PADDLE_LENGTH + 0.02), speed=0.2)
        move_to((*center, BOARD_TOP + depth + PADDLE_LENGTH), speed=0.05)
        _, _, thickness_pressed = dough_extent()
        move_to((*center, BOARD_TOP + thickness_pressed + PADDLE_LENGTH + 0.05), speed=0.1)
        gs.logger.info(
            f"Pressed: dough {100 * thickness_pressed:.1f} cm thick under the paddle, "
            f"{100 * dough_extent()[2]:.1f} cm once the paddle lifts."
        )

    def carry(bbox_min, bbox_max, waypoints):
        """Attach the particles in a box to the paddle, move the paddle through the waypoints, then release them."""
        dough.set_particle_constraints(
            dough.get_particles_in_bbox(bbox_min, bbox_max), paddle.links[0].idx, stiffness=1e5
        )
        for waypoint in waypoints:
            move_to(waypoint, speed=0.15)
        dough.remove_particle_constraints()

    if camera is not None:
        camera.start_recording(save_to_filename="out/mpm_knead_dough.mp4", fps=round(0.5 / DT))
    gs.logger.info(f"Lump {100 * dough_extent()[2]:.1f} cm thick.")

    # Rest the blade on the lump, then carry it onto the board by its top layer.
    move_to((*DOUGH_START, DOUGH_SIZE + PADDLE_LENGTH - 0.005), speed=0.08)
    carry(
        (DOUGH_START[0] - 0.06, DOUGH_START[1] - 0.06, DOUGH_SIZE - 0.025),
        (DOUGH_START[0] + 0.06, DOUGH_START[1] + 0.06, DOUGH_SIZE + 0.01),
        (
            (*DOUGH_START, DOUGH_SIZE + PADDLE_LENGTH + 0.06),
            (*BOARD_POS, BOARD_TOP + DOUGH_SIZE + PADDLE_LENGTH + 0.06),
            (*BOARD_POS, BOARD_TOP + DOUGH_SIZE + PADDLE_LENGTH),
        ),
    )
    move_to((*BOARD_POS, BOARD_TOP + DOUGH_SIZE + PADDLE_LENGTH + 0.04), speed=0.1)
    gs.logger.info(f"Lump {100 * dough_extent()[2]:.1f} cm thick on the board.")

    press(depth=0.025)
    # Fold the slab: lift its half on the -y side by the paddle and lay it over the other half.
    center, _, thickness = dough_extent()
    carry(
        (center[0] - 0.15, center[1] - 0.15, 0.0),
        (center[0] + 0.15, center[1], 0.2),
        (
            (*target_pos[:2], target_pos[2] + 0.04),
            (center[0], center[1] + 0.06, target_pos[2] + 0.04),
            (center[0], center[1] + 0.06, target_pos[2] + thickness),
        ),
    )
    gs.logger.info(f"Folded dough {100 * dough_extent()[2]:.1f} cm thick.")
    press(depth=0.03)
    if camera is not None:
        camera.stop_recording()

    if args.vis:
        is_running = True

        def move(dpos):
            target_pos[:] += dpos

        def stop():
            nonlocal is_running
            is_running = False

        dpos = 0.002
        scene.viewer.register_keybinds(
            Keybind("move_forward", Key.UP, KeyAction.HOLD, callback=move, args=((-dpos, 0, 0),)),
            Keybind("move_back", Key.DOWN, KeyAction.HOLD, callback=move, args=((dpos, 0, 0),)),
            Keybind("move_left", Key.LEFT, KeyAction.HOLD, callback=move, args=((0, -dpos, 0),)),
            Keybind("move_right", Key.RIGHT, KeyAction.HOLD, callback=move, args=((0, dpos, 0),)),
            Keybind("move_up", Key.K, KeyAction.HOLD, callback=move, args=((0, 0, dpos),)),
            Keybind("move_down", Key.J, KeyAction.HOLD, callback=move, args=((0, 0, -dpos),)),
            Keybind("quit", Key.ESCAPE, KeyAction.RELEASE, callback=stop),
        )
        gs.logger.info("Keys: arrows move the paddle in the horizontal plane, K/J move it up/down, Esc quits.")
        while is_running and scene.viewer.is_alive() and "PYTEST_VERSION" not in os.environ:
            track_target()


if __name__ == "__main__":
    main()
