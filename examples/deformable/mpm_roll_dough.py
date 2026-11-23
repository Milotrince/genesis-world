"""A Franka rolls a lump of dough out into a flat sheet with a rolling pin.

The dough is an elasto-plastic MPM material: past its yield stress it flows, so each pass of the pin spreads it wider
instead of springing back. The robot holds a rolling pin attached to its hand and sweeps it back and forth
over the lump on a board, lowering it after every pass. The run logs the thickness and the footprint of the dough
after each pass, read from the particle positions. MPM is slow on CPU, so pass -g to run on GPU.

With the viewer (-v), the scripted passes are skipped: the robot holds the pin above the lump, and the keys drive it.

Keyboard controls (-v):
Up/Down    - Move the pin along -x/+x
Left/Right - Move the pin along -y/+y
K/J        - Move the pin up/down
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

BOARD_POS = (0.55, 0.0)
BOARD_TOP = 0.03
DOUGH_SIZE = (0.08, 0.08, 0.06)
# Radius and half-length of the rolling pin, and the distance from the hand down to the axis of the pin.
PIN_RADIUS = 0.025
PIN_HALF_LENGTH = 0.12
PIN_DEPTH = 0.16
# Height of the bottom of the pin above the board on each pass, and half the length of a pass, centered on the board.
PASSES_HEIGHT = (0.035, 0.022, 0.012)
PASS_HALF_LENGTH = 0.09


def rolling_pin_mjcf():
    """MJCF model of the rolling pin, in the frame of the hand that holds it: a handle runs down its z axis to the pin,
    whose axis lies along its y axis."""
    mjcf = ET.Element("mujoco", model="rolling_pin")
    pin = ET.SubElement(ET.SubElement(mjcf, "worldbody"), "body", name="rolling_pin")
    # The handle passes between the fingers, so it is left out of collisions.
    ET.SubElement(
        pin,
        "geom",
        type="cylinder",
        fromto=f"0 0 0.03 0 0 {PIN_DEPTH - PIN_RADIUS}",
        size="0.01",
        contype="0",
        conaffinity="0",
        density="600",
        rgba="0.55 0.35 0.2 1",
    )
    ET.SubElement(
        pin,
        "geom",
        type="cylinder",
        fromto=f"0 {-PIN_HALF_LENGTH} {PIN_DEPTH} 0 {PIN_HALF_LENGTH} {PIN_DEPTH}",
        size=f"{PIN_RADIUS}",
        density="600",
        rgba="0.75 0.55 0.35 1",
    )
    return ET.tostring(mjcf, encoding="unicode")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", help="Show the interactive viewer")
    parser.add_argument("-g", "--gpu", action="store_true", help="Run on GPU instead of CPU")
    parser.add_argument("-r", "--record", action="store_true", help="Record the run to 'out/mpm_roll_dough.mp4'")
    args = parser.parse_args()

    gs.init(backend=gs.gpu if args.gpu else gs.cpu)

    # The camera looks down on the board at about 30 degrees, so that the dough shows as it spreads.
    camera_pos = (0.85, -0.55, 0.4)
    camera_lookat = (0.52, 0.0, 0.05)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=DT,
            substeps=10,
        ),
        mpm_options=gs.options.MPMOptions(
            grid_density=128,
            lower_bound=(0.36, -0.16, -0.03),
            upper_bound=(0.74, 0.16, 0.15),
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
            size=(0.32, 0.26, BOARD_TOP),
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
        spec=gs.entities.MPM(
            morph=gs.morphs.Box(
                pos=(*BOARD_POS, BOARD_TOP + 0.5 * DOUGH_SIZE[2]),
                size=DOUGH_SIZE,
            ),
            material=gs.materials.MPM.ElastoPlastic(
                E=5e4,
                nu=0.3,
                rho=1100.0,
                von_mises_yield_stress=800.0,
            ),
            surface=gs.surfaces.Rough(
                color=(0.95, 0.87, 0.65),
            ),
            vis=gs.options.EntityVisOptions(
                debug=True,
            ),
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
    # The pin has no joint of its own, so it moves with the hand only once its vertices follow the hand.
    pin = scene.add_entity(
        morph=gs.morphs.MJCF(
            file=rolling_pin_mjcf(),
            batch_fixed_verts=True,
        ),
    )
    pin.attach(franka, "hand", pos=(0.0, 0.0, 0.0))

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

    motors_dof = np.arange(7)
    fingers_dof = np.arange(7, 9)
    hand = franka.get_link("hand")
    # The hand points down with its y axis along world y, so the pin lies across the direction of the passes.
    hand_quat = np.array([0.0, 1.0, 0.0, 0.0])
    x_start = BOARD_POS[0] - PASS_HALF_LENGTH
    target_pos = np.array([x_start, BOARD_POS[1], BOARD_TOP + DOUGH_SIZE[2] + PIN_RADIUS + PIN_DEPTH + 0.03])
    franka.set_qpos(franka.inverse_kinematics(link=hand, pos=target_pos, quat=hand_quat)[:7], motors_dof)
    franka.set_dofs_kp([100.0, 100.0], fingers_dof)
    franka.set_dofs_kv([10.0, 10.0], fingers_dof)
    franka.control_dofs_position([0.01, 0.01], fingers_dof)

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

    def log_dough(label):
        dough_pos = tensor_to_array(dough.get_particles_pos())
        thickness = np.percentile(dough_pos[..., 2], 98) - BOARD_TOP
        # The footprint counts the cells of a 1 cm grid over the board that hold at least one particle.
        cells = np.unique(np.floor(dough_pos[..., :2] / 0.01), axis=0)
        gs.logger.info(f"{label}: dough {100 * thickness:.1f} cm thick, footprint {len(cells)} cm^2.")

    if camera is not None:
        camera.start_recording(save_to_filename="out/mpm_roll_dough.mp4", fps=round(1.0 / DT))
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
        gs.logger.info("Keys: arrows move the pin in the horizontal plane, K/J move it up/down, Esc quits.")
        while is_running and scene.viewer.is_alive():
            track_target()
    else:
        log_dough("Lump")
        for i_pass, height in enumerate(PASSES_HEIGHT):
            z_hand = BOARD_TOP + height + PIN_RADIUS + PIN_DEPTH
            # Set the pin down behind the lump, roll it forward over the dough, then back.
            move_to((x_start, BOARD_POS[1], z_hand), speed=0.1)
            move_to((BOARD_POS[0] + PASS_HALF_LENGTH, BOARD_POS[1], z_hand), speed=0.15)
            move_to((x_start, BOARD_POS[1], z_hand), speed=0.15)
            log_dough(f"Pass {i_pass + 1}, pin {100 * height:.1f} cm above the board")
        move_to((x_start, BOARD_POS[1], target_pos[2] + 0.06), speed=0.1)
    if camera is not None:
        camera.stop_recording()


if __name__ == "__main__":
    main()
