"""A Franka holding a hose nozzle sweeps a jet of water across a table and washes the debris off it.

The water is a material point method (MPM) liquid released by an emitter, which every step takes the position and the
direction of the nozzle on the hand. The jet pushes the light rigid debris across the table top and off its edges.
With '--spray', the nozzle releases a flat fan instead of a round jet. Particle scenes are slow on CPU, so '-g' speeds
the run up a lot.

With the viewer (-v), the scripted sweep is skipped: the nozzle starts aimed at the near edge of the table, the water
running all along, and the keyboard aims it.

Keyboard controls (-v):
Up        - Aim forward (+x)
Down      - Aim backward (-x)
Left      - Aim left (+y)
Right     - Aim right (-y)
Esc       - Quit
"""

import argparse
import os

import numpy as np

import genesis as gs
import genesis.utils.geom as gu
from genesis.utils.misc import tensor_to_array
from genesis.vis.keybindings import Key, KeyAction, Keybind


TABLE_SIZE = (0.4, 0.6, 0.3)
TABLE_POS = (0.6, 0.0)
NOZZLE_LENGTH = 0.08
# Distance along the hand axis from the origin of the hand to the tip of the nozzle it holds between its fingers.
NOZZLE_TIP_DEPTH = 0.1 + NOZZLE_LENGTH
# The jet leans forward, so that it pushes the debris towards the far edge of the table.
JET_DIRECTION = (0.5, 0.0, -1.0)
JET_SPEED = 3.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spray", action="store_true", help="Release a flat fan of water instead of a round jet")
    parser.add_argument("-v", "--vis", action="store_true", help="Show the interactive viewer")
    parser.add_argument("-g", "--gpu", action="store_true", help="Run on GPU instead of CPU")
    parser.add_argument("-r", "--record", action="store_true", help="Record the run to 'out/mpm_hose.mp4'")
    args = parser.parse_args()

    gs.init(backend=gs.gpu if args.gpu else gs.cpu, precision="32", seed=0)

    camera_pos = (1.5, -1.15, 1.1)
    camera_lookat = (0.45, 0.0, 0.35)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=4e-3,
            substeps=10,
        ),
        # The water runs off the table and spreads over the floor, so the domain reaches about a metre past the table on
        # every side, its walls out of frame. A coarser grid than the default keeps it affordable on CPU.
        mpm_options=gs.options.MPMOptions(
            grid_density=40,
            lower_bound=(-0.35, -1.05, 0.0),
            upper_bound=(1.75, 1.05, 0.85),
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=camera_pos,
            camera_lookat=camera_lookat,
            camera_fov=40,
        ),
        show_viewer=args.vis,
    )

    # A rough floor slows the water down as it spreads.
    scene.add_entity(
        gs.morphs.Plane(),
        material=gs.materials.Rigid(
            coup_friction=1.0,
        ),
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
    rng = np.random.default_rng(0)
    debris_entities = []
    for i_debris in range(8):
        size = rng.uniform(0.03, 0.045)
        debris_entities.append(
            scene.add_entity(
                gs.morphs.Box(
                    pos=(
                        TABLE_POS[0] + rng.uniform(-0.12, 0.08),
                        TABLE_POS[1] + rng.uniform(-0.2, 0.2),
                        TABLE_SIZE[2] + 0.5 * size,
                    ),
                    size=(size, size, size),
                    euler=(0.0, 0.0, rng.uniform(0.0, 90.0)),
                ),
                material=gs.materials.Rigid(
                    rho=300.0,
                    friction=0.3,
                ),
                surface=gs.surfaces.Rough(
                    color=(0.35, 0.3 + 0.05 * i_debris, 0.2),
                ),
            )
        )
    franka = scene.add_entity(
        gs.morphs.MJCF(
            file="xml/franka_emika_panda/panda.xml",
        ),
        # The water only meets the debris and the table, so the arm stays out of the coupling.
        material=gs.materials.Rigid(
            needs_coup=False,
            gravity_compensation=1.0,
        ),
    )
    nozzle = scene.add_entity(
        gs.morphs.Cylinder(
            radius=0.012,
            height=NOZZLE_LENGTH,
        ),
        material=gs.materials.Rigid(
            needs_coup=False,
        ),
        surface=gs.surfaces.Rough(
            color=(0.2, 0.5, 0.2),
        ),
    )
    nozzle.attach(franka, "hand", pos=(0.0, 0.0, NOZZLE_TIP_DEPTH - 0.5 * NOZZLE_LENGTH))
    emitter = scene.add_emitter(
        material=gs.materials.MPM.Liquid(),
        max_particles=20000,
        surface=gs.surfaces.Default(
            color=(0.35, 0.6, 0.95, 1.0),
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

    scene.build()

    hand = franka.get_link("hand")
    arm_dofs = range(7)
    step_scale = 1.0 if "PYTEST_VERSION" not in os.environ else 0.05
    franka.set_qpos((0.0, -0.4, 0.0, -2.4, 0.0, 2.0, 0.785, 0.0, 0.0))
    jet_direction = gu.normalize(np.array(JET_DIRECTION))
    nozzle_quat = gu.R_to_quat(gu.z_up_to_R(jet_direction))
    droplet_shape, droplet_size = ("rectangle", (0.02, 0.12)) if args.spray else ("circle", 0.04)

    def aim(target, emit):
        """Track a point the jet should hit on the table top for one step, releasing water if asked."""
        # The tip of the nozzle stays 35 cm back from the point it aims at, along the jet.
        hand_pos = np.array(target) - (0.35 + NOZZLE_TIP_DEPTH) * jet_direction
        qpos = franka.inverse_kinematics(link=hand, pos=hand_pos, quat=nozzle_quat)
        franka.control_dofs_position(qpos[:7], arm_dofs)
        if emit:
            hand_quat = tensor_to_array(hand.get_quat())
            direction = gu.transform_by_quat(np.array((0.0, 0.0, 1.0)), hand_quat)
            pos = tensor_to_array(hand.get_pos()) + NOZZLE_TIP_DEPTH * direction
            emitter.emit(droplet_shape, droplet_size, pos=pos, direction=direction, speed=JET_SPEED)
        scene.step()

    if camera is not None:
        camera.start_recording(save_to_filename="out/mpm_hose.mp4", fps=25)

    table_top = TABLE_SIZE[2]
    x_near = TABLE_POS[0] - 0.5 * TABLE_SIZE[0]
    if args.vis:
        # Keybind callbacks run on the viewer thread, so they only move the target that the main loop tracks.
        jet_target = np.array((x_near, 0.0, table_top))
        table_lower = np.array(TABLE_POS) - 0.5 * np.array(TABLE_SIZE[:2])
        table_upper = np.array(TABLE_POS) + 0.5 * np.array(TABLE_SIZE[:2])
        is_running = True

        def move(delta):
            # The nozzle follows the point it aims at, so keeping that point on the table keeps the water in the domain.
            jet_target[:2] = np.clip(jet_target[:2] + delta, table_lower, table_upper)

        def stop():
            nonlocal is_running
            is_running = False

        step = 0.003
        scene.viewer.register_keybinds(
            Keybind("aim_forward", Key.UP, KeyAction.HOLD, callback=move, args=((step, 0.0),)),
            Keybind("aim_backward", Key.DOWN, KeyAction.HOLD, callback=move, args=((-step, 0.0),)),
            Keybind("aim_left", Key.LEFT, KeyAction.HOLD, callback=move, args=((0.0, step),)),
            Keybind("aim_right", Key.RIGHT, KeyAction.HOLD, callback=move, args=((0.0, -step),)),
            Keybind("quit", Key.ESCAPE, KeyAction.RELEASE, callback=stop),
        )
        gs.logger.info("Keys: Up/Down/Left/Right aim the jet, Esc quits.")
        # The arm starts aimed at the target, so that the water lands on the table from the first step.
        hand_pos = jet_target - (0.35 + NOZZLE_TIP_DEPTH) * jet_direction
        franka.set_qpos(franka.inverse_kinematics(link=hand, pos=hand_pos, quat=nozzle_quat)[:7], arm_dofs)
        while scene.viewer.is_alive() and is_running:
            aim(jet_target, emit=True)
    else:
        # Aim at the near left corner without water, then sweep the table in passes across it, each one further forward.
        y_half = 0.5 * TABLE_SIZE[1] - 0.05
        for _ in range(int(100 * step_scale)):
            aim((x_near, y_half, table_top), emit=False)
        n_pass_steps = int(200 * step_scale)
        for i_pass, x in enumerate((x_near + 0.05, x_near + 0.12, x_near + 0.19, x_near + 0.26)):
            y_start, y_end = (y_half, -y_half) if i_pass % 2 == 0 else (-y_half, y_half)
            for i_step in range(n_pass_steps):
                aim((x, y_start + (y_end - y_start) * i_step / n_pass_steps, table_top), emit=True)
        for _ in range(int(50 * step_scale)):
            scene.step()

        n_washed = sum(entity.get_pos()[2].item() < table_top for entity in debris_entities)
        gs.logger.info(f"{n_washed} of {len(debris_entities)} pieces of debris were washed off the table.")
    if camera is not None:
        camera.stop_recording()


if __name__ == "__main__":
    main()
