"""A chain reaction of dominos, a ball, a seesaw, a piston and a pendulum ends by ringing a bell.

A row of dominos topples on a raised platform and pushes a heavy ball down a ramp. The ball drops onto the raised end
of a seesaw (a revolute joint), whose far end swings up and shoves a piston up its inclined rail (a prismatic joint).
The piston strikes the bob of a pendulum hanging from a ball joint (a spherical joint), which swings into a bell. The
jointed parts are one MJCF model built inline. Hibernation is on, so every part at rest, ahead of the reaction or
behind it, sleeps and costs nothing to simulate until something hits it.

With the viewer (-v), the machine is set back up once the scripted run ends. Drag the first domino towards the others
with the mouse to run the chain again, or poke any other part.
"""

import argparse
import math
import os
import xml.etree.ElementTree as ET

import genesis as gs


DT = 5e-3

# Dominos: thickness, width and height, the spacing between them and their number.
DOMINO_SIZE = (0.03, 0.12, 0.24)
DOMINO_PITCH = 0.12
N_DOMINOS = 6
BALL_RADIUS = 0.05

# Platform and ramp, their top surfaces being the path of the ball. The ramp starts a step below the platform edge, so
# that the ball pushed over the edge drops clear of the last domino.
PLATFORM_TOP = 0.6
PLATFORM_EDGE_X = -0.5
RAMP_START = (-0.49, 0.5)
RAMP_END = (0.02, 0.38)

# Seesaw: pivot and plank half-size. At rest its far end lies on the floor, and its near end is up.
SEESAW_PIVOT = (0.45, 0.15)
SEESAW_HALF_LENGTH = 0.4
SEESAW_HALF_THICKNESS = 0.01
SEESAW_TILT_DEG = math.degrees(math.asin((SEESAW_PIVOT[1] - SEESAW_HALF_THICKNESS) / SEESAW_HALF_LENGTH))

# Piston: rail inclination from the horizontal, rod length, and the travel along its rail.
PISTON_ANGLE_DEG = 45.0
PISTON_LENGTH = 0.3
PISTON_TRAVEL = 0.2

# Pendulum and bell.
PENDULUM_LENGTH = 0.5
BOB_RADIUS = 0.06
BELL_RADIUS = 0.06
BELL_HEIGHT = 0.12


def machine_mjcf():
    """MJCF model of the jointed parts (the seesaw, the piston, the pendulum and the bell) and of their stands."""
    mjcf = ET.Element("mujoco", model="rube_goldberg")
    worldbody = ET.SubElement(mjcf, "worldbody")

    # The seesaw rotates about +y. A positive angle about +y lowers the +x end, so the near (-x) end starts up.
    pivot_x, pivot_z = SEESAW_PIVOT
    ET.SubElement(
        worldbody,
        "geom",
        type="box",
        size=f"0.03 0.08 {0.5 * pivot_z - 0.03}",
        pos=f"{pivot_x} 0 {0.5 * pivot_z - 0.03}",
        rgba="0.35 0.35 0.38 1",
    )
    seesaw = ET.SubElement(
        worldbody, "body", name="seesaw", pos=f"{pivot_x} 0 {pivot_z}", euler=f"0 {SEESAW_TILT_DEG} 0"
    )
    ET.SubElement(seesaw, "joint", name="seesaw_hinge", type="hinge", axis="0 1 0", damping="0.01")
    ET.SubElement(
        seesaw,
        "geom",
        type="box",
        size=f"{SEESAW_HALF_LENGTH} 0.07 {SEESAW_HALF_THICKNESS}",
        density="600",
        rgba="0.75 0.55 0.3 1",
    )
    # A lip near the near end catches the ball, and a counterweight on the far half keeps that end down at rest.
    ET.SubElement(
        seesaw,
        "geom",
        type="box",
        size="0.01 0.07 0.03",
        pos=f"{-SEESAW_HALF_LENGTH + 0.12} 0 {SEESAW_HALF_THICKNESS + 0.03}",
        density="600",
        rgba="0.75 0.55 0.3 1",
    )
    ET.SubElement(
        seesaw,
        "geom",
        type="box",
        size="0.04 0.06 0.02",
        pos=f"{0.5 * SEESAW_HALF_LENGTH} 0 {SEESAW_HALF_THICKNESS + 0.02}",
        density="1000",
        rgba="0.3 0.3 0.32 1",
    )

    # The piston slides along its rail, tilted towards +x, from where its lower end sits just above the far end of the
    # seesaw at rest.
    angle = math.radians(PISTON_ANGLE_DEG)
    axis = (math.cos(angle), 0.0, math.sin(angle))
    tip_x = pivot_x + SEESAW_HALF_LENGTH * math.cos(math.radians(SEESAW_TILT_DEG))
    tip_z = pivot_z - SEESAW_HALF_LENGTH * math.sin(math.radians(SEESAW_TILT_DEG))
    piston_base = (tip_x - 0.03, tip_z + 0.06)
    piston_center = [piston_base[0] + 0.5 * PISTON_LENGTH * axis[0], piston_base[1] + 0.5 * PISTON_LENGTH * axis[2]]
    piston = ET.SubElement(
        worldbody,
        "body",
        name="piston",
        pos=f"{piston_center[0]} 0 {piston_center[1]}",
        euler=f"0 {90.0 - PISTON_ANGLE_DEG} 0",
    )
    ET.SubElement(
        piston, "joint", name="piston_slide", type="slide", axis="0 0 1", range=f"0 {PISTON_TRAVEL}", damping="0.05"
    )
    ET.SubElement(
        piston,
        "geom",
        type="cylinder",
        size=f"0.025 {0.5 * PISTON_LENGTH}",
        density="400",
        rgba="0.7 0.72 0.75 1",
    )

    # The bob hangs on the rail axis, a few centimetres ahead of the top of the piston.
    reach = 0.5 * PISTON_LENGTH + 0.06 + BOB_RADIUS
    bob = (piston_center[0] + reach * axis[0], piston_center[1] + reach * axis[2])
    pendulum = ET.SubElement(worldbody, "body", name="pendulum", pos=f"{bob[0]} 0 {bob[1] + PENDULUM_LENGTH}")
    ET.SubElement(pendulum, "joint", name="pendulum_ball", type="ball", damping="0.002")
    ET.SubElement(
        pendulum,
        "geom",
        type="capsule",
        fromto=f"0 0 0 0 0 {-PENDULUM_LENGTH + BOB_RADIUS}",
        size="0.008",
        density="1000",
        rgba="0.3 0.3 0.32 1",
    )
    ET.SubElement(
        pendulum,
        "geom",
        type="sphere",
        pos=f"0 0 {-PENDULUM_LENGTH}",
        size=f"{BOB_RADIUS}",
        density="400",
        rgba="0.8 0.2 0.15 1",
    )

    # The bell hangs from a hinge at the height of the pivot of the pendulum, half a bob radius clear of the bob at rest.
    pivot_z_top = bob[1] + PENDULUM_LENGTH
    bell_x = bob[0] + 1.5 * BOB_RADIUS + BELL_RADIUS
    bell_rod_length = PENDULUM_LENGTH - 0.5 * BELL_HEIGHT
    bell = ET.SubElement(worldbody, "body", name="bell", pos=f"{bell_x} 0 {pivot_z_top}")
    ET.SubElement(bell, "joint", name="bell_hinge", type="hinge", axis="0 1 0", damping="0.002")
    ET.SubElement(
        bell,
        "geom",
        type="capsule",
        fromto=f"0 0 0 0 0 {-bell_rod_length}",
        size="0.006",
        density="1000",
        rgba="0.3 0.3 0.32 1",
    )
    ET.SubElement(
        bell,
        "geom",
        type="cylinder",
        pos=f"0 0 {-bell_rod_length - 0.5 * BELL_HEIGHT}",
        size=f"{BELL_RADIUS} {0.5 * BELL_HEIGHT}",
        density="300",
        rgba="0.85 0.7 0.2 1",
    )

    # The gantry both hang from: a beam just above the two pivots, held by an arm from a post behind them.
    beam_z = pivot_z_top + 0.03
    for size, pos in (
        (f"{0.5 * (bell_x - bob[0]) + 0.05} 0.015 0.015", f"{0.5 * (bob[0] + bell_x)} 0 {beam_z}"),
        ("0.015 0.1 0.015", f"{0.5 * (bob[0] + bell_x)} 0.1 {beam_z}"),
        (f"0.015 0.015 {0.5 * beam_z}", f"{0.5 * (bob[0] + bell_x)} 0.2 {0.5 * beam_z}"),
    ):
        ET.SubElement(worldbody, "geom", type="box", size=size, pos=pos, rgba="0.35 0.35 0.38 1")
    return mjcf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--seconds", type=float, default=6.0, help="Simulated duration of the scripted run")
    parser.add_argument("-v", "--vis", action="store_true", help="Show the interactive viewer")
    parser.add_argument("-g", "--gpu", action="store_true", help="Run on GPU instead of CPU")
    parser.add_argument("-r", "--record", action="store_true", help="Record the run to 'out/rube_goldberg.mp4'")
    args = parser.parse_args()

    gs.init(backend=gs.gpu if args.gpu else gs.cpu)

    camera_pos = (0.0, -2.6, 1.2)
    camera_lookat = (0.0, 0.0, 0.5)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=DT,
        ),
        rigid_options=gs.options.RigidOptions(
            use_hibernation=True,
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
    platform_length = (N_DOMINOS + 1) * DOMINO_PITCH + 0.2
    scene.add_entity(
        gs.morphs.Box(
            pos=(PLATFORM_EDGE_X - 0.5 * platform_length, 0.0, 0.5 * PLATFORM_TOP),
            size=(platform_length, 0.3, PLATFORM_TOP),
            fixed=True,
        ),
        surface=gs.surfaces.Default(
            color=(0.45, 0.45, 0.5),
        ),
    )
    # The ramp is a thin slab whose top surface runs down to just above the near end of the seesaw.
    ramp_dx = RAMP_END[0] - RAMP_START[0]
    ramp_dz = RAMP_END[1] - RAMP_START[1]
    ramp_thickness = 0.04
    ramp_angle = math.atan2(-ramp_dz, ramp_dx)
    scene.add_entity(
        gs.morphs.Box(
            pos=(
                RAMP_START[0] + 0.5 * ramp_dx - 0.5 * ramp_thickness * math.sin(ramp_angle),
                0.0,
                RAMP_START[1] + 0.5 * ramp_dz - 0.5 * ramp_thickness * math.cos(ramp_angle),
            ),
            size=(math.hypot(ramp_dx, ramp_dz), 0.16, ramp_thickness),
            euler=(0.0, math.degrees(ramp_angle), 0.0),
            fixed=True,
        ),
        surface=gs.surfaces.Default(
            color=(0.45, 0.45, 0.5),
        ),
    )
    dominos = []
    for i_domino in range(N_DOMINOS):
        dominos.append(
            scene.add_entity(
                gs.morphs.Box(
                    pos=(
                        PLATFORM_EDGE_X - BALL_RADIUS - 0.1 - (N_DOMINOS - 1 - i_domino) * DOMINO_PITCH,
                        0.0,
                        PLATFORM_TOP + 0.5 * DOMINO_SIZE[2],
                    ),
                    size=DOMINO_SIZE,
                ),
                surface=gs.surfaces.Default(
                    color=(0.9, 0.9, 0.88),
                ),
            )
        )
    # The ball rests at the very edge of the platform, so the leaning last domino rolls it over.
    scene.add_entity(
        gs.morphs.Sphere(
            pos=(PLATFORM_EDGE_X - 0.005, 0.0, PLATFORM_TOP + BALL_RADIUS),
            radius=BALL_RADIUS,
        ),
        material=gs.materials.Rigid(
            rho=6000.0,
        ),
        surface=gs.surfaces.Default(
            color=(0.3, 0.3, 0.35),
        ),
    )
    machine = scene.add_entity(
        gs.morphs.MJCF(
            file=ET.tostring(machine_mjcf(), encoding="unicode"),
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

    bell_dof = machine.get_joint("bell_hinge").dofs_idx_local

    if camera is not None:
        camera.start_recording(save_to_filename="out/rube_goldberg.mp4", fps=40)
    # A flick of the first domino towards the others, about its bottom edge, as a push with the mouse gives it.
    dominos[0].set_dofs_velocity((0.3, 0.0, 0.0, 0.0, 2.5, 0.0))
    n_steps = round(args.seconds / DT) if "PYTEST_VERSION" not in os.environ else 50
    bell_swing_max = 0.0
    for _ in range(n_steps):
        scene.step()
        bell_swing_max = max(bell_swing_max, abs(float(machine.get_dofs_position(bell_dof)[0])))
    if camera is not None:
        camera.stop_recording()
    gs.logger.info(f"The bell swung by up to {math.degrees(bell_swing_max):.1f} degrees.")

    # The machine is set back up, ready for the user to push the first domino again.
    if args.vis:
        scene.reset()
        gs.logger.info("Drag the first domino towards the others to run the chain again.")
    while args.vis and scene.viewer.is_alive() and "PYTEST_VERSION" not in os.environ:
        scene.step()


if __name__ == "__main__":
    main()
