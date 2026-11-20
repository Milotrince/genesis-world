"""A soft-serve machine extrudes ice cream onto a cone that a Franka circles under the nozzle, so the serve coils up.

The serve is an elasto-plastic MPM material poured by an emitter at the nozzle: it yields and flows as it is laid down,
then holds its shape under its own weight. The robot holds the cone attached to its hand and moves it around a small
circle under the nozzle while lowering it, which lays the strand in rings stacked on top of each other, then pulls the
cone out of the machine and presents it. The run logs the height and the width of the serve, read from the particle
positions, after the serve and after the presentation. MPM is slow on CPU, so pass -g to run on GPU.

Keyboard controls (-v), once the scripted serve and presentation are over:
Up/Down    - Move the cone along -x/+x
Left/Right - Move the cone along -y/+y
K/J        - Move the cone up/down
Space      - Hold to extrude
Esc        - Quit
"""

import argparse
import math
import os
import xml.etree.ElementTree as ET

import numpy as np

import trimesh
from huggingface_hub import snapshot_download

import genesis as gs
from genesis.utils.misc import tensor_to_array
from genesis.vis.keybindings import Key, KeyAction, Keybind


DT = 4e-3

# Tip of the nozzle of the dispenser, placed so that the nozzle hangs at this point.
NOZZLE_POS = np.array([0.55, 0.0, 0.25])
NOZZLE_DIAMETER = 0.022
EXTRUSION_SPEED = 0.12
# Cone: radius of its rim, length from the rim down to its tip, and the distance from the rim down to where the fingers
# hold it. The hand holds the cone with its axis this far ahead of the hand.
CONE_RADIUS = 0.028
CONE_LENGTH = 0.11
CONE_GRIP_DEPTH = 0.045
CONE_REACH = 0.1
# Path of the rim center: the radius and the period of the circle under the nozzle, and the gap to the nozzle at start.
CIRCLE_RADIUS = 0.015
CIRCLE_PERIOD = 0.7
RIM_GAP = 0.025
# Rim center of the cone once the robot presents the serve, out of the machine and towards the camera.
PRESENT_POS = np.array([0.4, 0.05, 0.24])


def cone_mjcf():
    """MJCF model of the cone, in the frame of the hand that holds it horizontally: the axis of the cone runs along the
    hand x axis, from the rim on the negative side down to the tip."""
    cone = trimesh.creation.cone(radius=CONE_RADIUS, height=CONE_LENGTH, sections=24)
    # The trimesh cone stands on its base along +z, which becomes the rim along -x. Swapping the two axes mirrors the
    # mesh, so its faces are rewound.
    verts = cone.vertices[:, (2, 1, 0)] + (-CONE_GRIP_DEPTH, 0.0, CONE_REACH)
    mjcf = ET.Element("mujoco", model="cone")
    asset = ET.SubElement(mjcf, "asset")
    ET.SubElement(
        asset,
        "mesh",
        name="cone",
        vertex=" ".join(f"{coord:.6g}" for coord in verts.ravel()),
        face=" ".join(str(idx) for idx in cone.faces[:, ::-1].ravel()),
    )
    body = ET.SubElement(ET.SubElement(mjcf, "worldbody"), "body", name="cone")
    ET.SubElement(ET.SubElement(mjcf, "default"), "geom", density="300", rgba="0.85 0.6 0.3 1")
    ET.SubElement(body, "geom", type="mesh", mesh="cone")
    # A lip of capsules around the rim holds the base of the serve while the cone circles.
    angles = np.linspace(0.0, 2.0 * np.pi, 17)
    lip = (CONE_RADIUS - 0.004) * np.stack((np.sin(angles), np.cos(angles)), axis=-1) + (0.0, CONE_REACH)
    for (y_start, z_start), (y_end, z_end) in zip(lip[:-1], lip[1:]):
        x_lip = -CONE_GRIP_DEPTH - 0.004
        ET.SubElement(
            body, "geom", type="capsule", fromto=f"{x_lip} {y_start} {z_start} {x_lip} {y_end} {z_end}", size="0.004"
        )
    return ET.tostring(mjcf, encoding="unicode")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", help="Show the interactive viewer")
    parser.add_argument("-g", "--gpu", action="store_true", help="Run on GPU instead of CPU")
    parser.add_argument("-r", "--record", action="store_true", help="Record the run to 'out/mpm_ice_cream.mp4'")
    args = parser.parse_args()

    gs.init(backend=gs.gpu if args.gpu else gs.cpu)

    # The camera faces the robot from the open front of the machine, whose housing hides the cone from behind.
    camera_pos = (0.2, 0.36, 0.42)
    camera_lookat = (0.46, -0.06, 0.22)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=DT,
            substeps=10,
        ),
        mpm_options=gs.options.MPMOptions(
            grid_density=192,
            lower_bound=(PRESENT_POS[0] - 0.07, NOZZLE_POS[1] - 0.08, NOZZLE_POS[2] - 0.17),
            upper_bound=(NOZZLE_POS[0] + 0.08, PRESENT_POS[1] + 0.07, PRESENT_POS[2] + 0.12),
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=camera_pos,
            camera_lookat=camera_lookat,
            camera_fov=45,
        ),
        show_viewer=args.vis,
    )

    scene.add_entity(
        gs.morphs.Plane(),
    )
    # The dispenser of genesis/assets, without the ice creams modelled under two of its nozzles.
    asset_path = snapshot_download(
        repo_type="dataset",
        repo_id="Genesis-Intelligence/assets",
        revision="1fc8d7f42e83b7ce0ea0bdb1feecbabf8b173a2d",
        allow_patterns="icecream_dispenser/*",
        max_workers=1,
    )
    # The mesh is y-up and in centimeters. Its right nozzle sits 19.04 cm right of its origin and 6 cm ahead, with its
    # tip 25.2 cm high. Turned to face -x, the machine puts that tip at NOZZLE_POS.
    scene.add_entity(
        morph=gs.morphs.Mesh(
            file=f"{asset_path}/icecream_dispenser/icecream_dispenser.obj",
            scale=0.01,
            pos=(NOZZLE_POS[0] + 0.06, NOZZLE_POS[1] + 0.1904, NOZZLE_POS[2] - 0.252),
            euler=(0.0, 0.0, -90.0),
            fixed=True,
            collision=False,
            file_meshes_are_zup=False,
        ),
    )
    # The robot stands beside the machine, facing +y, and holds the cone in the open front of the machine.
    franka = scene.add_entity(
        morph=gs.morphs.MJCF(
            file="xml/franka_emika_panda/panda.xml",
            pos=(NOZZLE_POS[0] - CONE_REACH, NOZZLE_POS[1] - 0.55, 0.0),
            euler=(0.0, 0.0, 90.0),
        ),
        material=gs.materials.Rigid(
            gravity_compensation=1.0,
        ),
    )
    # The cone has no joint of its own, so it moves with the hand only once its vertices follow the hand.
    cone = scene.add_entity(
        morph=gs.morphs.MJCF(
            file=cone_mjcf(),
            batch_fixed_verts=True,
        ),
        material=gs.materials.Rigid(
            coup_friction=1.0,
        ),
    )
    cone.attach(franka, "hand", pos=(0.0, 0.0, 0.0))
    emitter = scene.add_emitter(
        spec=gs.entities.MPM(
            material=gs.materials.MPM.ElastoPlastic(
                E=2e4,
                nu=0.3,
                rho=600.0,
                von_mises_yield_stress=800.0,
            ),
            surface=gs.surfaces.Rough(
                color=(0.98, 0.95, 0.88),
            ),
            vis=gs.options.EntityVisOptions(
                debug=True,
            ),
        ),
        max_particles=8000,
    )

    camera = None
    if args.record:
        camera = scene.add_camera(
            res=(1280, 720),
            pos=camera_pos,
            lookat=camera_lookat,
            fov=45,
            debug=True,
        )

    scene.build()

    motors_dof = np.arange(7)
    fingers_dof = np.arange(7, 9)
    hand = franka.get_link("hand")
    # The hand points along +x with its fingers closing along y, so that the cone stands upright ahead of it.
    hand_quat = np.array([math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4), 0.0])
    # The rim center sits CONE_GRIP_DEPTH above the hand and CONE_REACH ahead of it. It starts on the circle it follows
    # under the nozzle.
    rim_center = NOZZLE_POS - (0.0, 0.0, RIM_GAP)
    rim_pos = rim_center + (CIRCLE_RADIUS, 0.0, 0.0)
    hand_offset = np.array([CONE_REACH, 0.0, CONE_GRIP_DEPTH])
    franka.set_qpos(franka.inverse_kinematics(link=hand, pos=rim_pos - hand_offset, quat=hand_quat)[:7], motors_dof)
    franka.set_dofs_kp([100.0, 100.0], fingers_dof)
    franka.set_dofs_kv([10.0, 10.0], fingers_dof)
    grip_radius = CONE_RADIUS * (1.0 - CONE_GRIP_DEPTH / CONE_LENGTH)
    franka.control_dofs_position([grip_radius, grip_radius], fingers_dof)

    def track_target():
        q = franka.inverse_kinematics(link=hand, pos=rim_pos - hand_offset, quat=hand_quat)[:7]
        franka.control_dofs_position(q, motors_dof)
        scene.step()

    def extrude():
        emitter.emit(
            droplet_shape="circle",
            droplet_size=NOZZLE_DIAMETER,
            pos=NOZZLE_POS,
            direction=(0.0, 0.0, -1.0),
            speed=EXTRUSION_SPEED,
        )

    def log_serve(label):
        serve_pos = tensor_to_array(emitter.entity.get_particles_pos())
        serve_pos = serve_pos[tensor_to_array(emitter.entity.get_particles_active())]
        height = np.percentile(serve_pos[:, 2], 98) - rim_pos[2]
        width = 2.0 * np.percentile(np.linalg.norm(serve_pos[:, :2] - rim_pos[:2], axis=-1), 98)
        gs.logger.info(f"{label}: serve {100 * height:.1f} cm high and {100 * width:.1f} cm wide on the rim.")

    if "PYTEST_VERSION" in os.environ:
        n_serve, n_settle, n_out, n_up, n_hold = 20, 10, 5, 5, 5
    else:
        n_serve, n_settle, n_out, n_up, n_hold = (round(duration / DT) for duration in (2.0, 1.0, 2.5, 2.5, 1.0))
    # Lower the cone as fast as a column as wide as the rim would grow at the flow rate of the nozzle, so that the nozzle
    # stays just above the top of the serve.
    flow_rate = 0.25 * math.pi * NOZZLE_DIAMETER**2 * EXTRUSION_SPEED
    sink_speed = flow_rate / (math.pi * CONE_RADIUS**2)

    if camera is not None:
        camera.start_recording(save_to_filename="out/mpm_ice_cream.mp4", fps=round(0.5 / DT))
    for i_step in range(n_serve):
        # The circle shrinks as the serve rises, so that the rings taper to a peak.
        angle = 2.0 * math.pi * i_step * DT / CIRCLE_PERIOD
        radius = CIRCLE_RADIUS * (1.0 - 0.8 * i_step / n_serve)
        rim_pos[:] = rim_center + (radius * math.cos(angle), radius * math.sin(angle), -sink_speed * i_step * DT)
        extrude()
        track_target()
    log_serve("Extruded")
    for _ in range(n_settle):
        track_target()
    log_serve("Settled")
    # Pull the cone straight out of the bay, then raise it towards the camera. Each move eases in and out, so the serve
    # stays on the cone.
    for pos_end, n_steps in ((np.array([PRESENT_POS[0], *rim_pos[1:]]), n_out), (PRESENT_POS, n_up)):
        pos_start = rim_pos.copy()
        for i_step in range(1, n_steps + 1):
            ratio = i_step / n_steps
            ratio = ratio * ratio * (3.0 - 2.0 * ratio)
            rim_pos[:] = (1.0 - ratio) * pos_start + ratio * pos_end
            track_target()
    for _ in range(n_hold):
        track_target()
    log_serve("Presented")
    if camera is not None:
        camera.stop_recording()

    if args.vis:
        is_running = True
        is_extruding = False

        def move(dpos):
            rim_pos[:] += dpos

        def toggle_extrusion(is_on):
            nonlocal is_extruding
            is_extruding = is_on

        def stop():
            nonlocal is_running
            is_running = False

        dpos = 0.001
        scene.viewer.register_keybinds(
            Keybind("move_forward", Key.UP, KeyAction.HOLD, callback=move, args=((-dpos, 0, 0),)),
            Keybind("move_back", Key.DOWN, KeyAction.HOLD, callback=move, args=((dpos, 0, 0),)),
            Keybind("move_left", Key.LEFT, KeyAction.HOLD, callback=move, args=((0, -dpos, 0),)),
            Keybind("move_right", Key.RIGHT, KeyAction.HOLD, callback=move, args=((0, dpos, 0),)),
            Keybind("move_up", Key.K, KeyAction.HOLD, callback=move, args=((0, 0, dpos),)),
            Keybind("move_down", Key.J, KeyAction.HOLD, callback=move, args=((0, 0, -dpos),)),
            Keybind("extrude", Key.SPACE, KeyAction.PRESS, callback=toggle_extrusion, args=(True,)),
            Keybind("stop_extrusion", Key.SPACE, KeyAction.RELEASE, callback=toggle_extrusion, args=(False,)),
            Keybind("quit", Key.ESCAPE, KeyAction.RELEASE, callback=stop),
        )
        gs.logger.info(
            "Keys: arrows move the cone in the horizontal plane, K/J move it up/down, hold Space to extrude, Esc quits."
        )
        while is_running and scene.viewer.is_alive() and "PYTEST_VERSION" not in os.environ:
            if is_extruding:
                extrude()
            track_target()


if __name__ == "__main__":
    main()
