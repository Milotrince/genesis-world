import argparse
import math
import os

import genesis as gs

parser = argparse.ArgumentParser()
parser.add_argument("-v", "--vis", action="store_true", help="Show the interactive viewer")
args = parser.parse_args()

gs.init(backend=gs.cpu)

dt = 0.01
scene = gs.Scene(
    sim_options=gs.options.SimOptions(
        dt=dt,
    ),
    viewer_options=gs.options.ViewerOptions(
        camera_pos=(0.8, -2.0, 1.4),
        camera_lookat=(0.9, 0.4, 0.1),
        camera_fov=50,
    ),
    show_viewer=args.vis,
)

plane = scene.add_entity(
    gs.morphs.Plane(),
)
franka = scene.add_entity(
    gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
    # cancel the weight of the links, so that a joint under force control only moves under the commanded torque
    material=gs.materials.Rigid(
        gravity_compensation=1.0,
    ),
)
dominoes = [
    scene.add_entity(
        gs.morphs.Box(
            size=(0.088, 0.016, 0.178),
            pos=(1.5 - math.cos(0.1 * i), math.sin(0.1 * i), 0.089),
            euler=(0.0, 0.0, -math.degrees(0.1 * i)),
        ),
    )
    for i in range(30)
]
scene.build()

arm_dofs = [franka.get_joint(f"joint{i}").dofs_idx_local[0] for i in range(1, 8)]
franka.set_dofs_kp((4500, 4500, 3500, 3500, 2000, 2000, 2000), arm_dofs)
franka.set_dofs_kv((450, 450, 350, 350, 200, 200, 200), arm_dofs)

ready = (0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785)
# the hand above the first domino, then beside its upper half, 4 cm behind it
hover = (-0.01, -0.1, -0.09, -2.27, -0.01, 2.17, 0.69)
beside = (-0.01, 0.19, -0.08, -2.39, 0.03, 2.58, 0.66)
n_steps = 100 if "PYTEST_VERSION" not in os.environ else 5

for torque, track_velocity in ((3.0, False), (20.0, True)):
    # setting the position teleports the arm, where controlling it only sets the target its motors drive towards
    franka.set_dofs_position(ready, arm_dofs)
    franka.control_dofs_position(ready, arm_dofs)
    tracking_error = 0.0
    for target in (hover, beside):
        start = franka.get_dofs_position(arm_dofs)
        delta = start.new_tensor(target) - start
        for k in range(1, n_steps + 1):
            # a smooth joint trajectory, whose velocity a PD target can follow to cut the tracking lag
            phase = math.pi * k / n_steps
            position = start + delta * (1 - math.cos(phase)) / 2
            velocity = delta * math.sin(phase) * math.pi / (2 * n_steps * dt)
            if track_velocity:
                franka.control_dofs_position_velocity(position, velocity, arm_dofs)
            else:
                franka.control_dofs_position(position, arm_dofs)
            scene.step()
            tracking_error = max(tracking_error, (franka.get_dofs_position(arm_dofs) - position).abs().max().item())
        for _ in range(n_steps // 3):
            scene.step()
    print(f"tracking error with{'' if track_velocity else 'out'} target velocity: {tracking_error:.3f} rad")

    # a torque pulse on the base joint swings the hand into the domino, the other joints holding their position
    franka.control_dofs_force(torque, arm_dofs[0])
    for k in range(3 * n_steps):
        if k == 10:
            franka.control_dofs_position(beside[0], arm_dofs[0])
        scene.step()
    print(f"push of {torque} N.m: {sum(domino.get_pos()[2] < 0.06 for domino in dominoes).item()} dominoes fallen")
