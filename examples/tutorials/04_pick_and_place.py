import argparse
import os

import genesis as gs

parser = argparse.ArgumentParser()
parser.add_argument("-v", "--vis", action="store_true", help="Show the interactive viewer")
args = parser.parse_args()

gs.init(backend=gs.cpu)

scene = gs.Scene(
    sim_options=gs.options.SimOptions(
        dt=0.01,
    ),
    viewer_options=gs.options.ViewerOptions(
        camera_pos=(1.6, -1.2, 0.9),
        camera_lookat=(0.45, 0.0, 0.1),
        camera_fov=40,
    ),
    show_viewer=args.vis,
)

plane = scene.add_entity(
    gs.morphs.Plane(),
)
# five dominoes stand in a supply row, their thin side along y, where the fingers close
n_dominoes = 5 if "PYTEST_VERSION" not in os.environ else 1
dominoes = [
    scene.add_entity(
        gs.morphs.Box(
            size=(0.088, 0.016, 0.178),
            pos=(0.2 + 0.13 * i, -0.3, 0.089),
        ),
    )
    for i in range(n_dominoes)
]
franka_spec = gs.entities.Rigid(
    morph=gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
)
# select the fingertip pads (geoms 1 to 5 of each finger link, geom 0 being the finger mesh) to raise their friction
fingers = franka_spec.select(gs.indices.Link.named("left_finger", "right_finger"))
fingers.select(gs.indices.Geom.at(slice(1, 6))).set_friction(1.5)
franka = scene.add_entity(
    spec=franka_spec,
)
scene.build()

hand = franka.get_link("hand")
fingers_dofs = range(7, 9)
# the MJCF drives the fingers through a tendon actuator, so position control needs PD gains of its own
franka.set_dofs_kp((100, 100), fingers_dofs)
franka.set_dofs_kv((10, 10), fingers_dofs)
step_scale = 1.0 if "PYTEST_VERSION" not in os.environ else 0.05


def move_hand(pos, steps, plan=False):
    """Move the hand, pointing down, to a position along a collision-free planned path or a straight line."""
    steps = int(steps * step_scale)
    if plan:
        path = franka.plan_path(franka.inverse_kinematics(link=hand, pos=pos, quat=(0, 1, 0, 0)), num_waypoints=steps)
    else:
        # one inverse kinematics solve per step keeps the hand on the line, where the joints would arc between its ends
        start = hand.get_pos()
        points = [start.lerp(start.new_tensor(pos), k / steps) for k in range(1, steps + 1)]
        path = [franka.inverse_kinematics(link=hand, pos=point, quat=(0, 1, 0, 0)) for point in points]
    for waypoint in path:
        franka.control_dofs_position(waypoint[:7], range(7))
        scene.step()
    # the PD controller lags behind the path, so give the hand time to reach its target
    for _ in range(int(40 * step_scale)):
        scene.step()


# grasp each domino near its top, then stand it on the run line, 10 cm after the previous one
for i, domino in enumerate(dominoes):
    x, y, _ = domino.get_pos().tolist()
    franka.control_dofs_position((0.04, 0.04), fingers_dofs)
    move_hand((x, y, 0.5), steps=200, plan=True)
    move_hand((x, y, 0.25), steps=80)
    franka.control_dofs_force((-5.0, -5.0), fingers_dofs)
    move_hand((x, y, 0.5), steps=80)
    move_hand((0.5, 0.1 * i, 0.5), steps=200, plan=True)
    move_hand((0.5, 0.1 * i, 0.245), steps=80)
    franka.control_dofs_position((0.04, 0.04), fingers_dofs)
    move_hand((0.5, 0.1 * i, 0.5), steps=80)

print("dominoes standing:", sum(domino.get_pos()[2] > 0.08 for domino in dominoes).item(), "of", n_dominoes)

# close the gripper and push the first domino over along the run
franka.control_dofs_position((0.0, 0.0), fingers_dofs)
move_hand((0.5, -0.06, 0.25), steps=200, plan=True)
move_hand((0.5, 0.02, 0.25), steps=60)
move_hand((0.5, -0.06, 0.5), steps=60)
print("dominoes fallen:", sum(domino.get_pos()[2] < 0.06 for domino in dominoes).item(), "of", n_dominoes)
