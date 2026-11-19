import argparse
import os

import torch

import genesis as gs

parser = argparse.ArgumentParser()
parser.add_argument("-v", "--vis", action="store_true", help="Show the interactive viewer")
args = parser.parse_args()

gs.init(backend=gs.cpu)

scene = gs.Scene(
    sim_options=gs.options.SimOptions(
        dt=0.01,
    ),
    vis_options=gs.options.VisOptions(
        # render the first 5 environments only, the cameras rendering each of them on its own
        rendered_envs_idx=list(range(5)),
        split_envs=True,
    ),
    viewer_options=gs.options.ViewerOptions(
        camera_pos=(3.5, -1.0, 2.5),
        camera_lookat=(0.0, 0.0, 0.5),
        camera_fov=40,
    ),
    show_viewer=args.vis,
)

plane = scene.add_entity(
    gs.morphs.Plane(),
)

franka = scene.add_entity(
    gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
)

cam = scene.add_camera(
    res=(320, 240),
    pos=(3.5, 0.0, 2.5),
    lookat=(0.0, 0.0, 0.5),
    fov=30,
)

# create 20 parallel environments
B = 20
scene.build(n_envs=B, env_spacing=(1.0, 1.0))

# control all the robots
franka.control_dofs_position(
    torch.tile(torch.tensor([0, 0, 0, -1.0, 0, 1.0, 0, 0.02, 0.02], device=gs.device), (B, 1)),
)

# 'envs_idx' narrows a command to a subset of the environments, leaving the rest on their previous target
franka.control_dofs_position(
    torch.zeros(3, 9, device=gs.device),
    envs_idx=torch.tensor([1, 5, 7], device=gs.device),
)

horizon = 1000 if "PYTEST_VERSION" not in os.environ else 5
for i in range(horizon):
    scene.step()

# one image per rendered environment, stacked along the first dimension
rgb, depth, segmentation, normal = cam.render(rgb=True, depth=True, segmentation=True, colorize_seg=True, normal=True)
print("rgb:", rgb.shape, "depth:", depth.shape, "segmentation:", segmentation.shape, "normal:", normal.shape)
