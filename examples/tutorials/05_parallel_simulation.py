import argparse
import os

import torch

import genesis as gs

parser = argparse.ArgumentParser()
parser.add_argument("-v", "--vis", action="store_true", help="Show the interactive viewer")
args = parser.parse_args()

gs.init(backend=gs.cpu)

n_envs = 512 if "PYTEST_VERSION" not in os.environ else 8
# the viewer draws 8 runs of the 512, their spacing growing from one to the next
rendered_envs_idx = list(range(0, n_envs, n_envs // 8))
scene = gs.Scene(
    sim_options=gs.options.SimOptions(
        dt=0.01,
    ),
    vis_options=gs.options.VisOptions(
        rendered_envs_idx=rendered_envs_idx,
    ),
    viewer_options=gs.options.ViewerOptions(
        camera_pos=(3.0, -1.2, 2.0),
        camera_lookat=(0.5, 1.3, 0.0),
        camera_fov=45,
    ),
    show_viewer=args.vis,
)

plane = scene.add_entity(
    gs.morphs.Plane(),
)
dominoes = [
    scene.add_entity(
        gs.morphs.Box(
            size=(0.088, 0.016, 0.178),
            pos=(0.0, 0.0, 0.089),
        ),
    )
    for _ in range(15)
]

# the rendered runs side by side, 25 cm apart, the other environments sharing their place on screen
scene.build(n_envs=n_envs, env_spacing=(0.25, 0.0), n_envs_per_row=n_envs // 8)

# one spacing per environment, every position setter taking one value per environment along the first dimension
spacing = torch.linspace(0.02, 0.2, n_envs, device=gs.device)
for i, domino in enumerate(dominoes):
    domino.set_pos(torch.stack((torch.zeros_like(spacing), i * spacing, torch.full_like(spacing, 0.089)), dim=-1))
# tip the first domino forward in every environment, at 3 rad/s about the x axis
dominoes[0].set_dofs_velocity(-3.0, dofs_idx_local=3)

for i in range(400):
    scene.step()

# the heights of all the dominoes in all the environments in one read, a domino lying flat once below 6 cm
links_idx = [domino.base_link_idx for domino in dominoes]
heights = scene.rigid_solver.get_links_pos(links_idx=links_idx)[..., 2]
is_run_fallen = (heights < 0.06).all(dim=1)
spacing_min, spacing_max = spacing[is_run_fallen].aminmax()
print(
    f"heights: {tuple(heights.shape)}, the whole run falls for a spacing from {spacing_min:.3f} to {spacing_max:.3f} m"
)

# the same read, restricted to the rendered environments
heights_rendered = scene.rigid_solver.get_links_pos(links_idx=links_idx, envs_idx=rendered_envs_idx)[..., 2]
print("fallen dominoes in the rendered runs:", (heights_rendered < 0.06).sum(dim=1).tolist())
