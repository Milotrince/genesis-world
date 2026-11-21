"""Import a refrigerator from a USD stage, then sweep each of its joints between its limits with PD control.

Mouse controls (-v): drag the doors and drawers.
"""

import argparse
import os

import numpy as np
from huggingface_hub import snapshot_download

import genesis as gs
import genesis.utils.geom as gu
from genesis.utils.misc import tensor_to_array


class JointAnimator:
    """
    A simple JointAnimator to animate the joints' positions of the scene.

    It uses the sin function to interpolate between the lower and upper limits of the joints.
    """

    def __init__(self, scene: gs.Scene):
        self.rigid_solver = scene.rigid_solver
        self.joint_lower, self.joint_upper = map(tensor_to_array, self.rigid_solver.get_dofs_limit())

        init_positions = tensor_to_array(self.rigid_solver.get_dofs_position())
        normalized_init_pos = np.where(
            (self.joint_upper - self.joint_lower) > gs.EPS,
            2.0 * (init_positions - self.joint_lower) / (self.joint_upper - self.joint_lower) - 1.0,
            0.0,
        )
        self.init_phase = np.arcsin(normalized_init_pos)

        self.rigid_solver.set_dofs_kp(gu.default_dofs_kp(self.rigid_solver.n_dofs))

    def animate(self, scene: gs.Scene):
        t = tensor_to_array(scene.get_time())
        theta = np.pi * t + self.init_phase
        target = (self.joint_upper + self.joint_lower + (self.joint_upper - self.joint_lower) * np.sin(theta)) / 2
        self.rigid_solver.control_dofs_position(target)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--steps",
        type=int,
        default=5000 if "PYTEST_VERSION" not in os.environ else 1,
        help="Number of simulation steps",
    )
    parser.add_argument("-v", "--vis", action="store_true", help="Show visualization GUI")
    parser.add_argument("-r", "--record", action="store_true", help="Record a video of the run to 'out/usd_fridge.mp4'")
    args = parser.parse_args()

    gs.init(backend=gs.cpu)

    camera_pos = (4.0, 2.0, 2.5)
    camera_lookat = (0.0, 0.0, 1.0)

    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=camera_pos,
            camera_lookat=camera_lookat,
            camera_fov=40,
        ),
        profiling_options=gs.options.ProfilingOptions(
            show_FPS=False,
        ),
        show_viewer=args.vis,
    )

    asset_path = snapshot_download(
        repo_type="dataset",
        repo_id="Genesis-Intelligence/assets",
        revision="c50bfe3e354e105b221ef4eb9a79504650709dd2",
        allow_patterns="usd/Refrigerator055/*",
        max_workers=1,
    )

    plane = scene.add_entity(
        gs.morphs.Plane(),
    )
    entities = scene.add_stage(
        morph=gs.morphs.USD(
            file=f"{asset_path}/usd/Refrigerator055/Refrigerator055.usd",
            pos=(0, 0, 0.9),
            euler=(0, 0, 180),
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
    if camera is not None:
        camera.start_recording(save_to_filename="out/usd_fridge.mp4", fps=round(1.0 / scene.dt))

    if args.vis:
        while scene.viewer.is_alive():
            scene.step()
    else:
        joint_animator = JointAnimator(scene)
        for _ in range(args.steps):
            joint_animator.animate(scene)
            scene.step()
    if camera is not None:
        camera.stop_recording()


if __name__ == "__main__":
    main()
