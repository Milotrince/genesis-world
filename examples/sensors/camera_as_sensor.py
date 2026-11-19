"""
Example demonstrating camera sensors with different rendering backends.

Creating cameras as sensors using add_sensor() with three backends
Rasterizer, Raytracer and BatchRenderer.
Test the attachment, add light, batch rendering functionalities.

A backend whose optional dependency is missing is skipped: the Raytracer needs LuisaRenderPy, and the BatchRenderer
needs gs_madrona and a CUDA GPU (-g). With the viewer (-v), drag the sphere or the box with the mouse.
"""

import argparse
import importlib.util
import os
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np

import genesis as gs
from genesis.options.sensors import BatchRendererCameraOptions, RasterizerCameraOptions, RaytracerCameraOptions
from genesis.utils.misc import tensor_to_array


class CameraView(NamedTuple):
    """A camera viewpoint, rendered by a camera sensor of every enabled backend."""

    name: str
    """Name of the view, which suffixes the name of each camera sensor and its PNGs."""
    pos: tuple[float, float, float]
    """Position of the camera, in the frame of the sphere when attached to it, in the world frame otherwise."""
    lookat: tuple[float, float, float]
    """Point the camera looks at."""
    fov: float
    """Vertical field of view, in degrees."""
    is_attached: bool = False
    """Whether the camera rides the sphere."""
    offset_T: np.ndarray | None = None
    """Pose of the camera relative to the sphere, which takes priority over the position offset."""
    has_light: bool = False
    """Whether the camera adds a point light above the scene."""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", help="Show the interactive viewer")
    parser.add_argument("-g", "--gpu", action="store_true", help="Run on GPU instead of CPU")
    parser.add_argument("-s", "--steps", type=int, default=100, help="Number of simulation steps")
    parser.add_argument(
        "-o", "--output-dir", default="out/camera_as_sensor", help="Directory for the PNGs the camera sensors render"
    )
    parser.add_argument(
        "-r", "--record", action="store_true", help="Record a video of the run to 'out/camera_as_sensor.mp4'"
    )
    args = parser.parse_args()
    if "PYTEST_VERSION" in os.environ:
        args.steps = 11

    gs.init(seed=0, precision="32", backend=gs.gpu if args.gpu else gs.cpu, logging_level="info")

    enable_raytracer = importlib.util.find_spec("LuisaRenderPy") is not None
    if not enable_raytracer:
        gs.logger.info("LuisaRenderPy not available, skipping the Raytracer cameras.")
    enable_batch_renderer = importlib.util.find_spec("gs_madrona") is not None and gs.backend == gs.cuda
    if not enable_batch_renderer:
        gs.logger.info("gs_madrona or a CUDA backend not available, skipping the BatchRenderer cameras.")

    env_surface = gs.surfaces.Emission(
        emissive_texture=gs.textures.ColorTexture(
            color=(0.2, 0.3, 0.5),
        ),
    )
    if enable_raytracer:
        renderer = gs.renderers.RayTracer(
            env_surface=env_surface,
            env_radius=20.0,
        )
    else:
        renderer = gs.renderers.Rasterizer()

    camera_pos = (3.0, -3.0, 2.5)
    camera_lookat = (0.0, 0.0, 1.0)

    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=camera_pos,
            camera_lookat=camera_lookat,
        ),
        renderer=renderer,
        show_viewer=args.vis,
    )

    plane = scene.add_entity(
        morph=gs.morphs.Plane(),
        surface=gs.surfaces.Rough(
            color=(0.4, 0.4, 0.4),
        ),
    )

    sphere = scene.add_entity(
        morph=gs.morphs.Sphere(
            radius=0.5,
            pos=(0.0, 0.0, 2.0),
        ),
        surface=gs.surfaces.Smooth(
            color=(1.0, 0.5, 0.5),
        ),
    )

    box = scene.add_entity(
        morph=gs.morphs.Box(
            size=(0.3, 0.3, 0.3),
            pos=(1.0, 1.0, 1.0),
        ),
        surface=gs.surfaces.Rough(
            color=(0.5, 1.0, 0.5),
        ),
    )

    views = (
        CameraView("cam0", pos=(3.0, 0.0, 2.0), lookat=(0.0, 0.0, 1.0), fov=60.0, has_light=True),
        CameraView("cam1", pos=(0.0, 3.0, 2.0), lookat=(0.0, 0.0, 1.0), fov=60.0),
        CameraView("cam_attached", pos=(0.0, 0.0, 1.0), lookat=(0.0, 0.0, 0.0), fov=70.0, is_attached=True),
        CameraView(
            "cam_offset_T", pos=(0.0, 0.0, 1.0), lookat=(0.0, 0.0, 0.0), fov=70.0, is_attached=True, offset_T=np.eye(4)
        ),
    )
    res = (500, 600)
    light_pos = (2.0, 2.0, 5.0)

    cameras = {}
    for view in views:
        cameras[f"rasterizer_{view.name}"] = scene.add_sensor(
            RasterizerCameraOptions(
                res=res,
                pos=view.pos,
                lookat=view.lookat,
                fov=view.fov,
                lights=[{"pos": light_pos, "color": (1.0, 1.0, 1.0), "intensity": 1.0}] if view.has_light else [],
                offset_T=view.offset_T,
                entity_idx=sphere.idx if view.is_attached else -1,
                near=0.1,
            )
        )
    if enable_raytracer:
        for view in views:
            cameras[f"raytracer_{view.name}"] = scene.add_sensor(
                RaytracerCameraOptions(
                    res=res,
                    pos=view.pos,
                    lookat=view.lookat,
                    fov=view.fov,
                    lights=[{"pos": light_pos, "color": (10.0, 10.0, 10.0), "intensity": 1.0}]
                    if view.has_light
                    else [],
                    offset_T=view.offset_T,
                    entity_idx=sphere.idx if view.is_attached else -1,
                    spp=64,
                    env_surface=None if view.is_attached else env_surface,
                    env_radius=20.0,
                )
            )
    if enable_batch_renderer:
        for view in views:
            cameras[f"batch_render_{view.name}"] = scene.add_sensor(
                BatchRendererCameraOptions(
                    res=res,
                    pos=view.pos,
                    lookat=view.lookat,
                    fov=view.fov,
                    lights=(
                        [{"pos": light_pos, "color": (1.0, 1.0, 1.0), "intensity": 1.0, "directional": False}]
                        if view.has_light
                        else []
                    ),
                    offset_T=view.offset_T,
                    entity_idx=sphere.idx if view.is_attached else -1,
                )
            )
    gs.logger.info(f"Camera sensors: {', '.join(cameras)}")

    video_camera = None
    if args.record:
        video_camera = scene.add_camera(
            res=(1280, 720),
            pos=camera_pos,
            lookat=camera_lookat,
        )

    if args.vis:
        scene.viewer.add_plugin(
            gs.vis.viewer_plugins.MouseInteractionPlugin(),
        )

    scene.build(n_envs=1)

    os.makedirs(args.output_dir, exist_ok=True)
    if video_camera is not None:
        video_camera.start_recording(save_to_filename="out/camera_as_sensor.mp4", fps=round(1.0 / scene.dt))

    for i in range(args.steps):
        scene.step()
        if i % 10 == 0:
            for cam_name, camera in cameras.items():
                rgb = camera.read().rgb[0]
                plt.imsave(os.path.join(args.output_dir, f"{cam_name}_step{i:03d}.png"), tensor_to_array(rgb))
            gs.logger.info(f"Step {i}: saved {len(cameras)} images to '{args.output_dir}'")
    if video_camera is not None:
        video_camera.stop_recording()

    while args.vis and scene.viewer.is_alive() and "PYTEST_VERSION" not in os.environ:
        scene.step()


if __name__ == "__main__":
    main()
