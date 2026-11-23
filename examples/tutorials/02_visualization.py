import argparse
import math
import os

import genesis as gs
from genesis.utils.image_exporter import FrameImageExporter

parser = argparse.ArgumentParser()
parser.add_argument("-v", "--vis", action="store_true", help="Show the interactive viewer")
args = parser.parse_args()

gs.init(backend=gs.cpu)

scene = gs.Scene(
    viewer_options=gs.options.ViewerOptions(
        res=(1280, 960),
        camera_pos=(1.5, -2.5, 2.0),
        camera_lookat=(1.2, 0.5, 0.0),
        camera_fov=45,
    ),
    vis_options=gs.options.VisOptions(
        show_world_frame=True,
        plane_reflection=True,
    ),
    renderer=gs.renderers.Rasterizer(),
    show_viewer=args.vis,
)

plane = scene.add_entity(
    gs.morphs.Plane(),
)
franka = scene.add_entity(
    gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
    # cancel the weight of the links, so the idle arm stays upright instead of folding onto the plane
    material=gs.materials.Rigid(
        gravity_compensation=1.0,
    ),
)
# 30 dominoes along a half circle starting in front of the Franka, 10 cm apart, each turned to face the next one
for i in range(30):
    angle = 0.1 * i
    scene.add_entity(
        gs.morphs.Box(
            size=(0.088, 0.016, 0.178),
            pos=(1.5 - math.cos(angle), math.sin(angle), 0.089),
            # the first domino leans towards the second one, so the chain starts on its own
            euler=(-15.0 if i == 0 else 0.0, 0.0, -math.degrees(angle)),
        ),
    )

cam = scene.add_camera(
    res=(640, 480),
    pos=(1.5, -2.5, 2.0),
    lookat=(1.2, 0.5, 0.0),
    fov=45,
    GUI=args.vis,
)

scene.build()

cam.start_recording(save_to_filename="out/02_visualization.mp4", fps=50)
horizon = 350 if "PYTEST_VERSION" not in os.environ else 5
for i in range(horizon):
    # look across the half circle at the falling front of the chain
    angle = min(3.0 * i / 300, 3.0)
    cam.set_pose(
        pos=(1.5 + 0.6 * math.cos(angle), -0.6 * math.sin(angle), 1.0),
        lookat=(1.5 - math.cos(angle), math.sin(angle), 0.0),
    )
    scene.step()
cam.stop_recording()

# the fallen run from above, as color, distance to the camera and the entity seen at each pixel
cam.set_pose(pos=(1.5, -2.5, 2.0), lookat=(1.2, 0.5, 0.0))
rgb, depth, segmentation, _ = cam.render(rgb=True, depth=True, segmentation=True, colorize_seg=True)
FrameImageExporter("out/02_visualization", depth_clip_max=5.0).export_frame_single_camera(
    0, 0, rgb, depth, segmentation
)
