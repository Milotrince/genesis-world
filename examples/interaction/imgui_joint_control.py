"""Interactive joint control example using ImGui overlay.

Demonstrates:
- Simulation controls (play/pause/step/reset)
- Entity browser with joint sliders
- Visualization toggles
- Camera controls
- Custom user panels via register_panel()
- Scene rebuild (add entities, change scale)

A scripted sweep first drives every joint of the Panda through its range, then the overlay takes over until the viewer
closes.
"""

import argparse
import math
import os

import torch

import genesis as gs
from genesis.ext.pyrender.overlay import ImGuiOverlayPlugin


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r", "--record", action="store_true", help="Record a video of the run to 'out/imgui_joint_control.mp4'"
    )
    args = parser.parse_args()

    gs.init(backend=gs.cpu)

    camera_pos = (2.0, 2.0, 1.5)
    camera_lookat = (0.0, 0.0, 0.5)

    # enable_gui attaches the ImGui overlay and lets it manage scene editing internally, so no manual
    # InteractiveScene is needed: a plain Scene is the whole setup, and Rebuild Scene is handled inside step().
    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=camera_pos,
            camera_lookat=camera_lookat,
            enable_gui=True,
        ),
        show_viewer=True,
    )
    scene.add_entity(
        morph=gs.morphs.Plane(),
        name="Plane",
    )
    panda = scene.add_entity(
        morph=gs.morphs.MJCF(
            file="xml/franka_emika_panda/panda.xml",
        ),
        name="Panda",
    )
    scene.add_entity(
        morph=gs.morphs.Box(
            pos=(0, 0, 1.0),
            size=(0.2, 0.2, 0.2),
        ),
        name="Box",
    )

    camera = None
    if args.record:
        camera = scene.add_camera(
            res=(1280, 720),
            pos=camera_pos,
            lookat=camera_lookat,
        )

    scene.build()

    # Grab the auto-attached overlay to register a custom panel.
    plugin = next(p for p in scene.viewer.plugins if isinstance(p, ImGuiOverlayPlugin))

    def custom_panel(imgui):
        imgui.text("Custom Demo Panel")
        imgui.text("This panel was registered via register_panel()")

    plugin.register_panel(custom_panel)

    # The sweep moves every joint between its limits, each one out of phase with the previous one.
    dofs_lower, dofs_upper = panda.get_dofs_limit()
    dofs_mid = 0.5 * (dofs_lower + dofs_upper)
    dofs_amplitude = 0.4 * (dofs_upper - dofs_lower)
    dofs_phase = torch.arange(panda.n_dofs, dtype=gs.tc_float, device=gs.device)
    horizon = 5 if "PYTEST_VERSION" in os.environ else 300
    if camera is not None:
        camera.start_recording(save_to_filename="out/imgui_joint_control.mp4", fps=30)
    for i_step in range(horizon):
        phase = 2.0 * math.pi * i_step / horizon
        panda.control_dofs_position(dofs_mid + dofs_amplitude * torch.sin(phase + dofs_phase))
        scene.step()
    if camera is not None:
        camera.stop_recording()

    # step() honors the GUI: it advances only while playing, and applies a pending Rebuild Scene first. The viewer
    # paces the loop to real time (ViewerOptions.realtime_factor), so no manual sleep is needed here.
    while scene.viewer.is_alive() and "PYTEST_VERSION" not in os.environ:
        scene.step()


if __name__ == "__main__":
    main()
