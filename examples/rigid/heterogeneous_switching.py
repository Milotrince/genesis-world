"""
Morph Variant Switching Example
===============================

One entity is built from a box, a sphere and a cylinder, and every environment carries one of them. The variant an
environment carries switches at runtime through 'set_morph_variant', and both the physics and the rendering follow it
from the next step on.

Usage:
    python heterogeneous_switching.py -v  # SPACE gives every environment a random variant, ESC quits
    python heterogeneous_switching.py     # switches by itself every 50 steps
"""

import argparse
import os

import numpy as np

import genesis as gs
from genesis.vis.keybindings import Key, KeyAction, Keybind


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", help="Show the viewer, where SPACE switches the variants")
    parser.add_argument("-g", "--gpu", action="store_true", help="Run on GPU")
    parser.add_argument("-b", "--num-envs", type=int, default=4, help="Number of parallel environments")
    parser.add_argument("-s", "--steps", type=int, default=500, help="Number of steps to run without the viewer")
    args = parser.parse_args()

    gs.init(backend=gs.gpu if args.gpu else gs.cpu)

    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(3.0, 3.0, 2.0),
            camera_lookat=(0.0, 0.0, 0.2),
        ),
        show_viewer=args.vis,
    )
    scene.add_entity(
        gs.morphs.Plane(),
    )
    morphs = [
        gs.morphs.Box(
            size=(0.2, 0.2, 0.2),
            pos=(0.0, 0.0, 0.3),
        ),
        gs.morphs.Sphere(
            radius=0.12,
            pos=(0.0, 0.0, 0.3),
        ),
        gs.morphs.Cylinder(
            radius=0.1,
            height=0.25,
            pos=(0.0, 0.0, 0.3),
        ),
    ]
    entity = scene.add_entity(
        morph=morphs,
    )
    if args.vis:
        scene.viewer.add_plugin(
            gs.vis.viewer_plugins.MouseInteractionPlugin(
                use_force=True,
            )
        )
    scene.build(
        n_envs=args.num_envs,
        env_spacing=(1.0, 1.0),
    )

    rng = np.random.default_rng(seed=0)
    is_running = True
    is_switch_pending = False

    def request_switch():
        nonlocal is_switch_pending
        is_switch_pending = True

    def stop():
        nonlocal is_running
        is_running = False

    if args.vis:
        scene.viewer.register_keybinds(
            Keybind("switch_variants", Key.SPACE, KeyAction.PRESS, callback=request_switch),
            Keybind("quit", Key.ESCAPE, KeyAction.RELEASE, callback=stop),
        )
        print("\nSPACE gives every environment a random variant, drag the entity with the mouse, ESC quits.\n")
    n_steps = args.steps if "PYTEST_VERSION" not in os.environ else 5

    i_step = 0
    while is_running:
        # The viewer runs the keybind callbacks on its own thread, so the scene switches here, between two steps
        if is_switch_pending or (not args.vis and i_step % 50 == 0):
            is_switch_pending = False
            entity.set_morph_variant(rng.integers(len(morphs), size=max(args.num_envs, 1)))
            # The incoming variant drops from above the floor, at rest
            entity.set_pos((0.0, 0.0, 0.3))
            entity.zero_all_dofs_velocity()
            print("variants:", entity.get_morph_variant())
        scene.step()
        i_step += 1
        if not args.vis and i_step == n_steps:
            is_running = False


if __name__ == "__main__":
    main()
