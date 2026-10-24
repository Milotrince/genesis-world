"""Cabinet door held shut by a latch behavior, released by pressing the button next to it.

The latch is a behavior type defined here, the way a third-party plugin defines one: an options class declaring what
an entity holds, and a runtime class deriving from 'Behavior' that registers itself and acts on the simulation. It
holds the door hinge with a weld while engaged, releases it on the rising edge of a button press, then pushes the door
open with a torque pulse. Each environment has its own latch state.
"""

import argparse
import os
import tempfile

import torch

import genesis as gs
from genesis.engine.behaviors import Behavior
from genesis.utils.array_class import DataKind, iter_data

CABINET_MJCF = """
<mujoco model="cabinet">
  <compiler angle="radian"/>
  <worldbody>
    <body name="cabinet" pos="0 0 0.15">
      <geom name="shell" type="box" size="0.25 0.2 0.15" rgba="0.8 0.8 0.8 1"/>
      <body name="door" pos="-0.25 -0.215 0">
        <joint name="door_hinge" type="hinge" axis="0 0 1" range="-1.6 0" damping="0.2"/>
        <geom name="panel" type="box" size="0.2 0.01 0.14" pos="0.2 0 0" rgba="0.3 0.3 0.35 1"/>
      </body>
      <body name="button" pos="0.2 -0.215 0">
        <joint name="open_button" type="slide" axis="0 1 0" range="0 0.005" stiffness="50" damping="1"/>
        <geom name="cap" type="box" size="0.02 0.01 0.02" contype="0" conaffinity="0" rgba="0.8 0.2 0.2 1"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


class Latch(gs.behaviors.BehaviorOptions):
    """Hold a joint in place while engaged, and release it when another joint is pressed past a position.

    Parameters
    ----------
    joint : str
        The name of the latched joint.
    is_engaged : bool
        Whether the latch is engaged after a reset. Defaults to True.
    release_joint : str | None
        The name of the joint whose press releases the latch. None for a latch released by 'release' alone.
    release_position : float
        The position of 'release_joint' past which it counts as pressed.
    push_torque : float
        The torque about the latched joint axis applied on release, in N*m. Defaults to 0.
    push_duration : float
        How long the torque is applied after a release, in seconds. Defaults to 0.
    """

    joint: str
    is_engaged: bool = True
    release_joint: str | None = None
    release_position: float = 0.0
    push_torque: float = 0.0
    push_duration: float = 0.0


class LatchBehavior(Behavior[Latch]):
    def build(self):
        scene = self.entity.scene
        self._joint = self.entity.get_joint(self.options.joint)
        self._link = self._joint.link
        self._push_n_steps = round(self.options.push_duration / scene.dt)
        self._release_dofs_idx_local = None
        if self.options.release_joint is not None:
            self._release_dofs_idx_local = self.entity.get_joint(self.options.release_joint).dofs_idx_local

        self._is_engaged = torch.zeros((scene._B,), dtype=torch.bool, device=gs.device)
        self._is_pressed = torch.zeros((scene._B,), dtype=torch.bool, device=gs.device)
        self._push_steps_left = torch.zeros((scene._B,), dtype=gs.tc_int, device=gs.device)

    @property
    def data(self):
        # The welds travel with the rigid solver state, and these flags with the behavior, so a checkpoint keeps both
        yield from iter_data(self, "_is_engaged", DataKind.STATE)
        yield from iter_data(self, "_is_pressed", DataKind.STATE)
        yield from iter_data(self, "_push_steps_left", DataKind.STATE)

    @property
    def is_engaged(self) -> torch.Tensor:
        """Whether the latch is engaged, per environment."""
        return self._is_engaged.clone()

    def engage(self, envs_idx=None):
        """Lock the latched joint at its current position in the selected environments."""
        envs_idx = self.entity.scene._sanitize_envs_idx(envs_idx)
        envs_idx = envs_idx[~self._is_engaged[envs_idx]]
        if len(envs_idx) == 0:
            return
        self.entity.solver.add_weld_constraint(
            self._link.parent_idx, self._link.idx, envs_idx=envs_idx if self.entity.scene.n_envs > 0 else None
        )
        self._is_engaged[envs_idx] = True

    def release(self, envs_idx=None):
        """Free the latched joint in the selected environments, and start the opening push there."""
        envs_idx = self.entity.scene._sanitize_envs_idx(envs_idx)
        envs_idx = envs_idx[self._is_engaged[envs_idx]]
        if len(envs_idx) == 0:
            return
        self.entity.solver.delete_weld_constraint(
            self._link.parent_idx, self._link.idx, envs_idx=envs_idx if self.entity.scene.n_envs > 0 else None
        )
        self._is_engaged[envs_idx] = False
        self._push_steps_left[envs_idx] = self._push_n_steps

    def step(self):
        if self._release_dofs_idx_local is not None:
            release_qpos = self.entity.get_dofs_position(dofs_idx_local=self._release_dofs_idx_local)
            is_pressed = torch.atleast_2d(release_qpos)[:, 0] > self.options.release_position
            # A held press releases once, on the step the button crosses the release position
            self.release(torch.nonzero(is_pressed & ~self._is_pressed)[:, 0])
            self._is_pressed = is_pressed

        is_pushing = self._push_steps_left > 0
        if is_pushing.any():
            push_torque = self.options.push_torque * self._joint.get_anchor_axis()
            self._link.apply_external_torque(is_pushing[:, None] * push_torque)
            self._push_steps_left.sub_(1).clamp_(min=0)

    def reset(self, envs_idx):
        # A reset drops the welds added at runtime in the reset environments
        self._is_engaged[envs_idx] = False
        self._is_pressed[envs_idx] = False
        self._push_steps_left[envs_idx] = 0
        if self.options.is_engaged:
            self.engage(envs_idx)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", help="Show visualization GUI")
    parser.add_argument("-g", "--gpu", action="store_true", help="Run on GPU instead of CPU")
    args = parser.parse_args()

    gs.init(backend=gs.gpu if args.gpu else gs.cpu)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.01,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.6, -1.4, 0.8),
            camera_lookat=(0.0, 0.0, 0.15),
        ),
        show_viewer=args.vis,
    )
    scene.add_entity(gs.morphs.Plane())
    cabinet = scene.add_entity(
        options=gs.options.RigidEntityOptions(
            morph=gs.morphs.MJCF(
                file=CABINET_MJCF,
            ),
            behaviors={
                "door_latch": Latch(
                    joint="door_hinge",
                    release_joint="open_button",
                    release_position=0.003,
                    push_torque=-1.0,
                    push_duration=0.2,
                ),
            },
        ),
    )
    scene.build(n_envs=2)

    latch = cabinet.get_behavior("door_latch")
    door = cabinet.get_link("door")
    button_dofs_idx_local = cabinet.get_joint("open_button").dofs_idx_local
    cabinet.set_dofs_kp(200.0, dofs_idx_local=button_dofs_idx_local)

    def report(phase, cabinet):
        door_angle = cabinet.get_qpos(qs_idx_local=cabinet.get_joint("door_hinge").qs_idx_local)[:, 0]
        is_engaged = cabinet.get_behavior("door_latch").is_engaged
        print(f"{phase}: door angle {door_angle.tolist()} rad, latch engaged {is_engaged.tolist()}")

    # The engaged latch holds the door shut against a pull on its handle
    for _ in range(50):
        door.apply_external_torque((0.0, 0.0, -2.0))
        scene.step()
    report("Pulled while latched", cabinet)

    # Pressing the button in env 0 releases its latch alone, and the release pushes the door open
    cabinet.control_dofs_position([[0.005], [0.0]], dofs_idx_local=button_dofs_idx_local)
    for _ in range(100):
        scene.step()
    report("Button held in env 0", cabinet)

    # A checkpoint holds the latch state along with the welds and the rest of the scene
    checkpoint_dir = tempfile.TemporaryDirectory()
    checkpoint_path = os.path.join(checkpoint_dir.name, "latch.gstraj")
    scene.save_checkpoint(checkpoint_path)

    # Engaging again locks the door wherever it stands
    cabinet.control_dofs_position([[0.0], [0.0]], dofs_idx_local=button_dofs_idx_local)
    latch.engage()
    for _ in range(50):
        door.apply_external_torque((0.0, 0.0, -2.0))
        scene.step()
    report("Pulled after engaging again", cabinet)

    # A reset closes the door and restores the declared latch state in every environment
    scene.reset()
    report("Reset", cabinet)

    # The scene loaded from the checkpoint stands where it was saved, its door open and released in env 0 alone
    restored_scene = gs.Scene.load_checkpoint(checkpoint_path)
    restored_cabinet = restored_scene.entities[1]
    report("Loaded from the checkpoint", restored_cabinet)
    for _ in range(50):
        restored_cabinet.get_link("door").apply_external_torque((0.0, 0.0, -2.0))
        restored_scene.step()
    report("Pulled after loading", restored_cabinet)


if __name__ == "__main__":
    main()
