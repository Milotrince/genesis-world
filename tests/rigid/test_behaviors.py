import pytest
import torch

import genesis as gs
from genesis.engine.behaviors import Behavior
from genesis.recorders.trajectory import TRAJECTORY_FORMAT, Trajectory
from genesis.typing import Vec3FType
from genesis.utils.array_class import DataKind, iter_data

from ..utils.assertions import assert_allclose


@pytest.fixture(scope="module")
def thruster():
    class Thruster(gs.behaviors.BehaviorOptions):
        link: str
        force: Vec3FType
        is_active: bool = True

    class ThrusterBehavior(Behavior[Thruster]):
        def build(self):
            self._link = self.entity.get_link(self.options.link)
            self._force = torch.tensor(self.options.force, dtype=gs.tc_float, device=gs.device)
            self._is_active = torch.full((self.entity.scene._B,), self.options.is_active, device=gs.device)

        @property
        def data(self):
            yield from iter_data(self, "_is_active", DataKind.STATE)

        def deactivate(self, envs_idx=None):
            self._is_active[self.entity.scene._sanitize_envs_idx(envs_idx)] = False

        def step(self):
            self._link.apply_external_force(self._is_active[:, None] * self._force)

        def reset(self, envs_idx):
            self._is_active[envs_idx] = self.options.is_active

    return Thruster


@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0, 2])
def test_plugin_behavior(n_envs, thruster, show_viewer, tmp_path, tol):
    DT, GRAVITY, LIFT, N_STEPS = 0.01, 10.0, 5.0, 20

    class Unregistered(gs.behaviors.BehaviorOptions):
        pass

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=DT,
            gravity=(0.0, 0.0, -GRAVITY),
        ),
        rigid_options=gs.options.RigidOptions(
            integrator=gs.integrator.Euler,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.0, -3.0, 1.0),
            camera_lookat=(0.0, 0.0, 1.0),
        ),
        show_viewer=show_viewer,
    )
    lifted_box = scene.add_entity(
        morph=gs.morphs.Box(
            size=(0.1, 0.1, 0.1),
            pos=(-0.5, 0.0, 1.0),
        ),
        behaviors={
            "lift": thruster(
                link="box_baselink",
                force=(0.0, 0.0, LIFT),
            ),
        },
    )
    idle_box = scene.add_entity(
        morph=gs.morphs.Box(
            size=(0.1, 0.1, 0.1),
            pos=(0.5, 0.0, 1.0),
        ),
        behaviors={
            "lift": thruster(
                link="box_baselink",
                force=(0.0, 0.0, LIFT),
                is_active=False,
            ),
        },
    )
    lift = lifted_box.get_behavior("lift")
    with pytest.raises(gs.GenesisException, match="No behavior class registered for Unregistered"):
        scene.add_entity(morph=gs.morphs.Box(size=(0.1, 0.1, 0.1)), behaviors={"unregistered": Unregistered()})
    with pytest.raises(gs.GenesisException, match="maps a behavior name to its options"):
        scene.add_entity(morph=gs.morphs.Box(size=(0.1, 0.1, 0.1)), behaviors={"lift": None})
    with pytest.raises(gs.GenesisException, match="only applies to rigid and kinematic entities"):
        scene.add_entity(
            morph=gs.morphs.Box(size=(0.1, 0.1, 0.1)),
            material=gs.materials.MPM.Elastic(),
            behaviors={"lift": thruster(link="box_baselink", force=(0.0, 0.0, LIFT))},
        )
    with pytest.raises(gs.GenesisException, match="Available behavior names"):
        lifted_box.get_behavior("thrust")

    scene.export(tmp_path / "behaviors.gscene")
    restored = gs.Scene.load(tmp_path / "behaviors.gscene")
    assert restored.entities[0].get_behavior("lift").options == lift.options

    scene.build(n_envs=n_envs)
    assert lifted_box.get_behavior("lift") is lift
    lifted_dvz = (LIFT / lifted_box.get_mass() - GRAVITY) * N_STEPS * DT
    falling_dvz = -GRAVITY * N_STEPS * DT

    for _ in range(N_STEPS):
        scene.step()
    assert_allclose(lifted_box.get_vel()[..., 2], lifted_dvz, tol=tol)
    assert_allclose(idle_box.get_vel()[..., 2], falling_dvz, tol=tol)

    lifted_vz = lifted_box.get_vel()[..., 2]
    if n_envs > 0:
        lift.deactivate(envs_idx=[0])
        lifted_dvz_expected = torch.tensor([falling_dvz, lifted_dvz], dtype=gs.tc_float, device=gs.device)
    else:
        lift.deactivate()
        lifted_dvz_expected = falling_dvz
    for _ in range(N_STEPS):
        scene.step()
    assert_allclose(lifted_box.get_vel()[..., 2], lifted_vz + lifted_dvz_expected, tol=tol)

    checkpoint = scene.__getstate__()
    checkpoint_path = tmp_path / f"deactivated{TRAJECTORY_FORMAT}"
    scene.save_checkpoint(checkpoint_path)
    lifted_vz = lifted_box.get_vel()[..., 2]

    scene.reset()
    for _ in range(N_STEPS):
        scene.step()
    assert_allclose(lifted_box.get_vel()[..., 2], lifted_dvz, tol=tol)

    # A checkpoint and a checkpoint file put back the deactivation, which a reset undid
    scene.__setstate__(checkpoint)
    for _ in range(N_STEPS):
        scene.step()
    assert_allclose(lifted_box.get_vel()[..., 2], lifted_vz + lifted_dvz_expected, tol=tol)

    Trajectory(checkpoint_path, scene=scene).seek(-1)
    for _ in range(N_STEPS):
        scene.step()
    assert_allclose(lifted_box.get_vel()[..., 2], lifted_vz + lifted_dvz_expected, tol=tol)
