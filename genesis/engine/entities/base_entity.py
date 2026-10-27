from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

import genesis as gs
from genesis.engine.materials.base import BaseMaterial
from genesis.grad.bridge import QueriedStates
from genesis.options.entities import EntityOptions, HybridEntityOptions, KinematicEntityOptions
from genesis.options.morphs import Morph
from genesis.repr_base import RBC

if TYPE_CHECKING:
    from genesis.engine.scene import Scene
    from genesis.engine.sensors.base_sensor import Sensor


@dataclass
class EntityDescription:
    """Base class of what one entity of a scene is created from, whatever the solver simulating it.

    A scene names every entity it holds as one of these, so a kind of entity described later stands there too. The
    options the entity was constructed from stand here as authored, before any default was resolved, and decide the
    solver simulating it (see 'EntityOptions.scene_solver_options_cls'). A subclass holds what the resolution of
    those options changes or adds, such as the generated name.
    """

    # The prefix of the names generated for the entities of this kind, the morphs named after their own identifier,
    # and the identifier standing in for the morphs of any other kind (see 'resolve_name')
    _name_prefix = ""
    _name_identified_morphs = ()
    _name_fallback = "entity"

    options: EntityOptions = field(kw_only=True)

    @property
    def material(self) -> BaseMaterial | None:
        """The material of the entity, or None for an entity holding none.

        A kinematic entity takes no part in physics, and each part of a hybrid entity holds a material of its own.
        """
        if isinstance(self.options, (KinematicEntityOptions, HybridEntityOptions)):
            return None
        return self.options.material

    @classmethod
    def resolve_name(cls, options: EntityOptions) -> str:
        """Return the name the options give the entity, or otherwise one generated from its morph."""
        if options.name is not None:
            return options.name
        morphs = options.morph if isinstance(options.morph, tuple) else (options.morph,)
        if len(morphs) > 1:
            identifier = "heterogeneous"
        elif isinstance(morphs[0], cls._name_identified_morphs):
            identifier = morphs[0]._identifier()
        elif isinstance(morphs[0], gs.morphs.Mesh):
            identifier = Path(morphs[0].file).stem
        elif isinstance(morphs[0], gs.morphs.Nowhere):
            identifier = "emitter"
        else:
            identifier = cls._name_fallback
        return f"{cls._name_prefix}{identifier}_{gs.UID().short()}"


@dataclass(kw_only=True)
class VerticesDescription:
    init_positions: np.ndarray


class Entity(RBC):
    """
    Base class for all types of entities.
    """

    def __init__(self, idx, scene, solver, desc: EntityDescription, morph: Morph):
        # An entity is created from one morph. A kind of entity built from several passes the primary one here.
        if not isinstance(morph, Morph):
            gs.raise_exception(f"An entity is created from one morph, got {type(morph).__name__}.")
        uid = gs.UID()
        while any(entity.uid.match(uid, short_only=True) for entity in scene.entities):
            uid = gs.UID()
        self._uid = uid
        self._idx = idx
        self._scene: "Scene" = scene
        self._solver = solver
        self._desc = desc
        self._material = desc.material
        self._morph = morph
        self._surface = desc.surface
        self._sim = scene.sim

        # The states of this entity queried during the current rollout, registered by 'get_state' (see QueriedStates)
        self._queried_states = QueriedStates(self._sim)

        # The resolution of the description names the entity, generating a name when the options give none
        if any(entity.name == desc.name for entity in scene.entities):
            gs.raise_exception(f"Entity name '{desc.name}' already exists in scene.")
        self._name = desc.name

        gs.logger.info(
            f"Adding ~<{self.__repr_name__()}>~ ~~<{self._name}>~~. idx: ~<{self._idx}>~, "
            f"morph: ~<{morph}>~, material: ~<{self._material}>~."
        )

    # ------------------------------------------------------------------------------------
    # ----------------------------------- properties -------------------------------------
    # ------------------------------------------------------------------------------------

    @property
    def uid(self):
        return self._uid

    @property
    def idx(self):
        return self._idx

    @property
    def scene(self):
        return self._scene

    @property
    def sim(self):
        return self._sim

    @property
    def solver(self):
        return self._solver

    @property
    def surface(self):
        return self._surface

    @property
    def morph(self):
        return self._morph

    @property
    def desc(self) -> EntityDescription:
        """The description this entity was created from."""
        return self._desc

    @property
    def options(self) -> EntityOptions:
        """The options this entity was constructed from, as authored.

        Each access returns a copy of its own, so changing the morph or the material it holds leaves the entity as it
        is. The options passed to 'Scene.add_entity' may change or construct other entities afterwards. What the scene
        resolved from them lives in 'desc'.
        """
        return self._desc.options.model_copy(deep=True)

    @property
    def material(self):
        return self._material

    @property
    def is_built(self):
        return self._solver._scene._is_built

    def _repr_brief(self):
        return f"{self.__repr_name__()}, idx: {self.idx}, morph: {self._repr_morph()}, material: {self.material}"

    def _repr_morph(self):
        return f"{self.morph}"

    @property
    def name(self) -> str:
        """
        The name of this entity.

        Returns
        -------
        str
            The name given by the entity options, or otherwise the name generated from the morph as the entity was
            added.
        """
        return self._name

    @property
    def sensors(self) -> "gs.List[Sensor]":
        """List of sensors attached to this entity."""
        return self._sim._sensor_manager.get_sensors_by_entity(self._idx)

    def collect_output_grads(self):
        """Hand the gradients of the states of this entity queried at the current step to 'add_grad_from_state'."""
        for state in self._queried_states.get(self._sim.cur_step_global):
            self.add_grad_from_state(state)

    @gs.assert_built
    def read_sensors(self, envs_idx=None) -> "dict[type[Sensor], torch.Tensor]":
        """
        Read every sensor attached to this entity as a tensor per sensor class.

        Always returns a fresh tensor independent of the internal sensor storage; the caller is free to mutate the
        result.

        Parameters
        ----------
        envs_idx : array-like | int | slice | None
            Environment selection. Defaults to all environments.

        Returns
        -------
        dict[Type[Sensor], torch.Tensor]
            For each sensor class with at least one sensor on this entity, a tensor of shape
            (B, [history,] entity_cache_size_for_class).
        """
        return self._sim._sensor_manager.read_sensors(entity_idx=self._idx, envs_idx=envs_idx)
