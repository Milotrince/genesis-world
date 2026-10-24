from collections.abc import Iterator
from typing import TYPE_CHECKING

from genesis.options.behaviors import BehaviorOptions
from genesis.utils.array_class import DataItem

from .base_behavior import Behavior

if TYPE_CHECKING:
    from genesis.engine.entities.rigid_entity import KinematicEntity
    from genesis.engine.simulator import Simulator


class BehaviorManager:
    """Create the behaviors of every entity of a simulator, and run them through build, step and reset."""

    def __init__(self, sim: "Simulator"):
        self._sim = sim
        self._behaviors: list[Behavior] = []

    @property
    def data(self) -> Iterator[DataItem]:
        """Yield the state of every behavior, each item named after its entity and behavior (see 'Behavior.data').

        An item is named '<entity name>.<behavior name>.<item name>'.
        """
        for behavior in self._behaviors:
            for item in behavior.data:
                yield item._replace(name=f"{behavior.entity.name}.{behavior.name}.{item.name}")

    def create_behavior(self, options: BehaviorOptions, entity: "KinematicEntity", name: str) -> Behavior:
        behavior = Behavior.TYPES_MAP[type(options)](options, entity, name)
        self._behaviors.append(behavior)
        return behavior

    def build(self):
        for behavior in self._behaviors:
            behavior.build()

    def step(self):
        for behavior in self._behaviors:
            behavior.step()

    def reset(self, envs_idx=None):
        if not self._behaviors:
            return
        envs_idx = self._sim._scene._sanitize_envs_idx(envs_idx)
        for behavior in self._behaviors:
            behavior.reset(envs_idx)
