from collections.abc import Iterator
from typing import TYPE_CHECKING, ClassVar, Generic, TypeVar, get_args, get_origin

import torch

import genesis as gs
from genesis.options.behaviors import BehaviorOptions
from genesis.repr_base import RBC
from genesis.utils.array_class import DataItem

if TYPE_CHECKING:
    from genesis.engine.entities.rigid_entity import KinematicEntity

BehaviorOptionsT = TypeVar("BehaviorOptionsT", bound=BehaviorOptions)


class Behavior(RBC, Generic[BehaviorOptionsT]):
    """Base class of the runtime behavior of an entity, the handle that 'entity.get_behavior' returns.

    A behavior type registers itself by deriving from this class parameterized with its options class, e.g.
    'class LatchBehavior(Behavior[Latch])'. The simulator calls 'build' once the scene is built, 'step' at the start
    of every step and 'reset' whenever the scene is reset. Each one does nothing unless the behavior type overrides it.

    A behavior keeps whatever per-environment state it needs, allocated in 'build' and yielded by 'data', and its own
    methods form the API users act on it with, e.g. 'engage(envs_idx)' for a latch.
    """

    # Options class -> behavior class of every registered behavior type
    TYPES_MAP: ClassVar[dict[type[BehaviorOptions], type["Behavior"]]] = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # A class deriving from a registered behavior class inherits its bases, and registers nothing by itself
        if "__orig_bases__" not in cls.__dict__:
            return
        for base in cls.__orig_bases__:
            if get_origin(base) is Behavior:
                (options_cls,) = get_args(base)
                break
        else:
            return
        if isinstance(options_cls, TypeVar):
            return

        # A class defined again under the same qualified name (e.g. a notebook cell run twice) registers beside the
        # former one, which the entities already declared with it keep using
        for registered_options_cls in Behavior.TYPES_MAP:
            if registered_options_cls.__name__ == options_cls.__name__ and (
                registered_options_cls.__module__,
                registered_options_cls.__qualname__,
            ) != (options_cls.__module__, options_cls.__qualname__):
                gs.raise_exception(
                    f"Behavior options class name '{options_cls.__name__}' is already registered by "
                    f"'{registered_options_cls.__module__}'. A saved scene names a behavior type by this name, so it "
                    "must be unique."
                )
        Behavior.TYPES_MAP[options_cls] = cls

    def __init__(self, options: BehaviorOptionsT, entity: "KinematicEntity", name: str):
        self._options: BehaviorOptionsT = options
        self._entity: "KinematicEntity" = entity
        self._name: str = name

    def build(self):
        """Resolve the targets the options name, and allocate the per-environment state."""

    def step(self):
        """Act on the simulation at the start of a step, before physics, so applied forces take effect in that step."""

    def reset(self, envs_idx: torch.Tensor):
        """Restore the declared baseline in the selected environments, once their simulation state is restored.

        'envs_idx' holds the indices of those environments.
        """

    @property
    def data(self) -> Iterator[DataItem]:
        """Yield the state of this behavior, which checkpoints and trajectory frames carry.

        A behavior type whose state changes what the next step does yields every such tensor as STATE, e.g.
        'yield from iter_data(self, "_is_engaged", DataKind.STATE)', and a restore writes it back in place. Each item
        travels under the name of its entity and behavior (see 'BehaviorManager.data'). It yields nothing unless the
        behavior type overrides it.
        """
        yield from ()

    @property
    def options(self) -> BehaviorOptionsT:
        """The options this behavior was declared with."""
        return self._options

    @property
    def entity(self) -> "KinematicEntity":
        """The entity this behavior belongs to."""
        return self._entity

    @property
    def name(self) -> str:
        """The name this behavior is looked up by on its entity."""
        return self._name
