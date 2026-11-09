from typing import TYPE_CHECKING, ClassVar

from pydantic import StrictBool

from .base import Material

if TYPE_CHECKING:
    from genesis.engine.entities.base_entity import Entity


class Kinematic[EntityT: "Entity"](Material[EntityT]):
    """
    Visualization-only material for ghost/reference entities.

    Kinematic entities are rendered but do not participate in physics simulation, collision detection, or constraint
    solving. ``Scene.add_entity`` turns it into the options of a kinematic entity (see ``gs.entities.Kinematic``).

    Parameters
    ----------
    use_visual_raycasting : bool, optional
        Deprecated, see ``gs.entities.Kinematic``. Default is False.
    """

    _entity_options_name: ClassVar[str] = "gs.entities.Kinematic"
    _entity_names: ClassVar[tuple[str, ...]] = ("use_visual_raycasting",)

    use_visual_raycasting: StrictBool = False
