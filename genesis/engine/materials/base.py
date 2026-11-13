from typing import TYPE_CHECKING, Any, ClassVar

from pydantic import StrictBool

from genesis.options.options import Options
from genesis.utils.warnings import warn_once

if TYPE_CHECKING:
    from genesis.engine.entities.base_entity import Entity


class Material[EntityT: "Entity"](Options):
    """
    The base class of materials.

    Some fields of a material state an option of the entity rather than of its material, which the entity options state
    as well (see ``gs.entities``). Given explicitly, they are deprecated and set the entity option they stand for.

    Note
    ----
    This class should *not* be instantiated directly.
    """

    # The entity options the material is given to, and the deprecated fields of the material that they state, either as
    # their own fields or as fields of their solver options
    _entity_options_name: ClassVar[str] = ""
    _entity_names: ClassVar[tuple[str, ...]] = ()
    _solver_options_names: ClassVar[tuple[str, ...]] = ()

    use_visual_raycasting: StrictBool = False

    def get_entity_options_fields(self) -> dict[str, Any]:
        """Return the fields of the entity options that the deprecated fields given explicitly stand for, with a warning.

        The fields of the solver options are gathered under 'solver_options'.
        """
        fields, solver_fields = {}, {}
        for name in self.model_fields_set:
            if name in self._entity_names:
                fields[name] = getattr(self, name)
            elif name in self._solver_options_names:
                solver_fields[name] = getattr(self, name)
        if solver_fields:
            fields["solver_options"] = solver_fields
        if fields:
            names = sorted(self.model_fields_set.intersection((*self._entity_names, *self._solver_options_names)))
            warn_once(
                f"Setting {', '.join(map(repr, names))} on 'gs.materials.{type(self).__name__}' is deprecated. Set "
                f"them through 'options={self._entity_options_name}(...)' of 'add_entity' instead."
            )
        return fields
