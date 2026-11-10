from typing import TYPE_CHECKING, Any, ClassVar

from genesis.options.options import Options
from genesis.utils.warnings import warn_once

if TYPE_CHECKING:
    from genesis.engine.entities.base_entity import Entity


class Material[EntityT: "Entity"](Options):
    """
    The base class of materials, each holding the parameters of the entities of one solver.

    Some fields state an option of the entity rather than of its material, which the entity options state as well
    (see ``gs.entities``). The material still takes them, with a deprecation warning, as long as the entity options
    leave them unset.

    Note
    ----
    This class should *not* be instantiated directly.
    """

    # The entity options this material is given to, and the fields of the material that they state as well, either as
    # their own fields or as fields of their solver options
    _entity_options_name: ClassVar[str] = ""
    _entity_names: ClassVar[tuple[str, ...]] = ()
    _solver_options_names: ClassVar[tuple[str, ...]] = ()

    def __deepcopy__(self, memo: dict[int, Any] | None = None) -> "Material":
        # A material binds its kernels to itself as it is built, so a copy is built anew from the fields given
        # explicitly rather than holding the kernels of the original. The fields the solver sets are left out.
        return type(self)(
            **{
                name: value
                for name, value in self
                if name in self.model_fields_set and not type(self).model_fields[name].exclude
            }
        )

    @classmethod
    def get_entity_options_names(cls) -> tuple[str, ...]:
        """Return the names of the deprecated fields, which state options of the entity rather than of its material."""
        return (*cls._entity_names, *cls._solver_options_names)

    def get_entity_options_fields(self) -> dict[str, Any]:
        """Return the fields of the entity options that the deprecated fields given explicitly stand for, with a warning.

        The fields of the solver options are gathered under 'solver_options'.
        """
        fields, solver_fields = {}, {}
        for name, value in self:
            if name in self.model_fields_set:
                if name in self._entity_names:
                    fields[name] = value
                elif name in self._solver_options_names:
                    solver_fields[name] = value
        if solver_fields:
            fields["solver_options"] = solver_fields
        if fields:
            names = sorted(self.model_fields_set.intersection(self.get_entity_options_names()))
            warn_once(
                f"Setting {', '.join(map(repr, names))} on 'gs.materials' is deprecated and will be removed. Pass "
                f"'options={self._entity_options_name}(...)' to 'add_entity' and set them there instead."
            )
        return fields
