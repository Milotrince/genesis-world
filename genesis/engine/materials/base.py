from typing import TYPE_CHECKING, Any, ClassVar

import genesis as gs
from genesis.options.options import Options

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

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        names = sorted(self.model_fields_set.intersection(self.get_entity_options_names()))
        if names:
            gs.logger.warning(
                f"Setting {', '.join(map(repr, names))} on 'gs.materials' is deprecated and will be removed. Pass "
                f"'options={self._entity_options_name}(...)' to 'add_entity' and set them there instead."
            )

    @classmethod
    def get_entity_options_names(cls) -> tuple[str, ...]:
        """Return the names of the deprecated fields, which state options of the entity rather than of its material."""
        return (*cls._entity_names, *cls._solver_options_names)

    def split_entity_options_fields(self) -> tuple["Material", dict[str, Any]]:
        """Split off the deprecated fields given explicitly, as the fields of the entity options they stand for.

        Return the material holding the other fields given explicitly, and the fields of the entity options, those of
        the solver options gathered under 'solver_options'.
        """
        if self.model_fields_set.isdisjoint(self.get_entity_options_names()):
            return self, {}
        fields, solver_fields, material_fields = {}, {}, {}
        for name, value in self:
            if name not in self.model_fields_set:
                continue
            if name in self._entity_names:
                fields[name] = value
            elif name in self._solver_options_names:
                solver_fields[name] = value
            else:
                material_fields[name] = value
        if solver_fields:
            fields["solver_options"] = solver_fields
        return type(self)(**material_fields), fields
