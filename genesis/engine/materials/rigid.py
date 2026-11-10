from typing import TYPE_CHECKING, Annotated, Any, ClassVar

from pydantic import Field, StrictBool

import genesis as gs
from genesis.typing import NonNegativeFloat, PositiveFloat, StrictInt, StrArrayType, ValidFloat

from .kinematic import Kinematic

if TYPE_CHECKING:
    from genesis.engine.entities.rigid_entity import RigidEntity


class Rigid(Kinematic["RigidEntity"]):
    """
    The Rigid class represents a material used in rigid body simulation.

    The options of the links and of the geoms of a rigid entity take it as well, to override its density and friction
    coefficients (see ``gs.options.Link`` and ``gs.options.Geom``). The fields other than these state options of the
    entity, set through ``gs.entities.Rigid`` and its solver options (see ``gs.options.RigidEntitySolverOptions``).
    The material still takes them, with a deprecation warning. 'coup_links' and 'coup_collision_links' enable the
    coupling and the collision of the links of these names alone, through the solver options of these links (see
    ``gs.options.RigidLinkSolverOptions``).

    Parameters
    ----------
    rho : float or None, optional
        The density of the material used to estimate mass if necessary. When None, the default depends on context:
        1000 kg/m^3 if MuJoCo compatibility is enabled (``RigidOptions.enable_mujoco_compatibility``),
        otherwise 600 kg/m^3 for basic rigid objects (mug, table...) vs 1500 kg/m^3 for poly-articulated
        robots. An explicit value replaces the densities an asset authors per geom as well. Default is None.
    friction : float, optional
        Friction coefficient within the rigid solver, between 0.01 and 5. If None, a default of 1.0 may be used or
        parsed from file.
    friction_torsional : float, optional
        Torsional friction coefficient, resisting relative spin about the contact normal. Expressed in meters, as it
        stands for the effective contact patch radius over which sliding friction acts. Only effective when torsional
        friction is enabled at the scene level (see 'RigidOptions.enable_torsional_friction'). If None, parsed from
        file when available (MJCF), otherwise 0.005. Default is None.
    friction_rolling : float, optional
        Rolling friction coefficient, resisting rolling about the two contact tangent axes. Expressed in meters, like
        the torsional coefficient. Only effective when rolling friction is enabled at the scene level (see
        'RigidOptions.enable_rolling_friction'). If None, parsed from file when available (MJCF), otherwise 0.0001.
        Default is None.
    use_visual_raycasting, gravity_compensation : optional
        Deprecated, see ``gs.entities.Rigid``.
    needs_coup, coup_friction, coup_softness, coup_restitution, sdf_cell_size, sdf_min_res, sdf_max_res, coup_type,
    enable_coup_collision, contact_resistance : optional
        Deprecated, see ``gs.options.RigidEntitySolverOptions``.
    coup_links, coup_collision_links : tuple of str or None, optional
        Deprecated, see ``gs.options.RigidLinkSolverOptions``.
    """

    _entity_options_name: ClassVar[str] = "gs.entities.Rigid"
    _entity_names: ClassVar[tuple[str, ...]] = ("use_visual_raycasting", "gravity_compensation")
    _solver_options_names: ClassVar[tuple[str, ...]] = (
        "needs_coup",
        "coup_friction",
        "coup_softness",
        "coup_restitution",
        "sdf_cell_size",
        "sdf_min_res",
        "sdf_max_res",
        "coup_type",
        "coup_links",
        "enable_coup_collision",
        "coup_collision_links",
        "contact_resistance",
    )

    rho: ValidFloat | None = None
    friction: Annotated[ValidFloat, Field(ge=0.01, le=5.0)] | None = None
    friction_torsional: Annotated[ValidFloat, Field(ge=0.0)] | None = None
    friction_rolling: Annotated[ValidFloat, Field(ge=0.0)] | None = None
    needs_coup: StrictBool = True
    coup_friction: NonNegativeFloat = 0.1
    coup_softness: NonNegativeFloat = 0.002
    coup_restitution: Annotated[ValidFloat, Field(ge=0.0, le=1.0)] = 0.0
    sdf_cell_size: PositiveFloat = 0.005
    sdf_min_res: Annotated[StrictInt, Field(ge=16)] = 32
    sdf_max_res: Annotated[StrictInt, Field(ge=16)] = 128
    gravity_compensation: ValidFloat = 0.0
    coup_type: gs.coup_type | None = None
    coup_links: StrArrayType | None = None
    enable_coup_collision: StrictBool = True
    coup_collision_links: StrArrayType | None = None
    contact_resistance: PositiveFloat | None = None

    def get_entity_options_fields(self) -> dict[str, Any]:
        fields = super().get_entity_options_fields()
        solver_fields = fields.pop("solver_options", {})
        links_solver_fields: dict[str, dict[str, bool]] = {}
        for field_name, flag_name in (("coup_links", "needs_coup"), ("coup_collision_links", "enable_coup_collision")):
            links_name = solver_fields.pop(field_name, None)
            if links_name is None:
                continue
            if not solver_fields.get(flag_name, True):
                gs.raise_exception(f"'{field_name}' requires '{flag_name}=True'.")
            solver_fields[flag_name] = False
            for link_name in links_name:
                links_solver_fields.setdefault(link_name, {})[flag_name] = True
        if solver_fields:
            fields["solver_options"] = solver_fields
        if links_solver_fields:
            fields["links"] = {name: {"solver_options": flags} for name, flags in links_solver_fields.items()}
        return fields
