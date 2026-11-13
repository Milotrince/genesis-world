import glob
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, Literal

from pydantic import Field, StrictBool, model_validator

import genesis as gs
from genesis.typing import NonNegativeFloat, PositiveFloat, StrictInt, StrArrayType, ValidFloat

from .kinematic import Kinematic

if TYPE_CHECKING:
    from genesis.engine.entities.rigid_entity import RigidEntity

CoupType = Literal["two_way_soft_constraint", "external_articulation", "ipc_only"]


class Rigid(Kinematic["RigidEntity"]):
    """
    The Rigid class represents a material used in rigid body simulation.

    Note
    ----
    This class is intended for use with the rigid solver and provides parameters
    relevant to physical interactions such as friction, density, and signed distance fields (SDFs).

    Parameters
    ----------
    rho : float or None, optional
        The density of the material used to estimate mass if necessary. When None, the default depends on context:
        1000 kg/m^3 if MuJoCo compatibility is enabled (``RigidOptions.enable_mujoco_compatibility``),
        otherwise 600 kg/m^3 for basic rigid objects (mug, table...) vs 1500 kg/m^3 for poly-articulated
        robots. Default is None.
    friction : float, optional
        Friction coefficient within the rigid solver. If None, a default of 1.0 may be used or parsed from file.
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
        Deprecated. The links of these names alone are coupled, or collide in the IPC coupler, through the solver
        options of their link options (see ``gs.options.RigidLinkSolverOptions``).
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
    coup_type: CoupType | None = None
    coup_links: StrArrayType | None = None
    enable_coup_collision: StrictBool = True
    coup_collision_links: StrArrayType | None = None
    contact_resistance: PositiveFloat | None = None

    @model_validator(mode="after")
    def _check_ipc_only_gravity(self) -> "Rigid":
        # IPC applies gravity to an entity it simulates alone, so the gravity of Genesis would count twice
        if self.coup_type == "ipc_only" and abs(self.gravity_compensation) > gs.EPS:
            gs.raise_exception(
                "`gravity_compensation` must be 0 with coup_type='ipc_only'. IPC controls gravity for ipc_only entities."
            )
        return self

    def model_post_init(self, context: Any) -> None:
        if self.coup_type is not None and not self.needs_coup:
            gs.raise_exception(
                "`coup_type` is only supported with needs_coup=True. "
                f"Got needs_coup={self.needs_coup}, coup_type={self.coup_type!r}."
            )

        if self.coup_links is not None and (
            not self.needs_coup or self.coup_type not in (None, "two_way_soft_constraint")
        ):
            gs.raise_exception(
                "`coup_links` is only supported with needs_coup=True and "
                f"'two_way_soft_constraint' type in IPC. "
                f"Got needs_coup={self.needs_coup}, coup_type={self.coup_type!r}."
            )

        if self.coup_collision_links is not None and not self.enable_coup_collision:
            gs.raise_exception(
                "`coup_collision_links` is only effective when `enable_coup_collision=True`. "
                "Set `enable_coup_collision=False` to disable collision for all links."
            )

        if self.sdf_min_res > self.sdf_max_res:
            gs.raise_exception("`sdf_min_res` must be smaller than or equal to `sdf_max_res`.")

    def get_entity_options_fields(self) -> dict[str, Any]:
        fields = super().get_entity_options_fields()
        solver_fields = fields.pop("solver_options", {})
        links_solver_fields = []
        for field_name, flag_name in (("coup_links", "needs_coup"), ("coup_collision_links", "enable_coup_collision")):
            links_name = solver_fields.pop(field_name, None)
            if links_name is not None:
                solver_fields[flag_name] = False
                links_solver_fields.append((tuple(map(glob.escape, links_name)), flag_name))
        if solver_fields:
            fields["solver_options"] = solver_fields
        if links_solver_fields:
            fields["links_solver_options"] = links_solver_fields
        return fields
