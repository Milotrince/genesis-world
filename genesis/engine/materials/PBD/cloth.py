from typing import TYPE_CHECKING, ClassVar

from genesis.typing import NonNegativeFloat, PositiveFloat, ValidFloat

from .base import Base

if TYPE_CHECKING:
    from genesis.engine.entities.pbd_entity import PBD2DEntity


class Cloth(Base["PBD2DEntity"]):
    """
    The cloth material class for PBD.

    Parameters
    ----------
    rho : float, optional
        The density of the cloth. Default is 4.0.
        Note that this is kg/m^2, not kg/m^3, as cloth is a 2D material, so the entity mass will be calculated as rho * surface_area.
    static_friction : float, optional
        Static friction coefficient. Default is 0.15.
    kinetic_friction, stretch_compliance, bending_compliance, stretch_relaxation, bending_relaxation, air_resistance :
    optional
        Deprecated, see ``gs.options.PBDEntitySolverOptions``.
    """

    _solver_options_names: ClassVar[tuple[str, ...]] = (
        "kinetic_friction",
        "stretch_compliance",
        "bending_compliance",
        "stretch_relaxation",
        "bending_relaxation",
        "air_resistance",
    )

    rho: PositiveFloat = 4.0
    static_friction: NonNegativeFloat = 0.15
    kinetic_friction: NonNegativeFloat = 0.15
    stretch_compliance: NonNegativeFloat = 1e-7
    bending_compliance: NonNegativeFloat = 1e-5
    stretch_relaxation: ValidFloat = 0.3
    bending_relaxation: ValidFloat = 0.1
    air_resistance: NonNegativeFloat = 1e-3
