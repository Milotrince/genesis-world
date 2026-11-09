from typing import TYPE_CHECKING, ClassVar

from genesis.typing import NonNegativeFloat, PositiveFloat, ValidFloat

from .base import Base

if TYPE_CHECKING:
    from genesis.engine.entities.pbd_entity import PBD3DEntity


class Elastic(Base["PBD3DEntity"]):
    """
    The 3D elastic volumetric material class for PBD.

    Parameters
    ----------
    rho : float, optional
        The density of the elastic material (kg/m^3). Default is 1000.0.
    static_friction : float, optional
        Static friction coefficient. Default is 0.15.
    kinetic_friction, stretch_compliance, bending_compliance, volume_compliance, stretch_relaxation,
    bending_relaxation, volume_relaxation : optional
        Deprecated, see ``gs.options.PBDEntitySolverOptions``.
    """

    _solver_options_names: ClassVar[tuple[str, ...]] = (
        "kinetic_friction",
        "stretch_compliance",
        "bending_compliance",
        "volume_compliance",
        "stretch_relaxation",
        "bending_relaxation",
        "volume_relaxation",
    )

    rho: PositiveFloat = 1000.0
    static_friction: NonNegativeFloat = 0.15
    kinetic_friction: NonNegativeFloat = 0.15
    stretch_compliance: NonNegativeFloat = 0.0
    bending_compliance: NonNegativeFloat = 0.0
    volume_compliance: NonNegativeFloat = 0.0
    stretch_relaxation: ValidFloat = 0.1
    bending_relaxation: ValidFloat = 0.1
    volume_relaxation: ValidFloat = 0.1
