from typing import TYPE_CHECKING, ClassVar

import genesis as gs
from genesis.typing import PositiveFloat, ValidFloat

from .base import Base

if TYPE_CHECKING:
    from genesis.engine.entities.pbd_entity import PBDParticleEntity


class Liquid(Base["PBDParticleEntity"]):
    """
    The liquid material class for PBD.

    Parameters
    ----------
    rho : float, optional
        The rest density of the fluid in kg/m^3. Default is 1000.0.
    sampler, density_relaxation, viscosity_relaxation : optional
        Deprecated, see ``gs.options.PBDEntitySolverOptions``.
    """

    _solver_options_names: ClassVar[tuple[str, ...]] = ("sampler", "density_relaxation", "viscosity_relaxation")

    rho: PositiveFloat = 1000.0
    sampler: gs.particle_sampler = gs.particle_sampler.pbs
    density_relaxation: ValidFloat = 0.2
    viscosity_relaxation: ValidFloat = 0.01
