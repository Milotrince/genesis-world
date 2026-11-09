from typing import ClassVar

import genesis as gs
from genesis.typing import NonNegativeFloat, PositiveFloat

from .base import Base


class Liquid(Base):
    """
    The liquid material class for SPH.

    Parameters
    ----------
    rho : float, optional
        The rest density (kg/m^3). Default is 1000.0.
    stiffness : float, optional
        State stiffness (N/m^2). Controls how pressure increases with compression. Default is 50000.0.
    mu : float, optional
        The viscosity of the liquid. Default is 0.005.
    gamma : float, optional
        The surface tension of the liquid. Default is 0.01.
    exponent, sampler : optional
        Deprecated, see ``gs.options.SPHEntitySolverOptions``.
    """

    _solver_options_names: ClassVar[tuple[str, ...]] = (*Base._solver_options_names, "exponent")

    rho: PositiveFloat = 1000.0
    stiffness: PositiveFloat = 50000.0
    exponent: PositiveFloat = 7.0
    mu: NonNegativeFloat = 0.005
    gamma: NonNegativeFloat = 0.01
    sampler: gs.particle_sampler = gs.particle_sampler.regular
