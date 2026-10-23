from genesis.typing import NonNegativeFloat, PositiveFloat

from .base import BaseMaterial, PoissonRatio


class Snow(BaseMaterial):
    """A compressible solid that yields under load and hardens as it compacts.

    The material point method (MPM) simulates it.

    Parameters
    ----------
    density : float, optional
        The mass density, in kg/m^3. Default is 1000.
    youngs_modulus : float, optional
        The Young's modulus, in Pa. Default is 1e6.
    poisson_ratio : float, optional
        The Poisson ratio, between -1 and 0.5. Default is 0.2.
    critical_compression : float, optional
        The relative compression at which the snow yields. Default is 0.025.
    critical_stretch : float, optional
        The relative stretch at which the snow yields. Default is 0.0045.
    """

    density: PositiveFloat = 1000.0
    youngs_modulus: PositiveFloat = 1e6
    poisson_ratio: PoissonRatio = 0.2
    critical_compression: NonNegativeFloat = 2.5e-2
    critical_stretch: NonNegativeFloat = 4.5e-3
