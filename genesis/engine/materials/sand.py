from genesis.typing import PositiveFloat

from .base import BaseMaterial, PoissonRatio


class Sand(BaseMaterial):
    """A granular material whose grains slide over each other once the shear load overcomes their internal friction.

    The material point method (MPM) simulates it.

    Parameters
    ----------
    density : float, optional
        The mass density, in kg/m^3. Default is 1000.
    youngs_modulus : float, optional
        The Young's modulus, in Pa. Default is 1e6.
    poisson_ratio : float, optional
        The Poisson ratio, between -1 and 0.5. Default is 0.2.
    friction_angle : float, optional
        The internal friction angle, in degrees. A larger angle holds steeper piles, as wet sand does. Default is 45.
    """

    density: PositiveFloat = 1000.0
    youngs_modulus: PositiveFloat = 1e6
    poisson_ratio: PoissonRatio = 0.2
    friction_angle: PositiveFloat = 45.0
