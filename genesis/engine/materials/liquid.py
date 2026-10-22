from genesis.typing import NonNegativeFloat, PositiveFloat

from .base import BaseMaterial


class Liquid(BaseMaterial):
    """A fluid that flows under any shear load, such as water or honey.

    The material point method (MPM), smoothed particle hydrodynamics (SPH) and position-based dynamics (PBD) simulate
    it. PBD keeps its liquid incompressible and expresses viscosity through its own solver options, so it rejects an
    explicit bulk modulus, viscosity or surface tension.

    Parameters
    ----------
    density : float, optional
        The rest density, in kg/m^3. Default is 1000.
    bulk_modulus : float | None, optional
        The resistance to compression, in Pa. A larger value keeps the volume closer to constant but needs a finer
        time step to stay stable. If None, about 2.8e5 Pa under MPM and 3.5e5 Pa under SPH. Default is None.
    viscosity : float | None, optional
        The dynamic viscosity, in Pa*s. A larger value damps the flow, as honey does. If None, 0 under MPM, which
        simulates an inviscid liquid, and 0.005 under SPH. Default is None.
    surface_tension : float | None, optional
        The surface tension coefficient, which pulls the free surface together. SPH alone simulates it. If None, 0.01
        under SPH. Default is None.
    """

    density: PositiveFloat = 1000.0
    bulk_modulus: PositiveFloat | None = None
    viscosity: NonNegativeFloat | None = None
    surface_tension: NonNegativeFloat | None = None
