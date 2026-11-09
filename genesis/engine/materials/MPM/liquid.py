from pydantic import StrictBool

from .base import Base


class Liquid(Base):
    """
    The liquid material class for MPM.

    Parameters
    ----------
    E : float, optional
        Young's modulus. Default is 1e6.
    nu : float, optional
        Poisson ratio. Default is 0.2.
    rho : float, optional
        Density (kg/m^3). Default is 1000.
    viscous : bool, optional
        Whether the liquid is viscous. Simply sets mu to zero when non-viscous. Default is False.
    sampler : optional
        Deprecated, see ``gs.options.MPMEntitySolverOptions``.
    """

    viscous: StrictBool = False
