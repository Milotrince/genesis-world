from genesis.typing import NonNegativeFloat, PositiveFloat

from .base import BaseMaterial, PoissonRatio


class Elastic(BaseMaterial):
    """A solid that recovers its rest shape once the load is removed, such as rubber or soft tissue.

    The finite element method (FEM), the material point method (MPM) and position-based dynamics (PBD) simulate it.
    PBD expresses stiffness through its own solver options, so it rejects an explicit Young's modulus or Poisson ratio.

    Parameters
    ----------
    density : float, optional
        The mass density, in kg/m^3. Default is 1000.
    friction : float | None, optional
        The contact friction coefficient. If None, 0.1 under FEM and 0.15 under PBD. MPM has no contact friction of its
        own and rejects an explicit value. Default is None.
    youngs_modulus : float | None, optional
        The Young's modulus, in Pa. A stiffer solid deforms less but needs a finer time step to stay stable. If None,
        1e6 under FEM and 3e5 under MPM. Default is None.
    poisson_ratio : float, optional
        The Poisson ratio, between -1 and 0.5. Close to 0.5, the solid keeps its volume, at the cost of a stiffer
        response that needs a finer time step. Default is 0.2.
    """

    density: PositiveFloat = 1000.0
    friction: NonNegativeFloat | None = None
    youngs_modulus: PositiveFloat | None = None
    poisson_ratio: PoissonRatio = 0.2
