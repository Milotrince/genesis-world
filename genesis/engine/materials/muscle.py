from genesis.typing import NonNegativeFloat, PositiveFloat

from .base import BaseMaterial, PoissonRatio


class Muscle(BaseMaterial):
    """An elastic solid that also contracts along a fiber direction when actuated.

    The finite element method (FEM) and the material point method (MPM) simulate it. The entity groups its elements
    or particles into muscle groups (see the 'n_groups' solver option of its entity options), each actuated on its own.

    Parameters
    ----------
    density : float, optional
        The mass density, in kg/m^3. Default is 1000.
    friction : float | None, optional
        The contact friction coefficient. If None, 0.1 under FEM. MPM has no contact friction of its own and rejects an
        explicit value. Default is None.
    youngs_modulus : float, optional
        The Young's modulus, in Pa. It scales the active contraction as well. Default is 1e6.
    poisson_ratio : float, optional
        The Poisson ratio, between -1 and 0.5. Default is 0.2.
    """

    density: PositiveFloat = 1000.0
    friction: NonNegativeFloat | None = None
    youngs_modulus: PositiveFloat = 1e6
    poisson_ratio: PoissonRatio = 0.2
