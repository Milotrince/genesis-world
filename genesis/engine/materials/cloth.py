from genesis.typing import NonNegativeFloat, PositiveFloat

from .base import BaseMaterial, PoissonRatio


class Cloth(BaseMaterial):
    """A thin sheet that stretches and bends, such as fabric or paper.

    The finite element method (FEM) simulates it as a shell under the IPC coupler, and position-based dynamics (PBD)
    as a mesh of particles. PBD expresses stretching and bending through its own solver options, so it rejects an
    explicit Young's modulus, Poisson ratio or bending stiffness.

    Parameters
    ----------
    density : float | None, optional
        The mass density of the sheet, in kg/m^3, so that the mass per unit area is ``density * thickness``. If None,
        200 kg/m^3 under FEM, and 4.0 kg/m^2 of mass per unit area under PBD. Default is None.
    friction : float | None, optional
        The contact friction coefficient. If None, 0.1 under FEM and 0.15 under PBD. Default is None.
    youngs_modulus : float, optional
        The Young's modulus of the sheet, in Pa. A stiffer sheet stretches less. Default is 1e4.
    poisson_ratio : float, optional
        The Poisson ratio of the sheet, between -1 and 0.5. Default is 0.49.
    thickness : float, optional
        The thickness of the sheet, in m. It scales the mass per unit area given a density, and under FEM the
        resistance to bending as well, so PBD rejects an explicit value without a density. Default is 0.001.
    bending_stiffness : float | None, optional
        The resistance of the sheet to bending. If None, the sheet folds freely, like a thin fabric. Default is None.
    """

    friction: NonNegativeFloat | None = None
    youngs_modulus: PositiveFloat = 1e4
    poisson_ratio: PoissonRatio = 0.49
    thickness: PositiveFloat = 0.001
    bending_stiffness: NonNegativeFloat | None = None
