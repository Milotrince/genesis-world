from typing import ClassVar

from genesis.typing import PositiveInt

from .elastic import Elastic


class Muscle(Elastic):
    """
    The muscle material class for FEM.

    Parameters
    ----------
    E : float, optional
        Young's modulus, which controls stiffness, and scales the active stress of the muscle. Default is 1e6.
    nu : float, optional
        Poisson ratio. Default is 0.2.
    rho : float, optional
        Material density (kg/m^3). Default is 1000.
    friction_mu : float, optional
        Contact friction coefficient. Default is 0.1.
    model, n_groups, hydroelastic_modulus, contact_resistance : optional
        Deprecated, see ``gs.options.FEMEntitySolverOptions``.
    """

    _solver_options_names: ClassVar[tuple[str, ...]] = (*Elastic._solver_options_names, "n_groups")

    n_groups: PositiveInt = 1
