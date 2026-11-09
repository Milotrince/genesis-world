from typing import ClassVar

import genesis as gs
from genesis.typing import PositiveFloat, PositiveInt

from .elastic import Elastic


class Muscle(Elastic):
    """
    The muscle material class for MPM.

    Parameters
    ----------
    E : float, optional
        Young's modulus. Default is 1e6. The active stress of the muscle scales with the Young's modulus of its Lame
        parameters, which is E unless 'mu' or 'lam' is given.
    nu : float, optional
        Poisson ratio. Default is 0.2.
    rho : float, optional
        Density (kg/m^3). Default is 1000.
    model, n_groups, sampler : optional
        Deprecated, see ``gs.options.MPMEntitySolverOptions``.
    """

    _solver_options_names: ClassVar[tuple[str, ...]] = (*Elastic._solver_options_names, "n_groups")

    E: PositiveFloat = 1e6
    model: gs.mpm_model = gs.mpm_model.neohookean
    n_groups: PositiveInt = 1
