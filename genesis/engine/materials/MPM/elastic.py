from typing import Any, ClassVar

from pydantic import model_validator

import genesis as gs
from genesis.typing import PositiveFloat

from .base import Base


class Elastic(Base):
    """
    The elastic material class for MPM.

    Parameters
    ----------
    E : float, optional
        Young's modulus. Default is 3e5.
    nu : float, optional
        Poisson ratio. Default is 0.2.
    rho : float, optional
        Density (kg/m^3). Default is 1000.
    model, sampler : optional
        Deprecated, see ``gs.options.MPMEntitySolverOptions``.
    """

    _solver_options_names: ClassVar[tuple[str, ...]] = (*Base._solver_options_names, "model")

    E: PositiveFloat = 3e5
    model: gs.mpm_model = gs.mpm_model.corotation

    @model_validator(mode="before")
    @classmethod
    def _resolve_deprecated_model(cls, data: Any) -> Any:
        if isinstance(data, dict) and data.get("model") == "neohooken":
            gs.logger.warning("The 'neohooken' model is deprecated. Use 'neohookean' instead.")
            data["model"] = gs.mpm_model.neohookean
        return data
