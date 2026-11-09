from typing import Any

from pydantic import StrictBool, model_validator

import genesis as gs

from .base import SamplerType
from .elasto_plastic import ElastoPlastic


class Snow(ElastoPlastic):
    """
    The snow material class for MPM.

    Note
    ----
    Snow is a special type of ElastoPlastic that gets harder when compressed.
    It does not support von Mises yield criterion.

    Parameters
    ----------
    E : float, optional
        Young's modulus. Default is 1e6.
    nu : float, optional
        Poisson ratio. Default is 0.2.
    rho : float, optional
        Density (kg/m^3). Default is 1000.
    yield_lower : float, optional
        Lower bound of yield condition. Default is 2.5e-2.
    yield_higher : float, optional
        Upper bound of yield condition. Default is 4.5e-3.
    sampler : optional
        Deprecated, see ``gs.options.MPMEntitySolverOptions``.
    """

    sampler: SamplerType = "random"
    use_von_mises: StrictBool = False

    @model_validator(mode="before")
    @classmethod
    def _enforce_no_von_mises(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        if data.get("use_von_mises", False):
            gs.raise_exception("Snow does not support use_von_mises=True.")
        data["use_von_mises"] = False
        return data
