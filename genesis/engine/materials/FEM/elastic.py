from typing import Any, ClassVar

from pydantic import model_validator

import genesis as gs

from .base import Base


class Elastic(Base):
    """
    The elastic material class for FEM.

    Parameters
    ----------
    E : float, optional
        Young's modulus, which controls stiffness. Default is 1e6.
    nu : float, optional
        Poisson ratio, describing the material's volume change under stress. Default is 0.2.
    rho : float, optional
        Material density (kg/m^3). Default is 1000.
    friction_mu : float, optional
        Friction coefficient. Default is 0.1.
    model, hydroelastic_modulus, contact_resistance : optional
        Deprecated, see ``gs.options.FEMEntitySolverOptions``.
    """

    _solver_options_names: ClassVar[tuple[str, ...]] = (*Base._solver_options_names, "model")

    model: gs.fem_model = gs.fem_model.linear

    @model_validator(mode="before")
    @classmethod
    def _resolve_deprecated_model(cls, data: Any) -> Any:
        if isinstance(data, dict) and data.get("model") == "stable_neohooken":
            gs.logger.warning("The 'stable_neohooken' model is deprecated. Use 'stable_neohookean' instead.")
            data["model"] = gs.fem_model.stable_neohookean
        return data
