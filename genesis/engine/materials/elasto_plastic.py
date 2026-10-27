from typing import Literal

from pydantic import model_validator

import genesis as gs
from genesis.typing import NonNegativeFloat, PositiveFloat

from .base import BaseMaterial, PoissonRatio


class ElastoPlastic(BaseMaterial):
    """A solid that deforms elastically up to a yield limit and permanently beyond it, such as dough or clay.

    The material point method (MPM) simulates it.

    Parameters
    ----------
    density : float, optional
        The mass density, in kg/m^3. Default is 1000.
    youngs_modulus : float, optional
        The Young's modulus, in Pa. Default is 1e6.
    poisson_ratio : float, optional
        The Poisson ratio, between -1 and 0.5. Default is 0.2.
    yield_criterion : str, optional
        The law deciding when the solid yields. 'von_mises' yields once the shear stress exceeds 'yield_stress',
        whatever the pressure, as metals and dough do. 'stretch_clamp' yields once a principal stretch leaves the
        bounds 'critical_compression' and 'critical_stretch', as a crumbly material does. Each criterion rejects an
        explicit value of the parameters of the other. Default is 'von_mises'.
    yield_stress : float, optional
        The shear stress at which the solid yields under 'von_mises', in Pa. Default is 1e4.
    critical_compression : float, optional
        The relative compression at which the solid yields under 'stretch_clamp'. Default is 0.025.
    critical_stretch : float, optional
        The relative stretch at which the solid yields under 'stretch_clamp'. Default is 0.0045.
    """

    density: PositiveFloat = 1000.0
    youngs_modulus: PositiveFloat = 1e6
    poisson_ratio: PoissonRatio = 0.2
    yield_criterion: Literal["von_mises", "stretch_clamp"] = "von_mises"
    yield_stress: PositiveFloat = 1e4
    critical_compression: NonNegativeFloat = 2.5e-2
    critical_stretch: NonNegativeFloat = 4.5e-3

    @model_validator(mode="after")
    def _check_yield_parameters(self) -> "ElastoPlastic":
        if self.yield_criterion == "von_mises":
            names = sorted({"critical_compression", "critical_stretch"} & self.model_fields_set)
        else:
            names = sorted({"yield_stress"} & self.model_fields_set)
        if names:
            gs.raise_exception(f"Yield criterion '{self.yield_criterion}' ignores {', '.join(names)}.")
        return self
