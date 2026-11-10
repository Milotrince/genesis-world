from typing import Annotated, Any, ClassVar

import quadrants as qd
from pydantic import BeforeValidator

import genesis as gs

from genesis.typing import PositiveFloat

from .base import Base


# The stress model, which also takes the former spelling 'neohooken' of the Neo-Hookean model
ModelType = Annotated[
    gs.mpm_model, BeforeValidator(lambda model: gs.mpm_model.neohookean if model == "neohooken" else model)
]


@qd.data_oriented
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
        Density (kg/m³). Default is 1000.
    model : str, optional
        Stress model ('corotation', 'neohookean', also spelled 'neohooken'). Default is 'corotation'.
    """

    _solver_options_names: ClassVar[tuple[str, ...]] = (*Base._solver_options_names, "model")

    E: PositiveFloat = 3e5
    model: ModelType = gs.mpm_model.corotation

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)

        self.update_F_S_Jp = self._update_F_S_Jp_elastic
        if self.model == "corotation":
            self.update_stress = self._update_stress_corotation
            # corotation stress uses U @ V.T from SVD(F_tmp).
            self.needs_svd = True
        elif self.model == gs.mpm_model.neohookean:
            self.update_stress = self._update_stress_neohooken
            # neohooken stress only reads F_tmp and J=det(F_tmp); SVD can be skipped.
            self.needs_svd = False

    @qd.func
    def _update_F_S_Jp_elastic(self, J, F_tmp, U, S, V, Jp):
        F_new = F_tmp
        S_new = S
        Jp_new = Jp
        return F_new, S_new, Jp_new

    @qd.func
    def _update_stress_corotation(self, U, S, V, F_tmp, F_new, J, Jp, actu, m_dir):
        stress = 2 * self.mu * (F_new - U @ V.transpose()) @ F_new.transpose() + qd.Matrix.identity(
            gs.qd_float, 3
        ) * self.lam * J * (J - 1)
        return stress

    @qd.func
    def _update_stress_neohooken(self, U, S, V, F_tmp, F_new, J, Jp, actu, m_dir):
        stress = self.mu * (F_tmp @ F_tmp.transpose()) + qd.Matrix.identity(gs.qd_float, 3) * (
            self.lam * qd.log(J) - self.mu
        )
        return stress
