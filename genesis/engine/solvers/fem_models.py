"""Constitutive models of the finite element method (FEM) solver, bound from the material and the solver options."""

import quadrants as qd

import genesis as gs
import genesis.utils.geom as gu
from genesis.engine.materials.FEM import Cloth, Elastic, Muscle


@qd.data_oriented
class FEMModel:
    """The constitutive model of a material without volumetric response, whose physics the IPC coupler owns.

    The explicit solver compiles the stress of every model into its static loop over models, even one no element
    executes, so the stress of this model returns zero rather than raise, which quadrants cannot compile. The implicit
    solver takes 'hessian_product', the product of the Hessian of the energy density with a change of the deformation
    gradient, in place of the Hessian itself. 'pre_compute' computes the data of an element from its deformation
    gradient at the start of a substep, and 'pre_compute_grad' its reverse, returning the adjoint of that deformation
    gradient from the adjoint of the energy gradient at the end of the substep.

    Parameters
    ----------
    model : str | None
        The constitutive model its entity solver options choose, compiled in. None for a Cloth material.
    """

    def __init__(self, model: str | None) -> None:
        pass

    def build(self, fem_solver) -> None:
        pass

    @qd.func
    def pre_compute(self, i_e: int, i_b: int, J: float, F: qd.types.matrix(3, 3), m_dir: qd.types.vector(3)):
        pass

    @qd.func
    def pre_compute_grad(
        self,
        i_e: int,
        i_b: int,
        mu: float,
        lam: float,
        F_start: qd.types.matrix(3, 3),
        F: qd.types.matrix(3, 3),
        dF_damping: qd.types.matrix(3, 3),
        gradient_grad: qd.types.matrix(3, 3),
        actu: float,
        m_dir: qd.types.vector(3),
    ):
        return qd.Matrix.zero(gs.qd_float, 3, 3)

    @qd.func
    def update_stress(
        self,
        mu: float,
        lam: float,
        J: float,
        F: qd.types.matrix(3, 3),
        actu: float,
        m_dir: qd.types.vector(3),
    ):
        return qd.Matrix.zero(gs.qd_float, 3, 3)

    @qd.func
    def hessian_product(self, i_e: int, i_b: int, mu: float, lam: float, dF: qd.types.matrix(3, 3)):
        raise NotImplementedError

    @qd.func
    def compute_energy_gradient(
        self,
        i_e: int,
        i_b: int,
        mu: float,
        lam: float,
        J: float,
        F: qd.types.matrix(3, 3),
        actu: float,
        m_dir: qd.types.vector(3),
    ):
        raise NotImplementedError

    @qd.func
    def compute_energy(
        self,
        i_e: int,
        i_b: int,
        mu: float,
        lam: float,
        J: float,
        F: qd.types.matrix(3, 3),
        actu: float,
        m_dir: qd.types.vector(3),
    ):
        raise NotImplementedError


@qd.data_oriented
class FEMElasticModel(FEMModel):
    """The stress and energy of an Elastic material, under the constitutive model its entity solver options choose."""

    def __init__(self, model: str | None) -> None:
        super().__init__(model)
        # The rotation of each element under the linear corotated model, allocated once the element count is known
        self._R = None
        if model == gs.fem_model.linear:
            self.update_stress = self._update_stress_linear
            self.hessian_product = self._hessian_product_linear
            self.compute_energy_gradient = self._compute_energy_gradient_linear
            self.compute_energy = self._compute_energy_linear
        elif model == gs.fem_model.stable_neohookean:
            self.update_stress = self._update_stress_stable_neohookean
            self.hessian_product = self._hessian_product_stable_neohookean
            self.compute_energy_gradient = self._compute_energy_gradient_stable_neohookean
            self.compute_energy = self._compute_energy_stable_neohookean
        elif model == gs.fem_model.linear_corotated:
            self.build = self._build_linear_corotated
            self.pre_compute = self._pre_compute_linear_corotated
            self.pre_compute_grad = self._pre_compute_grad_linear_corotated
            self.update_stress = self._update_stress_linear_corotated
            self.hessian_product = self._hessian_product_linear_corotated
            self.compute_energy_gradient = self._compute_energy_gradient_linear_corotated
            self.compute_energy = self._compute_energy_linear_corotated

    def _build_linear_corotated(self, fem_solver) -> None:
        self._R = qd.field(dtype=gs.qd_mat3, shape=(fem_solver._B, fem_solver.n_elements))

    @qd.func
    def _pre_compute_linear_corotated(
        self, i_e: int, i_b: int, J: float, F: qd.types.matrix(3, 3), m_dir: qd.types.vector(3)
    ):
        U, S, V = gu.qd_svd(F)
        R = U @ V.transpose()
        self._R[i_b, i_e] = R

    @qd.func
    def _pre_compute_grad_linear_corotated(
        self,
        i_e: int,
        i_b: int,
        mu: float,
        lam: float,
        F_start: qd.types.matrix(3, 3),
        F: qd.types.matrix(3, 3),
        dF_damping: qd.types.matrix(3, 3),
        gradient_grad: qd.types.matrix(3, 3),
        actu: float,
        m_dir: qd.types.vector(3),
    ):
        # The adjoint of the rotation R, which enters the energy gradient 2 mu R E + lam tr(E) R with
        # E = sym(R^T F) - I and the damping stress mu D + mu R D^T R + lam (R : D) R with D = dF_damping.
        R = self._R[i_b, i_e]
        I_3 = qd.Matrix.identity(gs.qd_float, 3)
        E = 0.5 * (R.transpose() @ F + F.transpose() @ R) - I_3
        M = 0.5 * (R.transpose() @ gradient_grad + gradient_grad.transpose() @ R)
        gradient_dot_R = (gradient_grad * R).sum()
        R_grad = 2.0 * mu * (gradient_grad @ E + F @ M) + lam * (gradient_dot_R * F + E.trace() * gradient_grad)
        R_grad = R_grad + mu * (gradient_grad @ R.transpose() @ dF_damping + dF_damping @ R.transpose() @ gradient_grad)
        R_grad = R_grad + lam * (gradient_dot_R * dF_damping + (R * dF_damping).sum() * gradient_grad)

        # With F_start = U S V^T, a change dF turns the rotation R = U V^T by U W V^T, where W is skew with
        # W_ij = (A_ij - A_ji) / (S_ii + S_jj) and A = U^T dF V. The singular values are signed (see 'qd_svd'), so a
        # sum vanishes only for an element turned inside out, where the rotation is not differentiable: the adjoint
        # of the state then turns non-finite, which halts the backward pass (see 'StepOperation.backward').
        U, S, V = gu.qd_svd(F_start)
        K = U.transpose() @ R_grad @ V
        X = qd.Matrix.zero(gs.qd_float, 3, 3)
        for i_x, j_x in qd.static(qd.ndrange(3, 3)):
            if qd.static(i_x != j_x):
                X[i_x, j_x] = (K[i_x, j_x] - K[j_x, i_x]) / (S[i_x, i_x] + S[j_x, j_x])
        return U @ X @ V.transpose()

    # --- Linear model ---

    @qd.func
    def _update_stress_linear(
        self,
        mu: float,
        lam: float,
        J: float,
        F: qd.types.matrix(3, 3),
        actu: float,
        m_dir: qd.types.vector(3),
    ):
        I = qd.Matrix.identity(dt=gs.qd_float, n=3)
        stress = mu * (F + F.transpose() - 2 * I) + lam * (F - I).trace() * I
        return stress

    @qd.func
    def _hessian_product_linear(self, i_e: int, i_b: int, mu: float, lam: float, dF: qd.types.matrix(3, 3)):
        return mu * (dF + dF.transpose()) + lam * dF.trace() * qd.Matrix.identity(dt=gs.qd_float, n=3)

    @qd.func
    def _compute_energy_gradient_linear(
        self,
        i_e: int,
        i_b: int,
        mu: float,
        lam: float,
        J: float,
        F: qd.types.matrix(3, 3),
        actu: float,
        m_dir: qd.types.vector(3),
    ):
        I = qd.Matrix.identity(dt=gs.qd_float, n=3)
        eps = 0.5 * (F + F.transpose()) - I
        trEps = eps.trace()
        energy = mu * eps.norm_sqr() + 0.5 * lam * trEps**2
        gradient = 2.0 * mu * eps + lam * trEps * I
        return energy, gradient

    @qd.func
    def _compute_energy_linear(
        self,
        i_e: int,
        i_b: int,
        mu: float,
        lam: float,
        J: float,
        F: qd.types.matrix(3, 3),
        actu: float,
        m_dir: qd.types.vector(3),
    ):
        I = qd.Matrix.identity(dt=gs.qd_float, n=3)
        eps = 0.5 * (F + F.transpose()) - I
        trEps = eps.trace()
        energy = mu * eps.norm_sqr() + 0.5 * lam * trEps**2
        return energy

    # --- Stable Neo-Hookean model ---

    @qd.func
    def _update_stress_stable_neohookean(
        self,
        mu: float,
        lam: float,
        J: float,
        F: qd.types.matrix(3, 3),
        actu: float,
        m_dir: qd.types.vector(3),
    ):
        IC = F.norm_sqr()
        dJdF0 = F[:, 1].cross(F[:, 2])
        dJdF1 = F[:, 2].cross(F[:, 0])
        dJdF2 = F[:, 0].cross(F[:, 1])
        dJdF = qd.Matrix.cols([dJdF0, dJdF1, dJdF2])
        alpha = 1 + 0.75 * mu / lam
        stress = mu * (1 - 1 / (IC + 1)) * F + lam * (J - alpha) * dJdF
        return stress

    @qd.func
    def _hessian_product_stable_neohookean(self, i_e: int, i_b: int, mu: float, lam: float, dF: qd.types.matrix(3, 3)):
        raise NotImplementedError("Hessian computation is not implemented for stable_neohookean model.")

    @qd.func
    def _compute_energy_gradient_stable_neohookean(
        self,
        i_e: int,
        i_b: int,
        mu: float,
        lam: float,
        J: float,
        F: qd.types.matrix(3, 3),
        actu: float,
        m_dir: qd.types.vector(3),
    ):
        gs.raise_exception("Gradient computation is not implemented for stable_neohookean model.")

    @qd.func
    def _compute_energy_stable_neohookean(
        self,
        i_e: int,
        i_b: int,
        mu: float,
        lam: float,
        J: float,
        F: qd.types.matrix(3, 3),
        actu: float,
        m_dir: qd.types.vector(3),
    ):
        _lambda = lam + mu
        _alpha = 1.0 + mu / _lambda
        Ic = F.norm_sqr()
        Jminus1 = J - _alpha
        energy = 0.5 * (mu * (Ic - 3.0) + _lambda * Jminus1**2)
        return energy

    # --- Linear Corotated model ---

    @qd.func
    def _update_stress_linear_corotated(
        self,
        mu: float,
        lam: float,
        J: float,
        F: qd.types.matrix(3, 3),
        actu: float,
        m_dir: qd.types.vector(3),
    ):
        gs.raise_exception("Linear corotated stress update is not implemented yet.")

    @qd.func
    def _hessian_product_linear_corotated(self, i_e: int, i_b: int, mu: float, lam: float, dF: qd.types.matrix(3, 3)):
        # The Hessian depends on the rotation of the element alone, taken at the start of the substep (see pre_compute)
        R = self._R[i_b, i_e]
        return mu * dF + mu * R @ dF.transpose() @ R + lam * (R * dF).sum() * R

    @qd.func
    def _compute_energy_gradient_linear_corotated(
        self,
        i_e: int,
        i_b: int,
        mu: float,
        lam: float,
        J: float,
        F: qd.types.matrix(3, 3),
        actu: float,
        m_dir: qd.types.vector(3),
    ):
        F_hat = self._R[i_b, i_e].transpose() @ F
        eps = 0.5 * (F_hat + F_hat.transpose())
        for i in qd.static(range(3)):
            eps[i, i] -= 1.0
        trEps = eps.trace()
        energy = mu * eps.norm_sqr() + 0.5 * lam * trEps**2
        gradient = 2.0 * mu * self._R[i_b, i_e] @ eps + lam * trEps * self._R[i_b, i_e]
        return energy, gradient

    @qd.func
    def _compute_energy_linear_corotated(
        self,
        i_e: int,
        i_b: int,
        mu: float,
        lam: float,
        J: float,
        F: qd.types.matrix(3, 3),
        actu: float,
        m_dir: qd.types.vector(3),
    ):
        F_hat = self._R[i_b, i_e].transpose() @ F
        eps = 0.5 * (F_hat + F_hat.transpose())
        for i in qd.static(range(3)):
            eps[i, i] -= 1.0
        trEps = eps.trace()
        energy = mu * eps.norm_sqr() + 0.5 * lam * trEps**2
        return energy


@qd.data_oriented
class FEMMuscleModel(FEMElasticModel):
    """The model of a Muscle material: its elastic model plus an active stress along the fiber direction.

    The active stress scales with the Young's modulus of the material, recovered from the Lame parameters of the
    element. The implicit solver applies it along the direction of the fibers at the start of a substep, as the
    explicit one does, which leaves the energy quadratic in the positions at its end.
    """

    def __init__(self, model: str | None) -> None:
        super().__init__(model)
        # The small number added to the squared length of a fiber before taking its root, which keeps it positive, and
        # the direction of the fibers of each element under the implicit solver, both allocated at build
        self._fiber_length_eps = None
        self._fibers_dir = None
        self._update_stress_without_actuation = self.update_stress
        self.update_stress = self._update_stress_with_actuation
        self._build_without_actuation = self.build
        self.build = self._build_with_actuation
        self._pre_compute_without_actuation = self.pre_compute
        self.pre_compute = self._pre_compute_with_actuation
        self._pre_compute_grad_without_actuation = self.pre_compute_grad
        self.pre_compute_grad = self._pre_compute_grad_with_actuation
        self._compute_energy_gradient_without_actuation = self.compute_energy_gradient
        self.compute_energy_gradient = self._compute_energy_gradient_with_actuation
        self._compute_energy_without_actuation = self.compute_energy
        self.compute_energy = self._compute_energy_with_actuation

    def _build_with_actuation(self, fem_solver) -> None:
        self._build_without_actuation(fem_solver)
        self._fiber_length_eps = qd.field(dtype=gs.qd_float, shape=())
        self._fiber_length_eps[None] = 1e-12
        if fem_solver._use_implicit_solver:
            self._fibers_dir = qd.field(dtype=gs.qd_vec3, shape=(fem_solver._B, fem_solver.n_elements))

    @qd.func
    def _pre_compute_with_actuation(
        self, i_e: int, i_b: int, J: float, F: qd.types.matrix(3, 3), m_dir: qd.types.vector(3)
    ):
        self._pre_compute_without_actuation(i_e, i_b, J, F, m_dir)
        fiber = F @ m_dir
        self._fibers_dir[i_b, i_e] = fiber / fiber.norm(self._fiber_length_eps[None])

    @qd.func
    def _pre_compute_grad_with_actuation(
        self,
        i_e: int,
        i_b: int,
        mu: float,
        lam: float,
        F_start: qd.types.matrix(3, 3),
        F: qd.types.matrix(3, 3),
        dF_damping: qd.types.matrix(3, 3),
        gradient_grad: qd.types.matrix(3, 3),
        actu: float,
        m_dir: qd.types.vector(3),
    ):
        F_start_grad = self._pre_compute_grad_without_actuation(
            i_e, i_b, mu, lam, F_start, F, dF_damping, gradient_grad, actu, m_dir
        )
        # The fiber direction enters the energy gradient through the active stress youngs_modulus * actu * fiber_dir m^T
        youngs_modulus = mu * (3 * lam + 2 * mu) / (lam + mu)
        fiber_dir_grad = youngs_modulus * actu * gradient_grad @ m_dir
        fiber = F_start @ m_dir
        fiber_length = fiber.norm(self._fiber_length_eps[None])
        fiber_dir = fiber / fiber_length
        fiber_grad = (fiber_dir_grad - fiber_dir.dot(fiber_dir_grad) * fiber_dir) / fiber_length
        return F_start_grad + fiber_grad.outer_product(m_dir)

    @qd.func
    def _compute_energy_gradient_with_actuation(
        self,
        i_e: int,
        i_b: int,
        mu: float,
        lam: float,
        J: float,
        F: qd.types.matrix(3, 3),
        actu: float,
        m_dir: qd.types.vector(3),
    ):
        energy, gradient = self._compute_energy_gradient_without_actuation(i_e, i_b, mu, lam, J, F, actu, m_dir)
        youngs_modulus = mu * (3 * lam + 2 * mu) / (lam + mu)
        stress = youngs_modulus * actu * self._fibers_dir[i_b, i_e].outer_product(m_dir)
        return energy + (stress * F).sum(), gradient + stress

    @qd.func
    def _compute_energy_with_actuation(
        self,
        i_e: int,
        i_b: int,
        mu: float,
        lam: float,
        J: float,
        F: qd.types.matrix(3, 3),
        actu: float,
        m_dir: qd.types.vector(3),
    ):
        energy = self._compute_energy_without_actuation(i_e, i_b, mu, lam, J, F, actu, m_dir)
        youngs_modulus = mu * (3 * lam + 2 * mu) / (lam + mu)
        return energy + youngs_modulus * actu * self._fibers_dir[i_b, i_e].dot(F @ m_dir)

    @qd.func
    def _update_stress_with_actuation(
        self,
        mu: float,
        lam: float,
        J: float,
        F: qd.types.matrix(3, 3),
        actu: float,
        m_dir: qd.types.vector(3),
    ):
        stress = self._update_stress_without_actuation(mu, lam, J, F, actu, m_dir)

        l = (F @ m_dir).norm(self._fiber_length_eps[None])
        mmT = m_dir.outer_product(m_dir)
        youngs_modulus = mu * (3 * lam + 2 * mu) / (lam + mu)
        stress += youngs_modulus * (actu / l) * F @ mmT

        return stress


# The model class of each material, which compiles one model per constitutive model (see 'FEMModel')
FEM_MODEL_CLS_BY_MATERIAL = {Cloth: FEMModel, Elastic: FEMElasticModel, Muscle: FEMMuscleModel}
