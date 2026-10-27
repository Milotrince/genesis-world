def lame_parameters(youngs_modulus: float, poisson_ratio: float) -> tuple[float, float]:
    """Return the shear modulus and the first Lame parameter of an isotropic elastic material."""
    mu = youngs_modulus / (2.0 * (1.0 + poisson_ratio))
    lam = youngs_modulus * poisson_ratio / ((1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio))
    return mu, lam
