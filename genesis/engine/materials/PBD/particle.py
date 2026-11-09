from typing import TYPE_CHECKING, ClassVar

import genesis as gs
from genesis.typing import PositiveFloat

from .base import Base

if TYPE_CHECKING:
    from genesis.engine.entities.pbd_entity import PBDFreeParticleEntity


class Particle(Base["PBDFreeParticleEntity"]):
    """
    Particle-based entities with no inter-particle interaction.

    Note
    ----
    This creates particle-based entities that have no inter-particle interaction at all; i.e. they are only affected
    by external forces. This is useful for creating particle-based animations. Hosted under the PBD system.

    Parameters
    ----------
    rho : float, optional
        The rest density. Default is 1000.0.
    sampler : optional
        Deprecated, see ``gs.options.PBDEntitySolverOptions``.
    """

    _solver_options_names: ClassVar[tuple[str, ...]] = ("sampler",)

    rho: PositiveFloat = 1000.0
    sampler: gs.particle_sampler = gs.particle_sampler.pbs
