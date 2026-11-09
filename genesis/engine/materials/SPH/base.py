from typing import TYPE_CHECKING, ClassVar

import genesis as gs

from ..base import Material

if TYPE_CHECKING:
    from genesis.engine.entities.sph_entity import SPHEntity


class Base(Material["SPHEntity"]):
    """
    The base class of SPH materials.

    Note
    ----
    This class should *not* be instantiated directly.

    Parameters
    ----------
    sampler : optional
        Deprecated, see ``gs.options.SPHEntitySolverOptions``.
    """

    _entity_options_name: ClassVar[str] = "gs.entities.SPH"
    _solver_options_names: ClassVar[tuple[str, ...]] = ("sampler",)

    sampler: gs.particle_sampler = gs.particle_sampler.pbs
