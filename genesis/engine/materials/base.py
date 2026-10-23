from typing import Annotated

from pydantic import Field

from genesis.options.options import Options
from genesis.typing import PositiveFloat, ValidFloat


PoissonRatio = Annotated[ValidFloat, Field(gt=-1.0, lt=0.5)]


class BaseMaterial(Options):
    """The physical law of an entity and its parameters, whichever solver simulates it.

    A concrete subclass names one law, such as ``gs.materials.Elastic``. The entity options decide the solver, and each
    solver reads the parameters it supports. A parameter left to None takes the default of the solver simulating the
    entity, and the entity options raise an error on a parameter given explicitly that their solver cannot honour.

    Note
    ----
    This class should *not* be instantiated directly.

    Parameters
    ----------
    density : float | None, optional
        The mass density, in kg/m^3. If None, the solver decides.
    """

    density: PositiveFloat | None = None
