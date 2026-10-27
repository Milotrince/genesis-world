from typing import Annotated

from pydantic import Field

from genesis.typing import NonNegativeFloat, ValidFloat

from .base import BaseMaterial


class Material(BaseMaterial):
    """The material of a rigid entity, whose geometry never deforms.

    Parameters
    ----------
    density : float | None, optional
        The density used to estimate the mass of a link that states none, in kg/m^3. An explicit value replaces the
        densities an asset authors per geom as well. If None, the asset decides where it states a density, and
        otherwise 1000 kg/m^3 under MuJoCo compatibility (see ``RigidOptions.enable_mujoco_compatibility``), 1500
        kg/m^3 for an articulated robot and 600 kg/m^3 for a basic object such as a mug or a table. Default is None.
    friction : float | None, optional
        The sliding friction coefficient, which a rigid entity bounds (see ``RigidEntityOptions``). If None, the asset
        decides where it states one, and otherwise 1.0 for a rigid entity and 0 for a tool. Default is None.
    friction_torsional : float | None, optional
        The torsional friction coefficient, resisting the relative spin about the contact normal. Expressed in meters,
        as it stands for the radius of the contact patch over which sliding friction acts. Only effective when
        torsional friction is enabled (see ``RigidOptions.enable_torsional_friction``). If None, the asset decides
        where it states one (MJCF), and otherwise 0.005. Default is None.
    friction_rolling : float | None, optional
        The rolling friction coefficient, resisting the rolling about both contact tangent axes. Expressed in meters,
        like the torsional coefficient. Only effective when rolling friction is enabled (see
        ``RigidOptions.enable_rolling_friction``). If None, the asset decides where it states one (MJCF), and
        otherwise 0.0001. Default is None.
    """

    friction: NonNegativeFloat | None = None
    friction_torsional: Annotated[ValidFloat, Field(ge=0.0)] | None = None
    friction_rolling: Annotated[ValidFloat, Field(ge=0.0)] | None = None
