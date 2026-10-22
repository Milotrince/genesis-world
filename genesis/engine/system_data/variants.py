"""Morph variants of the heterogeneous entities of a solver, and the variant each environment carries."""

from dataclasses import dataclass

from genesis.utils import array_class

from . import SystemData, SystemDataDescription


@dataclass(frozen=True, kw_only=True, eq=False)
class VariantsData(SystemData):
    info: array_class.VariantsInfo
    state: array_class.VariantsState


@dataclass(frozen=True, kw_only=True, eq=False)
class VariantsEntityData(SystemData):
    """The variants of one heterogeneous entity, the entry 'heterogeneous_idx' of the entity tables of 'data'."""

    data: VariantsData
    heterogeneous_idx: int
    variant_start: int
    n_variants: int


@dataclass(frozen=True, kw_only=True)
class VariantsDescription(SystemDataDescription[VariantsData]):
    """Counts summed over every variant of every heterogeneous entity, a zero count of joints, dofs or qs padded to
    one entry.

    A kinematic solver has neither collision geoms nor inertial values, so their tables hold one padding entry.
    """

    _B: int
    n_entities_: int
    n_variants_: int
    n_links_: int
    n_joints_: int
    n_dofs_: int
    n_qs_: int
    is_dynamic: bool

    def allocate(self, owner) -> VariantsData:
        return VariantsData(
            owner=owner,
            info=array_class.get_variants_info(self),
            state=array_class.get_variants_state(self),
        )
