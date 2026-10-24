import dataclasses
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import torch

import genesis as gs
from genesis.engine.entities.rigid_entity.description import ContactParameters
from genesis.repr_base import RBC

if TYPE_CHECKING:
    from genesis.engine.solvers.rigid.rigid_solver import RigidSolver


@dataclasses.dataclass
class ContactTargetDescription:
    """Name one target of a contact pair: an entity, one of its links, or one collision geom of that link."""

    entity_name: str
    link_name: str | None = None
    # The index of the geom among the collision geoms of its link
    geom_idx_local: int | None = None


@dataclasses.dataclass
class ContactPairDescription:
    """Describe a contact pair declared on a scene, by the two targets it ties and the parameters it states."""

    targets: tuple[ContactTargetDescription, ContactTargetDescription]
    parameters: ContactParameters


class ContactPair(RBC):
    """The contact parameters stated for the contacts between the geoms of two targets (see 'Scene.add_contact_pair').

    A contact between a geom of one target and a geom of the other takes every coefficient the pair states. Where
    several pairs cover one contact, the pair whose two targets lie within those of the others states it.
    """

    def __init__(
        self, solver: "RigidSolver", idx: int, geoms_idx: tuple[frozenset[int], frozenset[int]], parameters, name
    ):
        self._solver = solver
        self._idx = idx
        self._geoms_idx = geoms_idx
        self._parameters: ContactParameters = parameters
        self._name: str = name

    def _assert_stated(self, i_component):
        if self._parameters.friction_coefficients[i_component] is None:
            gs.raise_exception(
                f"Contact pair {self._name} states no such friction coefficient. Declare it with 'add_contact_pair'."
            )

    def _set_friction(self, friction, envs_idx, i_component):
        self._assert_stated(i_component)
        friction = torch.as_tensor(friction, dtype=gs.tc_float, device=gs.device)
        self._solver.set_contact_pairs_friction(friction[..., None], [self._idx], envs_idx, i_component=i_component)

    def _get_friction(self, envs_idx, i_component):
        self._assert_stated(i_component)
        return self._solver.get_contact_pairs_friction(self._idx, envs_idx)[..., 0, i_component]

    @gs.assert_built
    def set_friction(self, friction, envs_idx=None):
        """
        Set the sliding friction coefficient of the contacts this pair governs, per environment.

        Parameters
        ----------
        friction : float | array_like
            The friction coefficient, broadcast over the environments.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.
        """
        self._set_friction(friction, envs_idx, i_component=0)

    @gs.assert_built
    def set_friction_torsional(self, friction_torsional, envs_idx=None):
        """Set the torsional friction coefficient of the contacts this pair governs, per environment."""
        self._set_friction(friction_torsional, envs_idx, i_component=1)

    @gs.assert_built
    def set_friction_rolling(self, friction_rolling, envs_idx=None):
        """Set the rolling friction coefficient of the contacts this pair governs, per environment."""
        self._set_friction(friction_rolling, envs_idx, i_component=2)

    @gs.assert_built
    def get_friction(self, envs_idx=None):
        """
        Get the sliding friction coefficient of the contacts this pair governs.

        Parameters
        ----------
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.

        Returns
        -------
        friction : torch.Tensor, shape () or (n_envs,)
            The friction coefficient.
        """
        return self._get_friction(envs_idx, i_component=0)

    @gs.assert_built
    def get_friction_torsional(self, envs_idx=None):
        """Get the torsional friction coefficient of the contacts this pair governs (see 'get_friction')."""
        return self._get_friction(envs_idx, i_component=1)

    @gs.assert_built
    def get_friction_rolling(self, envs_idx=None):
        """Get the rolling friction coefficient of the contacts this pair governs (see 'get_friction')."""
        return self._get_friction(envs_idx, i_component=2)

    @property
    def is_built(self) -> bool:
        """Whether the scene holding the pair is built."""
        return self._solver.is_built

    @property
    def idx(self) -> int:
        """The index of the pair among the contact pairs of the rigid solver."""
        return self._idx

    @property
    def parameters(self) -> ContactParameters:
        """The contact parameters the pair was declared with."""
        return self._parameters

    @property
    def geoms_idx(self) -> tuple[frozenset[int], frozenset[int]]:
        """The indices of the geoms of each of the two targets in the rigid solver."""
        return self._geoms_idx

    @property
    def name(self) -> str:
        """The targets of the pair, as error messages name them."""
        return self._name


class ContactClasses(NamedTuple):
    """What the contact pairs of a scene state, per class of geoms belonging to the same targets.

    'geoms_contact_class' gives the class of each geom. 'friction_pair_idx' gives, for each pair of classes, the index
    of the contact pair stating each of the sliding, torsional and rolling friction coefficients. 'is_collision_enabled'
    states whether the geoms of each pair of classes collide. Both hold -1 where no contact pair states the parameter.
    """

    geoms_contact_class: np.ndarray
    friction_pair_idx: np.ndarray
    is_collision_enabled: np.ndarray


def resolve_contact_pairs(pairs: list[ContactPair], n_geoms: int) -> ContactClasses:
    """Resolve which of the given contact pairs states each parameter for each pair of geom classes.

    Two geoms belonging to exactly the same targets share a class, so the table spans as many classes as there are
    distinct memberships, at most one more than the targets. For each parameter of each pair of classes, the pair
    stating it whose targets lie within those of every other pair stating it wins. A tie raises.
    """
    targets_idx = {geoms_idx: i_t for i_t, geoms_idx in enumerate(dict.fromkeys(g for p in pairs for g in p.geoms_idx))}
    pairs_targets = [tuple(targets_idx[geoms_idx] for geoms_idx in pair.geoms_idx) for pair in pairs]
    geoms_signature = [
        frozenset(i_t for geoms_idx, i_t in targets_idx.items() if i_g in geoms_idx) for i_g in range(n_geoms)
    ]
    signatures_class = {signature: i_c for i_c, signature in enumerate(dict.fromkeys(geoms_signature))}
    signatures = list(signatures_class)
    n_classes = len(signatures)

    friction_pair_idx = np.full((n_classes, n_classes, 3), -1, dtype=gs.np_int)
    is_collision_enabled = np.full((n_classes, n_classes), -1, dtype=gs.np_int)
    pairs_values = [(*pair.parameters.friction_coefficients, pair.parameters.is_collision_enabled) for pair in pairs]
    # Whether the two targets of each pair lie within the two targets of another, in either order
    pairs_is_within = [
        [
            (geoms_a <= other_a and geoms_b <= other_b) or (geoms_a <= other_b and geoms_b <= other_a)
            for other_a, other_b in (p.geoms_idx for p in pairs)
        ]
        for geoms_a, geoms_b in (p.geoms_idx for p in pairs)
    ]
    for i_ca, signature_a in enumerate(signatures):
        for i_cb in range(i_ca, n_classes):
            signature_b = signatures[i_cb]
            pairs_idx = [
                i_p
                for i_p, (i_ta, i_tb) in enumerate(pairs_targets)
                if (i_ta in signature_a and i_tb in signature_b) or (i_tb in signature_a and i_ta in signature_b)
            ]
            for i_value in range(4):
                stating_idx = [i_p for i_p in pairs_idx if pairs_values[i_p][i_value] is not None]
                winners_idx = [
                    i_p
                    for i_p in stating_idx
                    if not any(j_p != i_p and pairs_is_within[j_p][i_p] for j_p in stating_idx)
                ]
                if len(winners_idx) > 1:
                    gs.raise_exception(
                        f"Contact pairs {', '.join(pairs[i_p].name for i_p in winners_idx)} state the same parameter "
                        "for some contacts, and none of them lies within the others. Declare a pair between the "
                        "geoms they share to settle it."
                    )
                if not winners_idx:
                    continue
                (i_p,) = winners_idx
                if i_value < 3:
                    friction_pair_idx[i_ca, i_cb, i_value] = friction_pair_idx[i_cb, i_ca, i_value] = i_p
                else:
                    is_collision_enabled[i_ca, i_cb] = is_collision_enabled[i_cb, i_ca] = pairs_values[i_p][i_value]

    geoms_contact_class = np.array([signatures_class[signature] for signature in geoms_signature], dtype=gs.np_int)
    return ContactClasses(geoms_contact_class, friction_pair_idx, is_collision_enabled)
