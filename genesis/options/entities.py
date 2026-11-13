"""Typed options an entity is constructed from, each class deciding the kind of entity and the solver simulating it."""

import copy
from collections.abc import Mapping
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, Generic, Self, TypeVar

from pydantic import ConfigDict, Field, InstanceOf, StrictBool, model_validator

import genesis as gs
from genesis.engine import materials
from genesis.engine.materials.base import Material
from genesis.engine.materials.rigid import CoupType
from genesis.typing import FrozenDictType, NonNegativeFloat, PositiveFloat, StrictInt, ValidFloat

from .indices import GeomIndex, LinkIndex, VertexIndex, VGeomIndex
from .morphs import Morph
from .options import Options
from .solvers import FEMOptions, KinematicOptions, MPMOptions, PBDOptions, RigidOptions, SPHOptions
from .surfaces import Surface

if TYPE_CHECKING:
    from genesis.engine.entities import (
        Entity,
        FEMEntity,
        HybridEntity,
        KinematicEntity,
        MPMEntity,
        RigidEntity,
        SPHEntity,
    )
    from genesis.engine.entities.pbd_entity import PBDBaseEntity


EntityT = TypeVar("EntityT", bound="Entity")


def get_entity_options_cls(material: Material) -> type["BaseEntity"] | None:
    """Return the class of entity options of the solver simulating a material, or None if no solver simulates it.

    A material goes to the class of the nearest of its bases that the solvers of Genesis simulate, so a subclass a user
    declares goes where the material it derives from goes.
    """
    for material_cls in type(material).__mro__:
        options_cls = _OPTIONS_CLS_BY_MATERIAL_CLS.get(material_cls)
        if options_cls is not None:
            return options_cls
    return None


def _merge_material_fields(data: dict[str, Any], fields: dict[str, Any]) -> dict[str, Any]:
    """Return the fields given to entity options, completed by those the deprecated fields of their material state.

    'fields' are those of 'Material.get_entity_options_fields'. Giving one field two different values, one on each side,
    raises an error, while the same value twice is accepted, since entity options loaded from a scene file hold both.
    """
    if not fields:
        return data
    data = dict(data)
    for name, value in fields.items():
        if name == "solver_options":
            solver_options = data.get("solver_options") or {}
            if isinstance(solver_options, Options):
                solver_options = {n: getattr(solver_options, n) for n in solver_options.model_fields_set}
            for solver_name in value.keys() & solver_options.keys():
                if solver_options[solver_name] != value[solver_name]:
                    gs.raise_exception(f"'{solver_name}' is given both to the solver options and to the material.")
            data["solver_options"] = {**solver_options, **value}
        elif name == "links_solver_options":
            links = dict(data.get("links") or {})
            for links_name, flag_name in value:
                link_index = LinkIndex.named(*links_name)
                link_options = Link(solver_options=RigidLinkSolverOptions(**{flag_name: True}))
                if link_index in links:
                    link_options = Link.model_validate(links[link_index])
                    solver_options = link_options.solver_options or RigidLinkSolverOptions()
                    if getattr(solver_options, flag_name) is False:
                        gs.raise_exception(
                            f"'{flag_name}' of {link_index} is given both to the links and to the material."
                        )
                    solver_options = solver_options.model_copy(update={flag_name: True})
                    link_options = link_options.model_copy(update={"solver_options": solver_options})
                links[link_index] = link_options
            data["links"] = links
        elif name in data and data[name] != value:
            gs.raise_exception(f"'{name}' is given both to the entity options and to the material.")
        else:
            data[name] = value
    return data


class _FrozenOptions(Options):
    # Frozen, since assigning a field skips the validators
    model_config = ConfigDict(frozen=True)

    def model_copy(self, *, update: Mapping[str, Any] | None = None, deep: bool = False) -> Self:
        """Return a copy of these options with the fields 'update' gives, validated as a new instance would be.

        The fields given explicitly stay given, so the copy inherits from its context only what the original left
        unset. 'deep' copies the values held as well.
        """
        fields = {name: getattr(self, name) for name in self.model_fields_set}
        if deep:
            fields = copy.deepcopy(fields)
        return type(self)(**{**fields, **(update or {})})


class BaseEntity(_FrozenOptions, Generic[EntityT]):
    """Options an entity is constructed from, as passed to ``Scene.add_entity(options=...)``.

    A concrete subclass decides the kind of entity and the solver simulating it, and declares the materials it admits.
    The options are immutable: derive a variant with ``model_copy(update=...)``.

    Parameters
    ----------
    morph : Morph | tuple[Morph, ...]
        The geometry or asset of the entity. A sequence of morphs makes the entity heterogeneous: each parallel
        environment simulates one of them (see ``Scene.add_entity``). The options of a particle entity given to
        ``Scene.add_emitter`` leave it None, since the emitter sets it.
    surface : Surface | None, optional
        The appearance of the entity, its visualization mode included. If None, use ``gs.surfaces.Default()``.
    name : str | None, optional
        The name of the entity, unique within the scene. If None, a name is generated from the morph.
    """

    # Options class of the solver simulating the entity, None for a hybrid entity, whose parts are entities of their own
    _solver_options_cls: ClassVar[type[Options] | None] = None

    # Checked by instance alone, since validation would run the validators of 'Morph' on the sequence of morphs
    morph: InstanceOf[Morph] | Annotated[tuple[InstanceOf[Morph], ...], Field(strict=False)]
    surface: Surface | None = None
    name: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _check_material_solver(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        material = data.get("material")
        if isinstance(material, Material) and "material" in cls.model_fields:
            options_cls = get_entity_options_cls(material)
            if options_cls is None or not issubclass(cls, options_cls):
                options_name = "the entity options of its solver"
                if options_cls is not None:
                    options_name = f"gs.entities.{options_cls.__name__}"
                gs.raise_exception(
                    f"gs.entities.{cls.__name__} cannot take material {type(material).__name__} of another solver. "
                    f"Pass it to {options_name} instead."
                )
            data = _merge_material_fields(data, material.get_entity_options_fields())
        return data

    @property
    def morphs(self) -> tuple[Morph, ...]:
        """The morphs of the entity, one per variant of a heterogeneous entity and a single one otherwise."""
        return self.morph if isinstance(self.morph, tuple) else (self.morph,)


class _GeomFields(_FrozenOptions):
    rho: ValidFloat | None = None
    friction: Annotated[ValidFloat, Field(ge=0.01, le=5.0)] | None = None
    friction_torsional: Annotated[ValidFloat, Field(ge=0.0)] | None = None
    friction_rolling: Annotated[ValidFloat, Field(ge=0.0)] | None = None


class Geom(_GeomFields):
    """Options of the collision geoms of a rigid entity that a ``gs.GeomIndex`` selects.

    The fields given explicitly override those of the entity material for the selected geoms. Given in the 'geoms' of
    link options, they override the link options as well, while in the 'geoms' of entity options the link options
    override them. The fields left unset keep theirs.

    Parameters
    ----------
    rho : float | None, optional
        The density (kg/m^3). It estimates the mass and the inertia of a link the asset states none for, or of every
        moving link under the 'recompute_inertia' option of the morph. An aligned body is then simulated unaligned.
        The IPC coupler rejects it. If None, the geoms take the density of their link. Default is None.
    friction : float | None, optional
        The sliding friction coefficient of the contacts of the geoms (see ``gs.materials.Rigid.friction``). If None,
        the geoms take that of their link. Default is None.
    friction_torsional : float | None, optional
        The torsional friction coefficient (see ``gs.materials.Rigid.friction_torsional``). If None, the geoms take that
        of their link. Default is None.
    friction_rolling : float | None, optional
        The rolling friction coefficient (see ``gs.materials.Rigid.friction_rolling``). If None, the geoms take that of
        their link. Default is None.
    """


class RigidEntitySolverOptions(_FrozenOptions):
    """Options of the rigid solver specific to one rigid entity: its coupling with other solvers and its collision grid.

    Parameters
    ----------
    needs_coup : bool, optional
        Whether the links of the entity interact with the entities of other solvers, unless the solver options of a
        link override it (see ``RigidLinkSolverOptions``). Disabling it spares the cost of the coupling for an entity
        that never touches them. Default is True.
    coup_friction : float, optional
        The friction coefficient of the contacts with the particles of other solvers. Default is 0.1.
    coup_softness : float, optional
        The distance over which the entity pushes the particles of other solvers away, in m. A larger value holds the
        particles off more smoothly, at the cost of a visible gap. Default is 0.002.
    coup_restitution : float, optional
        The restitution coefficient of the contacts with the particles of other solvers, between 0 and 1. A non-zero
        value bounces particles off, at the cost of numerical stability. Default is 0.0.
    coup_type : str | None, optional
        How the IPC coupler simulates the entity, which requires a coupled link (see 'needs_coup').
        'two_way_soft_constraint' couples each link with a soft constraint, and alone couples some links of the entity
        without the others. 'external_articulation' couples the joint positions of a fixed-base articulated entity.
        'ipc_only' lets IPC simulate a non-articulated entity alone, gravity included. If None,
        'external_articulation' for a fixed-base articulated entity, 'two_way_soft_constraint' for a floating-base one
        and 'ipc_only' for a non-articulated one. Default is None.
    enable_coup_collision : bool, optional
        Whether the links of the entity collide in the IPC coupler, unless the solver options of a link override it.
        Disabling it keeps the coupling forces while dropping the contacts, whereas ``needs_coup=False`` leaves the
        entity out of the coupler. Default is True.
    contact_resistance : float | None, optional
        The contact stiffness of the entity in the IPC coupler. A stiffer contact lets less interpenetration through,
        at the cost of smaller time steps. If None, ``IPCCouplerOptions.contact_resistance``. Default is None.
    sdf_cell_size : float, optional
        The target cell size of the signed distance field (SDF) sampled around each collision geometry, in m. A finer
        grid resolves the contacts of non-convex geometry better, at the cost of memory and build time. Default is
        0.005.
    sdf_min_res : int, optional
        The minimum number of cells of the SDF along each axis, at least 16. Default is 32.
    sdf_max_res : int, optional
        The maximum number of cells of the SDF along each axis, at least 'sdf_min_res'. A primitive caps it at 32, which
        resolves its shape exactly. Default is 128.
    """

    needs_coup: StrictBool = True
    coup_friction: NonNegativeFloat = 0.1
    coup_softness: NonNegativeFloat = 0.002
    coup_restitution: Annotated[ValidFloat, Field(ge=0.0, le=1.0)] = 0.0
    coup_type: CoupType | None = None
    enable_coup_collision: StrictBool = True
    contact_resistance: PositiveFloat | None = None
    sdf_cell_size: PositiveFloat = 0.005
    sdf_min_res: Annotated[StrictInt, Field(ge=16)] = 32
    sdf_max_res: Annotated[StrictInt, Field(ge=16)] = 128

    @model_validator(mode="after")
    def _check_consistency(self) -> "RigidEntitySolverOptions":
        if self.sdf_min_res > self.sdf_max_res:
            gs.raise_exception("'sdf_min_res' must be smaller than or equal to 'sdf_max_res'.")
        if self.coup_restitution > gs.EPS:
            gs.logger.warning("Non-zero 'coup_restitution' could lead to instability. Use with caution.")
        return self


class RigidLinkSolverOptions(_FrozenOptions):
    """Options of the rigid solver specific to one link: its coupling with other solvers.

    A field left to None takes the value of the entity (see ``RigidEntitySolverOptions``).

    Parameters
    ----------
    needs_coup : bool | None, optional
        Whether the link interacts with the entities of other solvers. Under the IPC coupler, coupling some links of an
        entity without the others requires the 'two_way_soft_constraint' coupling. Default is None.
    enable_coup_collision : bool | None, optional
        Whether the link collides in the IPC coupler. Disabling it keeps the coupling forces of the link while dropping
        its contacts. Default is None.
    """

    needs_coup: StrictBool | None = None
    enable_coup_collision: StrictBool | None = None


class VGeom(_FrozenOptions):
    """Options of the visual geoms of a rigid or kinematic entity that a ``gs.VGeomIndex`` selects.

    Parameters
    ----------
    surface : gs.surfaces.Surface | None, optional
        The appearance of the selected visual geoms, which replaces that of their link and of the entity. The textures
        it leaves unset keep those the asset and the surface of the entity give the geoms. If None, the geoms take the
        surface of their link. Default is None.
    """

    surface: Surface | None = None


class Link(_GeomFields):
    """Options of the links of a rigid or kinematic entity that a ``gs.LinkIndex`` selects.

    The density and the friction coefficients given explicitly apply to every collision geom of the selected links,
    overriding those of the entity material and of the geom options of the entity. If None, the links keep those. A
    kinematic entity takes 'surface', 'vgeoms' and 'use_visual_raycasting' alone.

    Parameters
    ----------
    geoms : dict[gs.GeomIndex, gs.options.Geom], optional
        The options of collision geoms of each selected link, a ``gs.GeomIndex`` ordinal counting the collision geoms of
        that link alone (see ``gs.GeomIndex.at``). They override the options of the link. Default is no geom options.
    surface : gs.surfaces.Surface | None, optional
        The appearance of every visual geom of the selected links, which replaces that of the entity (see
        ``gs.options.VGeom.surface``). If None, the links take the surface of the entity. Default is None.
    vgeoms : dict[gs.VGeomIndex, gs.options.VGeom], optional
        The options of visual geoms of each selected link, a ``gs.VGeomIndex`` ordinal counting the visual geoms of that
        link alone, as its asset splits them (see ``gs.GeomIndex.at``). They override the options of the link. Default
        is no visual geom options.
    solver_options : gs.options.RigidLinkSolverOptions | None, optional
        The options of the rigid solver specific to the selected links. If None, the links take those of the entity.
        Default is None.
    use_visual_raycasting : bool | None, optional
        Whether to expose the visual geoms of the selected links to visual raycasting (see
        ``gs.entities.Kinematic.use_visual_raycasting``). If None, the links take the value of the entity. Default is
        None.
    """

    geoms: FrozenDictType[GeomIndex, Geom] = {}
    surface: Surface | None = None
    vgeoms: FrozenDictType[VGeomIndex, VGeom] = {}
    solver_options: RigidLinkSolverOptions | None = None
    use_visual_raycasting: StrictBool | None = None


class _ArticulatedEntity(BaseEntity[EntityT]):
    """Options shared by the entities made of links connected by joints, rigid and kinematic alike.

    See ``gs.entities.Kinematic`` for the parameters they add to ``gs.entities.BaseEntity``.
    """

    links: FrozenDictType[LinkIndex, Link] = {}
    vgeoms: FrozenDictType[VGeomIndex, VGeom] = {}
    use_visual_raycasting: StrictBool = False


class Rigid(_ArticulatedEntity["RigidEntity"]):
    """Options for an entity simulated with rigid-body dynamics.

    See ``gs.entities.Kinematic`` for the 'links' and 'vgeoms' parameters.

    Parameters
    ----------
    material : gs.materials.Rigid, optional
        The material of the entity. Defaults to ``gs.materials.Rigid()``.
    visualize_contact : bool, optional
        Whether to draw the contact forces applied to the entity as arrows in the viewer and rendered images. Each
        arrow is drawn every frame, so reserve it for the entities being inspected. Default is False.
    geoms : dict[gs.GeomIndex, gs.options.Geom], optional
        The options of the collision geoms each key selects among the collision geoms of the whole entity, by ordinal
        (see ``gs.GeomIndex.at``) or by name in any of its links, raising if it selects no geom. They override the
        material of the entity, and the options of the links override them in turn. Where two keys give one field of
        one geom, the first key applies, with a warning. Default is no geom options.
    solver_options : gs.options.RigidEntitySolverOptions, optional
        The options of the rigid solver specific to the entity. Defaults to ``RigidEntitySolverOptions()``.
    gravity_compensation : float, optional
        The fraction of gravity cancelled on the entity, 1.0 letting it float. An entity that IPC simulates alone
        (``coup_type='ipc_only'``) falls under the gravity of IPC instead, so it requires 0.0. Default is 0.0.
    """

    _solver_options_cls = RigidOptions

    material: materials.Rigid = Field(default_factory=materials.Rigid)
    visualize_contact: StrictBool = False
    geoms: FrozenDictType[GeomIndex, Geom] = {}
    solver_options: RigidEntitySolverOptions = Field(default_factory=RigidEntitySolverOptions)
    gravity_compensation: ValidFloat = 0.0

    @model_validator(mode="after")
    def _check_coupling(self) -> "Rigid":
        solver_options = self.solver_options
        if solver_options.coup_type == "ipc_only" and abs(self.gravity_compensation) > gs.EPS:
            gs.raise_exception("'gravity_compensation' must be 0 with coup_type='ipc_only', since IPC applies gravity.")
        links_needs_coup = [
            link_options.solver_options.needs_coup
            for link_options in self.links.values()
            if link_options.solver_options is not None and link_options.solver_options.needs_coup is not None
        ]
        if solver_options.coup_type is not None and not (solver_options.needs_coup or any(links_needs_coup)):
            gs.raise_exception(f"'coup_type' requires a coupled link, got coup_type={solver_options.coup_type!r}.")
        if any(needs_coup != solver_options.needs_coup for needs_coup in links_needs_coup) and (
            solver_options.coup_type not in (None, "two_way_soft_constraint")
        ):
            gs.raise_exception(
                "Coupling some links of an entity without the others requires coup_type='two_way_soft_constraint', "
                f"got coup_type={solver_options.coup_type!r}."
            )
        return self


class Kinematic(_ArticulatedEntity["KinematicEntity"]):
    """Options for an entity simulated with kinematics only, with no collision.

    Parameters
    ----------
    material : gs.materials.Kinematic, optional
        The material of the entity. Defaults to ``gs.materials.Kinematic()``.
    links : dict[gs.LinkIndex, gs.options.Link], optional
        The options of the links each key selects, which override those of the entity and, on a rigid entity, its
        material. A link the asset parser merges into its parent has no name of its own, so keep it with the
        'links_to_keep' option of the morph to select it. Where two keys give one field of one link, the first key
        applies, with a warning. A kinematic entity takes the 'surface', 'vgeoms' and 'use_visual_raycasting' of its
        link options alone. Default is no link options.
    vgeoms : dict[gs.VGeomIndex, gs.options.VGeom], optional
        The options of the visual geoms each key selects among the visual geoms of the whole entity, by ordinal (see
        ``gs.GeomIndex.at``) or by name in any of its links, raising if it selects no visual geom. They override the
        surface of the entity, and the options of the links override them in turn. Default is no visual geom options.
    use_visual_raycasting : bool, optional
        Whether to expose the visual geoms of the entity to every consumer that casts rays against visual geometry:
        raycaster sensors, and the viewer plugins built on ``RaycasterViewerPlugin`` with 'use_visual_geom' set. Each
        such consumer then scans the visual triangles of the entity again as it moves, so reserve it for the entities
        meant to be sensed or picked. Link options override it per link. Default is False.
    """

    _solver_options_cls = KinematicOptions

    material: materials.Kinematic = Field(default_factory=materials.Kinematic)

    @model_validator(mode="after")
    def _check_links(self) -> "Kinematic":
        for link_options in self.links.values():
            if (
                link_options.model_dump(include=set(_GeomFields.model_fields), exclude_none=True)
                or link_options.geoms
                or link_options.solver_options is not None
            ):
                gs.raise_exception(
                    "gs.entities.Kinematic cannot honour the density, the friction, the geoms and the solver options of "
                    "link options, which take no physics."
                )
        return self


class Vertex(_FrozenOptions):
    """Options of the vertices of a deformable entity that a ``gs.VertexIndex`` selects.

    Parameters
    ----------
    is_fixed : bool, optional
        Whether the vertices stay at their initial positions, as a hard constraint the solver enforces at every
        substep. It holds an entity in place by a few vertices, such as a soft pad clamped by one face. The vertices are
        vertex constraints from the build on, which ``FEMEntity.remove_vertex_constraints`` releases like any other. The
        IPC coupler does not support it, so a Cloth material rejects it. Default is False.
    """

    is_fixed: StrictBool = False


class FEM(BaseEntity["FEMEntity"]):
    """Options for an entity simulated with the finite element method (FEM).

    Parameters
    ----------
    material : gs.materials.FEM.Base
        The material of the entity, one of ``gs.materials.FEM.Elastic``, ``Cloth`` or ``Muscle``.
    vertices : dict[gs.VertexIndex, gs.options.Vertex], optional
        The options of the vertices each key selects among the vertices of the volume mesh. Where several keys give
        one field of one vertex, the first key applies, with a warning. A Cloth material takes none. Default is no
        vertex options.
    """

    _solver_options_cls = FEMOptions

    material: materials.FEM.Base
    vertices: FrozenDictType[VertexIndex, Vertex] = {}

    @model_validator(mode="after")
    def _check_vertices(self) -> "FEM":
        if isinstance(self.material, materials.FEM.Cloth) and self.vertices:
            gs.raise_exception("gs.entities.FEM cannot honour 'vertices' for a Cloth material.")
        return self


class _EmittableParticles(BaseEntity[EntityT]):
    """Options of the particle entities an emitter can hold, whose morph ``Scene.add_emitter`` sets."""

    morph: InstanceOf[Morph] | Annotated[tuple[InstanceOf[Morph], ...], Field(strict=False)] | None = None


class MPM(_EmittableParticles["MPMEntity"]):
    """Options for an entity simulated with the material point method (MPM).

    Parameters
    ----------
    material : gs.materials.MPM.Base
        The material of the entity, one of ``gs.materials.MPM.Elastic``, ``ElastoPlastic``, ``Liquid``, ``Sand``,
        ``Snow`` or ``Muscle``.
    """

    _solver_options_cls = MPMOptions

    material: materials.MPM.Base


class SPH(_EmittableParticles["SPHEntity"]):
    """Options for an entity simulated with smoothed particle hydrodynamics (SPH).

    Parameters
    ----------
    material : gs.materials.SPH.Base, optional
        The material of the liquid. Defaults to ``gs.materials.SPH.Liquid()``.
    """

    _solver_options_cls = SPHOptions

    material: materials.SPH.Base = Field(default_factory=materials.SPH.Liquid)


class PBD(_EmittableParticles["PBDBaseEntity"]):
    """Options for an entity simulated with position-based dynamics (PBD).

    Parameters
    ----------
    material : gs.materials.PBD.Base
        The material of the entity, one of ``gs.materials.PBD.Cloth``, ``Elastic``, ``Liquid`` or ``Particle``.
    """

    _solver_options_cls = PBDOptions

    material: materials.PBD.Base


class Hybrid(BaseEntity["HybridEntity"]):
    """Options for an entity simulated with a rigid skeleton inside a material point method (MPM) skin.

    Parameters
    ----------
    material : gs.materials.Hybrid
        The material of the skeleton and of the skin, and how they are coupled.
    """

    material: materials.Hybrid


# The class of entity options each material class goes to, a material taking that of the nearest of its bases
_OPTIONS_CLS_BY_MATERIAL_CLS: dict[type[Material], type[BaseEntity]] = {
    materials.Rigid: Rigid,
    materials.Kinematic: Kinematic,
    materials.FEM.Base: FEM,
    materials.MPM.Base: MPM,
    materials.SPH.Base: SPH,
    materials.PBD.Base: PBD,
    materials.Hybrid: Hybrid,
}
