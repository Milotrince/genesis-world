from dataclasses import dataclass

import numpy as np

import trimesh

import quadrants as qd

import genesis as gs
import genesis.utils.geom as gu
import genesis.utils.mesh as mu
from genesis.engine.entities.particle_entity import ParticleEntity, ParticleEntityDescription
from genesis.options.entities import PBDClothEntityOptions, PBDElasticEntityOptions


@dataclass(kw_only=True)
class PBDEntityDescription(ParticleEntityDescription):
    pass


@dataclass(kw_only=True)
class PBDMeshDescription(PBDEntityDescription):
    """The visual mesh of a cloth or an elastic entity, remeshed to the particle size, and the rest topology on it.

    A cloth holds the edges of its triangles and the inner edges its bending acts across, and an elastic entity holds
    the edges and the elements of its tetrahedralization.
    """

    mesh: gs.Mesh
    mass: float
    edges: np.ndarray
    edges_len_rest: np.ndarray
    inner_edges: np.ndarray
    inner_edges_len_rest: np.ndarray
    elems: np.ndarray
    elems_vol_rest: np.ndarray

    @classmethod
    def resolve(cls, morph, surface, particle_size, name, options):
        """Remesh the visual mesh of a cloth or an elastic entity placed at the morph pose, and build its topology.

        'options' are the entity options the other arguments were taken from, kept as authored in the description.
        """
        vmesh = cls.resolve_mesh(morph, surface)
        # The morph pose offset (e.g. an up-axis conversion) is composed onto the morph pose
        pos, quat = gu.transform_pos_quat_by_trans_quat(
            np.array(morph.offset_pos, dtype=gs.np_float),
            np.array(morph.offset_quat, dtype=gs.np_float),
            np.array(morph.pos, dtype=gs.np_float),
            np.array(morph.quat, dtype=gs.np_float),
        )
        vmesh.apply_transform(gu.trans_quat_to_T(pos, quat))
        mesh = vmesh.copy()
        mesh.remesh(edge_len_abs=particle_size, fix=isinstance(options, PBDElasticEntityOptions))
        inner_edges = np.zeros((0, 4), dtype=gs.np_int)
        inner_edges_len_rest = np.zeros((0,), dtype=gs.np_float)
        elems = np.zeros((0, 4), dtype=gs.np_int)
        elems_vol_rest = np.zeros((0,), dtype=gs.np_float)
        if isinstance(options, PBDClothEntityOptions):
            if vmesh.area < 1e-6:
                gs.raise_exception("Input mesh has zero surface area.")
            mass = vmesh.area * options.material_params.density
            particles = mesh.verts.astype(gs.np_float, copy=False)
            edges = mesh.get_unique_edges().astype(gs.np_int, copy=False)
            # Inner edges are two diagonal edges of each quadrilateral formed by adjacent face pairs
            adjacency, face_edges = trimesh.graph.face_adjacency(mesh=mesh.trimesh, return_edges=True)
            v3 = np.sum(mesh.faces[adjacency[:, 0]], axis=1) - face_edges[:, 0] - face_edges[:, 1]
            v4 = np.sum(mesh.faces[adjacency[:, 1]], axis=1) - face_edges[:, 0] - face_edges[:, 1]
            inner_edges = np.stack([face_edges[:, 0], face_edges[:, 1], v3, v4], axis=1, dtype=gs.np_int)
            inner_edges_len_rest = np.linalg.norm(particles[inner_edges[:, 2]] - particles[inner_edges[:, 3]], axis=1)
        else:
            if vmesh.volume < 1e-6:
                gs.raise_exception("Input mesh has zero volume.")
            mass = vmesh.volume * options.material_params.density
            particles, elems, *_ = mesh.tetrahedralize(mu.generate_tetgen_config_from_morph(morph))
            particles = particles.astype(gs.np_float, copy=False)
            elems = elems.astype(gs.np_int, copy=False)
            edge_i, edge_j = np.triu_indices(4, k=1)
            edges = np.stack((elems[:, edge_i], elems[:, edge_j]), axis=-1).reshape((-1, 2))
            edges = np.unique(np.sort(edges, axis=-1), axis=0)
            elems_vol_rest = np.linalg.det(particles[elems[:, 1:]] - particles[elems[:, :1]]) / 6.0
        edges_len_rest = np.linalg.norm(particles[edges[:, 0]] - particles[edges[:, 1]], axis=1)
        return cls(
            morph=morph,
            surface=vmesh.surface,
            name=name,
            particle_size=particle_size,
            has_skinning=True,
            sampler=None,
            init_positions=particles,
            origin=pos,
            vmesh=vmesh,
            vverts=vmesh.verts.astype(gs.np_float, copy=False),
            vfaces=vmesh.faces.astype(gs.np_int, copy=False),
            mesh_set_group_ids=None,
            mesh=mesh,
            mass=mass,
            edges=edges,
            edges_len_rest=edges_len_rest,
            inner_edges=inner_edges,
            inner_edges_len_rest=inner_edges_len_rest,
            elems=elems,
            elems_vol_rest=elems_vol_rest,
            options=options,
        )


class PBDBaseEntity(ParticleEntity):
    """
    Base class for PBD entity.
    """

    @property
    def desc(self) -> PBDEntityDescription:
        return self._desc

    def _add_to_solver(self):
        super()._add_to_solver()

        # Get UVs from vmesh (may be None if no texture)
        uvs = self._vmesh.uvs if self._vmesh is not None else None
        if uvs is None or len(uvs) != self.n_vverts:
            # No UVs available, pass empty array (kernel will skip copy)
            uvs = np.zeros((0, 2), dtype=gs.np_float)

        self._solver.add_uvs_and_faces(self._vvert_start, self._vface_start, uvs, self._vfaces)

    @gs.assert_built
    def fix_particles_to_link(self, link_idx, particles_idx_local=None, envs_idx=None):
        """
        Attach some particles to a rigid link, which carries them along until they are released.

        Parameters
        ----------
        link_idx : int
            Global index of the rigid link.
        particles_idx_local : int | array_like, shape ([M,] N)
            Index of the particles relative to this entity.
        envs_idx : None | int | array_like, shape (M,), optional
            The indices of the environments to set. If None, all environments will be set. Defaults to None.
        """
        particles_idx = self._get_global_idx(
            particles_idx_local, self._n_particles, self._particle_start, unsafe=True, batched_idx=True
        )
        self._solver.fix_particles_to_link(link_idx, particles_idx, envs_idx)

    @gs.assert_built
    def fix_particles(self, particles_idx_local=None, envs_idx=None, zero_velocity=True):
        """
        Fix the position of some particles in the simulation.

        Parameters
        ----------
        particles_idx_local : int | array_like, shape ([M,] N)
            Index of the particles relative to this entity.
        envs_idx : None | int | array_like, shape (M,), optional
            The indices of the environments to set. If None, all environments will be set. Defaults to None.
        zero_velocity : bool, optional
            Whether to zero the velocity of the particles. Defaults to True.
        """
        particles_idx = self._get_global_idx(
            particles_idx_local, self._n_particles, self._particle_start, unsafe=True, batched_idx=True
        )
        if zero_velocity:
            self._solver.set_particles_vel(0.0, particles_idx, envs_idx)
        self._solver.fix_particles(particles_idx, envs_idx)

    @gs.assert_built
    def release_particle(self, particles_idx_local=None, envs_idx=None):
        """
        Release some of the attached particles, allowing them to move freely again.

        Parameters
        ----------
        particles_idx_local : int | array_like, shape ([M,] N)
            Index of the particles relative to this entity.
        envs_idx : None | int | array_like, shape (M,), optional
            The indices of the environments to set. If None, all environments will be set. Defaults to None.
        """
        particles_idx = self._get_global_idx(
            particles_idx_local, self._n_particles, self._particle_start, unsafe=True, batched_idx=True
        )
        self._solver.release_particles(particles_idx, envs_idx)

    # ------------------------------------------------------------------------------------
    # --------------------------------- naming methods -----------------------------------
    # ------------------------------------------------------------------------------------

    def _get_morph_identifier(self) -> str:
        return f"pbd_{super()._get_morph_identifier()}"


@qd.data_oriented
class PBDTetEntity(PBDBaseEntity):
    """
    PBD entity represented by tetrahedral elements.

    Parameters
    ----------
    scene : Scene
        The simulation scene this entity is part of.
    solver : Solver
        The PBD solver instance managing this entity.
    desc : PBDMeshDescription
        The remeshed mesh and its rest topology, with the options, morph and surface they are resolved from.
    idx : int
        Unique index of this entity within the scene.
    particle_start : int
        Starting index of this entity's particles in the global particle buffer.
    edge_start : int
        Starting index of this entity's edges in the global edge buffer.
    vvert_start : int
        Starting index of this entity's visual vertices.
    vface_start : int
        Starting index of this entity's visual faces.
    """

    def __init__(
        self, scene, solver, desc: PBDMeshDescription, idx, particle_start, edge_start, vvert_start, vface_start
    ):
        super().__init__(scene, solver, desc, idx, particle_start, vvert_start, vface_start)
        self._edge_start = edge_start
        self._mesh = desc.mesh
        self._mass = desc.mass
        self._particle_mass = desc.mass / self.n_particles
        self._edges = desc.edges
        self._edges_len_rest = desc.edges_len_rest
        self._inner_edges = desc.inner_edges
        self._inner_edges_len_rest = desc.inner_edges_len_rest
        self._elems = desc.elems
        self._elems_vol_rest = desc.elems_vol_rest

    def _add_particles_to_solver(self):
        self._solver.add_particles(
            self._sim.cur_substep_local,
            self._particle_start,
            active=True,
            material_type=self._material_type,
            mass=float(self._particle_mass),
            rho_rest=0.0,
            mu_s=self._desc.options.material_params.friction,
            mu_k=self._desc.options.material_params.kinetic_friction,
            air_resistance=self._desc.options.solver_options.air_resistance
            if isinstance(self._desc.options, gs.options.PBDClothEntityOptions)
            else 0.0,
            density_relaxation=0.0,
            viscosity_relaxation=0.0,
            particles=self._particles,
        )
        self._solver.add_edges(
            self._edge_start,
            self._particle_start,
            self._desc.options.solver_options.stretch_compliance,
            self._desc.options.solver_options.stretch_relaxation,
            self._edges,
            self._edges_len_rest,
        )

    # ------------------------------------------------------------------------------------
    # ----------------------------------- properties -------------------------------------
    # ------------------------------------------------------------------------------------

    @property
    def desc(self) -> PBDMeshDescription:
        return self._desc

    @property
    def mesh(self):
        """Mesh."""
        return self._mesh

    @property
    def edges(self):
        """Edge array of the mesh."""
        return self._edges

    @property
    def n_edges(self):
        """Number of edges in the mesh."""
        return len(self._edges)

    @property
    def edge_start(self):
        """The starting index of the edges in the global solver."""
        return self._edge_start


@qd.data_oriented
class PBD2DEntity(PBDTetEntity):
    """
    PBD entity represented by a 2D mesh.

    Parameters
    ----------
    scene : Scene
        The simulation scene this entity is part of.
    solver : Solver
        The PBD solver instance managing this entity.
    desc : PBDMeshDescription
        The remeshed mesh and its rest topology, with the options, morph and surface they are resolved from.
    idx : int
        Unique index of this entity within the scene.
    particle_start : int
        Starting index of this entity's particles in the global particle buffer.
    edge_start : int
        Starting index of this entity's edges in the global edge buffer.
    inner_edge_start: int
        Starting index of this entity's inner edges in the global buffer.
    vvert_start : int
        Starting index of this entity's visual vertices.
    vface_start : int
        Starting index of this entity's visual faces.
    """

    def __init__(
        self,
        scene,
        solver,
        desc: PBDMeshDescription,
        idx,
        particle_start,
        edge_start,
        inner_edge_start,
        vvert_start,
        vface_start,
    ):
        super().__init__(scene, solver, desc, idx, particle_start, edge_start, vvert_start, vface_start)

        self._inner_edge_start = inner_edge_start
        self._material_type = int(self.solver.MATERIAL.CLOTH)

    def _add_particles_to_solver(self):
        super()._add_particles_to_solver()
        self._solver.add_inner_edges(
            self._inner_edge_start,
            self._particle_start,
            self._desc.options.solver_options.bending_compliance,
            self._desc.options.solver_options.bending_relaxation,
            self._inner_edges,
            self._inner_edges_len_rest,
        )

    @property
    def n_inner_edges(self):
        """The number of inner edges in the 2D mesh."""
        return len(self._inner_edges)

    @property
    def inner_edge_start(self):
        """The starting index of the inner edges in the global solver."""
        return self._inner_edge_start


@qd.data_oriented
class PBD3DEntity(PBDTetEntity):
    """
    PBD entity represented by a 3D mesh.

    Parameters
    ----------
    scene : Scene
        The simulation scene this entity is part of.
    solver : Solver
        The PBD solver instance managing this entity.
    desc : PBDMeshDescription
        The remeshed mesh and its rest topology, with the options, morph and surface they are resolved from.
    idx : int
        Unique index of this entity within the scene.
    particle_start : int
        Starting index of this entity's particles in the global particle buffer.
    edge_start : int
        Starting index of this entity's edges in the global edge buffer.
    elem_start: int
        Starting index of this entity's element in the global buffer.
    vvert_start : int
        Starting index of this entity's visual vertices.
    vface_start : int
        Starting index of this entity's visual faces.
    """

    def __init__(
        self,
        scene,
        solver,
        desc: PBDMeshDescription,
        idx,
        particle_start,
        edge_start,
        elem_start,
        vvert_start,
        vface_start,
    ):
        super().__init__(scene, solver, desc, idx, particle_start, edge_start, vvert_start, vface_start)

        self._elem_start = elem_start

        self._material_type = int(self.solver.MATERIAL.ELASTIC)

    def _add_particles_to_solver(self):
        super()._add_particles_to_solver()
        self._solver.add_elems(
            self._elem_start,
            self._particle_start,
            self._desc.options.solver_options.volume_compliance,
            self._desc.options.solver_options.volume_relaxation,
            self._elems,
            self._elems_vol_rest,
        )

    @property
    def n_elems(self):
        """The number of tetrahedral elements in the mesh."""
        return len(self._elems)

    @property
    def elem_start(self):
        """The starting index of the elements in the global solver."""
        return self._elem_start

    @property
    def elem_end(self):
        """The ending index of the elements in the global solver."""
        return self._elem_start + self.n_elems


@qd.data_oriented
class PBDParticleEntity(PBDBaseEntity):
    """
    PBD entity represented solely by particles.

    Parameters
    ----------
    scene : Scene
        The simulation scene this entity is part of.
    solver : Solver
        The PBD solver instance managing this entity.
    desc : PBDEntityDescription
        The resolved particles, with the options, morph and surface they are resolved from.
    idx : int
        Unique index of this entity within the scene.
    particle_start : int
        Starting index of this entity's particles in the global particle buffer.
    """

    def _add_particles_to_solver(self):
        self._solver.add_particles(
            self._sim.cur_substep_local,
            self._particle_start,
            active=self.active,
            material_type=int(self._solver.MATERIAL.LIQUID),
            mass=self._material.density,
            rho_rest=self._material.density,
            mu_s=0.0,
            mu_k=0.0,
            air_resistance=0.0,
            density_relaxation=self._desc.options.solver_options.density_relaxation,
            viscosity_relaxation=self._desc.options.solver_options.viscosity_relaxation,
            particles=self._particles,
        )

    @property
    def n_fluid_particles(self):
        """The number of fluid particles."""
        return self.n_particles


@qd.data_oriented
class PBDFreeParticleEntity(PBDBaseEntity):
    """
    PBD-based entity represented by non-physics particles

    Parameters
    ----------
    scene : Scene
        The simulation scene this entity is part of.
    solver : Solver
        The PBD solver instance managing this entity.
    desc : PBDEntityDescription
        The resolved particles, with the options, morph and surface they are resolved from.
    idx : int
        Unique index of this entity within the scene.
    particle_start : int
        Starting index of this entity's particles in the global particle buffer.
    """

    def _add_particles_to_solver(self):
        self._solver.add_particles(
            self._sim.cur_substep_local,
            self._particle_start,
            active=self.active,
            material_type=int(self._solver.MATERIAL.PARTICLE),
            mass=self._material.density,
            rho_rest=self._material.density,
            mu_s=0.0,
            mu_k=0.0,
            air_resistance=0.0,
            density_relaxation=0.0,
            viscosity_relaxation=0.0,
            particles=self._particles,
        )
