import enum

from pydantic_core import core_schema

URDF_FORMAT = ".urdf"
XACRO_FORMAT = ".xacro"
MJCF_FORMAT = ".xml"
# Root tags identifying the format of inline XML content passed as 'FileMorph.file'.
XML_ROOT_TAG_TO_FORMAT = {"mujoco": MJCF_FORMAT, "robot": URDF_FORMAT}
GLTF_FORMATS = (".glb", ".gltf")
MESH_FORMATS = (".obj", ".stl", ".dae", *GLTF_FORMATS)
USD_FORMATS = (".usd", ".usda", ".usdc", ".usdz")

# dynamic loading
ACTIVE = 1
INACTIVE = 0


class IntEnum(enum.IntEnum):
    def __repr__(self):
        return f"<gs.{self.__class__.__name__}.{self.name}: {self.value}>"

    def __format__(self, format_spec):
        return f"<{self.name}: {self.value}>"


class StrEnum(enum.StrEnum):
    """Named choice that options and public functions also accept spelled as its value."""

    def __repr__(self):
        return f"<gs.{self.__class__.__name__}.{self.name}: '{self.value}'>"

    @classmethod
    def _missing_(cls, value):
        choices = ", ".join(f"'{member.value}'" for member in cls)
        raise ValueError(f"Input should be one of {choices}")

    @classmethod
    def __get_pydantic_core_schema__(cls, source, handler):
        # Options validate in strict mode, which accepts members alone. Converting first also accepts their values.
        return core_schema.no_info_before_validator_function(cls, handler(source))


# geom type in rigid solver
class GEOM_TYPE(IntEnum):
    # Beware PLANE must be the first geometry type as this is assumed by MPR collision detection.
    PLANE = 0
    SPHERE = 1
    ELLIPSOID = 2
    CYLINDER = 3
    CAPSULE = 4
    BOX = 5
    MESH = 6
    TERRAIN = 7


# joint type in rigid solver, ranked by number of dofs
class JOINT_TYPE(IntEnum):
    FIXED = 0
    REVOLUTE = 1
    PRISMATIC = 2
    SPHERICAL = 3
    FREE = 4


class EQUALITY_TYPE(IntEnum):
    CONNECT = 0
    WELD = 1
    JOINT = 2
    SCREW = 3


class CTRL_MODE(IntEnum):
    POSITION = 0
    VELOCITY = 1
    FORCE = 2


######### User accessible constants do not capitalize #########
# rigid solver intergrator
class integrator(IntEnum):
    Euler = 0
    implicitfast = 1
    approximate_implicitfast = 2


# rigid solver constraint solver
class constraint_solver(IntEnum):
    CG = 0
    Newton = 1


# rigid solver contact friction cone
class friction_cone(IntEnum):
    """
    Contact friction cone model, trading numerical robustness for physical accuracy.

    'pyramidal' (the default) approximates the friction cone by a pyramid: robust and easy to solve, but the
    approximation makes friction anisotropic (the effective limit depends on the sliding direction). 'elliptic' is
    the exact cone: friction is isotropic and bounded by its true Euclidean limit sqrt(f_t1^2 + f_t2^2) <= mu * f_n
    in every direction, and with a high 'impratio' it holds resting stacks without the slow tangential creep of
    regularized friction, in return for being harder to solve and more sensitive numerically. Prefer pyramidal for
    robustness; choose elliptic when isotropic friction or firm static friction matters - e.g. objects that must stay
    put at rest instead of slowly creeping.
    """

    pyramidal = 0
    elliptic = 1


# rigid solver contact resolution
class contact_resolution(IntEnum):
    """
    How a contact's normal force and friction force are resolved against each other.

    'convex' poses the whole contact as a single smooth convex cost and lets the solver trade the normal residual
    against the tangential one. Because the friction limit mu * f_n bounds the pair jointly, a contact sliding fast
    enough that its friction rows demand more force than the cone allows can be answered by raising f_n instead: a body
    launched horizontally then lifts off a flat floor, by more the faster it slides. In exchange the whole problem stays
    one convex program, which converges predictably on stiff articulated chains and high mass ratios.

    'signorini' bounds friction against the normal force the contact has actually developed, so that force is set by the
    contact's own normal state rather than by tangential demand, and sliding can never inflate it - a sliding body
    decelerates at mu * g and stays down at any speed. Contacts are resolved by successive approximation, costing extra
    solver iterations and giving up the single-convex-program guarantee. Prefer it whenever sliding contact matters;
    choose 'convex' for parity with engines built on that formulation, or if a stiff scene converges better under it.

    'signorini' requires the elliptic friction cone, whose rows separate into a normal row and a friction disc - the
    pyramidal cone mixes the normal direction into every row and admits no such split - and the Newton constraint
    solver, the only one that reaches the fixed point of the resulting successive approximation. It implements the
    Coulomb complementarity problem eq. (C.22) of Alexis Duburcq, "Learning and Optimization of the Locomotion with an
    Exoskeleton for Paraplegic People", PhD thesis, Universite Paris Sciences et Lettres, 2022 (HAL tel-04166955),
    Appendix C, whose Signorini condition is what forbids the normal force from absorbing tangential demand.
    """

    convex = 0
    signorini = 1


# rigid solver broadphase traversal strategy
class broadphase_traversal(IntEnum):
    """
    Strategy for broad-phase collision detection in the rigid solver.

    Broad-phase quickly eliminates geometry pairs that cannot collide before the more expensive narrow-phase runs.

    At init time, geometry pairs that can never collide are filtered out (same-link, fixed-vs-fixed, contype/conaffinity
    mismatch, etc.), producing a list of *valid pairs*. The geometries of a link keep their pose in the link, so the
    links are swept first, each with the box bounding all its geometries, and the geometry pairs are only tested under
    the link pairs whose boxes overlap. The two strategies differ in how they sweep the links each step:

    Attributes
    ----------
    SAP : int
        Sweep-and-prune. Sorts the link boxes along one axis then only checks the link pairs that overlap on that axis.
        The sort and sweep are single-threaded per environment, which uses GPU cores poorly when there are few
        environments. The boxes stay nearly sorted from one step to the next, so the sort usually runs in near-linear
        time, up to O(n_links^2) after a large reordering, and the sweep checks the k link pairs overlapping along
        the axis.
    ALL_VS_ALL : int
        Checks every valid link pair every step, in parallel across GPU threads. The cost per step grows with the number
        of valid link pairs, which grows quadratically with the number of independent bodies in the scene. With
        heterogeneous entities, every environment also checks the link pairs of the variants it does not carry.

    Notes
    -----
    ``RigidOptions.broadphase_traversal`` defaults to ``None``, which lets the solver choose automatically:

    - **CPU backend** -> ``SAP`` (sequential sweep is efficient on CPU).
    - **GPU backend** -> ``ALL_VS_ALL`` (parallel pair checking is faster).
    - **GPU with heterogeneous entities** -> ``SAP`` (the links of the variants an environment does not carry have
      empty boxes, which yield no candidate pair).
    """

    SAP = 0
    ALL_VS_ALL = 1


class link_ref_frame(IntEnum):
    """
    Reference frame at which a per-link quantity is expressed.

    Each member fixes an origin, about which spatial quantities are translated, together with an orientation, by which
    quantities given in local coordinates are rotated. 'root_COM' is the center of mass of the whole kinematic tree the
    link belongs to, with world-aligned axes; the solver stores link velocities and generalized forces there natively,
    so it is the only frame reached without a moment arm. 'link_COM' is the center of mass of the link, with the axes of
    its inertial frame. 'link_origin' is the origin of the link, with the axes of the link frame.
    """

    root_COM = 0
    link_COM = 1
    link_origin = 2


# backend
class backend(IntEnum):
    cpu = 0
    gpu = 1
    cuda = 2
    amdgpu = 3
    metal = 4

    def __format__(self, format_spec):
        return f"gs.{self.name}"


# image types for visualization
class IMAGE_TYPE(StrEnum):
    RGB = "rgb"
    DEPTH = "depth"
    SEGMENTATION = "segmentation"
    NORMAL = "normal"

    def __format__(self, format_spec):
        return self.name


# parallelize
class PARA_LEVEL(IntEnum):
    NEVER = 0  # when using cpu
    PARTIAL = 1  # when using gpu for non-batched scene
    ALL = 2  # when using gpu for batched scene


# floating-point precision of the simulation
class float_precision(StrEnum):
    fp32 = "32"
    fp64 = "64"


# surface maps a dynamic texture binds to
class texture_map_type(StrEnum):
    diffuse = enum.auto()
    emissive = enum.auto()


# color encoding of a texture
class texture_encoding(StrEnum):
    srgb = enum.auto()
    linear = enum.auto()


# how a surface draws its entity
class vis_mode(StrEnum):
    visual = enum.auto()
    collision = enum.auto()
    particle = enum.auto()
    sdf = enum.auto()
    recon = enum.auto()


# surface reconstruction of particles
class recon_backend(StrEnum):
    splashsurf = enum.auto()
    levelset = enum.auto()


class metal_type(StrEnum):
    aluminium = enum.auto()
    gold = enum.auto()
    copper = enum.auto()
    brass = enum.auto()
    iron = enum.auto()
    titanium = enum.auto()
    vanadium = enum.auto()
    lithium = enum.auto()


class light_type(StrEnum):
    directional = enum.auto()
    point = enum.auto()
    ambient = enum.auto()


# what one segmentation index labels
class segmentation_level(StrEnum):
    entity = enum.auto()
    link = enum.auto()
    geom = enum.auto()


class render_particle_as(StrEnum):
    sphere = enum.auto()
    tet = enum.auto()


class camera_model(StrEnum):
    pinhole = enum.auto()
    thinlens = enum.auto()
    fisheye = enum.auto()


# how a tactile sensor measures the depth of a contact
class contact_depth_query(StrEnum):
    sdf = enum.auto()
    raycast = enum.auto()


# how particles fill the volume of a morph
class particle_sampler(StrEnum):
    pbs = enum.auto()
    random = enum.auto()
    regular = enum.auto()


# shape of the droplets an emitter emits
class droplet_shape(StrEnum):
    circle = enum.auto()
    sphere = enum.auto()
    square = enum.auto()
    rectangle = enum.auto()


class subterrain_type(StrEnum):
    flat_terrain = enum.auto()
    fractal_terrain = enum.auto()
    random_uniform_terrain = enum.auto()
    sloped_terrain = enum.auto()
    pyramid_sloped_terrain = enum.auto()
    discrete_obstacles_terrain = enum.auto()
    wave_terrain = enum.auto()
    stairs_terrain = enum.auto()
    pyramid_stairs_terrain = enum.auto()
    stepping_stones_terrain = enum.auto()


class drone_model(StrEnum):
    CF2X = "CF2X"
    CF2P = "CF2P"
    RACE = "RACE"


# stress model of an FEM material
class fem_model(StrEnum):
    linear = enum.auto()
    stable_neohookean = enum.auto()
    linear_corotated = enum.auto()


# stress model of an MPM material
class mpm_model(StrEnum):
    corotation = enum.auto()
    neohooken = enum.auto()


class yield_criterion(StrEnum):
    von_mises = enum.auto()
    stretch_clamp = enum.auto()


class pressure_solver(StrEnum):
    WCSPH = "WCSPH"
    DFSPH = "DFSPH"


# how the IPC coupler couples a rigid entity
class coup_type(StrEnum):
    two_way_soft_constraint = enum.auto()
    external_articulation = enum.auto()
    ipc_only = enum.auto()


# contact model of a pair of SAP coupler surfaces
class sap_contact_type(StrEnum):
    tet = enum.auto()
    vert = enum.auto()
    none = enum.auto()


class linear_system_solver(StrEnum):
    linear_pcg = enum.auto()
    direct = enum.auto()


class contact_constitution(StrEnum):
    ipc = enum.auto()
    isometric = enum.auto()


class collision_detection_method(StrEnum):
    linear_bvh = enum.auto()
    spatial_hash = enum.auto()


# CoACD convex decomposition
class preprocess_mode(StrEnum):
    on = enum.auto()
    off = enum.auto()
    auto = enum.auto()


class apx_mode(StrEnum):
    ch = enum.auto()
    box = enum.auto()


# what a trajectory recorder stores of each frame
class trajectory_mode(StrEnum):
    exact = enum.auto()
    minimal = enum.auto()


class raytracer_logging_level(StrEnum):
    debug = enum.auto()
    info = enum.auto()
    warning = enum.auto()


# side of the polar decomposition the rotation multiplies
class polar_side(StrEnum):
    right = enum.auto()
    left = enum.auto()
