"""
LAMMPS/LMGC90-style simulation setup interface for SoftFlow.

Provides a declarative, sequential API for configuring and running
LBM-IBM capsule simulations. Commands are issued in order:

    domain → boundary → obstacles → fluid → particle types →
    physics → coupling → interactions → regions → generate →
    timestep → output → run

Example (basic)
---------------
    from pysoftflow import SoftFlowSimulation

    sim = SoftFlowSimulation()
    sim.domain(nx=400, ny=80)
    sim.boundary(x="periodic", y="wall")
    sim.fluid(tau=0.8)
    sim.body_force(5e-6)
    sim.particle_type("rbc", model="skalak", G_s=0.06, C_skalak=10.0,
                      k_bend=0.003, k_area=0.8, k_perimeter=0.08)
    sim.region("inlet", x=(20, 150), y=(10, 70))
    sim.generate("rbc", count=20, region="inlet", radius=(4.0, 6.0))
    sim.output(format="vtk", directory="output", interval=200)
    sim.run(20000)

Example (biomedical with adhesion + scalar transport)
-----------------------------------------------------
    sim = SoftFlowSimulation()
    sim.domain(nx=400, ny=80)
    sim.boundary(x="periodic", y="wall")
    sim.fluid(tau=0.8, collision="regularized")
    sim.body_force(5e-6)
    sim.ibm(iterations=2)
    sim.lubrication(enabled=True)
    sim.adhesion(enabled=True, k_on=0.001, k_off=0.01, k_bond=0.05)
    sim.scalar_transport(enabled=True, diffusivity=0.01)
    sim.viscosity_contrast(enabled=True, update_interval=10)
    sim.metrics(interval=5000)
    sim.checkpoint(interval=10000)
    sim.particle_type("rbc", model="skalak", G_s=0.06, viscosity_ratio=5.0)
    sim.particle_type("tumor", model="neo_hookean", G_s=0.15)
    sim.region("channel", x=(10, 390), y=(10, 70))
    sim.generate("rbc", count=20, region="channel", radius=(4, 6))
    sim.generate("tumor", count=5, region="channel", radius=(3, 5))
    sim.run(50000)
"""

from __future__ import annotations

import os
import sys
import math
import warnings
from typing import Optional, Tuple, Union, List

import numpy as np
import softflow_core as _sc


def _fmt_time(seconds: float) -> str:
    """Format seconds into a human-readable string (e.g. '1m 23s')."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        m, s = divmod(int(seconds), 60)
        return f"{m}m {s:02d}s"
    else:
        h, rem = divmod(int(seconds), 3600)
        m, s = divmod(rem, 60)
        return f"{h}h {m:02d}m"


# ═══════════════════════════════════════════════════════════════
# Fluid type presets
# ═══════════════════════════════════════════════════════════════

_FLUID_PRESETS = {
    "water":  {"tau": 0.8,  "rho0": 1.0},
    "air":    {"tau": 0.55, "rho0": 1.0},
    "custom": {},
    "none":   {},
}

# Membrane model name → enum mapping
_MEMBRANE_MODELS = {
    "hookean": _sc.MembraneModel.HOOKEAN,
    "neo_hookean": _sc.MembraneModel.NEO_HOOKEAN,
    "skalak": _sc.MembraneModel.SKALAK,
    "wlc": _sc.MembraneModel.WLC,
}

# Capsule shape name → enum mapping
_CAPSULE_SHAPES = {
    "circle":    _sc.CapsuleShape.CIRCLE,
    "ellipse":   _sc.CapsuleShape.ELLIPSE,
    "biconcave": _sc.CapsuleShape.BICONCAVE,
    "fiber":     _sc.CapsuleShape.FIBER,
}

# Collision model name → enum mapping
_COLLISION_MODELS = {
    "bgk": _sc.CollisionModel.BGK,
    "mrt": _sc.CollisionModel.MRT,
    "regularized": _sc.CollisionModel.REGULARIZED,
}


class SoftFlowSimulation:
    """LAMMPS/LMGC90-style declarative simulation setup.

    Commands must follow a logical ordering (domain first, run last).
    The simulation is not built until ``initialize()`` or ``run()``
    is called.
    """

    def __init__(self):
        # Domain
        self._domain: Optional[dict] = None
        self._boundary: dict = {"x": "inlet_outlet", "y": "wall"}

        # Geometry
        self._obstacles: list = []
        self._polygon_obstacles: list = []
        self._polygon_domains: list = []   # exterior → SOLID

        # Fluid
        self._fluid_config: dict = {}
        self._fluid_fill_mode: str = "all"
        self._fluid_fill_region: Optional[str] = None
        self._fluid_phases: int = 1
        self._shan_chen_G: float = -5.0
        self._shan_chen_eos: int = 0       # 0=exponential, 1=Carnahan-Starling
        self._shan_chen_cs: dict = {}      # C-S EOS parameters

        # Particle types (named registry)
        self._particle_types: dict = {}
        self._particle_type_ids: dict = {}  # name → int id
        self._next_type_id: int = 0

        # Regions (named registry)
        self._regions: dict = {}

        # Particle generation commands (deferred)
        self._particle_specs: list = []

        # Physics
        self._gravity: Tuple[float, float] = (0.0, 0.0)
        self._gravity_target: str = "fluid"
        self._body_force: Tuple[float, float] = (0.0, 0.0)

        # Coupling
        self._coupling: dict = {
            "method": "ibm",
            "mode": "two_way",
            "delta_function": "peskin_4pt",
        }

        # New physics configuration (stored as dicts, applied in _build_params)
        self._ibm_config: dict = {}
        self._lubrication_config: dict = {}
        self._adhesion_config: dict = {}
        self._scalar_config: dict = {}
        self._viscosity_contrast_config: dict = {}
        self._metrics_config: dict = {}
        self._checkpoint_config: dict = {}
        self._moving_wall_config: dict = {}
        self._scalar_sources: list = []  # (type_name, release, absorption)

        # Data output configuration (disabled by default; use data_output() to enable)
        self._data_output_config: dict = {
            "enabled": False,         # master switch — only generate when asked
            "interval": 0,            # 0 = use thermo interval
            "trajectory": True,       # per-capsule (step, id, type, x, y, D, R)
            "timeseries": True,       # global metrics (u_max, D_large, D_small, bonds)
            "positions": True,        # final capsule positions
            "bonds": True,            # final adhesion bonds
            "node_positions": False,  # full membrane node coordinates (large files!)
            "velocity_field": False,  # fluid velocity field snapshots (very large!)
            "format": "csv",          # "csv" or "dat"
            "directory": None,        # separate dir for data files (None = vtk dir)
        }

        # Interactions
        self._interactions: dict = {}

        # Timestep
        self._dt: float = 1.0

        # Unit conversion (set by units())
        # _unit_dx, _unit_dt, _unit_rho are created only when units() is called

        # Output
        self._outputs: list = []
        self._thermo_interval: int = 0

        # Runtime
        self._initialized: bool = False
        self._core: Optional[_sc.Simulation] = None
        self._params: Optional[_sc.SimulationParams] = None

    # ══════════════════════════════════════════════════════════
    # 1. Domain & geometry
    # ══════════════════════════════════════════════════════════

    def domain(self, nx: int, ny: int) -> "SoftFlowSimulation":
        """Define the simulation domain size in lattice units.

        Parameters
        ----------
        nx : int
            Domain width (number of columns).
        ny : int
            Domain height (number of rows).
        """
        if nx < 3 or ny < 3:
            raise ValueError("Domain must be at least 3x3")
        self._domain = {"nx": nx, "ny": ny}
        return self

    def boundary(self, x: str = "inlet_outlet", y: str = "wall",
                 capsules: str = "follow") -> "SoftFlowSimulation":
        """Set boundary conditions.

        Parameters
        ----------
        x : str
            X-direction: ``"periodic"``, ``"inlet_outlet"`` (default),
            or ``"closed"`` (solid walls on all 4 sides).
        y : str
            Y-direction: ``"wall"`` (default), ``"periodic"``, or ``"open"``.
        capsules : str
            Capsule streamwise behaviour: ``"follow"`` (default) uses the same
            x-BC as the fluid; ``"periodic"`` recirculates capsules in x
            *independently* of the fluid. Use ``capsules="periodic"`` with
            ``x="inlet_outlet"`` for a finite suspension that recirculates at
            sustained concentration while the fluid is velocity/pressure-driven.
        """
        self._require_domain("boundary")
        valid_x = ("periodic", "inlet_outlet", "closed")
        valid_y = ("wall", "periodic", "open")
        valid_caps = ("follow", "periodic")
        if x not in valid_x:
            raise ValueError(f"x boundary must be one of {valid_x}, got {x!r}")
        if y not in valid_y:
            raise ValueError(f"y boundary must be one of {valid_y}, got {y!r}")
        if capsules not in valid_caps:
            raise ValueError(f"capsules must be one of {valid_caps}, got {capsules!r}")
        self._boundary = {"x": x, "y": y, "capsules": capsules}
        return self

    def obstacle(self, shape: str, **kwargs) -> "SoftFlowSimulation":
        """Add a solid obstacle to the domain.

        Parameters
        ----------
        shape : str
            ``"circle"``, ``"rect"``, or ``"polygon"``.
        **kwargs
            For circle: ``center=(cx, cy)``, ``radius=r``.
            For rect: ``p1=(x0, y0)``, ``p2=(x1, y1)``.
            For polygon: ``vertices=[(x0,y0), (x1,y1), ...]``.
        """
        self._require_domain("obstacle")
        if shape == "circle":
            center = kwargs.get("center")
            radius = kwargs.get("radius")
            if center is None or radius is None:
                raise ValueError("Circle obstacle requires center=(cx,cy) and radius=r")
            self._obstacles.append({"shape": "circle", "cx": center[0],
                                     "cy": center[1], "radius": radius})
        elif shape == "rect":
            p1 = kwargs.get("p1")
            p2 = kwargs.get("p2")
            if p1 is None or p2 is None:
                raise ValueError("Rect obstacle requires p1=(x0,y0) and p2=(x1,y1)")
            self._obstacles.append({"shape": "rect", "x0": p1[0], "y0": p1[1],
                                     "x1": p2[0], "y1": p2[1]})
        elif shape == "polygon":
            vertices = kwargs.get("vertices")
            if vertices is None or len(vertices) < 3:
                raise ValueError("Polygon obstacle requires vertices=[(x0,y0), ...] with at least 3 vertices")
            self._polygon_obstacles.append(vertices)
        else:
            raise ValueError(f"Unknown obstacle shape: {shape!r}. Use 'circle', 'rect', or 'polygon'.")
        return self

    def polygon_domain(self, vertices) -> "SoftFlowSimulation":
        """Confine flow to the interior of a polygon — exterior cells become SOLID.

        Parameters
        ----------
        vertices : list of (x, y) tuples
            Convex or concave polygon (≥3 vertices) in lattice units.
            Cells outside the polygon (ray-cast test) are marked SOLID.
        """
        self._require_domain("polygon_domain")
        if len(vertices) < 3:
            raise ValueError("polygon_domain requires at least 3 vertices")
        self._polygon_domains.append(list(vertices))
        return self

    # ══════════════════════════════════════════════════════════
    # 2. Fluid definition
    # ══════════════════════════════════════════════════════════

    def fluid(
        self,
        type: str = "custom",
        method: str = "lbm",
        tau: Optional[float] = None,
        density: Optional[float] = None,
        viscosity: Optional[float] = None,
        inlet_velocity: Optional[float] = None,
        outlet_density: Optional[float] = None,
        pressure: Optional[float] = None,
        use_mrt: bool = False,
        collision: str = "bgk",
        max_lattice_force: Optional[float] = None,
    ) -> "SoftFlowSimulation":
        """Configure the fluid solver.

        Parameters
        ----------
        type : str
            Fluid preset: ``"water"``, ``"air"``, ``"custom"``, or ``"none"``.
        method : str
            Solver method (currently only ``"lbm"``).
        tau : float, optional
            BGK relaxation time. Overrides preset value if given.
        density : float, optional
            Initial fluid density. Overrides preset value if given.
        viscosity : float, optional
            Kinematic viscosity (alternative to tau: tau = 3*nu + 0.5).
        inlet_velocity : float, optional
            Inlet velocity for inlet/outlet BCs.
        outlet_density : float, optional
            Outlet density for inlet/outlet BCs.
        pressure : float, optional
            Reference pressure (stored for future use).
        use_mrt : bool
            Use MRT collision instead of BGK. Deprecated; use ``collision="mrt"``.
        collision : str
            Collision operator: ``"bgk"`` (default), ``"mrt"``, or ``"regularized"``.
            Regularized BGK (Latt & Chopard 2006) gives better stability at low tau.
        max_lattice_force : float, optional
            IBM/body-force per-cell magnitude cap (default 0.01). Membrane
            forces during dense collisions can exceed the default cap and
            cause silent truncation; raise to 0.02–0.05 for higher Reynolds
            or denser suspensions, or to a very large number to disable.
            Phase-1 ``checkStability`` will report how many nodes hit the
            cap each step.
        """
        self._require_domain("fluid")
        if type not in _FLUID_PRESETS:
            raise ValueError(f"Unknown fluid type: {type!r}. "
                             f"Choose from {list(_FLUID_PRESETS.keys())}")
        if collision not in _COLLISION_MODELS:
            raise ValueError(f"Unknown collision model: {collision!r}. "
                             f"Choose from {list(_COLLISION_MODELS.keys())}")

        config = {"type": type, "method": method}

        # Start from preset defaults
        preset = _FLUID_PRESETS[type]
        config["tau"] = preset.get("tau", 0.8)
        config["rho0"] = preset.get("rho0", 1.0)

        # Override with explicit values
        if viscosity is not None:
            config["tau"] = 3.0 * viscosity + 0.5
        if tau is not None:
            config["tau"] = tau
        if density is not None:
            config["rho0"] = density
        if inlet_velocity is not None:
            config["inlet_velocity"] = inlet_velocity
        if outlet_density is not None:
            config["outlet_density"] = outlet_density
        if pressure is not None:
            config["pressure"] = pressure

        # Collision model
        if use_mrt:
            config["collision"] = "mrt"
        else:
            config["collision"] = collision

        # Phase-1 IBM force cap. None → leave default (0.01) untouched.
        if max_lattice_force is not None:
            config["max_lattice_force"] = float(max_lattice_force)

        self._fluid_config = config
        return self

    def fluid_fill(self, mode: str = "all", name: Optional[str] = None) -> "SoftFlowSimulation":
        """Define the area to fill with fluid.

        Parameters
        ----------
        mode : str
            ``"all"`` fills entire domain (default).
            ``"region"`` fills only a named region.
        name : str, optional
            Region name (required if mode is ``"region"``).
        """
        if mode == "region" and name is None:
            raise ValueError("fluid_fill('region') requires name='region_name'")
        self._fluid_fill_mode = mode
        self._fluid_fill_region = name
        return self

    def fluid_phases(self, n: int = 1, G: float = -1.0,
                     eos: str = "carnahan_starling",
                     T: float = 0.06, a: float = 1.0,
                     b: float = 4.0, R: float = 1.0) -> "SoftFlowSimulation":
        """Configure single or two-phase fluid (Shan-Chen multiphase).

        Parameters
        ----------
        n : int
            Number of phases: 1 (default) or 2.
        G : float
            Shan-Chen interaction strength.
        eos : str
            Equation of state: ``"exponential"`` or ``"carnahan_starling"``.
        T, a, b, R : float
            Carnahan-Starling parameters.
        """
        if n not in (1, 2):
            raise ValueError("Number of fluid phases must be 1 or 2")
        self._fluid_phases = n
        self._shan_chen_G = G
        if eos == "carnahan_starling":
            self._shan_chen_eos = 1
            self._shan_chen_cs = {"a": a, "b": b, "T": T, "R": R}
        elif eos == "exponential":
            self._shan_chen_eos = 0
            self._shan_chen_cs = {}
        else:
            raise ValueError(f"Unknown EOS: {eos!r}. Use 'exponential' or 'carnahan_starling'")
        return self

    # ══════════════════════════════════════════════════════════
    # 3. Particle types (extended with membrane models)
    # ══════════════════════════════════════════════════════════

    def particle_type(
        self,
        name: str,
        model: str = "hookean",
        k_stretch: float = 0.1,
        G_s: Optional[float] = None,
        C_skalak: float = 10.0,
        k_bend: float = 0.005,
        k_area: float = 0.5,
        k_perimeter: float = 0.1,
        gamma_visc: float = 0.01,
        eta_membrane: float = 0.05,
        use_helfrich: bool = False,
        kappa_0: float = 0.0,
        viscosity_ratio: float = 1.0,
        wlc_L_max_ratio: float = 2.2,
        wlc_kBT_p: float = 0.0,
        wlc_k_pow: float = 0.0,
        density: float = 1.0,
        shape: str = "circle",
        aspect_ratio: float = 1.0,
        indent_depth: float = 0.4,
        is_rigid: bool = False,
    ) -> "SoftFlowSimulation":
        """Define a named particle (capsule) type with membrane properties.

        Parameters
        ----------
        name : str
            Type name (e.g. ``"rbc"``, ``"cancer_cell"``, ``"platelet"``).
        model : str
            Membrane constitutive model:

            - ``"hookean"`` — linear springs (default)
            - ``"neo_hookean"`` — nonlinear hyperelastic (Barthes-Biesel)
            - ``"skalak"`` — area-dilatation resistant (Skalak 1973, for RBCs)
            - ``"wlc"`` — Worm-Like Chain (Fedosov et al., Biophys J 2010,
              nonlinear polymer chain model for cytoskeletal networks)

        k_stretch : float
            Stretching stiffness (used only for ``"hookean"`` model).
        G_s : float, optional
            Surface shear modulus (used for ``"neo_hookean"``, ``"skalak"``,
            and ``"wlc"`` models). If not given, defaults to ``k_stretch``.
        C_skalak : float
            Skalak area dilatation parameter C (only for ``"skalak"`` model).
        k_bend : float
            Bending stiffness.
        k_area : float
            Area conservation stiffness.
        k_perimeter : float
            Perimeter conservation stiffness.
        gamma_visc : float
            Lab-frame nodal damping, F = -gamma_visc * v_node.  This is a
            numerical stabiliser, not a physical membrane property: it is
            not Galilean invariant and removes momentum from the fluid
            through the IBM spread, which slows a driven suspension.  Set
            it to 0 for quantitative flow studies; use eta_membrane for
            physical (strain-rate) membrane viscosity.
        eta_membrane : float
            Kelvin-Voigt membrane viscosity (strain-rate damping).
            Resists rapid deformation while allowing slow deformation.
            Higher values = more viscous response. Default 0.05.
        use_helfrich : bool
            Use Helfrich curvature-based bending.
        kappa_0 : float
            Spontaneous curvature for Helfrich bending.
        viscosity_ratio : float
            Interior/exterior viscosity ratio (RBCs: ~5).
        wlc_L_max_ratio : float
            WLC maximum extension ratio L_max/L0 (default 2.2).
            Controls how much the membrane can stretch before diverging.
            Typical: 2.0-3.0.  Only used for ``"wlc"`` model.
        wlc_kBT_p : float
            WLC thermal energy / persistence length.  If 0 (default),
            auto-computed from G_s to match shear modulus at small strain.
            Only used for ``"wlc"`` model.
        wlc_k_pow : float
            WLC repulsive power-law coefficient.  If 0 (default),
            auto-computed to balance WLC force at rest length.
            Only used for ``"wlc"`` model.
        density : float
            Capsule density in lattice units (default 1.0 = neutrally buoyant).
            When ``density != 1.0`` and gravity is set, a net buoyancy force
            ``F = (rho_cap - rho_fluid) * A * g`` is applied to each capsule.
            Use physical density values (e.g. 1090 for RBC) when ``units()``
            mapping is configured.
        """
        if model not in _MEMBRANE_MODELS:
            raise ValueError(f"Unknown membrane model: {model!r}. "
                             f"Choose from {list(_MEMBRANE_MODELS.keys())}")

        mp = _sc.MembraneParams()
        mp.model = _MEMBRANE_MODELS[model]

        # Stretching parameters
        if model == "hookean":
            if G_s is not None:
                raise ValueError(
                    "particle_type(model='hookean'): G_s is not used by the "
                    "Hookean law; set its stiffness with k_stretch, or use "
                    "model='neo_hookean'/'skalak' for a G_s-based law.")
            mp.k_stretch = k_stretch
        else:
            mp.G_s = G_s if G_s is not None else k_stretch
            if model == "skalak":
                mp.C_skalak = C_skalak
            elif model == "wlc":
                mp.wlc_L_max_ratio = wlc_L_max_ratio
                mp.wlc_kBT_p = wlc_kBT_p
                mp.wlc_k_pow = wlc_k_pow

        # Bending
        mp.k_bend = k_bend
        if use_helfrich:
            mp.use_helfrich_bending = True
            mp.kappa_0 = kappa_0

        # Conservation
        mp.k_area = k_area
        mp.k_perimeter = k_perimeter
        mp.gamma_visc = gamma_visc
        mp.eta_membrane = eta_membrane

        # Viscosity contrast
        mp.viscosity_ratio = viscosity_ratio

        # Density (1.0 = neutrally buoyant / massless IBM)
        mp.density = density

        # Shape
        mp.shape = _CAPSULE_SHAPES.get(shape, _sc.CapsuleShape.CIRCLE)
        mp.aspect_ratio = aspect_ratio
        mp.indent_depth = indent_depth
        if is_rigid:
            warnings.warn(
                "particle_type(is_rigid=True): node velocities are projected "
                "onto a rigid-body motion, but no force is spread to the fluid, "
                "so the particle is a passive tracer (it does not disturb the "
                "flow or migrate laterally).  For a hydrodynamically active "
                "near-rigid particle use a stiff elastic capsule (e.g. "
                "model='hookean' with large k_stretch and use_helfrich=True).",
                stacklevel=2)
        mp.is_rigid = is_rigid

        self._particle_types[name] = mp
        if name not in self._particle_type_ids:
            self._particle_type_ids[name] = self._next_type_id
            self._next_type_id += 1
        return self

    # ══════════════════════════════════════════════════════════
    # 4. Physical parameters
    # ══════════════════════════════════════════════════════════

    def gravity(self, gx: float = 0.0, gy: float = 0.0,
                apply_to: str = "all") -> "SoftFlowSimulation":
        """Set gravitational acceleration (lattice units).

        Parameters
        ----------
        gx, gy : float
            Gravitational acceleration components (lattice units).
        apply_to : str
            ``"fluid"`` — density-weighted body force on fluid only
            (standard LBM gravity: F = rho * g).
            ``"capsules"`` — buoyancy force on capsules only
            (F = (rho_cap - rho_fluid) * A * g on each capsule).
            ``"all"`` — both fluid gravity and capsule buoyancy (default).
            Capsules with ``density=1.0`` are unaffected by buoyancy.
        """
        if apply_to not in ("fluid", "capsules", "all"):
            raise ValueError(f"apply_to must be 'fluid', 'capsules', or 'all', got {apply_to!r}")
        self._gravity = (gx, gy)
        self._gravity_target = apply_to
        return self

    def body_force(self, fx: float = 0.0, fy: float = 0.0) -> "SoftFlowSimulation":
        """Set a constant uniform body force on the fluid (lattice units).

        Used to drive flow in periodic channels (Poiseuille).
        Unlike gravity, this is NOT multiplied by density.
        """
        self._body_force = (fx, fy)
        return self

    def pressure_gradient(self, dp_dx: float = 0.0, dp_dy: float = 0.0) -> "SoftFlowSimulation":
        """Set a pressure gradient (converted to equivalent body force).

        In LBM: F = -dp/dx / (rho * cs^2), where cs^2 = 1/3.
        """
        rho0 = self._fluid_config.get("rho0", 1.0)
        cs2 = 1.0 / 3.0
        self._body_force = (-dp_dx / (rho0 * cs2), -dp_dy / (rho0 * cs2))
        return self

    # ══════════════════════════════════════════════════════════
    # 5. Coupling (extended with IBM iterations)
    # ══════════════════════════════════════════════════════════

    def coupling(
        self,
        method: str = "ibm",
        mode: str = "two_way",
        delta_function: str = "peskin_4pt",
    ) -> "SoftFlowSimulation":
        """Configure fluid-particle coupling.

        Parameters
        ----------
        method : str
            Coupling method: ``"ibm"`` (immersed boundary) or ``"none"``.
        mode : str
            ``"two_way"`` (default): full bidirectional coupling.
            ``"one_way"``: fluid affects particles but not vice versa.
        delta_function : str
            IBM delta function: ``"peskin_4pt"`` (default).
        """
        valid_methods = ("ibm", "none")
        valid_modes = ("two_way", "one_way")
        if method not in valid_methods:
            raise ValueError(f"Coupling method must be one of {valid_methods}")
        if mode not in valid_modes:
            raise ValueError(f"Coupling mode must be one of {valid_modes}")
        self._coupling = {
            "method": method,
            "mode": mode,
            "delta_function": delta_function,
        }
        return self

    def ibm(self, iterations: int = 1) -> "SoftFlowSimulation":
        """Configure the IBM coupling.

        Parameters
        ----------
        iterations : int
            1 (default) = standard Peskin spread/interpolate coupling, the
            correct choice for deformable membranes, whose nodes move with
            the interpolated fluid velocity.  Values > 1 enable the
            multi-direct-forcing correction of Luo et al. (2007), whose
            target velocity is the node velocity of the previous step; in a
            driven flow this damps the fluid around deformable capsules and
            reduces the flow rate, so it is not recommended for them.
        """
        if iterations > 1:
            warnings.warn(
                "ibm(iterations>1): multi-direct forcing targets the previous-"
                "step node velocity and acts as a drag on deformable capsules "
                "in driven flows; use iterations=1 for deformable membranes.",
                stacklevel=2)
        self._ibm_config = {"iterations": iterations}
        return self

    # ══════════════════════════════════════════════════════════
    # 6. Interactions
    # ══════════════════════════════════════════════════════════

    def interaction(
        self,
        pair: str,
        style: str = "morse",
        **kwargs,
    ) -> "SoftFlowSimulation":
        """Define a pairwise interaction (like LAMMPS pair_style).

        Parameters
        ----------
        pair : str
            Interaction pair: ``"particle-particle"``,
            ``"particle-wall"``, ``"particle-obstacle"``.
        style : str
            ``"morse"`` — conservative power-law repulsion only (legacy).
            ``"dem"`` — DEM-style contact: the same repulsion *plus* optional
            normal viscoelastic damping (restitution < 1) and tangential
            Coulomb friction. Both styles share the repulsion parameters; the
            DEM terms are off unless ``damping_normal`` / ``friction_coeff`` > 0.
        **kwargs
            Repulsion: ``epsilon``, ``sigma``, ``r_cut``, ``power``.
            DEM contact: ``damping_normal`` (normal dashpot coefficient,
            gives restitution < 1) and ``friction_coeff`` (Coulomb mu;
            tangential force capped at mu*|F_normal|). Applied to inter-capsule,
            wall, and obstacle contacts.
        """
        valid_pairs = ("particle-particle", "particle-wall", "particle-obstacle")
        if pair not in valid_pairs:
            raise ValueError(f"Interaction pair must be one of {valid_pairs}")
        self._interactions[pair] = {"style": style, **kwargs}
        return self

    # ══════════════════════════════════════════════════════════
    # 7. Advanced physics modules
    # ══════════════════════════════════════════════════════════

    def lubrication(
        self,
        enabled: bool = True,
        h_threshold: float = 1.5,
        h_min: float = 0.1,
    ) -> "SoftFlowSimulation":
        """Enable lubrication corrections for close capsule interactions.

        Prevents capsule overlap in dense suspensions by adding sub-grid
        lubrication forces when gap h < h_threshold.

        Parameters
        ----------
        enabled : bool
            Enable lubrication corrections.
        h_threshold : float
            Gap distance below which lubrication force is applied (lattice units).
        h_min : float
            Minimum gap distance for regularization (prevents singularity).
        """
        self._lubrication_config = {
            "enabled": enabled,
            "h_threshold": h_threshold,
            "h_min": h_min,
        }
        return self

    def adhesion(
        self,
        enabled: bool = True,
        k_on: float = 0.001,
        k_off: float = 0.01,
        k_bond: float = 0.05,
        d_bond: float = 2.0,
        F_crit: float = 0.01,
        max_bonds_per_node: int = 3,
        wall_adhesion: bool = False,
        wall_k_on: float = 0.001,
        wall_k_off: float = 0.01,
        wall_k_bond: float = 0.05,
        bond_model: str = "bell",
        k_off_catch: float = 0.05,
        F_catch: float = 0.02,
        k_off_slip: float = 0.001,
        F_slip: float = 0.01,
        wall_receptor_spacing: float = 2.0,
        adhesion_matrix=None,
    ) -> "SoftFlowSimulation":
        """Enable cell adhesion model.

        Supports two bond dissociation models:

        **Bell model** (default): ``k_off(F) = k_off * exp(F/F_crit)``
            Pure slip bonds — force always weakens the bond.

        **Catch-slip model** (Thomas et al. 2008):
            ``k_off(F) = k_catch * exp(-F/F_catch) + k_slip * exp(F/F_slip)``
            At low force, catch pathway dominates → bond is STRENGTHENED.
            At high force, slip pathway dominates → bond breaks.
            Creates biphasic lifetime with optimal force for strongest bonds.
            Biological examples: P-selectin/PSGL-1, FimH/mannose.

        Parameters
        ----------
        enabled : bool
            Enable adhesion model.
        k_on : float
            Bond formation rate constant.
        k_off : float
            Bond dissociation rate constant (Bell model only).
        k_bond : float
            Bond spring stiffness.
        d_bond : float
            Maximum distance for bond formation.
        F_crit : float
            Critical force for Bell model slip dissociation.
        max_bonds_per_node : int
            Maximum bonds per membrane node.
        wall_adhesion : bool
            Enable capsule-wall adhesion.
        wall_k_on, wall_k_off, wall_k_bond : float
            Wall adhesion parameters.
        bond_model : str
            ``"bell"`` (default) or ``"catch_slip"``.
        k_off_catch : float
            Catch pathway dissociation rate (catch-slip model).
        F_catch : float
            Catch force scale — bond strengthens under this force.
        k_off_slip : float
            Slip pathway dissociation rate (catch-slip model).
        F_slip : float
            Slip force scale — bond weakens above this force.
        """
        self._adhesion_config = {
            "enabled": enabled,
            "k_on": k_on,
            "k_off": k_off,
            "k_bond": k_bond,
            "d_bond": d_bond,
            "F_crit": F_crit,
            "max_bonds_per_node": max_bonds_per_node,
            "wall_adhesion": wall_adhesion,
            "wall_k_on": wall_k_on,
            "wall_k_off": wall_k_off,
            "wall_k_bond": wall_k_bond,
            "bond_model": bond_model,
            "k_off_catch": k_off_catch,
            "F_catch": F_catch,
            "k_off_slip": k_off_slip,
            "F_slip": F_slip,
            "wall_receptor_spacing": wall_receptor_spacing,
            # Phase-1 contract: empty adhesion_matrix means no bonds. Many
            # users will never need cross-type pair gating, so when
            # adhesion is enabled and no matrix is supplied, default to
            # an all-pairs True matrix sized to the registered particle
            # types. Pass an explicit matrix to restrict.
            "adhesion_matrix": adhesion_matrix,
        }
        return self

    def scalar_transport(
        self,
        enabled: bool = True,
        diffusivity: float = 0.01,
        n_species: int = 1,
        initial_concentration: float = 0.0,
        inlet_concentration: float = None,
        periodic_y: bool = False,
    ) -> "SoftFlowSimulation":
        """Enable advection-diffusion solver for scalar transport.

        Models transport of dissolved species (drugs, glucose, biomarkers)
        advected by the fluid velocity field.

        Parameters
        ----------
        enabled : bool
            Enable scalar transport.
        diffusivity : float
            Diffusion coefficient (lattice units).
        n_species : int
            Number of scalar species to transport (default 1).
        initial_concentration : float
            Initial uniform concentration everywhere in the domain (default 0).
        inlet_concentration : float, optional
            Concentration imposed at the left inlet each step (inlet_outlet BC).
            Defaults to same as initial_concentration if not set.
        periodic_y : bool
            Use periodic scalar BC in y (default: no-flux walls).
        """
        self._scalar_config = {
            "enabled": enabled,
            "diffusivity": diffusivity,
            "n_species": n_species,
            "initial_concentration": initial_concentration,
            "inlet_concentration": inlet_concentration,
            "periodic_y": periodic_y,
        }
        return self

    def scalar_source(
        self,
        type_name: str,
        release_rate: float = 0.0,
        absorption_rate: float = 0.0,
        # ── Physics-based leaching (Fick, Gap A + C) ──────────────────────────
        k_leach: float = 0.0,
        C_eq: float = 0.0,
        M_p_initial: float = 0.0,
        # ── Langmuir adsorption/desorption (Gap B) ────────────────────────────
        k_adsorb: float = 0.0,
        k_desorb: float = 0.0,
        Gamma_max: float = 1.0,
    ) -> "SoftFlowSimulation":
        """Set scalar source/sink parameters for a capsule type.

        Parameters
        ----------
        type_name : str
            Particle type name (must be defined with ``particle_type()``).
        release_rate : float
            Constant scalar release rate per node per step (backward-compatible).
        absorption_rate : float
            Constant scalar absorption rate per node per step (backward-compatible).
        k_leach : float
            Mass transfer coefficient for Fick-type leaching.
            J = k_leach * (C_eq - C_surface).  Activates physics-based mode.
        C_eq : float
            Equilibrium / saturation concentration for leaching.
        M_p_initial : float
            Initial chemical mass in each particle of this type (0 = infinite).
        k_adsorb : float
            Langmuir adsorption rate k_a:  dΓ/dt += k_a * C_surface * (1 - Γ/Γ_max).
        k_desorb : float
            Langmuir desorption rate k_d:  dΓ/dt -= k_d * Γ.
        Gamma_max : float
            Maximum surface coverage Γ_max (default 1.0).

        Notes
        -----
        Physics-based mode (k_leach / k_adsorb) is activated when those params > 0.
        Constant-rate (release_rate / absorption_rate) is the backward-compatible fallback.
        Both can coexist: constant-rate is used for types without physics params.
        """
        src = {
            "type_name":       type_name,
            "release_rate":    release_rate,
            "absorption_rate": absorption_rate,
            "k_leach":         k_leach,
            "C_eq":            C_eq,
            "M_p_initial":     M_p_initial,
            "k_adsorb":        k_adsorb,
            "k_desorb":        k_desorb,
            "Gamma_max":       Gamma_max,
        }
        # Update existing entry for same type (e.g. called again between phase runs)
        for i, existing in enumerate(self._scalar_sources):
            if existing["type_name"] == type_name:
                self._scalar_sources[i] = src
                break
        else:
            self._scalar_sources.append(src)

        # If the C++ core is already running (called between two sim.run() calls),
        # push the params directly so they take effect immediately.
        if self._initialized and self._core is not None:
            tid = self._particle_type_ids.get(type_name)
            if tid is not None:
                if release_rate > 0:
                    self._core.setScalarReleaseRate(tid, release_rate)
                if absorption_rate > 0:
                    self._core.setScalarAbsorptionRate(tid, absorption_rate)
                if k_leach > 0:
                    self._core.setLeachingParams(tid, k_leach, C_eq)
                if M_p_initial > 0:
                    self._core.setParticleMass(tid, M_p_initial)
                if k_adsorb > 0 or k_desorb > 0:
                    self._core.setAdsorptionParams(tid, k_adsorb, k_desorb, Gamma_max)
        return self

    def viscosity_contrast(
        self,
        enabled: bool = True,
        update_interval: int = 10,
    ) -> "SoftFlowSimulation":
        """Enable spatially varying viscosity (interior vs exterior fluid).

        Uses ray-casting to classify lattice nodes inside/outside capsules,
        then sets local tau accordingly: tau_in = 3 * lambda * nu_out + 0.5.
        Each capsule type's ``viscosity_ratio`` controls its internal viscosity.

        Parameters
        ----------
        enabled : bool
            Enable viscosity contrast.
        update_interval : int
            Re-compute viscosity field every N steps (default 10).

        Warning
        -------
        In this release the local relaxation-time field is computed but
        none of the collision operators (BGK, MRT, regularised) reads it,
        so the interior viscosity contrast has no effect on the flow.
        """
        if enabled:
            warnings.warn(
                "viscosity_contrast(): the local tau field is not used by any "
                "collision operator in this release, so the viscosity ratio "
                "has no effect (lambda = 1 in practice).", stacklevel=2)
        self._viscosity_contrast_config = {
            "enabled": enabled,
            "update_interval": update_interval,
        }
        return self

    def metrics(self, interval: int = 5000) -> "SoftFlowSimulation":
        """Enable segregation metrics computation.

        Computes margination parameter, mixing entropy, cell-free layer
        thickness, radial distribution function, and deformation statistics
        at regular intervals.

        Parameters
        ----------
        interval : int
            Compute metrics every N steps.
        """
        self._metrics_config = {"interval": interval}
        return self

    def checkpoint(self, interval: int = 10000) -> "SoftFlowSimulation":
        """Enable periodic checkpoint saving for restart capability.

        Parameters
        ----------
        interval : int
            Save checkpoint every N steps.
        """
        self._checkpoint_config = {"interval": interval}
        return self

    def enable_free_surface(
        self,
        rho_atm: float = 1.0,
        threshold: float = 0.002,
    ) -> "SoftFlowSimulation":
        """Enable free-surface (wet-dry) LBM for dam-break / gravity-driven flows.

        Empty cells act as solid walls until adjacent fluid pressure exceeds
        rho_atm * (1 + threshold), then they convert to fluid cells.  Gravity
        builds hydrostatic pressure at the bottom of the water column, so
        bottom cells convert first — water fills from the bottom up.

        Call set_empty_region() after this to mark the "air" side of the dam.

        Parameters
        ----------
        rho_atm : float
            Atmospheric (reference) density for newly-wetted cells. Default 1.0.
        threshold : float
            Fractional pressure excess that triggers wetting (0.002 = 0.2%).
            Lower = more sensitive (wets sooner); higher = needs more pressure.

        Note on gravity units
        ---------------------
        In LBM, gravity is in LATTICE UNITS, not 9.81 m/s².
        Conversion:  g_lat = g_SI × dt² / dx
        For dx=1 mm and water (nu=1e-6 m²/s), tau=1.0:
            dt = sqrt(g_lat × dx / g_SI)
            For g_lat = 1e-4:  dt ≈ 3.2 µs/ts,  Ma ≈ 0.17  ✓
        So use sim.gravity(gy=-1e-4) for a realistic dam-break demo.
        """
        self._free_surface_config = {
            "enabled": True,
            "rho_atm": rho_atm,
            "threshold": threshold,
            "empty_regions": [],
        }
        return self

    def set_empty_region(
        self,
        x: tuple,
        y: tuple,
    ) -> "SoftFlowSimulation":
        """Mark a rectangular region as 'empty' (air side of the dam).

        Must call enable_free_surface() first.

        Parameters
        ----------
        x : (x0, x1)  inclusive x range in lattice units
        y : (y0, y1)  inclusive y range in lattice units
        """
        if not hasattr(self, "_free_surface_config"):
            self.enable_free_surface()
        self._free_surface_config["empty_regions"].append(
            {"x0": int(x[0]), "x1": int(x[1]),
             "y0": int(y[0]), "y1": int(y[1])}
        )
        return self

    def moving_wall(
        self,
        top_velocity: float = 0.0,
        bottom_velocity: float = 0.0,
    ) -> "SoftFlowSimulation":
        """Set moving wall boundary conditions (Couette flow).

        Modified bounce-back with wall velocity term:
        f_opp = f_q + 2*w_q*rho*(e_q . u_wall) / cs^2

        Parameters
        ----------
        top_velocity : float
            Top wall velocity in x-direction.
        bottom_velocity : float
            Bottom wall velocity in x-direction.
        """
        self._moving_wall_config = {
            "top_velocity": top_velocity,
            "bottom_velocity": bottom_velocity,
        }
        return self

    # ══════════════════════════════════════════════════════════
    # 8. Regions & particle generation
    # ══════════════════════════════════════════════════════════

    def region(
        self,
        name: str,
        x: Tuple[float, float] = None,
        y: Tuple[float, float] = None,
    ) -> "SoftFlowSimulation":
        """Define a named rectangular region.

        Parameters
        ----------
        name : str
            Region name for later reference.
        x : (x0, x1)
            X-range of the region.
        y : (y0, y1)
            Y-range of the region.
        """
        self._require_domain("region")
        nx = self._domain["nx"]
        ny = self._domain["ny"]
        x0, x1 = x if x is not None else (0, nx)
        y0, y1 = y if y is not None else (0, ny)
        self._regions[name] = (x0, y0, x1, y1)
        return self

    def generate(
        self,
        type_name: str,
        count: int,
        region: str,
        radius: Union[float, Tuple[float, float]] = 5.0,
        num_nodes: Optional[int] = None,
        method: str = "random",
        spacing: Optional[float] = None,
        velocity: Tuple[float, float] = (0.0, 0.0),
        seed: int = 12345,
        min_gap: float = 1.0,
        packing: str = "hex",
        jitter: float = 0.0,
    ) -> "SoftFlowSimulation":
        """Generate particles in a named region.

        Parameters
        ----------
        type_name : str
            Particle type name (must be defined with ``particle_type()``).
        count : int
            Number of particles to generate.
        region : str
            Named region (defined with ``region()``).
        radius : float or (min, max)
            Particle radius. Float for fixed, tuple for polydisperse.
        num_nodes : int, optional
            Number of membrane nodes per capsule.  If ``None`` (default),
            automatically computed so that node spacing ds ~ 0.75 dx.
        method : str
            ``"fill"`` (a.k.a. ``"dense"``) — **recommended for dense
            suspensions.** Deterministic close-packing onto a grid whose
            spacing guarantees the minimum gap by construction, so it packs to
            the region's geometric capacity with no stochastic rejection. Use
            ``packing`` to pick ``"hex"`` (default) or ``"square"``.
            ``"random"`` — random placement with overlap rejection. Fine for
            dilute fills; severely under-fills dense regions.
            ``"hexagonal"`` — hex grid with jitter + overlap rejection (the
            jitter can reject many near-threshold sites; prefer ``"fill"``).
            ``"lattice"`` — simple regular grid placement.
        spacing : float, optional
            Grid spacing (centre-to-centre) for ``"fill"``, ``"lattice"`` or
            ``"hexagonal"``. Defaults to ``2 * r_max + min_gap``.
        packing : str
            For ``method="fill"``: ``"hex"`` (default, denser) or ``"square"``.
        jitter : float
            For ``method="fill"``: per-capsule random displacement (LU). The
            grid spacing is enlarged by ``2*jitter`` so packing stays dense and
            non-overlapping, while different ``seed`` values give DISTINCT
            configurations (the base fill grid is otherwise deterministic).
        velocity : (vx, vy)
            Initial velocity of generated particles.
        seed : int
            Random seed for reproducible placement (default 12345).
        min_gap : float
            Minimum gap between capsule surfaces in lattice units
            (default 1.0).  Smaller values → denser packing.
        """
        if type_name not in self._particle_types:
            raise ValueError(
                f"Unknown particle type {type_name!r}. "
                f"Define it first with particle_type()")
        if region not in self._regions:
            raise ValueError(
                f"Unknown region {region!r}. "
                f"Define it first with region()")

        if isinstance(radius, (int, float)):
            rmin = rmax = float(radius)
        else:
            rmin, rmax = float(radius[0]), float(radius[1])

        self._particle_specs.append({
            "mode": "generate",
            "type_name": type_name,
            "count": count,
            "region": region,
            "rmin": rmin,
            "rmax": rmax,
            "num_nodes": num_nodes if num_nodes is not None else 0,  # 0 = auto
            "method": method,
            "spacing": spacing,
            "velocity": velocity,
            "seed": seed,
            "min_gap": min_gap,
            "packing": packing,
            "jitter": jitter,
        })
        return self

    def particle(
        self,
        type_name: str,
        center: Tuple[float, float],
        radius: float = 5.0,
        num_nodes: Optional[int] = None,
        velocity: Tuple[float, float] = (0.0, 0.0),
    ) -> "SoftFlowSimulation":
        """Place a single particle at an exact position.

        Parameters
        ----------
        type_name : str
            Particle type name (must be defined with ``particle_type()``).
        center : (x, y)
            Center position.
        radius : float
            Particle radius.
        num_nodes : int, optional
            Number of membrane nodes.  If ``None`` (default), auto-computed
            from radius (ds ~ 0.75 dx).
        velocity : (vx, vy)
            Initial velocity.
        """
        if type_name not in self._particle_types:
            raise ValueError(
                f"Unknown particle type {type_name!r}. "
                f"Define it first with particle_type()")

        self._particle_specs.append({
            "mode": "single",
            "type_name": type_name,
            "center": center,
            "radius": radius,
            "num_nodes": num_nodes if num_nodes is not None else 0,  # 0 = auto
            "velocity": velocity,
        })
        return self

    # ══════════════════════════════════════════════════════════
    # 9. Timestep & Unit Conversion
    # ══════════════════════════════════════════════════════════

    def timestep(self, dt: float = 1.0) -> "SoftFlowSimulation":
        """Set the timestep size (lattice units, default 1.0).

        In LBM the lattice timestep is always dt=1.  This is kept for
        backward compatibility but rarely needs to be changed.
        """
        if dt <= 0:
            raise ValueError("Timestep must be positive")
        self._dt = dt
        return self

    def units(
        self,
        dx: Optional[float] = None,
        dt_phys: Optional[float] = None,
        rho_phys: Optional[float] = None,
        length_ref: Optional[float] = None,
        velocity_ref: Optional[float] = None,
    ) -> "SoftFlowSimulation":
        """Define the mapping between lattice units and physical (SI) units.

        In LBM every simulation uses lattice units (dx=1, dt=1, rho0=1).
        To interpret results in SI, you define three independent scales.

        Two ways to set the mapping (pick **one**):

        **Method A – direct scales** (most common):

        .. code-block:: python

            sim.units(dx=0.5e-6, dt_phys=1e-7, rho_phys=1000)
              # dx      = physical length of one lattice spacing [m]
              # dt_phys = physical time of one lattice step     [s]
              # rho_phys = physical density for rho0=1           [kg/m^3]

        **Method B – match a reference length and velocity**:

        .. code-block:: python

            sim.units(length_ref=10e-6, velocity_ref=0.001, rho_phys=1000)
              # length_ref   = physical channel height (maps to NY) [m]
              # velocity_ref = physical reference velocity          [m/s]
              # rho_phys     = physical density                     [kg/m^3]

        After calling ``units()`` the conversion factors are stored and
        ``dimensionless_numbers()`` prints physical values alongside
        lattice values.

        Returns self for chaining.
        """
        if rho_phys is None:
            rho_phys = 1000.0  # default to water

        if dx is not None and dt_phys is not None:
            # Method A: direct scales
            self._unit_dx = dx
            self._unit_dt = dt_phys
            self._unit_rho = rho_phys
        elif length_ref is not None and velocity_ref is not None:
            # Method B: derive dx and dt from reference length/velocity
            ny = self._domain["ny"] if self._domain else 61
            dx_val = length_ref / ny
            # u_lattice = characteristic lattice velocity
            tau = self._fluid_config.get("tau", 0.8)
            u_inlet = self._fluid_config.get("inlet_velocity", None)
            if u_inlet is not None:
                u_lattice = u_inlet
            elif self._body_force[0] != 0:
                nu_l = (tau - 0.5) / 3.0
                H = ny - 2
                u_lattice = self._body_force[0] * H**2 / (8.0 * nu_l)
            else:
                u_lattice = 0.01  # fallback
            dt_val = dx_val * u_lattice / velocity_ref
            self._unit_dx = dx_val
            self._unit_dt = dt_val
            self._unit_rho = rho_phys
        else:
            raise ValueError(
                "Provide either (dx, dt_phys) or (length_ref, velocity_ref), "
                "plus optionally rho_phys.")
        return self

    # -- Conversion helpers (available after units() is called) --

    def to_physical_length(self, lattice_length: float) -> float:
        """Convert a length from lattice units to physical units [m]."""
        self._require_units()
        return lattice_length * self._unit_dx

    def to_physical_time(self, lattice_steps: float) -> float:
        """Convert a time (or step count) to physical time [s]."""
        self._require_units()
        return lattice_steps * self._unit_dt

    def to_physical_velocity(self, lattice_vel: float) -> float:
        """Convert a velocity from lattice units to physical [m/s]."""
        self._require_units()
        return lattice_vel * self._unit_dx / self._unit_dt

    def to_physical_force(self, lattice_force: float) -> float:
        """Convert a force density from lattice to physical [N/m^3]."""
        self._require_units()
        return lattice_force * self._unit_rho * self._unit_dx / self._unit_dt**2

    def to_physical_viscosity(self, nu_lattice: float) -> float:
        """Convert kinematic viscosity to physical [m^2/s]."""
        self._require_units()
        return nu_lattice * self._unit_dx**2 / self._unit_dt

    def to_lattice_length(self, phys_length: float) -> float:
        """Convert a physical length [m] to lattice units."""
        self._require_units()
        return phys_length / self._unit_dx

    def to_lattice_velocity(self, phys_vel: float) -> float:
        """Convert a physical velocity [m/s] to lattice units."""
        self._require_units()
        return phys_vel * self._unit_dt / self._unit_dx

    def _require_units(self):
        if not hasattr(self, "_unit_dx"):
            raise RuntimeError(
                "Unit conversion not set. Call sim.units(...) first.")

    # -- Dimensionless numbers --

    def dimensionless_numbers(self) -> dict:
        """Compute and print key dimensionless numbers for the simulation.

        Works in lattice units (always available).  If ``units()`` was
        called the physical equivalents are also shown.

        Returns a dict with the computed values.
        """
        if self._domain is None or not self._fluid_config:
            raise RuntimeError("Set domain and fluid before calling "
                               "dimensionless_numbers().")

        nx = self._domain["nx"]
        ny = self._domain["ny"]
        tau = self._fluid_config.get("tau", 0.8)
        rho0 = self._fluid_config.get("rho0", 1.0)
        nu_l = (tau - 0.5) / 3.0         # lattice kinematic viscosity
        mu_l = rho0 * nu_l                # lattice dynamic viscosity
        H = ny - 2                        # channel height (without walls)

        # --- Characteristic velocity ---
        u_inlet = self._fluid_config.get("inlet_velocity", None)
        fx = self._body_force[0]
        if u_inlet is not None:
            u_char = u_inlet
            u_source = "inlet"
        elif fx != 0:
            u_char = fx * H**2 / (8.0 * nu_l)   # Poiseuille centerline
            u_source = "Poiseuille centerline"
        else:
            u_char = 0.0
            u_source = "none (no driving force)"

        # --- Reynolds number ---
        Re = u_char * H / nu_l if nu_l > 0 else 0.0

        # --- Capillary number (per particle type) ---
        Ca_dict = {}
        for tname, mp in self._particle_types.items():
            G_s = getattr(mp, "G_s", 0.0)
            if G_s <= 0:
                G_s = getattr(mp, "k_stretch", 0.0)
            # Average radius from specs
            radii = []
            for spec in self._particle_specs:
                if spec["type_name"] == tname:
                    if spec["mode"] == "generate":
                        r_lo, r_hi = spec["radius_range"]
                        radii.append(0.5 * (r_lo + r_hi))
                    else:
                        radii.append(spec["radius"])
            R_avg = sum(radii) / len(radii) if radii else 5.0
            # shear rate ~ u_char / (H/2) for Poiseuille
            gamma_dot = u_char / (H / 2.0) if H > 0 else 0.0
            Ca = mu_l * gamma_dot * R_avg / G_s if G_s > 0 else 0.0
            confinement = 2.0 * R_avg / H if H > 0 else 0.0
            Ca_dict[tname] = {"Ca": Ca, "R": R_avg, "G_s": G_s,
                              "confinement": confinement}

        # --- Mach number (compressibility check) ---
        cs = 1.0 / 3.0**0.5  # lattice speed of sound
        Ma = u_char / cs

        # --- Build result dict ---
        result = {
            "Re": Re,
            "Ma": Ma,
            "u_char": u_char,
            "u_source": u_source,
            "nu": nu_l,
            "tau": tau,
            "H": H,
            "nx": nx,
            "ny": ny,
            "capsules": Ca_dict,
        }

        # --- Print ---
        print(f"\n{'='*60}")
        print("Dimensionless Numbers (Lattice Units)")
        print(f"{'='*60}")
        print(f"  Domain:             {nx} x {ny}  (H = {H})")
        print(f"  tau = {tau:.4f},  nu = {nu_l:.6f}")
        print(f"  Characteristic U:   {u_char:.6f}  ({u_source})")
        print(f"  Reynolds number:    Re = {Re:.4f}")
        print(f"  Mach number:        Ma = {Ma:.4f}  "
              f"({'OK' if Ma < 0.1 else 'WARNING: Ma > 0.1, compressibility errors'})")

        if Ca_dict:
            print(f"\n  {'Type':<12s} {'Ca':>10s} {'R':>6s} {'G_s':>8s} {'2R/H':>6s}")
            print(f"  {'-'*44}")
            for tname, cd in Ca_dict.items():
                print(f"  {tname:<12s} {cd['Ca']:>10.4f} {cd['R']:>6.1f} "
                      f"{cd['G_s']:>8.4f} {cd['confinement']:>6.3f}")

        # --- Physical units (if set) ---
        has_units = hasattr(self, "_unit_dx")
        if has_units:
            dx = self._unit_dx
            dt = self._unit_dt
            rho = self._unit_rho
            L_phys = H * dx
            U_phys = u_char * dx / dt
            nu_phys = nu_l * dx**2 / dt
            mu_phys = rho * nu_phys

            print(f"\n{'='*60}")
            print("Physical Units")
            print(f"{'='*60}")
            print(f"  dx = {dx:.3e} m,  dt = {dt:.3e} s,  rho = {rho:.1f} kg/m^3")
            print(f"  Channel height:     {L_phys:.3e} m  ({L_phys*1e6:.1f} um)")
            print(f"  Velocity:           {U_phys:.3e} m/s")
            print(f"  Viscosity:          nu = {nu_phys:.3e} m^2/s,  "
                  f"mu = {mu_phys:.3e} Pa.s")
            print(f"  Reynolds number:    Re = {Re:.4f}  (same in both units)")

            if Ca_dict:
                print(f"\n  {'Type':<12s} {'R_phys':>12s} {'G_s_phys':>12s}")
                print(f"  {'-'*36}")
                for tname, cd in Ca_dict.items():
                    R_phys = cd["R"] * dx
                    # G_s has dimensions of force/length in 2D
                    G_s_phys = cd["G_s"] * rho * dx**3 / dt**2
                    print(f"  {tname:<12s} {R_phys:.3e} m  {G_s_phys:.3e} N/m")

            # Total physical simulation time
            total_specs = sum(s.get("count", 1) for s in self._particle_specs)
            result["dx"] = dx
            result["dt"] = dt
            result["rho"] = rho
            result["L_phys"] = L_phys
            result["U_phys"] = U_phys
            result["nu_phys"] = nu_phys
            result["mu_phys"] = mu_phys

        print(f"{'='*60}\n")
        return result

    # ══════════════════════════════════════════════════════════
    # 10. Output
    # ══════════════════════════════════════════════════════════

    def output(
        self,
        format: str = "vtk",
        directory: str = "output",
        interval: int = 100,
    ) -> "SoftFlowSimulation":
        """Add an output writer (can be called multiple times).

        Parameters
        ----------
        format : str
            Output format: ``"vtk"`` or ``"csv"``.
        directory : str
            Output directory path.
        interval : int
            Write every *interval* timesteps.
        """
        valid_formats = ("vtk", "vtk_legacy", "csv")
        if format not in valid_formats:
            raise ValueError(f"Output format must be one of {valid_formats}")
        self._outputs.append({
            "format": format,
            "directory": directory,
            "interval": interval,
        })
        return self

    def thermo(self, interval: int = 1000) -> "SoftFlowSimulation":
        """Enable LAMMPS-style console output at regular intervals.

        Parameters
        ----------
        interval : int
            Print simulation statistics every *interval* steps.
        """
        self._thermo_interval = interval
        return self

    def data_output(self,
                    enabled: bool = True,
                    interval: int = 0,
                    trajectory: bool = True,
                    timeseries: bool = True,
                    positions: bool = True,
                    bonds: bool = True,
                    node_positions: bool = False,
                    velocity_field: bool = False,
                    format: str = "csv",
                    directory: Optional[str] = None,
                    ) -> "SoftFlowSimulation":
        """Configure data output for ML analysis.

        Controls what simulation data is saved to disk and how often.
        All files are saved to the VTK output directory.

        Parameters
        ----------
        enabled : bool
            Master switch — if ``False``, no data files are written.
        interval : int
            Save data every *interval* steps.  If ``0`` (default),
            uses the thermo interval.
        trajectory : bool
            Save per-capsule trajectory: ``trajectory.csv``
            Columns: step, capsule_id, type_id, x, y, D, R.
            This is the key file for ML (positions + deformation over time).
        timeseries : bool
            Save global metrics: ``capsule_data.dat``
            Columns: step, u_max, max_D, D_per_type, bonds.
        positions : bool
            Save final capsule snapshot: ``final_positions.dat``
        bonds : bool
            Save final adhesion bonds: ``bonds_final.csv``
        node_positions : bool
            Save full membrane node coordinates at each interval:
            ``nodes_XXXXXXXX.csv``.  **Warning: large files!**
            Columns: step, capsule_id, node_id, x, y.
        velocity_field : bool
            Save fluid velocity field snapshots: ``field_XXXXXXXX.csv``.
            **Warning: very large files** (NX × NY × 2 values per snapshot).
        format : str
            File format: ``"csv"`` (comma-separated) or ``"dat"`` (space-separated).

        Examples
        --------
        ::

            # Save everything for ML analysis
            sim.data_output(trajectory=True, node_positions=True, interval=2000)

            # Minimal output (just timeseries)
            sim.data_output(trajectory=False, bonds=False, positions=False)

            # Disable all data output
            sim.data_output(enabled=False)

            # Save fluid velocity field every 5000 steps (large!)
            sim.data_output(velocity_field=True, interval=5000)
        """
        self._data_output_config.update({
            "enabled": enabled,
            "interval": interval,
            "trajectory": trajectory,
            "timeseries": timeseries,
            "positions": positions,
            "bonds": bonds,
            "node_positions": node_positions,
            "velocity_field": velocity_field,
            "format": format,
            "directory": directory,
        })
        if enabled:
            parts = []
            if trajectory: parts.append("trajectory")
            if timeseries: parts.append("timeseries")
            if positions: parts.append("positions")
            if node_positions: parts.append("node_positions")
            if velocity_field: parts.append("velocity_field")
            dest = directory or "<vtk output dir>"
            print(f"Data output enabled: [{', '.join(parts)}] "
                  f"every {interval} steps → {dest} ({format})")
        return self

    # ══════════════════════════════════════════════════════════
    # 11. Simulation lifecycle
    # ══════════════════════════════════════════════════════════

    def initialize(self) -> "SoftFlowSimulation":
        """Build the C++ simulation from the declared configuration.

        This is called automatically by ``run()`` if not called explicitly.
        """
        if self._initialized:
            return self

        self._validate_config()
        self._build_params()
        self._build_simulation()
        self._initialized = True
        return self

    def step(self) -> None:
        """Advance one timestep."""
        if not self._initialized:
            self.initialize()
        self._core.step()

    def save_checkpoint(self, filename: Optional[str] = None) -> str:
        """Save full simulation state to a binary checkpoint file.

        Saves LBM distributions, capsule positions/velocities, adhesion bonds,
        and the current step counter.  The simulation can be restarted from
        this checkpoint later with ``restart()``.

        Parameters
        ----------
        filename : str, optional
            Path to checkpoint file.  If ``None``, auto-generates a name
            based on the output directory and current step:
            ``<output_dir>/checkpoint_<step>.sfck``

        Returns
        -------
        str
            The path to the saved checkpoint file.
        """
        if not self._initialized:
            raise RuntimeError("Simulation not initialized. Call run() first.")

        if filename is None:
            out_dir = self._outputs[0].get("directory", ".") if self._outputs else "."
            step = self._core.currentStep()
            filename = os.path.join(out_dir, f"checkpoint_{step:08d}.sfck")

        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)

        ok = self._core.saveCheckpoint(filename)
        if not ok:
            raise RuntimeError(f"Failed to save checkpoint: {filename}")
        return filename

    def restart(self, filename: str) -> None:
        """Restart simulation from a checkpoint file.

        Loads LBM distributions, capsule positions/velocities, adhesion bonds,
        and the step counter from a previously saved ``.sfck`` file.

        The simulation must already be set up with the same domain, capsule
        count, and particle types as when the checkpoint was saved.

        Parameters
        ----------
        filename : str
            Path to a ``.sfck`` checkpoint file.

        Example
        -------
        ::

            # First run — save checkpoint at end
            sim = SoftFlowSimulation()
            sim.domain(nx=400, ny=100)
            # ... setup ...
            sim.run(50000)
            sim.save_checkpoint("my_checkpoint.sfck")

            # Later — restart and continue
            sim2 = SoftFlowSimulation()
            sim2.domain(nx=400, ny=100)
            # ... same setup as before ...
            sim2.restart("my_checkpoint.sfck")
            sim2.run(50000)   # continues from step 50000
        """
        if not self._initialized:
            self.initialize()

        ok = self._core.loadCheckpoint(filename)
        if not ok:
            raise RuntimeError(f"Failed to load checkpoint: {filename}")

    def warmup(self, steps: int = 2000, ramp_steps: int = 1000) -> None:
        """Run a warm-up phase to equilibrate densely packed capsules.

        Phase 1 (``steps``): Run with ZERO body force / inlet velocity.
                 Repulsion and membrane forces push overlapping capsules apart.
        Phase 2 (``ramp_steps``): Linearly ramp body force from 0 to target.
                 Avoids sudden pressure shock.

        Call this BEFORE ``run()`` when capsules are densely packed.

        Example::

            sim.warmup(steps=3000, ramp_steps=1000)
            sim.run(200000)
        """
        if not self._initialized:
            self.initialize()

        # Save original forces
        fp = self._core.params().fluid
        orig_bfx = fp.body_force_x
        orig_bfy = fp.body_force_y
        orig_uin = fp.inlet_velocity

        def _run_phase(label, total, step_fn):
            """Run steps with progress bar."""
            bar_w = 30
            update_every = max(1, total // bar_w)  # update ~30 times
            for s in range(total):
                step_fn(s)
                if (s + 1) % update_every == 0 or s + 1 == total:
                    done = (s + 1) / total
                    filled = int(bar_w * done)
                    bar = "█" * filled + "░" * (bar_w - filled)
                    print(f"\r  {label:12s} |{bar}| {100*done:5.1f}%  ({s+1}/{total})",
                          end="", flush=True)
            print()

        # Phase 1: zero-force equilibration
        if steps > 0:
            fp.body_force_x = 0.0
            fp.body_force_y = 0.0
            fp.inlet_velocity = 0.0
            print(f"Warm-up ({steps} + {ramp_steps} steps)")
            _run_phase("Equilibrate", steps, lambda s: self._core.step())

        # Phase 2: ramp up body force gradually
        if ramp_steps > 0:
            def _ramp_step(s):
                frac = (s + 1) / ramp_steps
                fp.body_force_x = orig_bfx * frac
                fp.body_force_y = orig_bfy * frac
                fp.inlet_velocity = orig_uin * frac
                self._core.step()
            _run_phase("Ramp flow", ramp_steps, _ramp_step)

        # Restore original values
        fp.body_force_x = orig_bfx
        fp.body_force_y = orig_bfy
        fp.inlet_velocity = orig_uin

    def run(self, num_steps: int) -> None:
        """Run the simulation for *num_steps* timesteps.

        Everything is automatic:
          - Prints progress to screen (step, %, u_max, max_D, etc.)
          - Writes VTK/CSV files if ``output()`` was configured
          - Saves capsule data to ``<output_dir>/capsule_data.dat``
          - Saves final positions to ``<output_dir>/final_positions.dat``

        Parameters
        ----------
        num_steps : int
            Number of timesteps to run.

        Example
        -------
        ::
            sim = SoftFlowSimulation()
            sim.domain(nx=400, ny=80)
            sim.boundary(x="periodic", y="wall")
            sim.fluid(tau=0.8)
            sim.body_force(5e-6)
            sim.particle_type("rbc", model="skalak", G_s=0.06)
            sim.region("all", x=(10, 390), y=(10, 70))
            sim.generate("rbc", count=20, region="all", radius=5.0)
            sim.timestep(dt=1.0)
            sim.output(format="vtk", directory="output", interval=1000)
            sim.run(50000)
        """
        import time as _time

        if not self._initialized:
            self.initialize()

        # Determine print interval
        if self._thermo_interval > 0:
            thermo = self._thermo_interval
        else:
            thermo = max(1, num_steps // 20)

        # Determine output directory for data files
        out_dir = None
        for out in self._outputs:
            out_dir = out["directory"]
            break

        caps = self._core.capsules()
        ncaps = caps.numCapsules()

        # Identify capsule types for per-type reporting
        type_names = list(self._particle_type_ids.keys())
        type_ids = self._particle_type_ids
        n_types = len(type_names)

        # Print header
        print(f"\nRunning {num_steps} steps  (capsules={ncaps}, "
              f"print every {thermo})")

        header = f"{'Step':>10s}  {'%':>6s}  {'u_max':>8s}  {'max_D':>8s}"
        if n_types > 1:
            for tname in type_names:
                header += f"  {'D_'+tname:>10s}"
        if self._adhesion_config.get("enabled"):
            header += f"  {'bonds':>6s}"
        header += f"  {'elapsed':>9s}  {'ETA':>9s}"
        print(header, flush=True)
        print("-" * len(header), flush=True)

        # Data collection for automatic file output
        import csv as _csv
        run_data = []
        node_snapshots = []    # full membrane node positions (kept in-memory: node_positions files)
        segregation_ts = []    # segregation metrics time-series
        dconf = self._data_output_config
        data_interval = dconf.get("interval", 0) or thermo
        t_start = _time.time()

        # ── Open incremental CSV files before the run loop ─────────────────────
        # Written every data_interval steps; flush after each write so data is
        # safe even if the simulation is interrupted mid-run.
        _inc_files = {}       # name → file handle
        _inc_writers = {}     # name → csv.writer
        _headers_done = set() # names whose header row has been written

        def _open_inc(name, data_dir):
            path = os.path.join(data_dir, f"{name}.csv")
            fh = open(path, "a", newline="")
            _inc_files[name] = fh
            _inc_writers[name] = _csv.writer(fh)

        def _write_inc(name, cols, row_dict):
            if name not in _headers_done:
                _inc_writers[name].writerow(cols)
                _headers_done.add(name)
            _inc_writers[name].writerow([row_dict.get(c, "") for c in cols])

        def _flush_all():
            for fh in _inc_files.values():
                fh.flush()

        if dconf.get("enabled"):
            _data_dir_inc = dconf.get("directory") or out_dir
            os.makedirs(_data_dir_inc, exist_ok=True)
            if dconf.get("trajectory"):
                _open_inc("trajectory", _data_dir_inc)
            if dconf.get("timeseries"):
                _open_inc("timeseries", _data_dir_inc)
            if self._scalar_config.get("enabled"):
                _open_inc("concentration_timeseries", _data_dir_inc)
            if self._adhesion_config.get("enabled"):
                _open_inc("bond_timeseries", _data_dir_inc)
        # ───────────────────────────────────────────────────────────────────────

        import sys as _sys
        _sys.stdout.flush()

        for i in range(num_steps):
            _t0 = _time.time()
            self._core.step()
            _dt = _time.time() - _t0
            step_num = i + 1
            if _dt > 2.0:
                print(f"  [WARNING] step {step_num} took {_dt:.1f}s!", flush=True)
            step_total = self._core.currentStep()

            # Print + collect data at thermo interval
            if step_num % thermo == 0 or step_num == num_steps:
                pct = 100.0 * step_num / num_steps
                max_speed = self._core.getMaxSpeed()

                max_D = 0.0
                D_by_type = {tname: [] for tname in type_names}
                for c in range(ncaps):
                    cap = caps[c]
                    D = cap.deformationIndex()
                    if D > max_D:
                        max_D = D
                    # Collect per-type deformation
                    tid = cap.getTypeId()
                    for tname, expected_tid in type_ids.items():
                        if tid == expected_tid:
                            D_by_type[tname].append(D)
                            break

                line = (f"{step_total:>10d}  {pct:>5.1f}%  "
                        f"{max_speed:>8.5f}  {max_D:>8.5f}")

                row = {"step": step_total, "u_max": max_speed, "max_D": max_D}

                if n_types > 1:
                    for tname in type_names:
                        D_arr = D_by_type[tname]
                        D_mean = sum(D_arr) / len(D_arr) if D_arr else 0.0
                        line += f"  {D_mean:>10.5f}"
                        row[f"D_{tname}"] = D_mean

                if self._adhesion_config.get("enabled"):
                    try:
                        bonds = self._core.adhesion().getBonds()
                        line += f"  {len(bonds):>6d}"
                        row["bonds"] = len(bonds)
                    except Exception:
                        line += f"  {'?':>6s}"

                # Elapsed time and ETA
                elapsed = _time.time() - t_start
                if step_num < num_steps:
                    eta = elapsed * (num_steps - step_num) / step_num
                else:
                    eta = 0.0
                line += f"  {_fmt_time(elapsed):>9s}  {_fmt_time(eta):>9s}"

                print(line, flush=True)
                run_data.append(row)
                if "timeseries" in _inc_writers:
                    TS_COLS = list(row.keys())
                    _write_inc("timeseries", TS_COLS, row)
                    _inc_files["timeseries"].flush()

                # ── Collect concentration time-series ──
                if self._scalar_config.get("enabled"):
                    try:
                        ad = self._core.advectionDiffusion()
                        import numpy as _np
                        conc = _np.asarray(ad.concentration(0))
                        conc_row = {
                            "step": step_total,
                            "conc_max": float(conc.max()),
                            "conc_mean": float(conc.mean()),
                            "conc_total": float(conc.sum()),
                        }
                        # Per-capsule cumulative scalar
                        n_tracked = ad.numTrackedCapsules()
                        total_released = sum(ad.getCapsuleReleased(c)
                                             for c in range(min(n_tracked, ncaps)))
                        total_absorbed = sum(ad.getCapsuleAbsorbed(c)
                                             for c in range(min(n_tracked, ncaps)))
                        conc_row["total_released"] = total_released
                        conc_row["total_absorbed"] = total_absorbed
                        if "concentration_timeseries" in _inc_writers:
                            CONC_COLS = list(conc_row.keys())
                            _write_inc("concentration_timeseries", CONC_COLS, conc_row)
                            _inc_files["concentration_timeseries"].flush()
                    except Exception:
                        pass

                # ── Collect bond/contact statistics ──
                if self._adhesion_config.get("enabled"):
                    try:
                        all_bonds = self._core.adhesion().getBonds()
                        n_wall = sum(1 for b in all_bonds if b.capsule_j < 0)
                        n_cell = len(all_bonds) - n_wall
                        avg_force = (sum(b.force for b in all_bonds) / len(all_bonds)
                                     if all_bonds else 0.0)
                        avg_len = (sum(b.length for b in all_bonds) / len(all_bonds)
                                   if all_bonds else 0.0)
                        bond_row = {
                            "step": step_total,
                            "total_bonds": len(all_bonds),
                            "wall_bonds": n_wall,
                            "cell_bonds": n_cell,
                            "avg_force": avg_force,
                            "avg_length": avg_len,
                        }
                        if "bond_timeseries" in _inc_writers:
                            BOND_COLS = list(bond_row.keys())
                            _write_inc("bond_timeseries", BOND_COLS, bond_row)
                            _inc_files["bond_timeseries"].flush()
                    except Exception:
                        pass

                # ── Collect segregation metrics (at metrics_interval) ──
                try:
                    seg = self._core.lastSegregationResults()
                    if seg is not None:
                        segregation_ts.append({
                            "step": step_total,
                            "margination": seg.margination_parameter,
                            "mixing_entropy": seg.mixing_entropy,
                            "cfl_top": seg.cfl_top,
                            "cfl_bottom": seg.cfl_bottom,
                            "mean_deformation": seg.mean_deformation,
                            "separation_efficiency": seg.separation_efficiency,
                        })
                except Exception:
                    pass

                # ── Auto-stop on blow-up ──
                if max_speed > 1.0 or (max_speed == 0.0 and step_num > thermo):
                    print(f"\n  *** SIMULATION UNSTABLE at step {step_total} ***")
                    if max_speed > 1.0:
                        print(f"  *** u_max = {max_speed:.2f} (threshold: 1.0) ***")
                    else:
                        print(f"  *** u_max = 0.0 (fluid solver crashed) ***")
                    print(f"  *** Stopping early to avoid wasting compute ***")
                    print(f"  *** Check: capsule overlap, body_force too high,")
                    print(f"  ***        gap too narrow, or too many capsules ***\n")
                    # Save checkpoint before stopping
                    try:
                        ckpt_file = self.save_checkpoint()
                    except Exception:
                        pass
                    break

            # ── Collect per-capsule data (independent of thermo) ──
            if dconf["enabled"] and step_num % data_interval == 0:
                if dconf["trajectory"]:
                    for c in range(ncaps):
                        cap = caps[c]
                        cent = cap.centroid()
                        tid = cap.getTypeId()
                        tname = "unknown"
                        for tn, ti in type_ids.items():
                            if ti == tid:
                                tname = tn
                                break
                        vx_sum, vy_sum = 0.0, 0.0
                        nn = cap.numNodes()
                        for k in range(nn):
                            nv = cap.nodeVelocity(k)
                            vx_sum += nv.x
                            vy_sum += nv.y
                        r_eff = cap.effectiveRadius()
                        traj_row = {
                            "step": step_total,
                            "capsule_id": c,
                            "type_id": tid,
                            "type_name": tname,
                            "x": cent.x,
                            "y": cent.y,
                            "vx": vx_sum / nn,
                            "vy": vy_sum / nn,
                            "D": cap.deformationIndex(),
                            "R": r_eff,
                            "diameter": 2.0 * r_eff,
                            "area": cap.area(),
                            "perimeter": cap.perimeter(),
                            "scalar_released": 0.0,
                            "scalar_absorbed": 0.0,
                        }
                        # Per-capsule scalar tracking
                        try:
                            _ad = self._core.advectionDiffusion()
                            if _ad and c < _ad.numTrackedCapsules():
                                traj_row["scalar_released"] = _ad.getCapsuleReleased(c)
                                traj_row["scalar_absorbed"] = _ad.getCapsuleAbsorbed(c)
                        except Exception:
                            pass
                        if "trajectory" in _inc_writers:
                            TRAJ_COLS = ["step","capsule_id","type_id","type_name",
                                         "x","y","vx","vy","D","R","diameter",
                                         "area","perimeter","scalar_released","scalar_absorbed"]
                            _write_inc("trajectory", TRAJ_COLS, traj_row)
                    if "trajectory" in _inc_files:
                        _inc_files["trajectory"].flush()

                if dconf["node_positions"]:
                    for c in range(ncaps):
                        cap = caps[c]
                        pos = cap.node_positions_array()
                        for ni in range(cap.numNodes()):
                            node_snapshots.append({
                                "step": step_total,
                                "capsule_id": c,
                                "node_id": ni,
                                "x": pos[ni, 0],
                                "y": pos[ni, 1],
                            })

                if dconf["velocity_field"]:
                    field = self._core.field()
                    vel_dir = dconf.get("directory") or out_dir
                    os.makedirs(vel_dir, exist_ok=True)
                    vel_path = os.path.join(vel_dir, f"field_{step_total:08d}.csv")
                    rho_arr = field.density()
                    ux_arr = field.velocity_x()
                    uy_arr = field.velocity_y()
                    fx_arr = field.force_x()
                    fy_arr = field.force_y()
                    with open(vel_path, "w") as vf:
                        vf.write("x,y,ux,uy,rho,pressure,Fx,Fy\n")
                        for iy in range(rho_arr.shape[0]):
                            for ix in range(rho_arr.shape[1]):
                                rho = rho_arr[iy, ix]
                                pressure = rho / 3.0
                                vf.write(
                                    f"{ix},{iy},"
                                    f"{ux_arr[iy,ix]:.8f},"
                                    f"{uy_arr[iy,ix]:.8f},"
                                    f"{rho:.8f},"
                                    f"{pressure:.8f},"
                                    f"{fx_arr[iy,ix]:.8f},"
                                    f"{fy_arr[iy,ix]:.8f}\n"
                                )

            # Auto-checkpoint if configured
            ckpt_interval = self._checkpoint_config.get("interval", 0)
            if ckpt_interval > 0 and step_num % ckpt_interval == 0:
                try:
                    ckpt_file = self.save_checkpoint()
                except Exception as e:
                    print(f"  [checkpoint failed: {e}]")

        total_elapsed = _time.time() - t_start
        steps_per_sec = num_steps / total_elapsed if total_elapsed > 0 else 0
        print(f"Run complete.  {self._core.currentStep()} steps in "
              f"{_fmt_time(total_elapsed)}  ({steps_per_sec:.0f} steps/s)")

        # ── Close incremental files ─────────────────────────────────────────
        for _name, _fh in _inc_files.items():
            _fh.close()
            if _inc_files:
                _dir = dconf.get("directory") or out_dir or "."
                print(f"Data saved: {os.path.join(_dir, _name + '.csv')}")
        # ───────────────────────────────────────────────────────────────────

        # ── Auto-save end-of-run snapshot files ──────────────────────────────
        # Timeseries/trajectory/concentration/bond files are written
        # incrementally above.  Only truly end-of-run snapshots go here.
        if out_dir is not None and dconf["enabled"]:
            data_dir = dconf.get("directory") or out_dir
            os.makedirs(data_dir, exist_ok=True)
            sep = "," if dconf["format"] == "csv" else "  "
            ext = ".csv" if dconf["format"] == "csv" else ".dat"

            # Final capsule positions snapshot
            if dconf["positions"]:
                pos_path = os.path.join(data_dir, f"positions_final{ext}")
                with open(pos_path, "w") as f:
                    f.write(sep.join(["id", "type", "x", "y", "D", "R"]) + "\n")
                    for c in range(ncaps):
                        cap = caps[c]
                        cent = cap.centroid()
                        D = cap.deformationIndex()
                        tid = cap.getTypeId()
                        tname = "unknown"
                        for tn, ti in type_ids.items():
                            if ti == tid:
                                tname = tn
                                break
                        f.write(sep.join([
                            str(c), tname, f"{cent.x:.4f}", f"{cent.y:.4f}",
                            f"{D:.6f}", f"{cap.effectiveRadius():.4f}"
                        ]) + "\n")
                print(f"Positions saved to {pos_path}")

            # Membrane node positions (accumulated in memory, written here)
            if dconf["node_positions"] and node_snapshots:
                node_path = os.path.join(data_dir, f"node_positions{ext}")
                cols = ["step", "capsule_id", "node_id", "x", "y"]
                with open(node_path, "w") as f:
                    f.write(sep.join(cols) + "\n")
                    for row in node_snapshots:
                        vals = [f"{row[c]}" for c in cols]
                        f.write(sep.join(vals) + "\n")
                print(f"Node positions saved to {node_path} "
                      f"({len(node_snapshots)} records)")

            # Final adhesion bonds snapshot
            if dconf["bonds"] and self._adhesion_config.get("enabled"):
                try:
                    bonds = self._core.adhesion().getBonds()
                    bond_path = os.path.join(data_dir, f"bonds_final{ext}")
                    with open(bond_path, "w") as f:
                        f.write(sep.join([
                            "capsule_i", "node_i", "capsule_j", "node_j",
                            "length", "force"
                        ]) + "\n")
                        for b in bonds:
                            f.write(sep.join([
                                str(b.capsule_i), str(b.node_i),
                                str(b.capsule_j), str(b.node_j),
                                f"{b.length:.6f}", f"{b.force:.6f}"
                            ]) + "\n")
                    print(f"Bonds saved to {bond_path} ({len(bonds)} bonds)")
                except Exception:
                    pass

            # Segregation time-series (kept in memory — typically small)
            if segregation_ts:
                seg_ts_path = os.path.join(data_dir, f"segregation_timeseries{ext}")
                cols = list(segregation_ts[0].keys())
                with open(seg_ts_path, "w") as f:
                    f.write(sep.join(cols) + "\n")
                    for row in segregation_ts:
                        vals = [f"{row[c]}" for c in cols]
                        f.write(sep.join(vals) + "\n")
                print(f"Segregation time-series saved to {seg_ts_path} "
                      f"({len(segregation_ts)} records)")

        # Write PVD collection files for ParaView time-series
        try:
            self._core.finalize()
        except Exception:
            pass

        print(f"VTK files in {out_dir}/" if out_dir else "No output directory configured.")

    # ══════════════════════════════════════════════════════════
    # Data access properties
    # ══════════════════════════════════════════════════════════

    @property
    def field(self) -> _sc.LatticeField:
        """Access the fluid lattice field."""
        self._ensure_initialized()
        return self._core.field()

    @property
    def capsules(self) -> _sc.CapsuleSystem:
        """Access the capsule system."""
        self._ensure_initialized()
        return self._core.capsules()

    @property
    def lbm_solver(self) -> _sc.LBMSolver:
        """Access the LBM solver."""
        self._ensure_initialized()
        return self._core.lbmSolver()

    @property
    def current_step(self) -> int:
        """Current simulation timestep."""
        self._ensure_initialized()
        return self._core.currentStep()

    @property
    def params(self) -> _sc.SimulationParams:
        """The simulation parameters struct."""
        return self._params

    @property
    def core(self) -> _sc.Simulation:
        """Direct access to the underlying C++ Simulation object."""
        self._ensure_initialized()
        return self._core

    @property
    def nx(self) -> int:
        return self._domain["nx"] if self._domain else 0

    @property
    def ny(self) -> int:
        return self._domain["ny"] if self._domain else 0

    def get_segregation_results(self):
        """Get latest segregation metrics results.

        Returns the SegregationResults object with fields:
        margination_parameter, mixing_entropy, cfl_bottom, cfl_top,
        mean_deformation, lateral_distribution, separation_efficiency,
        rdf_r, rdf_g.
        """
        self._ensure_initialized()
        return self._core.lastSegregationResults()

    def get_concentration(self, species: int = 0) -> np.ndarray:
        """Get scalar concentration field as numpy array.

        Parameters
        ----------
        species : int
            Species index (default 0).
        """
        self._ensure_initialized()
        ad = self._core.advectionDiffusion()
        return np.asarray(ad.concentration(species))

    def get_adhesion_bonds(self) -> list:
        """Get list of active adhesion bonds."""
        self._ensure_initialized()
        adh = self._core.adhesion()
        return adh.getBonds()

    # ══════════════════════════════════════════════════════════
    # Info / summary
    # ══════════════════════════════════════════════════════════

    def info(self) -> str:
        """Return a LAMMPS-style simulation summary string."""
        lines = []
        lines.append("=" * 60)
        lines.append("SoftFlow Simulation Summary")
        lines.append("=" * 60)

        # Thread info
        try:
            import multiprocessing
            omp_threads = os.environ.get("OMP_NUM_THREADS", "")
            n_cpu = multiprocessing.cpu_count()
            if omp_threads:
                lines.append(f"  Threads:      {omp_threads} (OMP_NUM_THREADS)")
            else:
                lines.append(f"  Threads:      {n_cpu} (auto, {n_cpu} cores)")
        except Exception:
            pass

        if self._domain:
            lines.append(f"  Domain:       {self._domain['nx']} x {self._domain['ny']}")
        lines.append(f"  Boundary:     x={self._boundary['x']}, y={self._boundary['y']}")

        if self._fluid_config:
            fc = self._fluid_config
            tau = fc.get("tau", 0.8)
            rho = fc.get("rho0", 1.0)
            nu = (tau - 0.5) / 3.0
            ftype = fc.get("type", "custom")
            coll = fc.get("collision", "bgk")
            if ftype == "none":
                lines.append("  Fluid:        NONE (dry particles)")
            else:
                lines.append(f"  Fluid:        {fc.get('method','lbm').upper()} "
                             f"(tau={tau:.3f}, nu={nu:.4f}, collision={coll})")
            if "inlet_velocity" in fc:
                lines.append(f"                inlet_velocity={fc['inlet_velocity']}")
            if self._fluid_phases > 1:
                lines.append(f"  Phases:       {self._fluid_phases} (Shan-Chen G={self._shan_chen_G})")
        else:
            lines.append("  Fluid:        NOT CONFIGURED")

        # IBM
        ibm_iter = self._ibm_config.get("iterations", 1)
        lines.append(f"  Coupling:     {self._coupling['method']} "
                     f"{self._coupling['mode']} (IBM iter={ibm_iter})")

        if self._body_force != (0.0, 0.0):
            lines.append(f"  Body force:   ({self._body_force[0]}, {self._body_force[1]})")
        if self._gravity != (0.0, 0.0):
            lines.append(f"  Gravity:      ({self._gravity[0]}, {self._gravity[1]})")

        # Moving wall
        if self._moving_wall_config:
            top_v = self._moving_wall_config.get("top_velocity", 0)
            bot_v = self._moving_wall_config.get("bottom_velocity", 0)
            lines.append(f"  Moving wall:  top={top_v}, bottom={bot_v}")

        # Advanced physics
        if self._lubrication_config.get("enabled"):
            lines.append(f"  Lubrication:  h_threshold={self._lubrication_config['h_threshold']}")
        if self._adhesion_config.get("enabled"):
            lines.append(f"  Adhesion:     k_on={self._adhesion_config['k_on']}, "
                         f"k_off={self._adhesion_config['k_off']}")
        if self._scalar_config.get("enabled"):
            lines.append(f"  Scalar:       D={self._scalar_config['diffusivity']}, "
                         f"n_species={self._scalar_config['n_species']}")
        if self._viscosity_contrast_config.get("enabled"):
            lines.append(f"  Visc contrast: update every {self._viscosity_contrast_config['update_interval']} steps")
        if self._metrics_config:
            lines.append(f"  Metrics:      every {self._metrics_config['interval']} steps")
        if self._checkpoint_config:
            lines.append(f"  Checkpoint:   every {self._checkpoint_config['interval']} steps")

        # Count particles by type
        counts = {}
        total = 0
        for spec in self._particle_specs:
            tname = spec["type_name"]
            n = spec.get("count", 1)
            counts[tname] = counts.get(tname, 0) + n
            total += n
        if total > 0:
            detail = " + ".join(f"{v} {k}" for k, v in counts.items())
            lines.append(f"  Particles:    {total} total ({detail})")
            # Show model info per type
            for tname, mp in self._particle_types.items():
                model_name = [k for k, v in _MEMBRANE_MODELS.items() if v == mp.model][0]
                lines.append(f"    {tname}: model={model_name}")
        else:
            lines.append("  Particles:    0")

        n_obs = len(self._obstacles) + len(self._polygon_obstacles)
        if n_obs > 0:
            n_circle = sum(1 for o in self._obstacles if o["shape"] == "circle")
            n_rect = sum(1 for o in self._obstacles if o["shape"] == "rect")
            n_poly = len(self._polygon_obstacles)
            lines.append(f"  Obstacles:    {n_obs} "
                         f"({n_circle} circle, {n_rect} rect, {n_poly} polygon)")

        if self._interactions:
            pairs = ", ".join(f"{k} ({v['style']})" for k, v in self._interactions.items())
            lines.append(f"  Interactions: {pairs}")

        if self._outputs:
            for out in self._outputs:
                lines.append(f"  Output:       {out['format']} every {out['interval']} "
                             f"steps -> {out['directory']}/")
        if self._thermo_interval > 0:
            lines.append(f"  Thermo:       every {self._thermo_interval} steps")

        lines.append(f"  Timestep:     {self._dt}")
        lines.append("=" * 60)
        return "\n".join(lines)

    # ══════════════════════════════════════════════════════════
    # Internal: validation
    # ══════════════════════════════════════════════════════════

    def _require_domain(self, command: str):
        if self._domain is None:
            raise RuntimeError(
                f"'{command}()' requires domain to be set first. "
                f"Call sim.domain(nx, ny) before sim.{command}().")

    def _ensure_initialized(self):
        if not self._initialized or self._core is None:
            raise RuntimeError(
                "Simulation not initialized. Call sim.initialize() or sim.run() first.")

    def _validate_config(self):
        """Check that all required configuration is present."""
        if self._domain is None:
            raise RuntimeError("Domain not set. Call sim.domain(nx, ny).")

        # Fluid is required unless "none"
        if not self._fluid_config:
            raise RuntimeError("Fluid not configured. Call sim.fluid(...).")

        # Check particle types referenced in specs
        for spec in self._particle_specs:
            tname = spec["type_name"]
            if tname not in self._particle_types:
                raise RuntimeError(
                    f"Particle type {tname!r} used in generate/particle "
                    f"but never defined with particle_type().")

        # Check regions referenced in specs
        for spec in self._particle_specs:
            if spec["mode"] == "generate":
                rname = spec["region"]
                if rname not in self._regions:
                    raise RuntimeError(
                        f"Region {rname!r} used in generate() "
                        f"but never defined with region().")

        # Check scalar sources reference valid types
        for src in self._scalar_sources:
            if src["type_name"] not in self._particle_types:
                raise RuntimeError(
                    f"Scalar source references unknown particle type {src['type_name']!r}.")

    # ══════════════════════════════════════════════════════════
    # Internal: build C++ simulation
    # ══════════════════════════════════════════════════════════

    def _build_params(self):
        """Create SimulationParams from stored configuration."""
        p = _sc.SimulationParams()
        p.nx = self._domain["nx"]
        p.ny = self._domain["ny"]
        p.dt = self._dt

        # Fluid parameters
        fc = self._fluid_config
        fp = p.fluid
        if fc.get("type") != "none":
            fp.tau = fc.get("tau", 0.8)
            fp.rho0 = fc.get("rho0", 1.0)
            if "inlet_velocity" in fc:
                fp.inlet_velocity = fc["inlet_velocity"]
            if "outlet_density" in fc:
                fp.outlet_density = fc["outlet_density"]

        # Collision model
        coll = fc.get("collision", "bgk")
        if coll == "mrt":
            fp.use_mrt = True
            fp.mrt_se = fc.get("mrt_se", 1.4)
            fp.mrt_s_eps = fc.get("mrt_s_eps", 1.4)
            fp.mrt_sq = fc.get("mrt_sq", 1.4)
        elif coll == "regularized":
            fp.collision_model = _sc.CollisionModel.REGULARIZED

        # Body force
        fp.body_force_x = self._body_force[0]
        fp.body_force_y = self._body_force[1]

        # Gravity
        fp.gravity_x = self._gravity[0]
        fp.gravity_y = self._gravity[1]
        fp.apply_gravity_to_fluid = self._gravity_target in ("fluid", "all")
        fp.apply_gravity_to_capsules = self._gravity_target in ("capsules", "all")

        # Boundary type
        if self._boundary["x"] == "periodic":
            fp.boundary_type = _sc.BoundaryType.PERIODIC
        elif self._boundary["x"] == "closed":
            fp.boundary_type = _sc.BoundaryType.CLOSED
        else:
            fp.boundary_type = _sc.BoundaryType.INLET_OUTLET

        # Periodic Y
        if self._boundary["y"] == "periodic":
            fp.periodic_y = True

        # Independent capsule recirculation in x (periodic capsules while the
        # fluid uses a non-periodic, e.g. inlet/outlet, BC).
        if self._boundary.get("capsules") == "periodic":
            fp.capsule_periodic_x = True

        # Moving walls
        if self._moving_wall_config:
            fp.top_wall_velocity = self._moving_wall_config.get("top_velocity", 0.0)
            fp.bottom_wall_velocity = self._moving_wall_config.get("bottom_velocity", 0.0)

        # Viscosity contrast
        if self._viscosity_contrast_config.get("enabled"):
            fp.viscosity_contrast = True
            fp.viscosity_update_interval = self._viscosity_contrast_config.get("update_interval", 10)

        p.fluid = fp

        # Phase-1 IBM/body-force per-cell magnitude cap. The user-supplied
        # value (via sim.fluid(max_lattice_force=...)) overrides the C++
        # default (0.01) directly on the SimulationParams struct, since
        # max_lattice_force lives at the top level (not under .fluid).
        if "max_lattice_force" in self._fluid_config:
            p.max_lattice_force = self._fluid_config["max_lattice_force"]

        # IBM iterations (multi-direct forcing)
        if self._ibm_config:
            p.ibm_iterations = self._ibm_config.get("iterations", 1)

        # Repulsion from interactions
        rp = p.repulsion
        for pair_name in ("particle-particle", "particle-wall", "particle-obstacle"):
            if pair_name in self._interactions:
                inter = self._interactions[pair_name]
                if inter["style"] in ("morse", "dem"):
                    rp.epsilon = inter.get("epsilon", rp.epsilon)
                    rp.sigma = inter.get("sigma", rp.sigma)
                    rp.r_cut = inter.get("r_cut", rp.r_cut)
                    rp.power = inter.get("power", rp.power)
                    # DEM-style dissipative/frictional contact (Path A).
                    rp.damping_normal = inter.get("damping_normal", rp.damping_normal)
                    rp.friction_coeff = inter.get("friction_coeff", rp.friction_coeff)
        p.repulsion = rp

        # Lubrication
        lp = p.lubrication
        if self._lubrication_config.get("enabled"):
            lp.enabled = True
            lp.h_threshold = self._lubrication_config.get("h_threshold", 1.5)
            lp.h_min = self._lubrication_config.get("h_min", 0.1)
        else:
            lp.enabled = False
        p.lubrication = lp

        # Adhesion
        if self._adhesion_config.get("enabled"):
            ap = p.adhesion
            ap.enabled = True
            ap.k_on = self._adhesion_config.get("k_on", 0.001)
            ap.k_off = self._adhesion_config.get("k_off", 0.01)
            ap.k_bond = self._adhesion_config.get("k_bond", 0.05)
            ap.d_bond = self._adhesion_config.get("d_bond", 2.0)
            ap.F_crit = self._adhesion_config.get("F_crit", 0.01)
            ap.max_bonds_per_node = self._adhesion_config.get("max_bonds_per_node", 3)
            # Wall adhesion
            ap.wall_adhesion = self._adhesion_config.get("wall_adhesion", False)
            ap.wall_k_on = self._adhesion_config.get("wall_k_on", 0.001)
            ap.wall_k_off = self._adhesion_config.get("wall_k_off", 0.01)
            ap.wall_k_bond = self._adhesion_config.get("wall_k_bond", 0.05)
            ap.wall_receptor_spacing = self._adhesion_config.get("wall_receptor_spacing", 2.0)
            # Catch-slip bond model
            if self._adhesion_config.get("bond_model") == "catch_slip":
                ap.use_catch_slip = True
                ap.k_off_catch = self._adhesion_config.get("k_off_catch", 0.05)
                ap.F_catch = self._adhesion_config.get("F_catch", 0.02)
                ap.k_off_slip = self._adhesion_config.get("k_off_slip", 0.001)
                ap.F_slip = self._adhesion_config.get("F_slip", 0.01)
            # Adhesion-matrix auto-fill. Phase-1 contract is "empty
            # matrix means no adhesion"; the friendly wrapper restores
            # the ergonomic default by filling an all-pairs True matrix
            # sized to the number of registered particle types when no
            # explicit matrix is provided.
            user_matrix = self._adhesion_config.get("adhesion_matrix")
            if user_matrix is None:
                T = max(1, len(self._particle_types))
                ap.adhesion_matrix = [[True] * T for _ in range(T)]
            else:
                ap.adhesion_matrix = [list(map(bool, row))
                                       for row in user_matrix]
            p.adhesion = ap

        # Scalar transport
        if self._scalar_config.get("enabled"):
            sp = p.scalar
            sp.enabled = True
            n_sp = self._scalar_config.get("n_species", 1)
            diff_val = self._scalar_config.get("diffusivity", 0.01)
            # diffusivity is a list (one per species)
            if isinstance(diff_val, (list, tuple)):
                sp.diffusivity = list(diff_val)
            else:
                sp.diffusivity = [float(diff_val)] * n_sp
            sp.n_species = n_sp
            # inlet_concentration: concentration imposed at the left inlet BC
            # Falls back to initial_concentration if not explicitly set
            init_conc  = self._scalar_config.get("initial_concentration", 0.0)
            inlet_conc = self._scalar_config.get("inlet_concentration", None)
            if inlet_conc is None:
                inlet_conc = init_conc   # backward-compatible default
            if isinstance(inlet_conc, (list, tuple)):
                sp.inlet_concentration = [float(c) for c in inlet_conc]
            else:
                sp.inlet_concentration = [float(inlet_conc)] * n_sp
            sp.periodic_y = self._scalar_config.get("periodic_y", False)
            p.scalar = sp

        # Metrics
        if self._metrics_config:
            p.metrics_interval = self._metrics_config.get("interval", 5000)

        # Checkpoint — handled by Python run() loop, disable C++ checkpoint
        # to avoid double-saving
        p.checkpoint_interval = 0

        # Shan-Chen multiphase
        if self._fluid_phases > 1:
            sc = p.shan_chen
            sc.enabled = True
            sc.G = self._shan_chen_G
            if self._shan_chen_eos == 1:
                sc.eos_type = 1
                if self._shan_chen_cs:
                    sc.cs_a = self._shan_chen_cs.get("a", 0.25)
                    sc.cs_b = self._shan_chen_cs.get("b", 4.0)
                    sc.cs_T = self._shan_chen_cs.get("T", 0.04)
                    sc.cs_R = self._shan_chen_cs.get("R", 1.0)
            p.shan_chen = sc
            # Shan-Chen multiphase: density deviations from rho0 are expected
            # (liquid ~2x, vapor ~0.3x). Relax the checker to avoid false warnings.
            p.max_density_deviation = 10.0  # effectively disabled for multiphase

        # Output — use the first VTK output as primary
        for out in self._outputs:
            if out["format"] in ("vtk", "vtk_legacy"):
                p.output_dir = out["directory"]
                p.output_interval = out["interval"]
                p.output_format = "vtk"
                if out["format"] == "vtk_legacy":
                    p.vtk_format = "legacy"
                break
        else:
            for out in self._outputs:
                p.output_dir = out["directory"]
                p.output_interval = out["interval"]
                p.output_format = out["format"]
                break

        if not self._outputs:
            p.output_interval = 0  # disable output

        self._params = p

    def _build_simulation(self):
        """Create and initialize the C++ Simulation object."""
        p = self._params

        # Channel builder
        builder = _sc.ChannelBuilder(p.nx, p.ny)

        # Walls
        if self._boundary["y"] == "wall":
            builder.addWalls()

        # Boundary type
        if self._boundary["x"] == "periodic":
            builder.setBoundaryType(_sc.BoundaryType.PERIODIC)
        elif self._boundary["x"] == "closed":
            builder.setBoundaryType(_sc.BoundaryType.CLOSED)

        # Standard obstacles
        for obs in self._obstacles:
            if obs["shape"] == "circle":
                builder.addCirclePillar(obs["cx"], obs["cy"], obs["radius"])
            elif obs["shape"] == "rect":
                builder.addRectPillar(obs["x0"], obs["y0"], obs["x1"], obs["y1"])

        # Create simulation
        self._core = _sc.Simulation(p)
        self._core.setChannelBuilder(builder)

        # Apply polygon obstacles to lattice field after builder
        # (polygons need the field to be created first, so we do this after init)

        # Add particles
        for spec in self._particle_specs:
            tname = spec["type_name"]
            mp = self._particle_types[tname]
            tid = self._particle_type_ids[tname]

            if spec["mode"] == "single":
                cx, cy = spec["center"]
                self._core.addCapsule(
                    _sc.Vec2d(cx, cy),
                    spec["radius"],
                    spec["num_nodes"],
                    mp, tid,
                )
            elif spec["mode"] == "generate":
                x0, y0, x1, y1 = self._regions[spec["region"]]

                if spec["method"] == "random":
                    self._core.addCapsuleRandom(
                        spec["count"],
                        x0, y0, x1, y1,
                        spec["rmin"], spec["rmax"],
                        spec["num_nodes"],
                        mp, tid,
                        spec.get("seed", 12345),
                        spec.get("min_gap", 1.0),
                    )
                elif spec["method"] == "hexagonal":
                    self._generate_hexagonal(spec, mp, tid, x0, y0, x1, y1)
                elif spec["method"] == "lattice":
                    self._generate_lattice(spec, mp, tid, x0, y0, x1, y1)
                elif spec["method"] in ("fill", "dense"):
                    self._generate_fill(spec, mp, tid, x0, y0, x1, y1)
                elif spec["method"] == "pour":
                    self._generate_pour(spec, mp, tid, x0, y0, x1, y1)
                else:
                    raise ValueError(f"Unknown generation method: {spec['method']!r}")

        # Enable free-surface BEFORE initialize() so empty regions are applied
        # inside initialize() after the LBM is set up
        if hasattr(self, "_free_surface_config") and self._free_surface_config.get("enabled"):
            fsc = self._free_surface_config
            self._core.enableFreeSurface(fsc["rho_atm"], fsc["threshold"])
            for reg in fsc["empty_regions"]:
                self._core.setEmptyRegion(
                    reg["x0"], reg["y0"], reg["x1"], reg["y1"]
                )

        # Initialize the C++ simulation
        self._core.initialize()

        # Re-initialize scalar domain to initial_concentration (may differ from
        # inlet_concentration which is stored in sp.inlet_concentration for BC use)
        if self._scalar_config.get("enabled"):
            _ad = self._core.advectionDiffusion()
            if _ad is not None:
                _init = self._scalar_config.get("initial_concentration", 0.0)
                if isinstance(_init, (list, tuple)):
                    _init = float(_init[0])
                _ad.initialize(float(_init))

        # Apply polygon obstacles (mark cells as SOLID)
        if self._polygon_obstacles:
            field = self._core.lbmSolver().field()
            for verts in self._polygon_obstacles:
                vec2d_verts = [_sc.Vec2d(v[0], v[1]) for v in verts]
                poly = _sc.PolygonObstacle(vec2d_verts)
                for y in range(p.ny):
                    for x in range(p.nx):
                        if poly.contains(x, y):
                            field.setCellType(x, y, _sc.CellType.SOLID)

        # Apply polygon domain (mark cells OUTSIDE polygon as SOLID)
        if self._polygon_domains:
            field = self._core.lbmSolver().field()
            for verts in self._polygon_domains:
                vec2d_verts = [_sc.Vec2d(v[0], v[1]) for v in verts]
                poly = _sc.PolygonObstacle(vec2d_verts)
                for y in range(p.ny):
                    for x in range(p.nx):
                        if not poly.contains(x, y):
                            field.setCellType(x, y, _sc.CellType.SOLID)

        # Set scalar source/sink rates (constant-rate + physics-based)
        for src in self._scalar_sources:
            tname = src["type_name"]
            tid = self._particle_type_ids[tname]
            # Constant-rate (backward-compatible)
            if src.get("release_rate", 0) > 0:
                self._core.setScalarReleaseRate(tid, src["release_rate"])
            if src.get("absorption_rate", 0) > 0:
                self._core.setScalarAbsorptionRate(tid, src["absorption_rate"])
            # Physics-based leaching (Fick, activates applyChemistry path)
            if src.get("k_leach", 0) > 0:
                self._core.setLeachingParams(tid, src["k_leach"], src.get("C_eq", 0.0))
            if src.get("M_p_initial", 0) > 0:
                self._core.setParticleMass(tid, src["M_p_initial"])
            # Langmuir adsorption/desorption
            if src.get("k_adsorb", 0) > 0 or src.get("k_desorb", 0) > 0:
                self._core.setAdsorptionParams(
                    tid,
                    src.get("k_adsorb", 0.0),
                    src.get("k_desorb", 0.0),
                    src.get("Gamma_max", 1.0),
                )

        # Set initial velocities for particles if specified
        caps = self._core.capsules()
        cap_idx = 0
        for spec in self._particle_specs:
            vx, vy = spec.get("velocity", (0.0, 0.0))
            count = spec.get("count", 1)
            if vx != 0.0 or vy != 0.0:
                for i in range(count):
                    if cap_idx + i < caps.numCapsules():
                        cap = caps[cap_idx + i]
                        for n in range(cap.numNodes()):
                            cap.setNodeVelocity(n, _sc.Vec2d(vx, vy))
            cap_idx += count

        # Print info
        print(self.info())

    def _generate_lattice(self, spec, mp, tid, x0, y0, x1, y1):
        """Generate particles on a regular lattice grid."""
        count = spec["count"]
        r = spec["rmin"]  # fixed radius for lattice
        spacing = spec.get("spacing")

        if spacing is None:
            width = x1 - x0
            height = y1 - y0
            aspect = width / max(height, 1e-10)
            ny_grid = max(1, int(math.sqrt(count / aspect)))
            nx_grid = max(1, int(math.ceil(count / ny_grid)))
            spacing_x = width / (nx_grid + 1)
            spacing_y = height / (ny_grid + 1)
        else:
            spacing_x = spacing_y = spacing
            nx_grid = max(1, int((x1 - x0) / spacing))
            ny_grid = max(1, int((y1 - y0) / spacing))

        placed = 0
        for iy in range(ny_grid):
            for ix in range(nx_grid):
                if placed >= count:
                    break
                cx = x0 + (ix + 1) * spacing_x
                cy = y0 + (iy + 1) * spacing_y
                if x0 + r < cx < x1 - r and y0 + r < cy < y1 - r:
                    self._core.addCapsule(
                        _sc.Vec2d(cx, cy), r,
                        spec["num_nodes"], mp, tid)
                    placed += 1
            if placed >= count:
                break

    def _generate_hexagonal(self, spec, mp, tid, x0, y0, x1, y1):
        """Generate particles on a hexagonal close-packed grid with jitter.

        Hexagonal packing gives ~15% higher density than square lattice.
        Odd rows are offset by half the spacing for close-packing.
        Small random jitter is added for polydisperse radii.
        """
        import random as _rnd

        count = spec["count"]
        rmin = spec["rmin"]
        rmax = spec["rmax"]
        r_avg = (rmin + rmax) / 2.0
        min_gap = spec.get("min_gap", 1.0)
        seed = spec.get("seed", 12345)
        spacing = spec.get("spacing")

        _rnd.seed(seed)

        if spacing is None:
            spacing = 2.0 * rmax + min_gap

        # Hexagonal: row spacing = spacing * sqrt(3)/2
        row_spacing = spacing * math.sqrt(3.0) / 2.0

        # Generate all candidate positions
        candidates = []
        margin = rmax + min_gap
        iy = 0
        cy = y0 + margin
        while cy < y1 - margin:
            offset = spacing / 2.0 if (iy % 2 == 1) else 0.0
            cx = x0 + margin + offset
            while cx < x1 - margin:
                candidates.append((cx, cy))
                cx += spacing
            cy += row_spacing
            iy += 1

        # Shuffle and pick 'count' positions (seeded)
        _rnd.shuffle(candidates)

        placed = 0
        placed_positions = []  # (cx, cy, r)
        for cx, cy in candidates:
            if placed >= count:
                break

            # Polydisperse radius
            r = _rnd.uniform(rmin, rmax)

            # Add small jitter (up to 20% of gap)
            jitter = min_gap * 0.2
            cx += _rnd.uniform(-jitter, jitter)
            cy += _rnd.uniform(-jitter, jitter)

            # Verify no overlap with already placed capsules (including
            # capsules from previous generate() calls)
            overlap = False
            for px, py, pr in placed_positions:
                dist = math.sqrt((cx - px)**2 + (cy - py)**2)
                if dist < r + pr + min_gap:
                    overlap = True
                    break

            # Also check existing capsules in the system
            if not overlap:
                ncaps = self._core.capsules().numCapsules()
                for c in range(ncaps):
                    cap = self._core.capsules()[c]
                    oc = cap.centroid()
                    or_ = cap.effectiveRadius()
                    dist = math.sqrt((cx - oc.x)**2 + (cy - oc.y)**2)
                    if dist < r + or_ + min_gap:
                        overlap = True
                        break

            if not overlap:
                nn = spec["num_nodes"]
                self._core.addCapsule(
                    _sc.Vec2d(cx, cy), r, nn, mp, tid)
                placed_positions.append((cx, cy, r))
                placed += 1

        print(f"Placed {placed}/{count} capsules "
              f"(hexagonal, {len(candidates)} sites available)")

    def _generate_fill(self, spec, mp, tid, x0, y0, x1, y1):
        """Deterministically fill a region with a dense, non-overlapping packing.

        This is the robust dense-packing method. Unlike ``random`` and
        ``hexagonal`` — which place candidates and then *reject* on overlap
        (the stochastic rejection, compounded by jitter, severely under-fills
        dense regions) — ``fill`` lays capsules on a regular grid whose spacing
        ``s = 2*rmax + min_gap`` guarantees the minimum surface gap *by
        construction*. No overlap test is needed, so the region packs to its
        geometric capacity every time.

        Packing (``spec["packing"]``):
          ``"hex"`` (default) — hexagonal close packing (rows offset by s/2,
              row pitch s*sqrt(3)/2); ~15 % denser than square.
          ``"square"`` — square lattice (row pitch s).

        Placement order is row-major (bottom-up), i.e. the region fills "part by
        part" from the inlet wall outward. If ``count`` exceeds the region
        capacity, the region is filled and a warning is logged; if ``count`` is
        smaller, the first ``count`` grid sites are used.

        Centres are kept at least ``rmax`` from the region edges so the membrane
        stays inside the region. For polydisperse radii the placed radius is
        ``rmin`` while the spacing uses ``rmax`` (conservative — never overlaps).
        """
        import logging
        import random as _rnd

        count = spec["count"]
        rmin = spec["rmin"]
        rmax = spec["rmax"]
        min_gap = spec.get("min_gap", 1.0)
        packing = spec.get("packing", "hex")
        # Optional seed-dependent jitter: displaces each capsule by up to
        # +-jitter (uniform) so different seeds give DISTINCT dense packings
        # (the base grid is otherwise deterministic). The base spacing already
        # carries ``min_gap`` of surface slack, so a jitter up to ``min_gap/2``
        # is absorbed by that slack and stays non-overlapping WITHOUT enlarging
        # the grid -- crucial, because enlarging the spacing would lower the
        # packing density and destroy the very contacts the clog study needs.
        jitter = min(float(spec.get("jitter", 0.0)), 0.5 * min_gap)
        spacing = spec.get("spacing") or (2.0 * rmax + min_gap)
        row_pitch = spacing * (math.sqrt(3.0) / 2.0) if packing == "hex" else spacing
        margin = rmax  # base grid; a +-jitter nudge past the seeding bound is harmless
        rng = _rnd.Random(spec.get("seed", 12345))

        # Build the full grid of valid centres (capacity), row-major.
        sites: list[tuple[float, float]] = []
        iy = 0
        cy = y0 + margin
        while cy <= y1 - margin + 1e-9:
            offset = (spacing / 2.0) if (packing == "hex" and iy % 2 == 1) else 0.0
            cx = x0 + margin + offset
            while cx <= x1 - margin + 1e-9:
                sites.append((cx, cy))
                cx += spacing
            cy += row_pitch
            iy += 1

        capacity = len(sites)
        n_place = min(count, capacity)
        for cx, cy in sites[:n_place]:
            if jitter > 0.0:
                cx += rng.uniform(-jitter, jitter)
                cy += rng.uniform(-jitter, jitter)
            self._core.addCapsule(_sc.Vec2d(cx, cy), rmin, spec["num_nodes"], mp, tid)

        if count > capacity:
            logging.getLogger("pysoftflow").warning(
                "fill: region capacity is %d capsules (%s packing, spacing=%.2f, "
                "jitter=%.2f); requested %d — placed %d (region saturated).",
                capacity, packing, spacing, jitter, count, n_place,
            )
        print(f"Placed {n_place}/{count} capsules "
              f"(fill/{packing}, jitter={jitter:.2f}, capacity {capacity})")

    def _generate_pour(self, spec, mp, tid, x0, y0, x1, y1):
        """Stack ALL requested capsules in a dense column for a gravity pour.

        Unlike ``fill`` (which caps at the region's single-layer capacity),
        ``pour`` uses the region only as the column *footprint* (its x-extent
        and floor ``y0``) and stacks hex-packed rows **upward as tall as needed**
        to hold all ``count`` capsules. Combined with ``sim.gravity(0, -g)`` the
        column then settles/flows downward — the hopper/silo "pour" used for
        granular-style filling and jamming studies.

        This guarantees every requested capsule is placed (the column simply
        grows taller), whereas ``fill`` saturates at the region area. If the
        column would rise above the domain top, only the rows that fit are
        placed and a warning is logged.
        """
        import logging

        count = spec["count"]
        rmin = spec["rmin"]
        rmax = spec["rmax"]
        min_gap = spec.get("min_gap", 1.0)
        spacing = spec.get("spacing") or (2.0 * rmax + min_gap)
        row_pitch = spacing * (math.sqrt(3.0) / 2.0)
        margin = rmax
        ny_domain = self._domain["ny"] if self._domain else y1

        # Number of columns that fit across the footprint width.
        usable_w = (x1 - x0) - 2.0 * margin
        n_cols = max(1, int(usable_w // spacing) + 1)

        placed = 0
        iy = 0
        cy = y0 + margin
        y_ceiling = ny_domain - margin
        while placed < count and cy <= y_ceiling + 1e-9:
            offset = (spacing / 2.0) if (iy % 2 == 1) else 0.0
            for ix in range(n_cols):
                if placed >= count:
                    break
                cx = x0 + margin + offset + ix * spacing
                if cx > x1 - margin + 1e-9:
                    break
                self._core.addCapsule(_sc.Vec2d(cx, cy), rmin, spec["num_nodes"], mp, tid)
                placed += 1
            cy += row_pitch
            iy += 1

        if placed < count:
            logging.getLogger("pysoftflow").warning(
                "pour: column reached the domain top after %d of %d capsules "
                "(widen the footprint or raise the domain).", placed, count,
            )
        print(f"Placed {placed}/{count} capsules "
              f"(pour, {n_cols}-wide column, {iy} rows)")

    # ══════════════════════════════════════════════════════════
    # Internal: thermo output
    # ══════════════════════════════════════════════════════════

    def _print_thermo(self, step_num: int):
        """Print LAMMPS-style thermo output."""
        caps = self._core.capsules()
        max_speed = self._core.getMaxSpeed()

        max_D = 0.0
        for c in range(caps.numCapsules()):
            D = caps[c].deformationIndex()
            if D > max_D:
                max_D = D

        line = (f"  Step {step_num:>8d} | max_speed={max_speed:.5f} | "
                f"capsules={caps.numCapsules()} | max_D={max_D:.4f}")

        # Add adhesion bond count if enabled
        if self._adhesion_config.get("enabled"):
            try:
                bonds = self._core.adhesion().getBonds()
                line += f" | bonds={len(bonds)}"
            except Exception:
                pass

        print(line)
