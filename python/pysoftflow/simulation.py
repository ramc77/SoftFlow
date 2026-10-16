"""
High-level Pythonic wrapper around the C++ SoftFlow Simulation.

Example
-------
    from pysoftflow import Simulation

    sim = Simulation(nx=400, ny=80)
    sim.set_fluid(tau=0.8, inlet_velocity=0.02)
    sim.add_wall("both")
    sim.add_capsules(10, region=(50, 10, 350, 70), radius_range=(4, 6),
                     model="skalak", G_s=0.06, C_skalak=10.0)
    sim.enable_lubrication()
    sim.set_output(directory="results", interval=200)
    sim.initialize()
    sim.run(20000, progress=True)
"""

from __future__ import annotations

import sys
from typing import Optional, Tuple, Union, List

import numpy as np

import softflow_core as _sc


# Membrane model name → enum mapping
_MEMBRANE_MODELS = {
    "hookean": _sc.MembraneModel.HOOKEAN,
    "neo_hookean": _sc.MembraneModel.NEO_HOOKEAN,
    "skalak": _sc.MembraneModel.SKALAK,
}

# Capsule shape name → enum mapping
_CAPSULE_SHAPES = {
    "circle":    _sc.CapsuleShape.CIRCLE,
    "ellipse":   _sc.CapsuleShape.ELLIPSE,
    "biconcave": _sc.CapsuleShape.BICONCAVE,
    "fiber":     _sc.CapsuleShape.FIBER,
}


class Simulation:
    """Pythonic facade over the C++ ``Simulation`` class."""

    def __init__(self, nx: int = 200, ny: int = 50, dx: float = 1.0):
        """Create a new simulation domain.

        Parameters
        ----------
        nx : int
            Lattice width (number of columns).
        ny : int
            Lattice height (number of rows).
        dx : float
            Lattice spacing (default 1.0 in lattice units).
        """
        self._params = _sc.SimulationParams()
        self._params.nx = nx
        self._params.ny = ny
        self._params.dt = dx
        self._builder = _sc.ChannelBuilder(nx, ny)
        self._capsule_specs: list[dict] = []
        self._polygon_obstacles: list = []
        self._scalar_sources: list = []
        self._initialized = False
        self._core: Optional[_sc.Simulation] = None

    # -- Fluid configuration --

    def set_fluid(
        self,
        viscosity: Optional[float] = None,
        density: Optional[float] = None,
        inlet_velocity: Optional[float] = None,
        outlet_density: Optional[float] = None,
        tau: Optional[float] = None,
        collision: str = "bgk",
    ) -> "Simulation":
        """Configure fluid parameters.

        Parameters
        ----------
        viscosity : float, optional
            Kinematic viscosity (sets tau = 3*viscosity + 0.5).
        density : float, optional
            Initial fluid density.
        inlet_velocity : float, optional
            Inlet velocity for inlet/outlet BCs.
        outlet_density : float, optional
            Outlet density for inlet/outlet BCs.
        tau : float, optional
            BGK relaxation time (overrides viscosity if both given).
        collision : str
            Collision operator: ``"bgk"``, ``"mrt"``, or ``"regularized"``.
        """
        fp = self._params.fluid
        if viscosity is not None:
            fp.tau = 3.0 * viscosity + 0.5
        if tau is not None:
            fp.tau = tau
        if density is not None:
            fp.rho0 = density
        if inlet_velocity is not None:
            fp.inlet_velocity = inlet_velocity
        if outlet_density is not None:
            fp.outlet_density = outlet_density
        if collision == "mrt":
            fp.use_mrt = True
        elif collision == "regularized":
            fp.collision_model = _sc.CollisionModel.REGULARIZED
        self._params.fluid = fp
        return self

    # -- Geometry helpers --

    def add_wall(self, which: str = "both") -> "Simulation":
        """Add solid walls (``"top"``, ``"bottom"``, or ``"both"``)."""
        if which in ("both", "top", "bottom"):
            self._builder.addWalls()
        else:
            raise ValueError(f"Unknown wall specification: {which!r}")
        return self

    def add_obstacle(
        self, obstacle: Union[_sc.CircleObstacle, _sc.RectObstacle, dict]
    ) -> "Simulation":
        """Add a solid obstacle to the channel."""
        if isinstance(obstacle, _sc.CircleObstacle):
            self._builder.addCirclePillar(
                obstacle.getCx(), obstacle.getCy(), obstacle.getRadius()
            )
        elif isinstance(obstacle, _sc.RectObstacle):
            self._builder.addRectPillar(
                obstacle.getX0(), obstacle.getY0(),
                obstacle.getX1(), obstacle.getY1()
            )
        elif isinstance(obstacle, dict):
            otype = obstacle.get("type", "")
            if otype == "circle":
                self._builder.addCirclePillar(
                    obstacle["cx"], obstacle["cy"], obstacle["radius"]
                )
            elif otype == "rect":
                self._builder.addRectPillar(
                    obstacle["x0"], obstacle["y0"],
                    obstacle["x1"], obstacle["y1"],
                )
            else:
                raise ValueError(f"Unknown obstacle type: {otype!r}")
        else:
            raise TypeError(f"Unsupported obstacle type: {type(obstacle)}")
        return self

    def add_circle_obstacle(
        self, cx: float, cy: float, radius: float
    ) -> "Simulation":
        """Shortcut: add a circular pillar."""
        self._builder.addCirclePillar(cx, cy, radius)
        return self

    def add_rect_obstacle(
        self, x0: float, y0: float, x1: float, y1: float
    ) -> "Simulation":
        """Shortcut: add a rectangular pillar."""
        self._builder.addRectPillar(x0, y0, x1, y1)
        return self

    def add_polygon_obstacle(self, vertices: List[Tuple[float, float]]) -> "Simulation":
        """Add a polygon obstacle defined by a list of (x, y) vertices."""
        self._polygon_obstacles.append(vertices)
        return self

    # -- Capsule (deformable particle) placement --

    def add_capsules(
        self,
        count: int,
        region: Tuple[float, float, float, float],
        radius_range: Union[Tuple[float, float], float] = 5.0,
        num_nodes: Optional[int] = None,
        *,
        model: str = "hookean",
        k_stretch: float = 0.1,
        G_s: Optional[float] = None,
        C_skalak: float = 10.0,
        k_bend: float = 0.005,
        k_area: float = 0.5,
        k_perimeter: float = 0.1,
        gamma_visc: float = 0.01,
        use_helfrich: bool = False,
        kappa_0: float = 0.0,
        viscosity_ratio: float = 1.0,
        capsule_type: int = 0,
        shape: str = "circle",
        aspect_ratio: float = 1.0,
        indent_depth: float = 0.4,
        is_rigid: bool = False,
    ) -> "Simulation":
        """Queue capsules for random placement inside *region*.

        Parameters
        ----------
        count : int
            Number of capsules.
        region : (x0, y0, x1, y1)
            Bounding box for random placement.
        radius_range : float or (min, max)
            Fixed radius or (min_radius, max_radius).
        num_nodes : int, optional
            Nodes per capsule membrane.
        model : str
            Membrane model: ``"hookean"``, ``"neo_hookean"``, or ``"skalak"``.
        k_stretch : float
            Stretching stiffness (hookean only).
        G_s : float, optional
            Surface shear modulus (neo_hookean/skalak).
        C_skalak : float
            Skalak area dilatation parameter.
        k_bend, k_area, k_perimeter, gamma_visc
            Membrane material parameters.
        use_helfrich : bool
            Use Helfrich curvature bending.
        kappa_0 : float
            Spontaneous curvature.
        viscosity_ratio : float
            Interior/exterior viscosity ratio.
        capsule_type : int
            Type tag for analysis.
        """
        if isinstance(radius_range, (int, float)):
            rmin = rmax = float(radius_range)
        else:
            rmin, rmax = radius_range

        self._capsule_specs.append(dict(
            count=count,
            region=region,
            rmin=rmin,
            rmax=rmax,
            num_nodes=num_nodes if num_nodes is not None else 0,
            model=model,
            k_stretch=k_stretch,
            G_s=G_s,
            C_skalak=C_skalak,
            k_bend=k_bend,
            k_area=k_area,
            k_perimeter=k_perimeter,
            gamma_visc=gamma_visc,
            use_helfrich=use_helfrich,
            kappa_0=kappa_0,
            viscosity_ratio=viscosity_ratio,
            capsule_type=capsule_type,
            shape=shape,
            aspect_ratio=aspect_ratio,
            indent_depth=indent_depth,
            is_rigid=is_rigid,
        ))
        return self

    def add_single_capsule(
        self,
        center: Tuple[float, float],
        radius: float,
        num_nodes: Optional[int] = None,
        *,
        model: str = "hookean",
        k_stretch: float = 0.1,
        G_s: Optional[float] = None,
        C_skalak: float = 10.0,
        k_bend: float = 0.005,
        k_area: float = 0.5,
        k_perimeter: float = 0.1,
        gamma_visc: float = 0.01,
        use_helfrich: bool = False,
        kappa_0: float = 0.0,
        viscosity_ratio: float = 1.0,
        capsule_type: int = 0,
        shape: str = "circle",
        aspect_ratio: float = 1.0,
        indent_depth: float = 0.4,
        is_rigid: bool = False,
    ) -> "Simulation":
        """Add a single capsule at an exact position."""
        self._capsule_specs.append(dict(
            count=1,
            center=center,
            radius=radius,
            num_nodes=num_nodes if num_nodes is not None else 0,
            model=model,
            k_stretch=k_stretch,
            G_s=G_s,
            C_skalak=C_skalak,
            k_bend=k_bend,
            k_area=k_area,
            k_perimeter=k_perimeter,
            gamma_visc=gamma_visc,
            use_helfrich=use_helfrich,
            kappa_0=kappa_0,
            viscosity_ratio=viscosity_ratio,
            capsule_type=capsule_type,
            shape=shape,
            aspect_ratio=aspect_ratio,
            indent_depth=indent_depth,
            is_rigid=is_rigid,
        ))
        return self

    # -- Output configuration --

    def set_output(
        self,
        directory: str = "output",
        format: str = "vtk",
        interval: int = 100,
    ) -> "Simulation":
        """Configure file output."""
        self._params.output_dir = directory
        self._params.output_format = format
        self._params.output_interval = interval
        return self

    # -- Optional physics modules --

    def enable_lubrication(
        self,
        h_threshold: float = 1.5,
        h_min: float = 0.1,
    ) -> "Simulation":
        """Enable lubrication corrections for close capsule interactions."""
        lp = self._params.lubrication
        lp.enabled = True
        lp.h_threshold = h_threshold
        lp.h_min = h_min
        self._params.lubrication = lp
        return self

    def enable_adhesion(
        self,
        k_on: float = 0.001,
        k_off: float = 0.01,
        k_bond: float = 0.05,
        d_bond: float = 2.0,
        F_crit: float = 0.01,
        max_bonds_per_node: int = 3,
    ) -> "Simulation":
        """Enable Bell model cell adhesion."""
        ap = self._params.adhesion
        ap.enabled = True
        ap.k_on = k_on
        ap.k_off = k_off
        ap.k_bond = k_bond
        ap.d_bond = d_bond
        ap.F_crit = F_crit
        ap.max_bonds_per_node = max_bonds_per_node
        self._params.adhesion = ap
        return self

    def enable_scalar_transport(
        self,
        diffusivity: float = 0.01,
        n_species: int = 1,
        initial_concentration: float = 0.0,
    ) -> "Simulation":
        """Enable advection-diffusion scalar transport."""
        sp = self._params.scalar
        sp.enabled = True
        sp.diffusivity = diffusivity
        sp.n_species = n_species
        sp.initial_concentration = initial_concentration
        self._params.scalar = sp
        return self

    def enable_viscosity_contrast(
        self,
        update_interval: int = 10,
    ) -> "Simulation":
        """Enable spatially varying viscosity (interior vs exterior)."""
        fp = self._params.fluid
        fp.viscosity_contrast = True
        fp.viscosity_update_interval = update_interval
        self._params.fluid = fp
        return self

    def enable_metrics(self, interval: int = 5000) -> "Simulation":
        """Enable segregation metrics computation."""
        self._params.metrics_interval = interval
        return self

    def set_ibm_iterations(self, iterations: int = 2) -> "Simulation":
        """Set multi-direct forcing IBM iterations."""
        self._params.ibm_iterations = iterations
        return self

    def set_moving_wall(
        self,
        top_velocity: float = 0.0,
        bottom_velocity: float = 0.0,
    ) -> "Simulation":
        """Set moving wall velocities for Couette flow."""
        fp = self._params.fluid
        fp.top_wall_velocity = top_velocity
        fp.bottom_wall_velocity = bottom_velocity
        self._params.fluid = fp
        return self

    def enable_ml_surrogate(
        self,
        warmup: int = 5000,
        retrain_interval: int = 2000,
        error_threshold: float = 0.05,
        hidden_size: int = 64,
        learning_rate: float = 0.001,
        training_epochs: int = 50,
        buffer_size: int = 10000,
    ) -> "Simulation":
        """Enable the ML-accelerated IBM surrogate."""
        ml = self._params.ml
        ml.enabled = True
        ml.warmup_steps = warmup
        ml.retrain_interval = retrain_interval
        ml.error_threshold = error_threshold
        ml.hidden_size = hidden_size
        ml.learning_rate = learning_rate
        ml.training_epochs = training_epochs
        ml.buffer_size = buffer_size
        self._params.ml = ml
        return self

    def enable_surface_tension(self, G: float = -5.0) -> "Simulation":
        """Enable Shan-Chen multiphase surface tension."""
        sc = self._params.shan_chen
        sc.enabled = True
        sc.G = G
        self._params.shan_chen = sc
        return self

    def set_periodic_x(
        self,
        body_force_x: float = 0.0,
        body_force_y: float = 0.0,
    ) -> "Simulation":
        """Use periodic boundary conditions in x-direction."""
        fp = self._params.fluid
        fp.boundary_type = _sc.BoundaryType.PERIODIC
        fp.body_force_x = body_force_x
        fp.body_force_y = body_force_y
        self._params.fluid = fp
        return self

    def set_periodic_y(self) -> "Simulation":
        """Enable periodic boundaries in y-direction."""
        fp = self._params.fluid
        fp.periodic_y = True
        self._params.fluid = fp
        return self

    def enable_checkpoint(self, interval: int = 10000) -> "Simulation":
        """Enable periodic checkpoint saving."""
        self._params.checkpoint_interval = interval
        return self

    # -- Simulation lifecycle --

    def _make_membrane_params(self, spec: dict) -> _sc.MembraneParams:
        """Create MembraneParams from a capsule spec dict."""
        mp = _sc.MembraneParams()
        model = spec.get("model", "hookean")
        if model in _MEMBRANE_MODELS:
            mp.model = _MEMBRANE_MODELS[model]
        if model == "hookean":
            mp.k_stretch = spec["k_stretch"]
        else:
            mp.G_s = spec["G_s"] if spec.get("G_s") is not None else spec["k_stretch"]
            if model == "skalak":
                mp.C_skalak = spec.get("C_skalak", 10.0)
        mp.k_bend = spec["k_bend"]
        mp.k_area = spec["k_area"]
        mp.k_perimeter = spec["k_perimeter"]
        mp.gamma_visc = spec["gamma_visc"]
        if spec.get("use_helfrich"):
            mp.use_helfrich_bending = True
            mp.kappa_0 = spec.get("kappa_0", 0.0)
        mp.viscosity_ratio = spec.get("viscosity_ratio", 1.0)
        shape = spec.get("shape", "circle")
        mp.shape = _CAPSULE_SHAPES.get(shape, _sc.CapsuleShape.CIRCLE)
        mp.aspect_ratio = spec.get("aspect_ratio", 1.0)
        mp.indent_depth = spec.get("indent_depth", 0.4)
        mp.is_rigid = spec.get("is_rigid", False)
        return mp

    def initialize(self) -> "Simulation":
        """Build the C++ simulation and initialize all subsystems."""
        self._core = _sc.Simulation(self._params)
        self._core.setChannelBuilder(self._builder)

        # Register capsules
        for spec in self._capsule_specs:
            mp = self._make_membrane_params(spec)

            if "center" in spec:
                cx, cy = spec["center"]
                self._core.addCapsule(
                    _sc.Vec2d(cx, cy),
                    spec["radius"],
                    spec["num_nodes"],
                    mp,
                    spec["capsule_type"],
                )
            else:
                x0, y0, x1, y1 = spec["region"]
                self._core.addCapsuleRandom(
                    spec["count"],
                    x0, y0, x1, y1,
                    spec["rmin"], spec["rmax"],
                    spec["num_nodes"],
                    mp,
                    spec["capsule_type"],
                )

        self._core.initialize()

        # Apply polygon obstacles
        if self._polygon_obstacles:
            field = self._core.lbmSolver().field()
            nx, ny = self._params.nx, self._params.ny
            for verts in self._polygon_obstacles:
                vec2d_verts = [_sc.Vec2d(v[0], v[1]) for v in verts]
                poly = _sc.PolygonObstacle(vec2d_verts)
                for y in range(ny):
                    for x in range(nx):
                        if poly.contains(x, y):
                            field.setCellType(x, y, _sc.CellType.SOLID)

        # Set scalar source/sink rates
        for src in self._scalar_sources:
            self._core.setScalarReleaseRate(src["type_id"], src["release_rate"])
            self._core.setScalarAbsorptionRate(src["type_id"], src["absorption_rate"])

        self._initialized = True
        return self

    def step(self) -> None:
        """Advance one timestep."""
        self._ensure_initialized()
        self._core.step()

    def run(self, num_steps: int, progress: bool = True) -> None:
        """Run *num_steps* timesteps."""
        self._ensure_initialized()

        if not progress:
            self._core.run(num_steps)
            return

        report_every = max(1, num_steps // 100)
        for i in range(num_steps):
            self._core.step()
            if (i + 1) % report_every == 0 or (i + 1) == num_steps:
                pct = 100.0 * (i + 1) / num_steps
                sys.stderr.write(
                    f"\r  step {self._core.currentStep():>8d}  "
                    f"({pct:5.1f}%)"
                )
                sys.stderr.flush()
        sys.stderr.write("\n")

    # -- Data access --

    @property
    def current_step(self) -> int:
        self._ensure_initialized()
        return self._core.currentStep()

    def particle_positions(self) -> np.ndarray:
        """Return capsule centroids as an (N, 2) numpy array."""
        self._ensure_initialized()
        cs = self._core.capsules()
        n = cs.numCapsules()
        out = np.empty((n, 2), dtype=np.float64)
        for i in range(n):
            c = cs[i].centroid()
            out[i, 0] = c.x
            out[i, 1] = c.y
        return out

    def fluid_density(self) -> np.ndarray:
        """Return the fluid density field as a (ny, nx) numpy array view."""
        self._ensure_initialized()
        return np.asarray(self._core.field().density())

    def fluid_velocity(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (ux, uy) fluid velocity fields as numpy array views."""
        self._ensure_initialized()
        field = self._core.field()
        return np.asarray(field.velocity_x()), np.asarray(field.velocity_y())

    def get_segregation_results(self):
        """Get latest segregation metrics results."""
        self._ensure_initialized()
        return self._core.lastSegregationResults()

    def get_concentration(self, species: int = 0) -> np.ndarray:
        """Get scalar concentration field as numpy array."""
        self._ensure_initialized()
        ad = self._core.advectionDiffusion()
        return np.asarray(ad.concentration(species))

    def save_checkpoint(self, path: str) -> None:
        """Save simulation state to file."""
        self._ensure_initialized()
        self._core.saveCheckpoint(path)

    def load_checkpoint(self, path: str) -> None:
        """Load simulation state from checkpoint."""
        self._ensure_initialized()
        self._core.loadCheckpoint(path)

    @property
    def core(self) -> _sc.Simulation:
        """Direct access to the underlying C++ ``Simulation`` object."""
        self._ensure_initialized()
        return self._core

    @property
    def params(self) -> _sc.SimulationParams:
        return self._params

    @property
    def nx(self) -> int:
        return self._params.nx

    @property
    def ny(self) -> int:
        return self._params.ny

    def _ensure_initialized(self) -> None:
        if not self._initialized or self._core is None:
            raise RuntimeError(
                "Simulation has not been initialized. Call .initialize() first."
            )
