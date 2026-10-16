# SoftFlow Python API reference

This is the full reference for the `SoftFlowSimulation` wrapper —
the LAMMPS-style declarative API that drives the C++ engine.

```python
from pysoftflow import SoftFlowSimulation

sim = SoftFlowSimulation()
# … chain configuration calls …
sim.run(50000)
```

Every configuration method returns `self`, so calls can chain. The
canonical setup order is:

1. **Geometry**: `domain()` → `boundary()` → `obstacle()` / `polygon_domain()`
2. **Fluid**: `fluid()` → optional `fluid_fill()`, `fluid_phases()`
3. **Forces**: `gravity()` / `body_force()` / `pressure_gradient()` / `moving_wall()`
4. **Coupling**: `coupling()` → optional `ibm()`
5. **Particle types**: `particle_type()` (one per species)
6. **Advanced physics**: `lubrication()`, `adhesion()`, `scalar_transport()`, `scalar_source()`, `viscosity_contrast()`, `metrics()`
7. **Regions & particles**: `region()` → `generate()` / `particle()`
8. **Units (optional)**: `units()`
9. **Output**: `output()`, `thermo()`, `data_output()`, `checkpoint()`
10. **Lifecycle**: `initialize()` (optional) → `warmup()` (optional) → `run()`

`initialize()` is called automatically by `run()` if you skip it.
`warmup()` is only needed for densely packed seedings.

---

## Quick reference

| Category | Method | Purpose |
|---|---|---|
| Geometry | [`domain`](#domain) | Set grid size |
|  | [`boundary`](#boundary) | x / y boundary conditions |
|  | [`obstacle`](#obstacle) | Solid obstacle (circle, rect, polygon) |
|  | [`polygon_domain`](#polygon_domain) | Confine flow to a polygon interior |
| Fluid | [`fluid`](#fluid) | LBM solver, viscosity, collision operator |
|  | [`fluid_fill`](#fluid_fill) | Restrict initial fluid to a region |
|  | [`fluid_phases`](#fluid_phases) | Shan-Chen multiphase |
| Forces | [`gravity`](#gravity) | Gravitational acceleration |
|  | [`body_force`](#body_force) | Constant body force on fluid |
|  | [`pressure_gradient`](#pressure_gradient) | Driving pressure gradient |
|  | [`moving_wall`](#moving_wall) | Couette-flow boundaries |
| Coupling | [`coupling`](#coupling) | Fluid-particle coupling |
|  | [`ibm`](#ibm) | Multi-direct-forcing IBM iterations |
| Particles | [`particle_type`](#particle_type) | Define a capsule species |
|  | [`region`](#region) | Named rectangular region |
|  | [`generate`](#generate) | Seed N particles into a region |
|  | [`particle`](#particle) | Place one particle at an exact position |
| Interactions | [`interaction`](#interaction) | Pairwise repulsion (LAMMPS-style) |
|  | [`lubrication`](#lubrication) | Close-contact lubrication |
|  | [`adhesion`](#adhesion) | Bell / catch-slip bonds |
| Chemistry | [`scalar_transport`](#scalar_transport) | Advection-diffusion solver |
|  | [`scalar_source`](#scalar_source) | Per-type leaching / Langmuir uptake |
| Free surface | [`enable_free_surface`](#enable_free_surface) | Wet-dry LBM |
|  | [`set_empty_region`](#set_empty_region) | Mark air region |
| Diagnostics | [`metrics`](#metrics) | Segregation/margination metrics |
|  | [`viscosity_contrast`](#viscosity_contrast) | Spatially-varying tau |
|  | [`dimensionless_numbers`](#dimensionless_numbers) | Re, Ca, Ma report |
| Units | [`units`](#units) | Map lattice ↔ SI |
|  | `to_physical_*` / `to_lattice_*` | Conversion helpers |
| Time | [`timestep`](#timestep) | Set dt (default 1.0) |
| Output | [`output`](#output) | VTK / CSV writer |
|  | [`thermo`](#thermo) | Console reporting |
|  | [`data_output`](#data_output) | CSV trajectory / fields for ML |
|  | [`checkpoint`](#checkpoint) | Periodic checkpoint cadence |
| Lifecycle | [`initialize`](#initialize) | Build the C++ simulation |
|  | [`warmup`](#warmup) | Equilibrate dense suspensions |
|  | [`run`](#run) | Run N steps |
|  | [`step`](#step) | Advance one step |
|  | [`save_checkpoint`](#save_checkpoint) / [`restart`](#restart) | Save/load state |

---

## Geometry

### `domain`

```python
sim.domain(nx: int, ny: int) -> SoftFlowSimulation
```

Set the lattice grid size in lattice units. Must be the **first**
call after creating the simulation.

| Parameter | Type | Description |
|---|---|---|
| `nx` | int | Number of columns (x extent). |
| `ny` | int | Number of rows (y extent). |

```python
sim.domain(nx=400, ny=80)        # narrow channel
sim.domain(nx=200, ny=200)       # square box
```

---

### `boundary`

```python
sim.boundary(x: str = "inlet_outlet", y: str = "wall") -> SoftFlowSimulation
```

Set boundary conditions on each axis.

| Parameter | Allowed values | Default | Description |
|---|---|---|---|
| `x` | `"periodic"`, `"inlet_outlet"`, `"closed"` | `"inlet_outlet"` | x-direction BC |
| `y` | `"wall"`, `"periodic"`, `"open"` | `"wall"` | y-direction BC |

`"closed"` makes all four sides solid walls. `"open"` removes the
y-walls (useful for free surface / dam break).

```python
sim.boundary(x="periodic", y="wall")     # Poiseuille channel
sim.boundary(x="inlet_outlet", y="wall") # driven by inlet velocity
```

---

### `obstacle`

```python
sim.obstacle(shape: str, **kwargs) -> SoftFlowSimulation
```

Add a solid obstacle inside the domain. The fluid sees it as a
no-slip wall (interpolated bounce-back for circles).

| `shape` | Required kwargs | Example |
|---|---|---|
| `"circle"` | `center=(cx, cy)`, `radius=r` | `sim.obstacle("circle", center=(100, 40), radius=8)` |
| `"rect"` | `p1=(x0, y0)`, `p2=(x1, y1)` | `sim.obstacle("rect", p1=(80, 30), p2=(120, 50))` |
| `"polygon"` | `vertices=[(x0,y0), ...]` (≥3) | `sim.obstacle("polygon", vertices=[(0,0),(10,0),(5,8)])` |

Call multiple times to build a pillar array, stenosis, etc.

```python
# DLD pillar array
for col in range(8):
    for row in range(4):
        cx = 50 + col * 35
        cy = 40 + (row - 1.5) * 14 + col * 3.5
        sim.obstacle("circle", center=(cx, cy), radius=4.0)
```

---

### `polygon_domain`

```python
sim.polygon_domain(vertices) -> SoftFlowSimulation
```

Confine the flow to the **interior** of a polygon — every lattice
cell outside the polygon is marked SOLID. Used for stenotic vessels,
bifurcations, organ-on-chip geometries.

```python
sim.polygon_domain(vertices=[(0,10),(80,10),(120,5),(180,10),(200,10),
                              (200,30),(180,30),(120,35),(80,30),(0,30)])
```

---

## Fluid

### `fluid`

```python
sim.fluid(
    type: str = "custom",
    method: str = "lbm",
    tau: float | None = None,
    density: float | None = None,
    viscosity: float | None = None,
    inlet_velocity: float | None = None,
    outlet_density: float | None = None,
    pressure: float | None = None,
    use_mrt: bool = False,
    collision: str = "bgk",
    max_lattice_force: float | None = None,
) -> SoftFlowSimulation
```

Configure the LBM fluid solver. **Either** `tau` **or** `viscosity`
(not both) — they're related by `tau = 3·ν + 0.5`.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `type` | str | `"custom"` | Preset: `"water"`, `"air"`, `"custom"`, `"none"`. |
| `tau` | float | from preset | BGK relaxation time. |
| `density` | float | from preset | Initial fluid density ρ₀ (lattice). |
| `viscosity` | float | derived from `tau` | Kinematic ν (lattice). |
| `inlet_velocity` | float | None | Inlet u for `inlet_outlet` BC. |
| `outlet_density` | float | None | Outlet ρ for `inlet_outlet` BC. |
| `collision` | str | `"bgk"` | `"bgk"`, `"mrt"`, or `"regularized"`. |
| `max_lattice_force` | float | 0.01 | Per-cell force cap (raise to 0.02–0.05 for dense / high-Re runs). |

```python
sim.fluid(tau=0.8)
sim.fluid(tau=0.7, collision="regularized", max_lattice_force=0.04)
sim.fluid(viscosity=1e-3, inlet_velocity=0.02)
```

**Stability cheat sheet:**
- `tau ≥ 0.6` recommended (BGK becomes unstable below ~0.55).
- `u_max ≤ 0.1` for `Ma < 0.17` (LBM compressibility).
- `regularized` collision is more stable than `bgk` at `tau` close to 0.5.

---

### `fluid_fill`

```python
sim.fluid_fill(mode: str = "all", name: str | None = None) -> SoftFlowSimulation
```

Restrict where the initial fluid is placed. Default fills the whole
domain. Use `"region"` for dam-break / partial-fill setups.

```python
sim.region("reservoir", x=(0, 100), y=(0, 60))
sim.fluid_fill(mode="region", name="reservoir")
```

---

### `fluid_phases`

```python
sim.fluid_phases(
    n: int = 1, G: float = -1.0, eos: str = "carnahan_starling",
    T: float = 0.06, a: float = 1.0, b: float = 4.0, R: float = 1.0,
) -> SoftFlowSimulation
```

Configure single-phase (default) or **Shan-Chen two-phase** fluid.
Two-phase enables droplet/bubble dynamics with a sharp interface.

```python
sim.fluid_phases(n=2, G=-5.5, eos="carnahan_starling")
```

---

## Forces

### `gravity`

```python
sim.gravity(gx: float = 0.0, gy: float = 0.0, apply_to: str = "all") -> SoftFlowSimulation
```

Gravitational acceleration in lattice units. With non-unit capsule
density, applies buoyancy `F = (ρ_cap − ρ_fluid) · A · g`.

| `apply_to` | Effect |
|---|---|
| `"fluid"` | Density-weighted body force on fluid only (`F = ρ·g`). |
| `"capsules"` | Buoyancy on capsules only. |
| `"all"` | Both (default). |

```python
sim.gravity(gy=-1e-4, apply_to="all")    # dam break
sim.gravity(gy=-5e-6, apply_to="capsules") # sedimentation
```

---

### `body_force`

```python
sim.body_force(fx: float = 0.0, fy: float = 0.0) -> SoftFlowSimulation
```

Constant uniform body force on the fluid (NOT density-weighted —
unlike `gravity`). The standard way to drive flow in a periodic
channel (Poiseuille).

```python
sim.body_force(fx=5e-6, fy=0)    # rightward Poiseuille drive
```

For a given target centerline velocity `U`, channel height `H`, and
viscosity `ν`: `fx = 8·ν·U / H²`.

---

### `pressure_gradient`

```python
sim.pressure_gradient(dp_dx: float = 0.0, dp_dy: float = 0.0) -> SoftFlowSimulation
```

Specify a pressure gradient; internally converted to an equivalent
body force `F = −∇p / (ρ·cs²)`.

---

### `moving_wall`

```python
sim.moving_wall(top_velocity: float = 0.0, bottom_velocity: float = 0.0) -> SoftFlowSimulation
```

Couette-style moving walls. Use with `boundary(y="wall")` and zero
body force for plane Couette shear.

```python
sim.moving_wall(top_velocity=0.01, bottom_velocity=-0.01)
```

---

## Coupling

### `coupling`

```python
sim.coupling(method: str = "ibm", mode: str = "two_way",
             delta_function: str = "peskin_4pt") -> SoftFlowSimulation
```

Configure how the fluid talks to the particles.

| Parameter | Allowed | Default | Description |
|---|---|---|---|
| `method` | `"ibm"`, `"none"` | `"ibm"` | Coupling method. |
| `mode` | `"two_way"`, `"one_way"` | `"two_way"` | Particles feed back to fluid (or not). |
| `delta_function` | `"peskin_4pt"` | `"peskin_4pt"` | IBM regularised delta. |

---

### `ibm`

```python
sim.ibm(iterations: int = 2) -> SoftFlowSimulation
```

Multi-direct-forcing IBM (Luo et al. 2007). `iterations=1` is plain
IBM; `2-3` reduces velocity slip at the membrane dramatically. For
dense suspensions, use `2`.

```python
sim.ibm(iterations=2)
```

---

## Particle types

### `particle_type`

```python
sim.particle_type(
    name: str,
    model: str = "hookean",
    k_stretch: float = 0.1,
    G_s: float | None = None,
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
) -> SoftFlowSimulation
```

Define a named capsule species and its membrane mechanics. Call once
per species *before* `generate()`. The species `name` is used later
when generating particles or attaching scalar sources.

#### Membrane models

| `model` | Constitutive law | Best for |
|---|---|---|
| `"hookean"` | Linear springs | Generic soft particle, smoke tests |
| `"neo_hookean"` | Barthes-Biesel hyperelastic | Polymer membranes |
| `"skalak"` | Area-dilatation resistant (Skalak 1973) | Red blood cells |
| `"wlc"` | Worm-Like Chain (Fedosov et al. 2010) | Cytoskeletal networks |

#### Common parameters

| Parameter | Used by | Description |
|---|---|---|
| `k_stretch` | hookean | Stretching stiffness (linear spring). |
| `G_s` | neo_hookean, skalak, wlc | Surface shear modulus. |
| `C_skalak` | skalak | Area-dilatation parameter C. |
| `k_bend` | all | Bending stiffness. |
| `k_area` | all | Area conservation. |
| `k_perimeter` | all | Perimeter conservation. |
| `gamma_visc` | all | Translational viscous damping. |
| `eta_membrane` | all | Kelvin-Voigt strain-rate damping. |
| `density` | all | Capsule density (1.0 = neutrally buoyant). |

```python
# RBC-like (soft, area-conserving)
sim.particle_type("rbc",
    model="skalak", G_s=0.03, C_skalak=10.0,
    k_bend=0.003, k_area=0.5, k_perimeter=0.05)

# Tumor cell-like (stiffer)
sim.particle_type("tumor",
    model="skalak", G_s=0.3, C_skalak=10.0,
    k_bend=0.02, k_area=1.0, k_perimeter=0.1)

# Sedimenting denser particle
sim.particle_type("dense_bead",
    model="hookean", k_stretch=0.5,
    density=1.5)
```

---

## Regions & particle generation

### `region`

```python
sim.region(name: str, x: tuple[float, float] = None,
           y: tuple[float, float] = None) -> SoftFlowSimulation
```

Define a named rectangular region. Used as a target for `generate()`
and `fluid_fill()`. Coordinates default to the whole domain.

```python
sim.region("inlet", x=(5, 50), y=(8, 72))
sim.region("downstream", x=(300, 380), y=(0, 80))
```

---

### `generate`

```python
sim.generate(
    type_name: str,
    count: int,
    region: str,
    radius: float | tuple[float, float] = 5.0,
    num_nodes: int | None = None,
    method: str = "random",
    spacing: float | None = None,
    velocity: tuple[float, float] = (0.0, 0.0),
    seed: int = 12345,
    min_gap: float = 1.0,
) -> SoftFlowSimulation
```

Generate `count` particles of the named type into a region.

| Parameter | Description |
|---|---|
| `type_name` | Must match a previous `particle_type(name=...)`. |
| `count` | Number to attempt to place. |
| `region` | Named region from `region()`. |
| `radius` | Float for fixed; `(rmin, rmax)` for polydisperse. |
| `num_nodes` | Membrane nodes per capsule (auto if `None` — gives `ds≈0.75·dx`). |
| `method` | `"random"` (RSA), `"hexagonal"` (close-packed), or `"lattice"` (grid). |
| `spacing` | Required for `"lattice"`; default for `"hexagonal"`. |
| `velocity` | Initial velocity per particle. |
| `seed` | RNG seed (reproducible). |
| `min_gap` | Minimum surface-to-surface gap in lattice units. |

```python
# Random monodisperse
sim.generate("rbc", count=40, region="inlet",
             radius=3.0, method="random", seed=42, min_gap=1.5)

# Polydisperse with size range
sim.generate("rbc", count=60, region="all",
             radius=(2.0, 4.0), method="random", seed=11)

# Dense hexagonal pack
sim.generate("tumor", count=80, region="cluster",
             radius=2.5, method="hexagonal")
```

**RSA jamming.** Random placement saturates around φ ≈ 0.55 in 2D.
If `count` exceeds that, late placements will fail (you'll see
`Placed 0/1 capsules (500 attempts)`). Lower `min_gap`, enlarge the
region, or switch to `"hexagonal"`.

---

### `particle`

```python
sim.particle(type_name: str, center: tuple[float, float],
             radius: float = 5.0, num_nodes: int | None = None,
             velocity: tuple[float, float] = (0.0, 0.0)) -> SoftFlowSimulation
```

Place a single capsule at an exact position. Useful for tests and
single-capsule benchmarks (tank-treading, parachute shape).

```python
sim.particle("rbc", center=(100, 40), radius=5.0)
```

---

## Pairwise interactions

### `interaction`

```python
sim.interaction(pair: str, style: str = "morse", **kwargs) -> SoftFlowSimulation
```

Define LAMMPS-style pairwise repulsion.

| `pair` | Between |
|---|---|
| `"particle-particle"` | Capsule ↔ capsule |
| `"particle-wall"` | Capsule ↔ channel wall |
| `"particle-obstacle"` | Capsule ↔ obstacle |

For `style="morse"`, supply `epsilon`, `sigma`, `r_cut`, `power`.

```python
sim.interaction("particle-particle", style="morse",
                epsilon=0.01, sigma=2.0, r_cut=4.0, power=2)
```

---

### `lubrication`

```python
sim.lubrication(enabled: bool = True, h_threshold: float = 1.5,
                h_min: float = 0.1) -> SoftFlowSimulation
```

Sub-grid lubrication correction for close (h < `h_threshold`)
capsule pairs. Prevents unphysical overlap when the LBM grid is too
coarse to resolve the squeeze film.

```python
sim.lubrication(enabled=True, h_threshold=1.5, h_min=0.1)
```

---

### `adhesion`

```python
sim.adhesion(
    enabled: bool = True,
    k_on: float = 0.001, k_off: float = 0.01,
    k_bond: float = 0.05, d_bond: float = 2.0,
    F_crit: float = 0.01, max_bonds_per_node: int = 3,
    wall_adhesion: bool = False,
    wall_k_on: float = 0.001, wall_k_off: float = 0.01,
    wall_k_bond: float = 0.05,
    bond_model: str = "bell",
    k_off_catch: float = 0.05, F_catch: float = 0.02,
    k_off_slip: float = 0.001, F_slip: float = 0.01,
    wall_receptor_spacing: float = 2.0,
    adhesion_matrix=None,
) -> SoftFlowSimulation
```

Reversible force-dependent bonds between membrane nodes.

#### Bond models

- **Bell model** (default): pure slip bond. Force always weakens the
  bond. `k_off(F) = k_off · exp(F/F_crit)`.
- **Catch-slip** (Thomas et al. 2008): biphasic. Low force
  *strengthens* the bond; high force breaks it. Biological examples:
  P-selectin/PSGL-1, FimH/mannose.
  `k_off(F) = k_off_catch·exp(−F/F_catch) + k_off_slip·exp(F/F_slip)`.

| Key parameter | Description |
|---|---|
| `k_on` | Bond formation rate. |
| `k_off` | Spontaneous dissociation (Bell). |
| `k_bond` | Spring stiffness of an active bond. |
| `d_bond` | Maximum distance for new bond formation. |
| `F_crit` | Bell-model critical force. |
| `max_bonds_per_node` | Cap on simultaneous bonds per node. |
| `bond_model` | `"bell"` or `"catch_slip"`. |
| `wall_adhesion` | Enable capsule-wall bonds. |
| `adhesion_matrix` | N×N bool matrix of which pairs can bond (None = all-pairs True). |

```python
# Simple homophilic adhesion (Bell)
sim.adhesion(enabled=True, k_on=0.05, k_off=0.001,
             k_bond=0.05, d_bond=2.0, F_crit=0.01,
             max_bonds_per_node=3, bond_model="bell")

# Catch-slip bonds (CTC cluster paper, Aceto 2014)
sim.adhesion(enabled=True, bond_model="catch_slip",
             k_on=0.05, k_off_catch=0.05, F_catch=0.02,
             k_off_slip=0.001, F_slip=0.01,
             k_bond=0.05, d_bond=6.0)
```

---

## Chemistry

### `scalar_transport`

```python
sim.scalar_transport(
    enabled: bool = True,
    diffusivity: float = 0.01,
    n_species: int = 1,
    initial_concentration: float = 0.0,
    inlet_concentration: float | None = None,
    periodic_y: bool = False,
) -> SoftFlowSimulation
```

Enable the advection-diffusion solver. Concentration is advected by
the LBM velocity field and diffuses with `D = diffusivity`.

```python
sim.scalar_transport(enabled=True, diffusivity=0.05,
                     n_species=1, initial_concentration=0.0,
                     inlet_concentration=1.0)
```

Stability: `D ≤ 1/6` (or the equivalent CFL).

---

### `scalar_source`

```python
sim.scalar_source(
    type_name: str,
    release_rate: float = 0.0,
    absorption_rate: float = 0.0,
    k_leach: float = 0.0, C_eq: float = 0.0, M_p_initial: float = 0.0,
    k_adsorb: float = 0.0, k_desorb: float = 0.0, Gamma_max: float = 1.0,
) -> SoftFlowSimulation
```

Attach a chemical source / sink to a particle type. Three coexisting
mechanisms:

| Mechanism | Activate by | Effect |
|---|---|---|
| Constant rate | `release_rate>0` or `absorption_rate>0` | Per-node constant flux. |
| Fick leaching | `k_leach>0` | `J = k_leach·(C_eq − C_surface)`; finite payload via `M_p_initial`. |
| Langmuir adsorption | `k_adsorb>0` and/or `k_desorb>0` | Surface coverage Γ with saturation `Γ_max`. |

```python
# Microplastic that leaches a contaminant
sim.scalar_source("mp", k_leach=0.005, C_eq=1.0, M_p_initial=100.0)

# RBC that adsorbs a biomarker
sim.scalar_source("rbc", k_adsorb=0.01, k_desorb=0.001, Gamma_max=1.0)
```

---

### `viscosity_contrast`

```python
sim.viscosity_contrast(enabled: bool = True, update_interval: int = 10) -> SoftFlowSimulation
```

Use a different `tau` inside vs outside capsules
(`τ_in = 3·λ·ν_out + 0.5`, with λ from each type's `viscosity_ratio`).
Adds a per-step ray-cast cost; refresh every `update_interval` steps.

```python
sim.particle_type("rbc", model="skalak", G_s=0.06, viscosity_ratio=5.0)
sim.viscosity_contrast(enabled=True, update_interval=10)
```

---

## Free-surface flow

### `enable_free_surface`

```python
sim.enable_free_surface(rho_atm: float = 1.0, threshold: float = 0.002) -> SoftFlowSimulation
```

Wet-dry LBM. "Empty" cells act as solid walls until adjacent fluid
pressure exceeds `rho_atm · (1 + threshold)`, then they wet. With
gravity, this gives a free surface from the bottom up.

### `set_empty_region`

```python
sim.set_empty_region(x: tuple, y: tuple) -> SoftFlowSimulation
```

Mark a rectangular region as initially empty (air). Call after
`enable_free_surface()`.

```python
sim.enable_free_surface(rho_atm=1.0, threshold=0.002)
sim.set_empty_region(x=(100, 200), y=(0, 60))   # air column
sim.gravity(gy=-1e-4)
```

---

## Diagnostics

### `metrics`

```python
sim.metrics(interval: int = 5000) -> SoftFlowSimulation
```

Enable in-situ segregation diagnostics: margination, mixing entropy,
cell-free-layer thickness, RDF, deformation. Results accessed via
`sim.get_segregation_results()`.

### `dimensionless_numbers`

```python
result = sim.dimensionless_numbers() -> dict
```

Prints Re, Ma, Ca (per particle type), and physical-unit equivalents
if `units()` was set. Returns the same values as a dict.

---

## Units and conversion

### `units`

```python
sim.units(
    dx: float | None = None,
    dt_phys: float | None = None,
    rho_phys: float | None = None,
    length_ref: float | None = None,
    velocity_ref: float | None = None,
) -> SoftFlowSimulation
```

Map lattice units to SI. Pick **one** method:

- **Direct scales**: `sim.units(dx=0.5e-6, dt_phys=1e-7, rho_phys=1000)`
- **Reference length + velocity**: `sim.units(length_ref=10e-6, velocity_ref=1e-3, rho_phys=1000)`

After `units()`, these helpers become available:

```python
sim.to_physical_length(L_lattice)       # → metres
sim.to_physical_time(steps)             # → seconds
sim.to_physical_velocity(u_lattice)     # → m/s
sim.to_physical_force(F_lattice)        # → N/m^3
sim.to_physical_viscosity(nu_lattice)   # → m^2/s
sim.to_lattice_length(L_phys)
sim.to_lattice_velocity(u_phys)
```

---

## Time

### `timestep`

```python
sim.timestep(dt: float = 1.0) -> SoftFlowSimulation
```

Set the lattice timestep. In LBM dt is essentially fixed at 1.0;
this is kept for backward compatibility.

---

## Output

### `output`

```python
sim.output(format: str = "vtk", directory: str = "output",
           interval: int = 100) -> SoftFlowSimulation
```

Add an output writer. Can be called multiple times for different
formats / directories.

| `format` | Files written |
|---|---|
| `"vtk"` | `fluid/*.vti`, `particles/*.vtp`, `*.pvd` time-series — open in ParaView. |
| `"vtk_legacy"` | Legacy `.vtk` ASCII files. |
| `"csv"` | Per-step CSV snapshots. |

```python
sim.output(format="vtk", directory="output/run01", interval=200)
```

---

### `thermo`

```python
sim.thermo(interval: int = 1000) -> SoftFlowSimulation
```

Print a one-line per-step status (step, %, u_max, deformation, ETA)
every `interval` steps. LAMMPS-style.

---

### `data_output`

```python
sim.data_output(
    enabled: bool = True,
    interval: int = 0,
    trajectory: bool = True,
    timeseries: bool = True,
    positions: bool = True,
    bonds: bool = True,
    node_positions: bool = False,
    velocity_field: bool = False,
    format: str = "csv",
    directory: str | None = None,
) -> SoftFlowSimulation
```

Write incremental CSV files for downstream ML / analysis. `interval=0`
inherits the thermo interval. **Warning:** `node_positions` and
`velocity_field` produce large files.

```python
sim.data_output(trajectory=True, node_positions=False,
                velocity_field=False, interval=2000)
```

Files written (in the VTK output dir unless `directory` is set):
`trajectory.csv`, `timeseries.csv`, `bond_timeseries.csv`,
`concentration_timeseries.csv`, `nodes_########.csv`,
`field_########.csv`.

---

### `checkpoint`

```python
sim.checkpoint(interval: int = 10000) -> SoftFlowSimulation
```

Enable periodic binary checkpoint dumps (full simulation state). The
files are written to the VTK output directory and named
`checkpoint_<step>.sfck`. See `save_checkpoint` / `restart` below for
manual control.

---

## Simulation lifecycle

### `initialize`

```python
sim.initialize() -> SoftFlowSimulation
```

Build the C++ `Simulation` from the declared configuration. Called
automatically by `run()`. Call it manually if you need access to
`sim.core` (the underlying C++ object) before running, e.g. to set
a step callback:

```python
sim.initialize()
sim.core.setStepCallback(my_callback)
sim.run(50000)
```

---

### `step`

```python
sim.step() -> None
```

Advance one timestep. Useful inside custom Python loops; `run()` is
preferred for normal use because it handles all the I/O and reporting.

---

### `warmup`

```python
sim.warmup(steps: int = 2000, ramp_steps: int = 1000) -> None
```

Two-phase equilibration for densely packed seedings:

1. **Phase 1 (`steps`):** zero body force, no inlet velocity.
   Repulsion + membrane forces relax overlapping capsules.
2. **Phase 2 (`ramp_steps`):** linearly ramp the body force / inlet
   velocity from 0 to its configured target.

```python
sim.warmup(steps=2000, ramp_steps=1000)
sim.run(50000)
```

---

### `run`

```python
sim.run(num_steps: int) -> None
```

Run `num_steps` timesteps with all configured output. This is the
normal entry point.

```python
sim = SoftFlowSimulation()
sim.domain(nx=400, ny=80)
sim.boundary(x="periodic", y="wall")
sim.fluid(tau=0.8)
sim.body_force(fx=5e-6, fy=0)
sim.particle_type("rbc", model="skalak", G_s=0.06)
sim.region("all", x=(10, 390), y=(10, 70))
sim.generate("rbc", count=20, region="all", radius=5.0)
sim.output(format="vtk", directory="out", interval=1000)
sim.thermo(interval=1000)
sim.run(50000)
```

---

### `save_checkpoint` / `restart`

```python
sim.save_checkpoint(filename: str | None = None) -> str
sim.restart(filename: str) -> None
```

Save and restore full simulation state (LBM distributions, capsule
positions/velocities, adhesion bonds, step counter).

```python
# Run 1: save
sim.run(50000)
sim.save_checkpoint("ckpt_50k.sfck")

# Run 2 (later, fresh process): restart and continue
sim2 = SoftFlowSimulation()
sim2.domain(nx=400, ny=100)             # same setup as before
# ... (all the same configuration calls) ...
sim2.restart("ckpt_50k.sfck")
sim2.run(50000)                          # continues from step 50000
```

---

## Property accessors

After `initialize()` or `run()`, the underlying C++ object is
exposed for advanced use:

| Property | Returns |
|---|---|
| `sim.core` | The C++ `Simulation` object (full power). |
| `sim.field` | LBM lattice field (densities, velocities). |
| `sim.capsules` | Capsule system. |
| `sim.lbm_solver` | LBM solver. |
| `sim.params` | Resolved parameter struct. |
| `sim.current_step` | Current timestep number. |
| `sim.nx`, `sim.ny` | Domain dimensions. |

```python
sim.initialize()
sim.core.setStepCallback(my_callback)      # called every step

# After run:
print("Final step:", sim.current_step)
u = sim.field.velocity()                   # numpy array (NY, NX, 2)
```

#### Result extraction

```python
sim.get_segregation_results()    # margination, mixing entropy, etc.
sim.get_concentration(species=0) # scalar field as numpy array
sim.get_adhesion_bonds()         # list of active bonds
```

---

## Common patterns

### Periodic Poiseuille channel with RBCs

```python
sim = SoftFlowSimulation()
sim.domain(nx=400, ny=80)
sim.boundary(x="periodic", y="wall")
sim.fluid(tau=0.8, collision="regularized")
sim.body_force(fx=3e-5, fy=0)
sim.coupling(method="ibm", mode="two_way")
sim.ibm(iterations=2)
sim.particle_type("rbc", model="skalak", G_s=0.06)
sim.region("seed", x=(20, 380), y=(8, 72))
sim.generate("rbc", count=30, region="seed", radius=3.0,
             method="random", seed=42, min_gap=1.5)
sim.output(format="vtk", directory="out", interval=500)
sim.thermo(interval=500)
sim.run(50000)
```

### Inlet-outlet channel with an obstacle

```python
sim = SoftFlowSimulation()
sim.domain(nx=600, ny=120)
sim.boundary(x="inlet_outlet", y="wall")
sim.fluid(tau=0.7, inlet_velocity=0.02, outlet_density=1.0)
sim.obstacle("circle", center=(200, 60), radius=15)
sim.particle_type("bead", model="hookean", k_stretch=0.5)
sim.region("inlet_band", x=(10, 50), y=(20, 100))
sim.generate("bead", count=40, region="inlet_band", radius=3.0)
sim.output(format="vtk", directory="out_obstacle", interval=200)
sim.run(20000)
```

### Two species with catch-slip adhesion (CTC clusters)

```python
sim = SoftFlowSimulation()
sim.domain(nx=400, ny=80)
sim.boundary(x="periodic", y="wall")
sim.fluid(tau=0.7, max_lattice_force=0.04, collision="regularized")
sim.body_force(fx=2e-5, fy=0)
sim.ibm(iterations=2)
sim.lubrication(enabled=True)

sim.particle_type("ctc", model="skalak", G_s=0.06,
                   k_bend=0.01, k_area=0.5, k_perimeter=0.075)

sim.adhesion(enabled=True, bond_model="catch_slip",
             k_on=0.05, k_off_catch=0.05, F_catch=0.02,
             k_off_slip=0.001, F_slip=0.01,
             k_bond=0.05, d_bond=6.0, max_bonds_per_node=3)

sim.region("seed", x=(20, 380), y=(8, 72))
sim.generate("ctc", count=9, region="seed",
             radius=4.0, method="hexagonal", seed=11)

sim.output(format="vtk", directory="out_ctc", interval=200)
sim.warmup(steps=1000, ramp_steps=1000)
sim.run(30000)
```

### Stenotic vessel with leaching microplastics

```python
sim = SoftFlowSimulation()
sim.domain(nx=600, ny=80)
sim.boundary(x="periodic", y="wall")
sim.fluid(tau=0.7, collision="regularized", max_lattice_force=0.04)
sim.body_force(fx=2e-5, fy=0)
sim.ibm(iterations=2)

# Stenosis (narrowing) — confine flow to a constricted region
sim.polygon_domain(vertices=[
    (0, 10), (250, 10), (300, 25), (350, 10), (600, 10),
    (600, 70), (350, 70), (300, 55), (250, 70), (0, 70),
])

sim.particle_type("mp",  model="skalak", G_s=0.1, density=1.05)
sim.particle_type("rbc", model="skalak", G_s=0.03)

sim.scalar_transport(enabled=True, diffusivity=0.05,
                     initial_concentration=0.0)
sim.scalar_source("mp", k_leach=0.005, C_eq=1.0, M_p_initial=100.0)
sim.scalar_source("rbc", k_adsorb=0.01, k_desorb=0.001)

sim.region("seed_rbc", x=(10, 240), y=(15, 65))
sim.region("seed_mp",  x=(10, 240), y=(15, 65))
sim.generate("rbc", count=30, region="seed_rbc", radius=3.0, seed=42)
sim.generate("mp",  count=5,  region="seed_mp",  radius=2.0, seed=43)

sim.output(format="vtk", directory="out_steno", interval=500)
sim.thermo(interval=500)
sim.warmup(steps=500, ramp_steps=1000)
sim.run(30000)
```

---

## See also

- [`examples/`](../../examples/) — five canonical demos (Poiseuille,
  segregation, diagnostics, drug delivery, tumour growth).
- [`docs/theory/`](../theory/) — discretisation and constitutive
  derivations referenced by each module.
