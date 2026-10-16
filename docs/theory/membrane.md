# Membrane mechanics — spring network with selectable constitutive law

## Discretization

A capsule is a closed loop of `N` Lagrangian nodes
`X_k = (x_k, y_k)`, connected by edges. Forces are accumulated on
each node from four sources:

1. **Stretching** (selectable model)
2. **Bending** (Laplacian or Helfrich curvature)
3. **Area conservation**
4. **Perimeter conservation** (optional)
5. **Membrane viscosity** (Kelvin–Voigt strain-rate damping)

(`src/membrane/capsule.h:`, `src/membrane/capsule.cpp`.)

All force computations use a **periodic-x minimum-image** convention
(`Capsule::minImageDiff`, `capsule.h:124–132`) so capsules straddling
the periodic seam still see consistent geometry.

## Stretching laws (`MembraneParams::model`)

Let `λ = L / L₀` for an edge with current length `L` and rest length
`L₀`. The implemented force-along-edge laws are:

| Model         | Force                                            | Reference                        |
| ------------- | ------------------------------------------------ | -------------------------------- |
| `HOOKEAN`     | `F = k_stretch (L/L₀ − 1)`                       | linear, default                   |
| `NEO_HOOKEAN` | `F = G_s (λ − 1/λ³)`                             | Barthes-Biesel 2016              |
| `SKALAK`      | `F = G_s λ (λ² − 1) (1 + C_skalak)`              | Skalak, Tozeren, Zarda, Chien 1973 |
| `WLC`         | `F = (k_BT/p) [ 1/(4(1−x)²) − ¼ + x ] − k_pow / L`, `x = L/L_max` | Fedosov, Caswell, Karniadakis 2010 |

(`capsule.cpp:286, 312–313, 350, 396–400`.) For biological capsules
(red blood cells, drug carriers) **Skalak** is the recommended
default: it is strain-hardening and area-stiff, matching the RBC
membrane response observed in single-cell stretch experiments. WLC
auto-tunes its parameters to match Skalak shear modulus at rest
(`capsule.cpp:84–99`).

## Bending

Two options:

- **Discrete Laplacian (default):** `F_i = k_bend (d_{i−1} + d_{i+1})`
  with `d_j = X_j − X_i`. Cheap, isotropic for closed loops with
  uniform spacing.
  (`capsule.cpp:430–431`.)
- **Helfrich curvature:** discrete turning-angle `θ` per node gives
  `κ = θ / ds`; the force is `F_i = k_b (κ − κ₀) n̂ ds` where `κ₀` is
  spontaneous curvature. Sign convention uses `atan2(cross, dot)` to
  stay continuous through `±π`.
  (`capsule.cpp:478–479`.)

## Area & perimeter conservation

Reference area `A₀` and perimeter `P₀` are computed at construction
(Shoelace formula with periodic-x unwrap). Penalties:

```
F_area      = − k_area      (A − A₀) / A₀ · n̂_node ds_node,
F_perimeter = − k_perimeter (P − P₀) / P₀ · t̂_node.
```

(`capsule.cpp:489–533`.)

## Membrane viscosity (Kelvin–Voigt)

A strain-rate damping along each edge:

```
F_visc = η_membrane (L − L_prev) / L₀.
```

`L_prev` is initialized on the first call (no impulse on init).
(`capsule.cpp:583–600`.)

## Time integration

Forward-Euler: `X_k(t+Δt) = X_k(t) + U_k(t) Δt`, where `U_k` is
interpolated from the lattice via the IBM. (`capsule.cpp:245`,
called from `simulation.cpp:389`.) Adequate for the soft-spring time
scales and the IBM-LBM Δt = 1; a velocity-Verlet upgrade is on the
Phase-2 list for stiff-membrane / drug-delivery cases.

## Wall clamping

After each step, `simulation.cpp:392–402` clamps node y-positions
to `[1.5, ny−1.5]` when `periodic_y` is off. This silently masks
LBM-IBM blow-ups; a future fix should log occurrences and prefer
soft wall repulsion (already available via `RepulsionForce`).

## V&V cases

`tests/verification/test_capsule_shear.cpp` (Skotheim–Secomb tank-
treading frequency vs Capillary number) is listed in
[`tests/verification/TODO.md`](../../tests/verification/TODO.md).

## References

- R. Skalak, A. Tozeren, R. P. Zarda, S. Chien, *Biophys. J.* **13**,
  245 (1973).
- D. Barthès-Biesel, *Annu. Rev. Fluid Mech.* **48**, 25 (2016).
- D. A. Fedosov, B. Caswell, G. E. Karniadakis, *Biophys. J.* **98**,
  2215 (2010).
- W. Helfrich, *Z. Naturforsch.* **28c**, 693 (1973).
- C. Pozrikidis, *Modeling and Simulation of Capsules and Biological
  Cells* (Chapman & Hall, 2003).
