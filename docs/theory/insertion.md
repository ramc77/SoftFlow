# Particle insertion module (Phase 2)

The insertion module places capsules into a SoftFlow simulation
either once at setup (static fills) or every timestep (dynamic
inserters). It addresses the project specification verbatim and is intended to
be the principal substrate for the Phase-3 segregation diagnostics
and the Phase-4 / 5 drug-delivery and tumour-aggregation examples.

The implementation lives in `src/core/insertion/`; the Python facade
is `pysoftflow.insertion`; the showcase example is
`examples/02_bidisperse_segregation/`.

## 1. Architecture

Three orthogonal abstract classes:

```
IRegion              — where placements may land
ISizeDistribution    — what radius each placement gets
IInserter            — static: how the placements are arranged
IDynamicInserter     — dynamic: per-step placement events
```

Concrete classes are registered in shared_ptr-shaped pybind11 holders
so the same C++ object can be passed to multiple inserters
(e.g. the same `RectRegion` to a hex-lattice and a constant-flux
inserter) and round-trip across the C++/Python boundary cleanly.

We rejected `std::variant<…>` strategy holders because (a) the
runtime-dispatch cost is invisible — insertion is rare or per-N-step,
not per-cell; (b) graduate students can subclass `IInserter` /
`IRegion` for their own geometries without recompiling the engine;
(c) pybind11 trampolines compose cleanly with virtual bases but not
with `std::visit`-style dispatch.

## 2. Static-fill strategies

Every static inserter consumes an `InsertionContext` (walls,
obstacles, existing capsules, periodic-x state, `min_gap`) and
returns a `vector<Placement>`. The Simulation never mutates state
during the inserter call — it commits the result via `addCapsule`
afterwards.

| Strategy                  | Algorithm                              | Reference                                                                 |
| ------------------------- | -------------------------------------- | ------------------------------------------------------------------------- |
| `SquareLatticeInserter`   | Regular grid + optional jitter         | textbook                                                                  |
| `HexagonalLatticeInserter`| Close-packed hex (rows offset s/2)     | textbook (φ_max ≈ 0.9069 for monodisperse touching disks)                  |
| `RSAInserter`             | Random Sequential Addition             | Widom 1966; Talbot, Tarjus, Schaaf 2000 (φ_J ≈ 0.547 for 2-D disks)        |
| `PoissonDiskInserter`     | Bridson background-grid sampling       | Bridson 2007 (*ACM SIGGRAPH sketches*)                                    |

Tests pin (see `tests/unit/test_insertion_lattice.cpp`,
`test_insertion_rsa.cpp`):

- exact analytic count for the structured layouts,
- pairwise no-overlap and wall-envelope respect for all four,
- bit-exact reproducibility under a fixed seed,
- saturation density approaching `φ_J` for RSA at large
  `max_attempts`,
- Bridson's invariant (every pair separated by `≥ r_min`) for the
  Poisson-disk inserter.

## 3. Size distributions

| Distribution     | Constructor                                          | Test moment           |
| ---------------- | ---------------------------------------------------- | --------------------- |
| `Monodisperse`   | `(r)`                                                | trivial               |
| `Bidisperse`     | `(r_small, r_large, fraction_small)`                 | binomial fraction     |
| `Lognormal`      | `(mu_log, sigma_log, r_min, r_max)` (hard truncation)| truncated mean        |
| `UserDiscrete`   | `(radii, weights)` with `std::discrete_distribution` | per-bin probabilities |

The continuous "user-PDF" case in the project specification is currently
expressible only via `UserDiscrete` (after binning the PDF). A
Python-callable PDF (`SizeDistribution.from_callable(fn)`) would
require a pybind11 trampoline and per-call Python invocation — fine
for static fills, prohibitive for dynamic inserters. We're holding
off until a user actually needs it.

## 4. Dynamic-fill strategies

Each registered dynamic inserter is invoked by `Simulation::step()`
between the y-clamp (post-step 10) and ML/metrics — so newly placed
capsules see full physics on the *next* step but are visible to the
*current* step's diagnostics.

| Strategy                       | Per-step semantics                                                             |
| ------------------------------ | ------------------------------------------------------------------------------ |
| `PoissonStochasticInserter`    | Draw `N ~ Poisson(rate · dt)` placements per step.                            |
| `ConstantFluxInserter`         | Measure φ in `region`; if below `target_phi`, RSA-fill to close the deficit.   |
| `ConveyorInserter`             | Measure count in `region`; if below `target_count`, RSA-fill to top up.       |

The "constant flux" name is historical: we don't impose a flux
directly; we maintain φ. At steady state the implied flux is
`φ · ū / region_length` for an inlet-shaped region.

`Simulation::registerDynamicInserter` takes a `std::shared_ptr` for
the inserter (the simulation is the long-lived owner) and a `seed_tag`
that derives an independent `std::mt19937_64` sub-stream from
`params.rng_seed` via FNV-1a. Multiple registrations are supported;
they fire in registration order.

## 5. Region masks

| Region            | Built-in?      | Use case                                                  |
| ----------------- | -------------- | --------------------------------------------------------- |
| `RectRegion`      | yes            | the 90 % case                                              |
| `CircleRegion`    | yes            | radial / annular placements, drug carriers                  |
| `PolygonRegion`   | yes            | hand-crafted irregular geometries (concave polygon OK)    |
| `ImageMaskRegion` | yes (PGM P2/P5)| irregular inlets imported from microfluidic device masks    |

`ImageMaskRegion::fromPGM(path, origin, scale, threshold)` reads
ASCII (P2) or 8-bit binary (P5) graymaps; max value > 255 (16-bit
graymaps) and packed-bit PBM (P1/P4) are explicitly rejected. The
y-axis flip means "up" in the image matches "+y" in the lattice.

## 6. Reproducibility

A complete run is reproducible from a single number:
`SimulationParams::rng_seed`. Each inserter draws a deterministic
sub-stream via

```
mt19937_64 rng(fnv1a64(seed_tag, params.rng_seed));
```

so different `seed_tag` strings produce statistically independent
sequences while the same `(seed, tag)` always reproduces bit-exact.
The tag is captured in the per-call log line and in the
`run_manifest.json` (Phase 1).

## 7. Limitations and follow-ups

- **No Python user-PDF callback.** Add when a user asks. Trampoline
  for `ISizeDistribution::sample` over pybind11 takes ~30 lines.
- **No lognormal area-weighted sampling.** `Lognormal` samples by
  number; for area-weighted use the user must bin via `UserDiscrete`.
- **Conveyor in INLET_OUTLET BCs.** The current measure is
  count-in-region; for natural inlet refresh under outflow BCs the
  appropriate quantity is the total-domain count, with the inserter
  region bounding *where* refills land. A small refactor.
- **No GPU-friendly SoA layout.** Each placement is two doubles plus
  a radius — not on the hot path.

## 8. References

- B. Widom, *J. Chem. Phys.* **44**, 3888 (1966) — RSA.
- J. Talbot, G. Tarjus, P. Viot, P. Schaaf, *Adv. Colloid Interface
  Sci.* **165**, 1 (2000) — RSA jamming asymptotics.
- R. Bridson, *ACM SIGGRAPH 2007 sketches*, 22 — Bridson's algorithm.
- D. Sutherland, J. Comput. **5**, 1 (1974) — point-in-polygon
  ray casting.
- I. Aitchison, J. Brown, *The Lognormal Distribution* (Cambridge
  Univ. Press, 1957).
