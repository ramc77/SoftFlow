# SoftFlow examples — one runnable showcase per phase

Five canonical entry points, one per project phase. Each writes
ParaView-readable `*.pvd` + `*.vti`/`*.vtp` time-series and a
`config/run_manifest.json` reproducibility record.

| Phase | Topic | Run | Open in ParaView |
|---|---|---|---|
| 1 | LBM hardening, run-manifest, Poiseuille | [01_poiseuille_lbm/run.py](01_poiseuille_lbm/run.py) | `output/01_poiseuille_lbm/{fluid,particles}.pvd` |
| 2 | Particle-insertion module | [02_bidisperse_segregation/run.py](02_bidisperse_segregation/run.py) | `output/02_bidisperse_segregation/{fluid,particles}.pvd` |
| 3 | Segregation + jamming diagnostics | [03_segregation_diagnostics/run.py](03_segregation_diagnostics/run.py) | `output/03_segregation_diagnostics/{fluid,particles,diagnostics}/*.pvd` |
| 4 | Drug delivery | [04_drug_delivery/run.py](04_drug_delivery/run.py) | `output/04_drug_delivery/<mode>/{fluid,particles}.pvd` |
| 5 | Tumour-growth proxy | [05_tumor_growth/run.py](05_tumor_growth/run.py) | `output/05_tumor_growth/{fluid,particles}.pvd` |

Each script accepts `--smoke` for a ~30–60 s CI/visualisation
smoke-test and runs at full length otherwise (a few minutes on a
laptop).

## Quick start

Make sure the C++ extension is built (Phase 1+):

```bash
cmake -S . -B build_phase2 -DBUILD_PYTHON=ON \
                            -DBUILD_TESTS=OFF \
                            -DPYTHON_EXECUTABLE=$(which python3)
cmake --build build_phase2 -j
```

Then run any of the five:

```bash
python examples/01_poiseuille_lbm/run.py --smoke         # ~10 s
python examples/02_bidisperse_segregation/run.py --smoke # ~5 s
python examples/03_segregation_diagnostics/run.py --smoke # ~10 s
python examples/04_drug_delivery/run.py --mode first_order --smoke  # ~5 s
python examples/05_tumor_growth/run.py --smoke           # ~5 s
```

Drop the `--smoke` for the production run (a few minutes each).

## Opening in ParaView

Each example prints the `.pvd` files it produced at the end of its
run. In ParaView:

1. **File → Open** → select the `.pvd` files (one or several at once).
2. Click the **green Apply** button.
3. Use the **time controls** at the top to animate.
4. To overlay multiple `.pvd` (e.g. fluid + particles + diagnostic
   field for Phase 3), repeat steps 1–2 for each.

For a quick sanity render:

- Fluid: colour by `velocity_x` (Phase 1 should look like a clean
  parabola in y); switch the representation to *Surface*.
- Particles: colour by `radius` or `type`; switch to *Surface*.
- Phase 3 diagnostic field: colour by `packing_fraction`.

## What each phase showcases

### Phase 1 — `01_poiseuille_lbm/`

Body-force-driven Poiseuille channel with six soft Skalak capsules.
Demonstrates:

- Build provenance in `run_manifest.json` (git SHA, compiler flags,
  OpenMP thread count, RNG seed) — the script prints it at the end.
- Configurable `max_lattice_force` cap with the per-step capped-node
  counter wired into `checkStability()`.
- The Phase-1 V&V Poiseuille profile — visible in ParaView as a
  clean parabolic streamwise-velocity field once the simulation has
  relaxed.

### Phase 2 — `02_bidisperse_segregation/`

50 / 50 small-vs-large suspension. Hex-lattice fill for the small
species (Skalak, soft) + RSA fill for the large (Neo-Hookean,
stiffer) using Phase-2's insertion module. Periodic body-force-driven
flow shows size-driven margination at long times.

### Phase 3 — `03_segregation_diagnostics/`

Same simulation as Phase 2, but additionally:

- Samples ~20 snapshots across the run.
- Computes Phase-3 diagnostics on each: packing field φ(x,y),
  Lacey index, lane order parameter, Hoshen–Kopelman cluster sizes,
  contact number Z̄, per-type contact matrix Z_ij.
- Writes the packing field as an animated `.vti` time-series
  (a separate ParaView dataset you can colour by `packing_fraction`).
- Writes the scalar diagnostics (Z, Lacey, lane order, …) as a CSV
  for Plot Over Time / Spreadsheet view in ParaView.

### Phase 4 — `04_drug_delivery/`

Carrier capsules released upstream, target absorber on the bottom
wall, off-target on the top wall. CLI flag selects the release
kinetic:

```bash
python examples/04_drug_delivery/run.py --mode diffusion
python examples/04_drug_delivery/run.py --mode first_order
python examples/04_drug_delivery/run.py --mode shear
python examples/04_drug_delivery/run.py --mode ph
python examples/04_drug_delivery/run.py --mode burst
```

A `sweep.py` script in the same directory runs a stiffness × release-
rate grid and produces a matplotlib heatmap of η + off-target fraction.

### Phase 5 — `05_tumor_growth/`

**Coarse-grained mechano-chemical proxy. Not a validated cancer
model.** See the example's `README.md` for the strong-language
limitations.

Five adhesive capsules in a deliberately narrow channel. Bell-model
adhesion (with optional `--catch-slip` toggle), stress-and-nutrient
gated stochastic division, and an embolization detector watching a
downstream cross-section for spanning-cluster + flow-rate-drop
events.

## Reproducing a run

Every example writes a `run_manifest.json` containing the git SHA,
compiler ID and resolved flags, OpenMP thread count, the canonical
RNG seed, and every nested `SimulationParams` field. Two runs sharing
the same manifest produce identical output. To diff manifests:

```bash
diff <(jq -S . run_a/config/run_manifest.json) \
     <(jq -S . run_b/config/run_manifest.json)
```

## Older / scratch examples

The flat-layout `*.py` files (`two_particle_types_microchannel.py`,
`shape_dynamics.py`, `density_driven_particles.py`,
`microplastic_blood_vessel.py`, `multi_density_particles.py`,
`plot_microplastic_blood.py`) are pre-Phase-1 scratch demos. They
still run but use older API patterns and do **not** produce a
`run_manifest.json`. Prefer the numbered examples above.
