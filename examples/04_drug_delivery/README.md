# Drug-delivery showcase (Phase 4)

Carriers seeded upstream, target absorber on the bottom wall, off-
target absorber on the top wall, in a periodic body-force-driven
microchannel. Selectable release kinetics:

| Mode | Class | Behaviour |
|---|---|---|
| `diffusion` | `DiffusionControlled` | Higuchi-style Fick: `J = k(C_eq − C_surf)` |
| `first_order` | `FirstOrder` | `M(t) = M₀ exp(−k_rel t)` |
| `shear` | `ShearTriggered` | sigmoid in local fluid shear rate |
| `ph` | `PhTriggered` | sigmoid in a designated trigger species |
| `burst` | `Burst` | one-shot release at `t = release_time` |

## Run

```bash
# Single mode, ~5 s smoke
python examples/04_drug_delivery/run.py --mode first_order --smoke

# Full single-mode run (~1 min)
python examples/04_drug_delivery/run.py --mode shear

# Stiffness × release-rate sweep, 3×3 grid (~3 min)
python examples/04_drug_delivery/sweep.py

# Full 5×5 grid (~10 min)
python examples/04_drug_delivery/sweep.py --full
```

Output goes under `output/04_drug_delivery/<mode>/` (or a `/tmp/`
subdirectory in `--smoke` mode). Every run writes `run_manifest.json`
(Phase 1) for full reproducibility.

## What's in the output

Per single-mode run:
- `history.npz` — per-step (M_p, target_absorbed, off_target_absorbed)
- `*.pvd` + `*.vti`/`*.vtp` — fluid + carrier trajectory for ParaView
- `config/run_manifest.json` — git SHA, compiler flags, RNG seed,
  every nested parameter

Per sweep:
- `results.npz` — η and off-target fraction on the (G_s, k) grid
- `dose_vs_parameter.png` — matplotlib heatmaps of both
- One `output/04_drug_delivery/sweep/G_*/` directory per cell

## Dimensionless numbers

Printed at startup in single-mode runs:

| Number | Definition | Healthy here |
|---|---|---|
| Re | ū H / ν | ~ 5–20 |
| Pe | ū H / D | ~ 10–100 |
| Da_I | k_release · L / ū | ~ 0.01–1 (transport-limited) |
| Da_II | k_uptake · L / ū | ~ 0.01–1 (target-limited) |
| Ma | u_lb / cs | < 0.05 |
| χ | 2 r / H | ~ 0.08 (carrier confinement) |

If `Ma > 0.3` the LBM compressibility error is no longer small —
results are unreliable.

## Limitations

Strong language constraint per the project specification (in spirit) and per the
[`docs/drug_delivery.md`](../../docs/drug_delivery.md) module
documentation:

1. **Carriers are 2-D coarse-grained spring-network capsules**
   (Skalak / Neo-Hookean / WLC), not realistic liposomes / micelles
   / RBC ghosts. They have membrane mechanics but no internal
   structure.

2. **"pH-triggered" is a misnomer** kept for user-familiarity. The
   actual implementation is sigmoid sensitivity to a designated
   trigger species. We don't simulate pH chemistry. For real
   pH-responsive carriers, the trigger species should encode a
   calibrated proxy (e.g. H+ equivalent concentration mapped to
   lattice units).

3. **"Shear-triggered" is a sigmoid in the magnitude of the local
   rate-of-strain tensor** at the carrier centroid. It does not
   represent any specific molecular shear-sensitive bond. Use as a
   phenomenological knob.

4. **Wall absorbers are boundary-condition sinks.** They are not
   validated tissue-uptake models. They have no spatial sub-structure,
   no transporter saturation beyond MM, no compartment exchange.
   Clinical relevance requires a separately validated tissue model
   coupled to the SoftFlow output.

5. **Non-Fickian release uses a 3×3 cell uniform spreading** — a
   coarse approximation, not the full Peskin 4-pt kernel that the
   C++ Fick path uses. Fine for relative comparisons (which release
   mode delivers more), not for cell-by-cell quantitative deposition
   profiles.

6. **No 3-D, no patient-specific geometry, no full chemical-kinetics
   network** (the project specification).

This pipeline is for **methodological exploration and student
training**, not clinical claims.

## Reproducing a sweep

Two sweeps with the same `--mode` and identical `params.rng_seed`
(default `0xBEEFCAFE` in [run.py](run.py)) produce bit-exact carrier
placements and bit-exact metrics. Diff the manifests with

```bash
diff <(jq -S . output/04_drug_delivery/<mode>/config/run_manifest.json) \
     <(jq -S . other_run/config/run_manifest.json)
```

to verify identical compile flags, git SHA, and resolved parameters.
