# Tumour-growth / aggregate-formation showcase (Phase 5)

> **This is a coarse-grained mechano-chemical proxy for circulating-
> cell aggregation under flow, NOT a validated cancer model.** Per
> the project specification, the language constraint is mandatory in code,
> docstrings, and outputs. Use this for methodological exploration
> and student training only — never clinical claims.

## Setup

A deliberately narrow 200×40 lattice channel, body-force-driven and
streamwise-periodic. 5 adhesive capsules (Skalak membrane, soft) are
seeded upstream. Adhesion is the Phase-0 Bell model with optional
catch/slip kinetics.

A `StressNutrientDivision` kinetic stochastically duplicates a
capsule when the local fluid shear is below `stress_max` and the
local nutrient concentration is above `nutrient_min`. An
`EmbolizationDetector` watches the cross-section at x=140 for
events where (a) the contact graph spans the channel and (b) the
volumetric flow rate has dropped below 50 % of baseline.

## Run

```bash
# Quick smoke run (~5 s)
python examples/05_tumor_growth/run.py --smoke

# Full single run (~5 min, 5000 steps)
python examples/05_tumor_growth/run.py

# Catch/slip bond kinetics
python examples/05_tumor_growth/run.py --catch-slip

# Plot the time series
python examples/05_tumor_growth/analyse.py
```

Output goes to `output/05_tumor_growth/` (or `/tmp/...` in `--smoke`
mode):

- `*.pvd` + `*.vti`/`*.vtp` — fluid + capsule trajectory for ParaView
- `history.npz` — per-step (capsule count, divisions, cluster size,
  flow rate, cluster span)
- `events.npz` — recorded embolization events (only present when
  events occurred)
- `config/run_manifest.json` — Phase-1 reproducibility record

`analyse.py` reads `history.npz` and produces three figures:
- `division_history.png` — capsule count + cumulative divisions
- `cluster_size.png` — largest connected cluster vs time
- `flow_rate.png` — Q(t)/Q₀ + cluster y-span overlay

## Limitations (mandatory reading)

The language constraint from the project specification is repeated here verbatim:

1. **2-D coarse-grained spring-network capsules**, not realistic
   tumour cells. They have membrane mechanics but no internal
   cell-cycle structure, mitogen pathways, oncogenic signalling, or
   tissue-of-origin specificity.

2. **"Tumour" terminology is kept** for project-specification alignment. The
   internal language uses "aggregate", "cluster", "circulating-cell"
   to keep readers from over-reading the model.

3. **Division is Poisson-clocked stochastic insertion** of a daughter
   capsule when local thresholds are met. It does NOT represent any
   specific cell-cycle checkpoint, mitogen-receptor pathway, micro-
   environment cue, or tumour heterogeneity.

4. **"Stress" is the magnitude of the local fluid rate-of-strain
   tensor** at the parent's centroid. It is NOT membrane stress, NOT
   any specific molecular shear-sensitive bond, and NOT a calibrated
   mechano-transducer signal.

5. **"Nutrient" is a generic scalar species concentration**. The
   user is responsible for whatever physical quantity that scalar
   represents — we don't simulate glucose / oxygen / pH chemistry,
   we simulate sigmoid sensitivity to a designated trigger species.

6. **Adhesion uses Bell + optional catch/slip** (Phase 0). These are
   coarse-grained kinetic bond models, not validated for any
   specific receptor-ligand pair.

7. **Embolization detection** combines spanning-cluster detection
   with a flow-rate gate. It reports *events*, not clinical embolic
   outcomes.

8. **No 3-D, no patient-specific geometry, no full cell-signalling
   network** (the project specification).

## Reproducibility

Two runs with the same `params.rng_seed` (default `0xCAB00D1E` in
[run.py](run.py)) and the same orchestrator seed (`rng_seed ^
0x5A5A5A5A`) produce bit-exact divisions and bit-exact metrics. Diff
the manifests with

```bash
diff <(jq -S . output/05_tumor_growth/config/run_manifest.json) \
     <(jq -S . other_run/config/run_manifest.json)
```

to verify identical compile flags, git SHA, and resolved parameters.
