# Tumour-growth / aggregate-formation extensions (Phase 5)

> **This is a coarse-grained mechano-chemical proxy for circulating-
> cell aggregation under flow, not a validated cancer model.** Per
> the project specification the language constraint is mandatory across code,
> docstrings, and any generated text. The pipeline is for
> methodological exploration and student training, not clinical
> claims.

Phase 5 adds three Python pieces on top of the existing C++
adhesion machinery (Bell + catch/slip from Phase 0) and the Phase-3
`hoshen_kopelman` / `force_percolation` diagnostics:

```
python/pysoftflow/tumor_growth/
├── division.py       — DivisionKinetic ABC + StressNutrientDivision
├── daughters.py      — DaughterPlacer (volume-exclusion-aware)
├── embolization.py   — EmbolizationDetector (Q(t) + cluster span + events)
└── runner.py         — TumorGrowthRun orchestrator
```

The C++ engine is **untouched**.

## Strong-language constraint (mandatory, repeated everywhere)

the project specification: "this is a coarse-grained mechano-chemical proxy,
**not** a validated oncology model. Language in outputs and papers
must reflect that." Every Phase-5 module docstring, every class
docstring, the example README, and this file open with the same
caveat. Concretely:

- **Capsules are 2-D coarse-grained spring-network particles**
  (Skalak / Neo-Hookean / WLC). They have membrane mechanics but no
  internal cell-cycle structure.
- **"Tumour-growth" terminology is kept** for project-specification alignment;
  internally we use "aggregate", "cluster", "circulating-cell"
  language.
- **Division is Poisson-clocked stochastic insertion** of a daughter
  capsule when local thresholds are met. It does not represent any
  specific cell-cycle checkpoint, mitogen pathway, or tumour
  micro-environment.
- **"Stress" is the magnitude of the local rate-of-strain tensor**
  γ̇ at the parent's centroid (same Phase-4 shear probe). It does
  not represent any specific molecular shear-sensitive bond or
  mechano-transducer.
- **"Nutrient" is a designated scalar species concentration**.
  The user is responsible for whatever physical quantity that
  scalar represents in their study.
- **Embolization detection** combines Phase-3's spanning-cluster
  detection with a flow-rate ratio gate. It reports *events*, not
  clinical embolic outcomes.
- **No 3-D, no patient-specific geometry, no full
  cell-signalling network** (the project specification).

## 1. Division kinetics (`division.py`)

```python
StressNutrientDivision(
    k_div=0.005,        # Poisson rate per unit lattice time
    stress_max=8e-4,    # block division above this γ̇
    nutrient_min=0.05,  # block division below this C
    max_divisions=None, # optional per-capsule lineage cap
)
```

Per-step logic:

1. If parent has hit `max_divisions`: block.
2. If γ̇ > `stress_max`: block (high shear is mechanically
   incompatible with division).
3. If C < `nutrient_min`: block (the cell starves).
4. Else fire with probability `1 − exp(−k_div · dt)` (closed-form
   Poisson, exact for any dt).

10 unit tests (`test_tumor_division.py`) — threshold gates, Poisson
firing-rate verification within a 4σ binomial envelope, lineage cap,
seeded-RNG reproducibility, boundary cases at exactly the threshold.

## 2. Daughter placement (`daughters.py`)

```python
DaughterPlacer(
    ring_radius_factor=1.05,
    max_attempts=12,
    min_gap=0.5,
    daughter_radius_factor=1.0,
)
```

`propose(parent_pos, parent_radius, …)` tries `max_attempts` random
angles around a ring at distance
`ring_radius_factor · (parent_r + daughter_r + min_gap)` (the
`min_gap` term is essential — without it the daughter lands inside
the parent's exclusion envelope). Each candidate is checked against
walls, obstacles (signed-distance callable), and existing capsules
under periodic-x minimum-image. Returns the first valid placement, or
None if every angle fails.

10 tests (`test_tumor_daughters.py`) — free-space success, narrow-
channel rejection, dense-neighbourhood rejection, periodic-x wrap,
obstacle blocking, reproducibility, smaller-daughter scaling.

## 3. Embolization detection (`embolization.py`)

```python
EmbolizationDetector(
    x_section=140,
    flow_drop_threshold=0.5,
    contact_cutoff=1.0,
    band_fraction=0.10,
)
detector.baseline(sim)            # call once at t=0
detector.step(sim, snap)           # call every step
```

Q(t) is the row-summed `u_x` at the chosen cross-section.
`force_percolation` (Phase 3) tells us whether any cluster spans the
top and bottom near-wall bands. An `EmbolizationEvent` fires only
when *both* gates fire simultaneously: Q(t)/Q₀ < threshold AND
spanning cluster present.

`flow_rate_history`, `cluster_span_history`, `events` accessors give
the per-step time series.

9 tests (`test_tumor_embolization.py`) — full gate combinations,
history accumulation, single-x_section flow integration, boundary
band validation.

## 4. Orchestrator (`runner.py`)

```python
run = TumorGrowthRun(sim=sim)
run.add_division_kinetic(type_id=0, kinetic=…, mparams=…,
                          placer=…, num_nodes=18)
run.add_embolization_detector(detector)
run.set_seed(42)
run.attach()
sim.initialize()
for s in range(n_steps): sim.step()
print(run.summary())
```

Same shape as Phase-4's `DrugDeliveryRun`. The single per-step
callback (1) probes shear + nutrient at every eligible parent's
centroid, (2) calls the kinetic, (3) on True asks the placer for an
offset and on success appends the daughter via
`CapsuleSystem.addCapsule(...)`, (4) feeds a `SimulationSnapshot` to
each detector, (5) appends a `RunStepRecord` to `history`.

Daughters added during step `k` are tracked from step `k+1` onward —
they cannot themselves divide in the same step they were created.

6 tests (`test_tumor_runner.py`) — end-to-end against a live
`Simulation`, high-stress block, low-nutrient block, monotonic
division total, idempotent attach, embolization-detector pipeline.

## 5. Showcase example

[`examples/05_tumor_growth/`](../examples/05_tumor_growth/):

- `run.py` — 200×40 narrow channel, body-force-driven, 5 adhesive
  capsules, default Bell adhesion + `--catch-slip` toggle, a 1.0
  uniform nutrient prefill, target embolization-detection at
  x=140. Outputs VTK + `history.npz` (capsule count, division
  totals, cluster sizes, flow rate, cluster span) + `events.npz`.
- `analyse.py` — three matplotlib plots (division history, cluster
  size, flow rate / cluster span overlay).
- `README.md` — strong-language Limitations block.

3 smoke tests (`test_tumor_example.py`) — default + catch/slip +
analyse-script round-trip.

## References

- G. I. Bell, *Science* **200**, 618 (1978) — Bell adhesion model.
- W. E. Thomas, V. Vogel, E. Sokurenko, *Annu. Rev. Biophys.*
  **37**, 399 (2008) — catch/slip bonds.
- N. Aceto et al., *Cell* **158**, 1110 (2014) — circulating tumour-
  cell clusters under flow.
- S. Au et al., *Proc. Natl. Acad. Sci. USA* **113**, 4947 (2016) —
  CTC embolic events in microvasculature.
- D. Stauffer & A. Aharony, *Introduction to Percolation Theory*
  (1994) — spanning-cluster detection.
- J. Hoshen & R. Kopelman, *Phys. Rev. B* **14**, 3438 (1976) —
  cluster labelling.

## Limitations / known follow-ups

- **Per-capsule strain accessor** — Phase-5 division uses fluid γ̇
  as the stress proxy. Adding a per-capsule membrane-strain
  accessor (max edge stretch, or area dilation) would let users
  use a more cell-mechanical proxy. Small C++ binding addition.
- **Daughter inherits zero bonds** — physically reasonable (mitosis
  releases adhesion contacts) but a follow-up could let daughters
  partially inherit parent bonds.
- **Single-x flow-rate measurement** — robust enough at the chosen
  cross-section but a multi-section averaged Q(t) is a small
  follow-up if local fluctuations become a problem.
- **Per-Peskin-cell daughter mass spreading** — currently the
  daughter is a fully-formed capsule from t=0. A "growing" daughter
  that interpolates from r → r_target over a few steps would be
  more physical for soft membranes; small follow-up.
- **3-D, patient-specific geometry, full cell-signalling network**
  — out-of-scope per the project specification.
