# Drug-delivery extensions (Phase 4)

Five release-kinetic models, wall-region absorbers, and four
delivery metrics, all post-processing on top of the existing C++
chemistry path (Phase-1's Fick leaching with finite reservoir M_p,
Langmuir adsorption, Peskin-spread fluxes).

The C++ engine is **untouched**. Every Phase-4 component is pure
Python + numpy and lives under
[`python/pysoftflow/drug_delivery/`](../python/pysoftflow/drug_delivery/).

## Strong language constraint

Per the project specification (in spirit; §7.3 in letter), this entire module is
a **methodological-exploration toolkit**, not a validated drug-carrier
or pharmacokinetic model. The constraints below appear verbatim in
module docstrings and in `examples/04_drug_delivery/README.md`:

- Carriers are 2-D coarse-grained spring-network capsules. They have
  selectable membrane mechanics (Skalak / Neo-Hookean / WLC) but no
  internal structure — not realistic liposomes, micelles, or RBC
  ghosts.
- Wall absorbers are **boundary-condition sinks**, not validated
  tissue-uptake models. Clinical relevance requires a separate,
  properly validated tissue model coupled to the SoftFlow output.
- "pH-triggered" really means **second-scalar-triggered**: we don't
  simulate pH chemistry, we simulate sigmoid sensitivity to a
  designated trigger species. The user is responsible for whatever
  proxy mapping they choose.
- "Shear-triggered" is a sigmoid in the **magnitude of the local
  rate-of-strain tensor** at the carrier centroid. It does not
  represent any specific molecular shear-sensitive bond.
- The non-Fickian release path (FirstOrder / Shear / pH / Burst)
  spreads released mass uniformly across a 3×3 cell window centred
  on the carrier — a coarse approximation, not the full Peskin
  4-pt kernel that the C++ Fick path uses. Fine for relative
  comparisons; not for cell-by-cell quantitative deposition profiles.

## 1. Release kinetics (`kinetics.py`)

All implement `update(carrier_state, fluid_probe, dt) -> released_mass`.

| Class | Behaviour | Reference |
|---|---|---|
| `DiffusionControlled(k_leach, C_eq)` | Adapter to existing C++ Fick path: `J = k · (C_eq − C_surface)` | Higuchi 1961 |
| `FirstOrder(k_rel)` | `M_p(t) = M₀ exp(−k_rel · t)` via exact closed-form `ΔM_p = M_p (1 − exp(−k_rel · dt))` | textbook |
| `ShearTriggered(k_max, γ_thresh, sharpness)` | `k_eff = k_max · σ(sharpness · (γ̇ − γ_thresh))` where σ is the logistic sigmoid; γ̇ from the local rate-of-strain tensor | Bao & Suresh 2003 (mechanosensitive) |
| `PhTriggered(k_max, C_thresh, species, sharpness)` | Same sigmoid, but on the local concentration of a designated species | Schmaljohann 2006 (stimuli-responsive) |
| `Burst(release_time, fraction)` | One-shot drop of `fraction · M₀` at `t ≥ release_time`, then zero | textbook |

All five classes raise `ValueError` for malformed parameters and are
exercised by 17 unit tests (`tests/python/test_drug_kinetics.py`)
that include closed-form decay verification, sigmoid asymptotics,
threshold sharpness, and per-carrier independence for `Burst`.

## 2. Wall absorbers (`absorbers.py`)

```python
WallAbsorber(
    i_range=(i_lo, i_hi),     # lattice columns, half-open
    j_range=(j_lo, j_hi),     # lattice rows
    species=0,
    mode="first_order",        # or "michaelis_menten"
    k=0.01,                    # rate constant or k_cat
    K_M=1.0,                   # MM only
    label="target",
)
```

Per-step semantics:
```
first_order:        ΔC = min(C, k · C · dt)
michaelis_menten:   ΔC = min(C, k · C / (K_M + C) · dt)
```

The patch operates on a writable numpy view of the C++ scalar field
(`AdvectionDiffusion.concentration(species)`). `cumulative_absorbed`
tracks the running total; `history` is the per-step uptake series
(useful for residence-time and dose-vs-time plots).

13 tests (`test_drug_absorbers.py`) cover analytic decay, mass
conservation between field loss and cumulative counter, MM
saturation at high C / linearity at low C, and the over-depletion
clamp.

## 3. Metrics (`metrics.py`)

| Function | Definition |
|---|---|
| `delivery_efficiency(target, carriers)` | η = `target.cumulative_absorbed / Σ M_p_initial` |
| `off_target_fraction(off_targets, carriers)` | `(Σ off_target absorbed) / Σ M_p_initial` |
| `residence_time_distribution(snapshots, target_band, …)` | Per-carrier cumulative time in target_band, plus histogram + mean + median |
| `spatial_dose_map(absorbers, nx, ny)` | Cumulative dose per lattice cell; sums across overlapping absorbers |

14 tests (`test_drug_metrics.py`) cover idealised mass balance
(η + OTF + remaining ≈ 1), histogram / type-filter behaviour for the
RTD, and overlap behaviour for the dose map.

## 4. Orchestrator (`runner.py`)

```python
run = DrugDeliveryRun(sim=sim)
run.add_carrier_type(type_id=0, kinetic=FirstOrder(k_rel=0.005),
                      initial_mass=1.0)
run.add_target(target_absorber)
run.add_off_target(off_target_absorber)
run.attach()
sim.initialize()
for s in range(n_steps):
    sim.step()
print(run.summary())   # η, OTF, remaining, n_carriers
```

`attach()` registers a single per-step callback on `Simulation` that
(1) refreshes per-carrier `M_p` from C++, (2) builds a `FluidProbe`
(centroid concentration + shear rate + species probes), (3) calls
each kinetic's `update()`, (4) Peskin-spreads released mass into
the local scalar field for non-Fickian models, (5) applies all
absorbers, and (6) appends a `RunStepRecord` to `history`.

3 tests (`test_drug_runner.py`) exercise the full pipeline against
a live `Simulation` with multiple carriers + target + off-target.

## 5. Showcase example

[`examples/04_drug_delivery/`](../examples/04_drug_delivery/):

- `run.py` — single-configuration showcase. CLI `--mode {diffusion,
  first_order, shear, ph, burst}` selects the release kinetic.
  Outputs `history.npz` + the standard SoftFlow VTK + `run_manifest.json`.
- `sweep.py` — runs `run.py`'s `build_simulation` over a (G_s × k)
  grid and produces `results.npz` + a matplotlib heatmap of η and
  off-target fraction. `--smoke` for a 2×2 CI grid, default 3×3,
  `--full` for the 5×5 grid (~10 minutes on a laptop).
- `README.md` — limitations section repeated, dimensionless numbers
  printed at startup.

6 smoke tests (`test_drug_example.py`) parametrise every release
mode through `run.py` and verify the sweep's mass balance.

## References

- T. Higuchi, *J. Pharm. Sci.* **50**, 874 (1961) — diffusion-
  controlled release.
- D. Schmaljohann, *Adv. Drug Deliv. Rev.* **58**, 1655 (2006) —
  stimuli-responsive carriers.
- G. Bao & S. Suresh, *Nat. Mater.* **2**, 715 (2003) — mechano-
  sensitive bonds.
- L. Michaelis & M. L. Menten, *Biochem. Z.* **49**, 333 (1913) —
  enzyme kinetics (MM uptake).
- P. Decuzzi & M. Ferrari, *Biomaterials* **29**, 377 (2008) — drug-
  carrier transport in vasculature.
