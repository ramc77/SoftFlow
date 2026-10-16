# V&V — known follow-ups

## Currently disabled regression cases

### `test_taylor_dispersion.cpp` — parked

Symptom: measured `D_eff` is ≈ 10× the Aris–Taylor analytical
`D + Pe² D / 210` (specifically 0.233 vs 0.024 at Pe ≈ 6.6).

Diagnosis: the failure persists with both first- and second-order ADR
equilibria, so the equilibrium order is **not** the cause. The most
likely culprits, in order of plausibility:

1. **Periodic-x wrap polluting the variance estimate.** At late times
   the Gaussian tails cross the periodic seam; the lab-frame
   `Σ c·(x−x̄)²` calculation includes wrapped mass at small `x` while
   the centroid is near `nx − k`, inflating the apparent variance.
2. **Insufficient long-time regime.** The current test uses
   `t_window = 5 × t_mix(loose)` ≈ 30 000 steps, but `t_mix(strict)
   = a²/(π²D)` is shorter — and Aris–Taylor only kicks in after several
   *strict* mixing times, not loose ones.
3. **Numerical-diffusion contamination from advection of a pulse with
   significant streamwise gradient at the edges.**

To re-enable, rewrite the test with **either** (a) `INLET_OUTLET`
streamwise BCs and a wide enough domain that the pulse never reaches
the outlet during the sample window, **or** (b) a circular-statistics
treatment of the pulse position with the comoving frame anchored on
`u_mean × t`.

### `test_capsule_shear.cpp` — Skotheim–Secomb tank-treading (the project specification)

Not yet implemented. Needs a single Skalak capsule in linear shear,
running long enough to reach steady tank-treading, then comparing the
tank-treading frequency vs Capillary number against the published
Skotheim–Secomb (or Pozrikidis 2003) curves. Tolerance per the project specification
§6: < 5%. **Estimated work: half a day.**

### `test_segre_silberberg.cpp` — single rigid disk equilibrium (the project specification)

Not yet implemented. Needs a single rigid disk (set
`MembraneParams.is_rigid = true` and very stiff repulsion against the
walls) in fully developed Poiseuille flow at finite Re, and verification
that the disk's steady radial position matches Segré–Silberberg's
0.6 × R_pipe equilibrium within 10 %. Tolerance per the project specification:
qualitative + < 10 % radial position. **Estimated work: 1 day.**

## Currently active suite

| Test                  | Tolerance     | Wall-clock |
| --------------------- | ------------- | ---------- |
| `test_poiseuille`     | < 1 % L2      | ~9 s       |
| `test_taylor_green`   | < 2 % decay   | ~1 s       |
| `test_pure_diffusion` | < 2 % σ²      | ~3 s       |

Plus unit tests under `tests/unit/`:

| Test                  | Wall-clock |
| --------------------- | ---------- |
| `test_lattice`        | < 1 s      |
| `test_run_manifest`   | ~1 s       |
