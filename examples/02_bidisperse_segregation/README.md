# Bidisperse segregation showcase (Phase 2)

Two-species suspension flowing through a periodic body-force-driven
microchannel. Demonstrates the Phase-2 insertion module:

- **Small species** (Skalak membrane, soft, RBC-like) — seeded by a
  hex-lattice fill in a band near the channel centre.
- **Large species** (Neo-Hookean, stiffer, CTC-like) — seeded by RSA
  in a wider band so the two species mix.

Both fills draw from the same canonical `params.rng_seed` but use
distinct `seed_tag`s, so the streams are reproducible *and*
independent.

## Run

```bash
# Quick smoke run (~50 steps, ~5 s on a laptop)
python examples/02_bidisperse_segregation/run.py --smoke

# Full run (~2000 steps, a few minutes)
python examples/02_bidisperse_segregation/run.py
```

Output goes to `output/02_bidisperse_segregation/` (or a `/tmp/`
subdirectory in `--smoke` mode). Each run writes `config/run_manifest.json`
with the full reproducibility record (git SHA, compiler flags, RNG
seed, every nested `SimulationParams` field).

Open the resulting `.pvd` file in [ParaView](https://www.paraview.org/)
to visualise fluid + capsule trajectories.

## Dimensionless numbers (printed at startup)

| Number          | Meaning                            | Healthy range here |
| --------------- | ---------------------------------- | ------------------ |
| Re   = ū H / ν  | channel Reynolds                   | ~ 5–20             |
| Ca_s = μ ū / G_s | capillary number (Skalak modulus)  | ~ 0.01–0.1         |
| χ_s  = 2 r_s/H  | small-particle confinement        | ~ 0.06–0.10        |
| χ_L  = 2 r_L/H  | large-particle confinement        | ~ 0.13–0.20        |
| Ma   = ū / cs   | LBM Mach (must stay < 0.3)        | < 0.05             |

## Limitations

1. **2-D**, no third-dimension hydrodynamic relaxation — agreement
   with 3-D experiment is qualitative only.
2. The Phase-1 ADR equilibrium is first-order in u; no scalar
   transport is enabled here.
3. Segregation is currently only **visualised**. Quantitative
   diagnostics — Lacey index, Danckwerts intensity of segregation,
   lane order parameter, contact-asymmetry — are part of Phase 3
   (the project specification).

## What you should see

- Hex-lattice initialisation produces the expected ~170 small
  capsules in the central strip, ~55 large capsules scattered
  through the wider band.
- After a few hundred steps, large capsules drift toward the channel
  centre while small ones marginate toward the walls — the classic
  size-driven segregation pattern.
