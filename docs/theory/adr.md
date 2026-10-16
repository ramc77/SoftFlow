# Advection–diffusion–reaction (ADR) for passive scalars

## Solver

A second D2Q9 distribution `g_q` is evolved alongside the fluid `f_q`
to transport a passive scalar concentration `C(x, t)`. It is BGK with
relaxation time

```
τ_s = 3 D + ½,        D ≥ 0,
ω_s = 1 / τ_s.
```

(`src/lbm/advection_diffusion.cpp:27`.)

The collision uses **first-order-in-velocity** equilibrium (Krüger
2017 §8.5.2):

```
g_q^{eq}(C, u) = w_q C ( 1 + (e_q · u) / cs² ).
```

(`advection_diffusion.cpp:`–`collide`.) Higher-order velocity terms
in the ADR equilibrium do *not* extend the stable Péclet range — they
introduce u² corrections to the recovered diffusivity that violate
the Aris–Taylor result. For genuinely high-Pe transport, the right
tools are MRT-ADR (Yoshida & Nagaoka, *J. Stat. Phys.* 2010) or grid
refinement. **Phase-1 status:** an earlier draft used a second-order
equilibrium and was reverted after the Taylor-dispersion test
diagnosed the regression.

## Three-pass stepping (Phase 1)

```
1. collide(species, fluid)         g_ → g_tmp (first-order BGK)
2. streamWithBC(species, fluid)    g_tmp → g_  (pull + halfway BB)
3. computeConcentration(species)   C = Σ_q g_q
```

(`advection_diffusion.cpp:84–177`.) The pre-Phase-1 implementation
fused collide+stream into a single pass and applied solid bounce-back
by writing into `g_[species]` from inside an OpenMP parallel region —
a real race condition. The split removes the race entirely (writers
and readers are disjoint within each pass).

## Boundary conditions

- **Solid walls:** halfway bounce-back at the fluid cell:
  `g_[n*Q + q] = g_tmp_[n*Q + opp[q]]`. Zero-flux Neumann.
- **Periodic-x:** always on (matches the project specification streamwise PBC).
- **Periodic-y:** opt-in via `ScalarParams::periodic_y`.
- **Inlet (`CellType::INLET`):** Dirichlet via re-imposing
  `C = inlet_concentration[species]` before computing `g^{eq}`. A
  follow-up will replace this with a proper anti-bounce-back inlet to
  reduce the truncation error at the inlet plane.

## Recovered macroscopic equation

Chapman–Enskog gives, to second order in the lattice spacing,

```
∂_t C + ∇ · (u C) = ∇ · ( D ∇C ),
```

with `D = (τ_s − ½) cs²`. Numerical anisotropy ∝ u_i u_j is below
0.01 % at typical u ~ 10⁻³ lattice units and is safely ignored.

## Chemistry extension (`applyChemistry`)

The drug-delivery / microplastic infrastructure overlays two
node-resolved processes on the bare ADR:

1. **Fick-type leaching** with finite reservoir `M_p`:

   ```
   J_leach = k_leach (C_eq − C_surface),
   dM_p/dt = − max(J_leach, 0).
   ```

2. **Langmuir adsorption / desorption** with surface coverage Γ
   per capsule node:

   ```
   dΓ/dt = k_a C_surface (1 − Γ/Γ_max) − k_d Γ.
   ```

Surface concentration `C_surface` and flux spreading both use the
same Peskin 4-point kernel as the IBM. `applyChemistry` is invoked
when any of `k_leach`, `k_adsorb`, `k_desorb` is set; otherwise the
backward-compatible constant-rate `applySourceSink` runs.

(`advection_diffusion.cpp:260–408`.)

## V&V cases that exercise this module

- `tests/verification/test_pure_diffusion.cpp` — Gaussian variance
  σ² = σ₀² + 2 D t, < 2 % error.
- `tests/verification/test_taylor_dispersion.cpp` — **parked**, see
  [tests/verification/TODO.md](../../tests/verification/TODO.md).

## References

- T. Krüger et al., *The Lattice Boltzmann Method* (Springer, 2017),
  Ch. 8.
- M. Sukop, D. Thorne, *Lattice Boltzmann Modeling* (Springer, 2006).
- H. Yoshida, M. Nagaoka, *J. Stat. Phys.* **141**, 1003 (2010).
- R. Aris, *Proc. R. Soc. A* **235**, 67 (1956).
- G. I. Taylor, *Proc. R. Soc. A* **219**, 186 (1953).
- I. Langmuir, *J. Am. Chem. Soc.* **40**, 1361 (1918).
