# LBM core — D2Q9 with BGK / MRT / regularized collision

## Lattice and equilibrium

SoftFlow uses the standard D2Q9 lattice with discrete velocities

```
e = { (0,0), (±1,0), (0,±1), (±1,±1) }
w = { 4/9,  1/9 ×4,             1/36 ×4 }
cs² = 1/3.
```

(`src/lbm/lattice.h:22–31`.) The equilibrium distribution is the
second-order Hermite expansion (Krüger 2017 Eq. 4.1):

```
f_q^{eq}(ρ, u) = w_q ρ [ 1 + (e_q·u)/cs²
                          + (e_q·u)²/(2 cs⁴)
                          - (u·u)/(2 cs²) ].
```

(`src/lbm/lattice.h:35–39`.)

## Macroscopic moments with Guo half-force correction

Forces enter through `Lattice_Field::Fx`, `Fy`. With Guo (2002) forcing
the velocity is corrected by half the body force per unit density:

```
ρ      = Σ_q f_q,
ρ u    = Σ_q e_q f_q + ½ F.
```

(`src/lbm/lattice_field.cpp:64–71`.)

## Collision operators

### BGK + Guo forcing (default)

```
f_q^{*} = (1 − ω) f_q + ω f_q^{eq} + (1 − ω/2) S_q,
S_q     = w_q [ (e_q − u)/cs² + (e_q·u)/cs⁴ e_q ] · F.
```

`ω = 1/τ`. (`src/lbm/bgk_collision.cpp:67–78`. Reference: Guo, Zheng &
Shi, Phys. Rev. E **65**, 046308 (2002); Krüger 2017 Eq. 5.30.)

### MRT (Lallemand–Luo 2000)

Moments `m = M f`, equilibrium moments `m^{eq}`, relaxation diagonal
`S = diag(0, s_e, s_ε, 0, s_q, 0, s_q, s_ν, s_ν)`. Guo forcing is
applied in moment space (Li, Luo & Krafczyk 2012):

```
m_q^{*} = m_q − s_q (m_q − m_q^{eq}) + (1 − s_q/2) Sb_q.
```

(`src/lbm/mrt_collision.cpp:77–127`.)

### Regularized BGK (Latt & Chopard 2006)

The non-equilibrium distribution is projected onto the second-order
polynomial spanned by `Q_q^{αβ} = e_q^α e_q^β − cs² δ^{αβ}`:

```
Π^{neq}_{αβ} = Σ_q (f_q − f_q^{eq}) e_q^α e_q^β,
f_q^{neq,reg} = w_q / (2 cs⁴) · Q_q^{αβ} Π^{neq}_{αβ}.
```

Then the standard BGK update with forcing:

```
f_q^{*} = f_q^{eq} + (1 − ω) f_q^{neq,reg} + S_q.
```

(`src/lbm/regularized_bgk.cpp:52–81`.)

## Streaming

Pull-style with precomputed neighbour indices, one q-plane per pass:

```
f_tmp[q*N + n] = f[q*N + neighbour_idx[q*N + n]].
```

Periodic-x and optional periodic-y are folded into the neighbour
table at construction. Out-of-bounds non-periodic upstream cells
self-reference (the bounce-back step then writes the correct value).
(`src/lbm/lbm_solver.cpp:64–96`, `:204–227`.)

## Boundary conditions

| BC                                       | Implementation                                  | Reference |
| ---------------------------------------- | ----------------------------------------------- | --------- |
| Halfway bounce-back (default solid)      | `bounce_back.cpp:18–60`                         | textbook   |
| Bouzidi–Firdaouss–Lallemand interpolated | `interpolated_bounce_back.cpp:36–88`            | BFL 2001 |
| Zou–He velocity inlet / pressure outlet  | `zou_he_boundary.cpp:29–79`                     | Zou & He 1997 |
| Ladd moving wall                         | `moving_wall.cpp:34–36`                         | Ladd 1994 |

When `params.fluid.use_interpolated_bb=true`, IBB is applied to all
obstacle links; otherwise halfway BB is used (staircase, **O(Δx)**).
The Phase-1 plan is to default IBB on whenever a curved obstacle is
registered.

## Recovered macroscopic equations

Chapman–Enskog gives the (compressible) Navier–Stokes equations:

```
∂_t ρ + ∇·(ρ u) = 0,
∂_t (ρ u) + ∇·(ρ u u) = −∇p + ∇·[ μ (∇u + ∇uᵀ) ] + F,
```

with `p = ρ cs²` and kinematic viscosity `ν = (τ − ½) cs²`.
`SimulationParams::kinematicViscosity()` exposes this relation.

## V&V cases that exercise this module

- `tests/verification/test_poiseuille.cpp` — parabolic profile, L2 < 1 %.
- `tests/verification/test_taylor_green.cpp` — exponential decay rate, < 2 %.

## References

- T. Krüger et al., *The Lattice Boltzmann Method* (Springer, 2017).
- Z. Guo, C. Zheng, B. Shi, *Phys. Rev. E* **65**, 046308 (2002).
- P. Lallemand, L.-S. Luo, *Phys. Rev. E* **61**, 6546 (2000).
- J. Latt, B. Chopard, *Math. Comput. Simul.* **72**, 165 (2006).
- M. Bouzidi, M. Firdaouss, P. Lallemand, *Phys. Fluids* **13**, 3452 (2001).
- Q. Zou, X. He, *Phys. Fluids* **9**, 1591 (1997).
- A. J. C. Ladd, *J. Fluid Mech.* **271**, 285 (1994).
