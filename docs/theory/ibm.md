# Immersed boundary method — Peskin 4-pt direct forcing

## Coupling

A capsule is represented by a closed loop of Lagrangian nodes
`X_k(t)`, with arc-length segment `Δs_k`. The internal membrane forces
`f_k` (springs, bending, area, perimeter, viscosity) are spread to
the Eulerian lattice as a body force, and the lattice velocity is
interpolated back to advect the nodes:

```
F(x) = Σ_k δ(x − X_k) f_k Δs_k          (spread)
U(X_k) = Σ_x δ(x − X_k) u(x) Δx²        (interpolate)
dX_k / dt = U(X_k).                      (move)
```

(`src/coupling/immersed_boundary.cpp:25–189`.) The body force `F`
enters the LBM via Guo forcing (see [lbm.md](lbm.md)).

## Regularized δ

We use Peskin's 4-point regularized delta (Peskin 2002):

```
φ(r) = ⅛ [ 3 − 2|r| + √(1 + 4|r| − 4r²) ],          0 ≤ |r| ≤ 1,
       ⅛ [ 5 − 2|r| − √(−7 + 12|r| − 4r²) ],        1 ≤ |r| ≤ 2,
       0,                                            |r| > 2,
δ_2D(x, y) = φ(x) φ(y).
```

(`src/coupling/delta_function.h:18–58`.) The same kernel is used for
spreading and interpolation, so momentum exchange is symmetric.

A 1024-bin lookup table avoids the `sqrt` in the inner loop with
< 0.1 % relative interpolation error.

## Direct vs multi-direct forcing

For `params.ibm_iterations == 1` we use **direct forcing** (Uhlmann
2005): one spread, one collision, one interpolate per timestep.

For `> 1`, we apply **multi-direct forcing** (Luo, Han, Wang & Tao 2007):
after each spread, the lattice velocity is recomputed and a
correction force `Δf = 2ρ (u_desired − u_interp) / Δt` is spread, for
the configured number of inner iterations. (`src/coupling/immersed_
boundary.cpp:231–286`.) Three iterations restores no-slip to ~1 % at
moderate resolution; the cost is ~3× the IBM step.

## Periodic-x minimum-image

Both arc-length differences (`minImageDs`) and the spread/interpolate
loops wrap the x coordinate through `params.nx`. Y is **not** wrapped
(channel walls are at y=0 and y=ny−1). (`immersed_boundary.cpp:16–23,
119–121`.)

## Known limitations (to revisit)

- **δ-stencil clipping at y-walls.** When a capsule node sits within
  2 lattice units of a wall, the part of the 4×4 stencil that would
  fall in `y < 0` or `y ≥ ny` is silently dropped, breaking momentum
  conservation. Soft wall repulsion ordinarily keeps nodes off the
  wall; a Phase-2 fix should add a wall-image extension or report the
  closest-approach distance per step.
- **No Guo correction inside IBM.** The IBM spread uses the standard
  direct-forcing formulation, not the modified scheme of
  Guo et al. (Eur. J. Mech. B/Fluids 2008) which includes a
  sub-lattice viscous correction. Acceptable for qualitative work;
  expect a few-percent under-prediction of viscous drag at low
  Reynolds.
- **Forward-Euler node integration.** `capsules.moveAllNodes(dt)`
  advances `X_k += U(X_k) Δt` without any velocity-Verlet or
  symplectic step. Adequate for the soft-spring time scales used here
  but fragile for stiff Skalak coefficients.

## V&V cases

A capsule-in-shear regression (Skotheim–Secomb tank-treading,
the project specification) is listed in `tests/verification/TODO.md` and is
the principal IBM-coupling test for the next phase.

## References

- C. S. Peskin, *Acta Numerica* **11**, 479 (2002).
- M. Uhlmann, *J. Comput. Phys.* **209**, 448 (2005).
- K. Luo, Z. Han, J. Wang, J.-Q. Tao, *Phys. Rev. E* **75**, 026706 (2007).
- T. Krüger, PhD thesis, RWTH Aachen (2012).
