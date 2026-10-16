# Lubrication — Brenner-style regularized 1/h correction

## Why lubrication?

The IBM cannot resolve the diverging hydrodynamic stress in the thin
fluid film between two close-approaching surfaces, because the film
drops below the lattice resolution Δx. Without an explicit
correction, capsules can spuriously interpenetrate or merge in the
streaming step. The textbook fix is a sub-lattice closure — a
short-range correction force that recovers the analytical
lubrication divergence as the gap `h` shrinks (Brenner 1961; Aidun
& Clausen, *Annu. Rev. Fluid Mech.* 2010).

## 2-D pair force

For a pair of soft particles with effective radii `r_i, r_j`,
relative velocity `v_n` projected onto the centre-to-centre normal
`n̂`, and surface gap `h` (regularized below), the 2-D lubrication
force is

```
F_lub = − 6 π μ a_eff v_n / max(h, h_min),
a_eff = r_i r_j / (r_i + r_j).
```

(`src/membrane/lubrication.cpp:112`.) Sign convention: `v_n > 0` for
approaching surfaces, so `F_lub < 0` — an outward repulsive normal
force that diverges as `h → h_min`.

## 2-D capsule–wall force

Same form, with `a_eff = r` (capsule radius) and `n̂` the wall normal:

```
F_lub^{wall} = − 6 π μ r v_n / max(h, h_min).
```

(`lubrication.cpp:143–152`.)

## Regularization

The bare `1/h` term diverges as the surfaces touch. SoftFlow uses a
clamp `h ← max(h, h_min)` (`lubrication.cpp:101`) with `h_min = 0.1`
lattice units by default. This is a smooth Brenner-style cutoff
rather than a hard wall — the correction stays finite but very large
as `h → h_min`, which combined with the soft repulsion is sufficient
to prevent overlap for the time-step / stiffness ratios used here.

`h_threshold` (default 2 lattice units) controls when the correction
turns on at all — above the threshold the lubrication contribution
is exponentially small (sub-lattice) and the bare LBM-IBM stress is
adequate.

## Cell-list acceleration

`LubricationCorrection::computeAll` uses the same spatial-hash cell
list as `RepulsionForce` for `O(N)` neighbour lookup
(`lubrication.cpp:25–119`). The parallel implementation is **one-
sided**: each thread owns the target capsule and writes only into
its own forces, so there are no race conditions on the shared
neighbour structure.

## Limitations

- 2-D lubrication has different prefactors than 3-D
  (`6 π μ a²/h` vs `6 π μ a²/h` with different a_eff scaling). The
  current implementation uses the standard 2-D form; a Phase-3
  follow-up may add the 3-D form behind a build flag for
  comparison-with-3-D-codes runs.
- The regularization is a hard clamp; a smoother stress relaxation
  (e.g. Kromkamp et al. 2006) is on the long-term wishlist.

## V&V

There is no isolated regression test for the lubrication module
today. The pragmatic check is "do close-approach pairs avoid overlap
in the standard `examples/two_particle_types_microchannel.py` run",
which is exercised in CI by building+running the binary. A
quantitative test against the Brenner two-sphere drag analytic for a
single approach event is on the V&V follow-up list.

## References

- H. Brenner, *Chem. Eng. Sci.* **16**, 242 (1961).
- C. K. Aidun, J. R. Clausen, *Annu. Rev. Fluid Mech.* **42**, 439
  (2010).
- J. Kromkamp, D. T. M. van den Ende, D. Kandhai, R. G. M. van der
  Sman, R. M. Boom, *J. Fluid Mech.* **529**, 253 (2005).
