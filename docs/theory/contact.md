# Contact mechanics: repulsion, normal damping, tangential friction

This note documents the short-range **contact** model between capsule
membranes (and between capsules and walls/obstacles), implemented in
`src/membrane/repulsion.cpp`. It complements the hydrodynamic near-contact
treatment in [`lubrication.md`](lubrication.md) and the bonded interactions in
[`adhesion.md`](adhesion.md).

## Role in a resolved IBM-capsule code

The capsule **deformation** is resolved explicitly by the membrane model
([`membrane.md`](membrane.md)) and the near-contact **hydrodynamic
dissipation** by the lubrication correction + the IBM/LBM fluid. The contact
force here is therefore *not* a full Hertzian elastic contact (that would
double-count the membrane); it is a short-range regularization that (i)
prevents membrane interpenetration and (ii) optionally adds the two pieces of
contact physics the bare repulsion omits — **restitution** (normal damping)
and **friction** (tangential).

This is the "Path A" contact: a conservative spring + a DEM-style
dashpot + Coulomb friction, evaluated **node to node** (each pair of membrane
markers within the cutoff).

## Force law

For a target node `a` (position `x_a`, velocity `v_a`) interacting with a
source node/surface `b` (`x_b`, `v_b`), let

    d   = x_a - x_b           (minimum-image in x when periodic)
    r   = |d|,   n = d / r    (unit normal, from b toward a)

The contact is active only for `r < r_cut`. The force on node `a` is

    F = F_n * n  +  F_t

with

    F_rep = epsilon * (sigma / r)^power            (conservative repulsion)
    v_n   = (v_a - v_b) . n                         (>0 separating, <0 approaching)
    F_n   = max(0,  F_rep - gamma_n * v_n)          (normal: spring + dashpot)
    v_t   = (v_a - v_b) - v_n * n                   (tangential relative velocity)
    F_t   = -mu * F_n * v_t / |v_t|     (|v_t| > 0)  (Coulomb friction)

By Newton's third law `F` acts as `-F` on `b`; the implementation satisfies
`pairForce(a,b) = -pairForce(b,a)`, so the two-sided, one-sided, and
cell-list code paths are consistent. Walls and obstacles are treated as static
surfaces (`v_b = 0`) with `n` the surface normal.

### Parameters (`RepulsionParams`)

| symbol | field | meaning |
|---|---|---|
| `epsilon` | `epsilon` | repulsion strength |
| `sigma`   | `sigma`   | repulsion range |
| `r_cut`   | `r_cut`   | interaction cutoff (LU) |
| `power`   | `power`   | repulsion exponent |
| `gamma_n` | `damping_normal` | normal dashpot coefficient (**0 = off**) |
| `mu`      | `friction_coeff` | Coulomb friction coefficient (**0 = off**) |

`damping_normal = friction_coeff = 0` recovers the legacy conservative
repulsion exactly, so existing simulations are unaffected unless the DEM terms
are explicitly switched on.

## Physical behaviour

- **Normal damping (`gamma_n > 0`).** The dashpot opposes the normal relative
  velocity, so it *adds* to the push while two surfaces approach and *subtracts*
  while they separate. The `max(0, .)` clamp makes the contact **cohesionless**:
  the normal force never becomes attractive (a bare dashpot would briefly pull
  the pair together at the end of a collision). The net effect over an
  approach/separation cycle is energy loss, i.e. a **coefficient of restitution
  below unity**. (Cundall & Strack 1979; Brilliantov et al. 1996.)
- **Tangential friction (`mu > 0`).** A kinetic Coulomb force opposes sliding,
  with magnitude capped at `mu * |F_n|`. This is what lets dense suspensions
  build **load-bearing, friction-stabilised arches** at a constriction — the
  mechanism behind clogging and granular jamming.

## Validity / caveats

- This is a contact *regularization* for fully-resolved soft capsules, **not** a
  Hertzian overlap model. For nearly-rigid grains where the deformation is not
  resolved, a coarse-grained Hertz-Mindlin DEM particle would be the correct
  model instead (not implemented here).
- The tangential law is kinetic Coulomb friction without a stored tangential
  (Mindlin) spring, so it does not model true static friction / stick; it
  captures sliding resistance, which is the dominant effect for flowing
  suspensions.

## Tests

`tests/python/test_contact_friction_damping.py` pins: legacy equivalence at
zero coefficients, inertness at zero relative velocity, dashpot resists
approach / clamps on separation, and Coulomb friction opposes sliding within
the `mu*|F_n|` cap.

## References

- P. A. Cundall and O. D. L. Strack, *A discrete numerical model for granular
  assemblies*, Géotechnique **29**, 47–65 (1979).
- N. V. Brilliantov, F. Spahn, J.-M. Hertzsch, T. Pöschel, *Model for collisions
  in granular gases*, Phys. Rev. E **53**, 5382 (1996).
- S. Luding, *Cohesive, frictional powders: contact models for tension*,
  Granular Matter **10**, 235 (2008).
