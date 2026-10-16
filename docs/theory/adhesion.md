# Adhesion — Bell-model bonds with optional catch-slip kinetics

## Bond model

Bonds are stored persistently in
`AdhesionModel::bonds_` (`src/membrane/adhesion.h:12–19`) as

```
struct Bond {
    int  capsule_i, node_i;     // capsule_j = -1 → bottom wall, -2 → top wall
    int  capsule_j, node_j;
    Real rest_length;
    Real current_force;
};
```

A bond exerts a Hookean spring force between the two attachment
points:

```
F = k_bond (d − ℓ_rest),     d = | X_i − X_j |.
```

(`adhesion.cpp:206`.)

## Stochastic formation

For each candidate (i, j) pair within `d_bond`, a Poisson process
forms a bond per timestep with probability

```
p_on = k_on Δt.
```

(`adhesion.cpp:49`.) Type-pair gating uses `adhesion_matrix[i][j]`;
when the matrix is empty, the current behaviour is **all types can
bond with all types** — a Phase-1 follow-up tightens this to "no
adhesion when matrix is empty".

## Stochastic dissociation — Bell (1978) and catch-slip

The standard Bell model gives an exponential force-dependent off-rate:

```
k_off(F) = k_off^0 exp( F / F_crit ),
p_off    = k_off(F) Δt.
```

(`adhesion.cpp:166–168`.)

When `params.adhesion.use_catch_slip == true`, the off-rate uses the
biphasic catch-slip form (Pereverzev et al. 2005, Thomas et al. 2008):

```
k_off(F) = k_catch exp( − F / F_catch ) + k_slip exp( F / F_slip ).
```

(`adhesion.cpp:162–163`.) At low force the catch term dominates — the
bond strengthens — while at high force the slip term takes over and
the bond breaks rapidly. This is the model used for circulating-cell
adhesion under physiological shear.

## Wall adhesion

`capsule_j = −1` means the bond attaches to the bottom wall at
fixed `(x, y_bottom)`; `capsule_j = −2` is the top wall. Wall bonds
use independent kinetic constants (`wall_k_on`, `wall_k_off`,
`wall_k_bond`) and a `wall_receptor_spacing` to discretize the wall
into pseudo-receptors. (`adhesion.cpp:124–193`.)

## Cluster detection

Connected-component labelling on the bond graph via union-find with
path compression returns:

- maximum cluster size (largest connected aggregate),
- mean cluster size,
- per-bond cluster ID for visualization.

(`adhesion.cpp:227–267`.) This is the basis for the embolization
metric — a cluster spanning the channel cross-section in flow.

## OpenMP & periodic-x safety

- Bond enumeration is serial; the per-step work is dominated by the
  Hookean spring update which is parallel-safe (one-sided writes).
- All inter-capsule and capsule-wall distances use the periodic-x
  minimum-image convention (`adhesion.cpp:78–81`).

## V&V

There is no closed-form analytical V&V case for stochastic adhesion;
acceptable validation comes from agreement with published simulation
results (e.g. Aceto et al. 2014, Au et al. 2016 for circulating-cell
clusters). A Phase-2 task is to compare cluster-size distributions
against the Müller–Fedosov–Gompper 2014 margination/clustering data.

## Modelling caveat (must be reflected in any output text)

Per the project specification, the tumour-aggregation infrastructure built on
top of this module is a **coarse-grained mechano-chemical proxy for
circulating-cell aggregation under flow**, not a validated cancer
model. The strong language constraint applies to docstrings, paper
text, and any auto-generated summary.

## References

- G. I. Bell, *Science* **200**, 618 (1978).
- M. Dembo, D. C. Torney, K. Saxman, D. Hammer, *Proc. R. Soc. B*
  **234**, 55 (1988).
- O. V. Pereverzev, E. V. Prezhdo, M. Forero, E. V. Sokurenko, W. E.
  Thomas, *Biophys. J.* **89**, 1446 (2005).
- W. E. Thomas, V. Vogel, E. Sokurenko, *Annu. Rev. Biophys.* **37**,
  399 (2008).
- N. Aceto et al., *Cell* **158**, 1110 (2014).
- S. Au et al., *Proc. Natl. Acad. Sci. USA* **113**, 4947 (2016).
