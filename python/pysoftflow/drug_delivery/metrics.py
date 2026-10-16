"""Drug-delivery metrics: η, off-target fraction, RTD, dose map.

All four are pure functions on lists of carriers and absorbers
(``CarrierState`` / ``WallAbsorber``) that the orchestrator owns.
None of them touch the C++ engine.

Definitions
-----------

  delivery_efficiency
      η = target.cumulative_absorbed / Σ_i M_p_initial(i)

      Fraction of total payload that ended up in the designated
      target absorber. Bounded in [0, 1] when the target is the only
      sink and no payload is lost — but in general η + off-target +
      M_remaining ≈ 1 only under certain idealisations (see
      Limitations in the docs).

  off_target_fraction
      OTF = Σ_a (a.cumulative_absorbed) / Σ_i M_p_initial(i)

      where the sum runs over all absorbers labelled "off_target"
      (or any user-selected list).

  residence_time_distribution(snapshots, target_band, type_filter)
      For each carrier, the cumulative time spent inside the target
      band across the snapshot list. Returned as both the per-carrier
      array and a histogram.

  spatial_dose_map
      Cumulative scalar flux into each lattice cell, integrated over
      the run. We approximate this by the *cumulative drop in C*
      across all absorbers' patches plus the standing scalar field
      at end-of-run, which is not a true cell-by-cell flux integral.
      For a more accurate dose map, save C(t) snapshots and use the
      Phase-3 ``packing_field``-style coarse-graining helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from .absorbers import WallAbsorber
from .kinetics import CarrierState


__all__ = [
    "delivery_efficiency",
    "off_target_fraction",
    "ResidenceTimeResult",
    "residence_time_distribution",
    "spatial_dose_map",
]


def _total_loaded(carriers: Iterable[CarrierState]) -> float:
    return float(sum(c.M_p_initial for c in carriers))


def delivery_efficiency(
    target: WallAbsorber,
    carriers: Sequence[CarrierState],
) -> float:
    """η = target.cumulative_absorbed / Σ M_p_initial.

    Returns ``nan`` if no payload was loaded.
    """
    loaded = _total_loaded(carriers)
    if loaded <= 0.0:
        return float("nan")
    return float(target.cumulative_absorbed / loaded)


def off_target_fraction(
    off_targets: Iterable[WallAbsorber],
    carriers: Sequence[CarrierState],
) -> float:
    """Σ off_target absorbed / Σ M_p_initial."""
    loaded = _total_loaded(carriers)
    if loaded <= 0.0:
        return float("nan")
    absorbed = sum(a.cumulative_absorbed for a in off_targets)
    return float(absorbed / loaded)


@dataclass(frozen=True)
class ResidenceTimeResult:
    """Result bundle for the residence-time distribution."""
    per_carrier:    np.ndarray   # (N,) cumulative time inside target_band
    bin_edges:      np.ndarray   # histogram bin edges
    counts:         np.ndarray   # histogram counts (per bin)
    mean:           float
    median:         float
    target_band:    tuple[float, float]


def residence_time_distribution(
    snapshots,
    target_band: tuple[float, float],
    *,
    type_filter: int | None = None,
    n_bins: int = 20,
) -> ResidenceTimeResult:
    """Cumulative time each carrier spent inside ``target_band`` (in y).

    A carrier is "inside" if its centroid's y-coordinate is in
    ``[y_lo, y_hi]``. The total time is the sum of ``Δt = t_{k+1} -
    t_k`` over all consecutive snapshot pairs where the carrier was
    inside at the *start* of the interval (i.e. at ``t_k``).

    Parameters
    ----------
    snapshots : sequence of SimulationSnapshot
        Must share a fixed particle set. Time-ordered.
    target_band : (y_lo, y_hi)
    type_filter : int, optional
        If given, restrict to particles of this type. Otherwise all.
    n_bins : int
        Histogram bins for the returned distribution.
    """
    if len(snapshots) < 2:
        raise ValueError("need at least 2 snapshots for an RTD")
    y_lo, y_hi = target_band
    n_particles = snapshots[0].n_particles

    if type_filter is None:
        select = np.ones(n_particles, dtype=bool)
    else:
        select = snapshots[0].types == int(type_filter)
    n_sel = int(select.sum())
    if n_sel == 0:
        return ResidenceTimeResult(
            per_carrier=np.empty(0), bin_edges=np.linspace(0, 1, n_bins + 1),
            counts=np.zeros(n_bins, dtype=np.int64),
            mean=float("nan"), median=float("nan"),
            target_band=target_band,
        )

    cumulative = np.zeros(n_sel, dtype=np.float64)
    for k in range(len(snapshots) - 1):
        sk, sk1 = snapshots[k], snapshots[k + 1]
        if sk.n_particles != n_particles or sk1.n_particles != n_particles:
            raise ValueError("RTD requires a constant particle set")
        dt = sk1.time - sk.time
        if dt <= 0.0:
            continue
        y = sk.positions[select, 1]
        in_band = (y >= y_lo) & (y <= y_hi)
        cumulative[in_band] += dt

    counts, edges = np.histogram(
        cumulative, bins=n_bins,
        range=(0.0, max(snapshots[-1].time - snapshots[0].time, 1.0)))
    return ResidenceTimeResult(
        per_carrier=cumulative,
        bin_edges=edges,
        counts=counts.astype(np.int64),
        mean=float(cumulative.mean()),
        median=float(np.median(cumulative)),
        target_band=target_band,
    )


def spatial_dose_map(
    absorbers: Iterable[WallAbsorber],
    *,
    nx: int,
    ny: int,
) -> np.ndarray:
    """Cumulative dose absorbed per lattice cell, summed across absorbers.

    Returns an ``(ny, nx)`` array with each cell holding the total
    mass that passed through any absorber covering it. Cells outside
    every absorber are zero.

    This is a simple area-summed approximation: each absorber
    distributes its ``cumulative_absorbed`` uniformly across its own
    patch cells. If two absorbers overlap, both contributions are
    summed (and the total may exceed the actual mass deposited there
    — keep absorber patches disjoint to avoid this).
    """
    dose = np.zeros((ny, nx), dtype=np.float64)
    for a in absorbers:
        i_lo, i_hi = a.i_range
        j_lo, j_hi = a.j_range
        if a.cumulative_absorbed <= 0.0 or a.n_cells == 0:
            continue
        per_cell = a.cumulative_absorbed / a.n_cells
        dose[j_lo:j_hi, i_lo:i_hi] += per_cell
    return dose
