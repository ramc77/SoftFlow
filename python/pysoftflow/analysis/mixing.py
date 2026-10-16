"""Mixing / segregation indices: Lacey, Danckwerts, contact-asymmetry.

All three are scalar summaries of how well two species are mixed in a
snapshot. Higher = better mixed for the Lacey index, lower = better
mixed for the Danckwerts intensity.

Definitions
-----------

For a domain partitioned into ``n_bins`` bins along the chosen
``axis``, let p_k be the fraction of type-A particles among all
particles in bin k, and p̄ the global type-A fraction. Then:

  σ²       = (1/n_bins) Σ_k (p_k - p̄)²       sample variance over bins
  σ²_R     = p̄ (1 - p̄) / N̄                  binomial random-mix variance
                                              (N̄ = avg particles per bin)
  σ²_M     = p̄ (1 - p̄)                      fully segregated variance

Lacey (1954)::

  M_L = (σ²_M - σ²) / (σ²_M - σ²_R)         M_L → 1 means random / well-mixed
                                              M_L → 0 means fully segregated

Danckwerts (1952) intensity of segregation::

  I_S = σ² / (p̄ (1 - p̄))                   I_S → 0 means random / well-mixed
                                              I_S → 1 means fully segregated

The two are related by
``M_L ≈ 1 - I_S``  modulo the σ²_R correction.

Contact asymmetry (per the project specification): for the type-A and type-B
populations under a contact cutoff δ_c (centre-distance < r_i + r_j +
δ_c)::

  A_contact = (n_AA - n_BB) / (n_AA + n_BB + 1e-30)

negative values mean B-B contacts dominate, positive means A-A
contacts dominate. Useful for detecting size-driven segregation when
combined with the per-type contact matrix in `jamming.per_type_contact_stats`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .snapshot import SimulationSnapshot


__all__ = [
    "MixingIndex",
    "lacey_index",
    "danckwerts_intensity",
    "contact_asymmetry",
]


@dataclass(frozen=True)
class MixingIndex:
    """Result bundle for the variance-based mixing indices."""
    n_bins:        int
    axis:          str
    type_a:        int
    type_b:        int
    n_total:       int
    p_global:      float                    # global type-A fraction
    sigma2:        float                    # sample variance across bins
    sigma2_random: float                    # binomial baseline
    sigma2_max:    float                    # fully-segregated value
    M_lacey:       float                    # Lacey 1954
    I_danckwerts:  float                    # Danckwerts 1952


def _bin_axis_extents(snap: SimulationSnapshot, axis: str) -> tuple[float, float]:
    if axis == "x":
        return 0.0, snap.Lx
    if axis == "y":
        return 0.0, snap.Ly
    raise ValueError(f"axis must be 'x' or 'y' (got {axis!r})")


def _binned_fractions(
    snap: SimulationSnapshot,
    *,
    axis: str,
    n_bins: int,
    type_a: int,
    type_b: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Compute (n_a_per_bin, n_total_per_bin, p_global)."""
    if n_bins < 2:
        raise ValueError("n_bins must be ≥ 2")
    lo, hi = _bin_axis_extents(snap, axis)
    if not (hi > lo):
        raise ValueError("snapshot box has zero length along the chosen axis")

    coord = snap.positions[:, 0 if axis == "x" else 1]
    mask_a = snap.types == int(type_a)
    mask_b = snap.types == int(type_b)
    sel    = mask_a | mask_b
    if not np.any(sel):
        raise ValueError("no particles of type_a or type_b in snapshot")

    edges = np.linspace(lo, hi, n_bins + 1)
    bin_idx_all = np.clip(
        np.digitize(coord[sel], edges[1:-1]),
        0, n_bins - 1)
    bin_idx_a   = bin_idx_all[mask_a[sel]]

    n_total = np.bincount(bin_idx_all, minlength=n_bins).astype(np.float64)
    n_a     = np.bincount(bin_idx_a,   minlength=n_bins).astype(np.float64)

    n_total_global = int(np.sum(sel))
    n_a_global     = int(np.sum(mask_a))
    p_global       = n_a_global / n_total_global

    return n_a, n_total, p_global


def lacey_index(
    snap: SimulationSnapshot,
    *,
    type_a: int = 0,
    type_b: int = 1,
    axis: str = "y",
    n_bins: int = 10,
) -> MixingIndex:
    """Lacey (1954) mixing index. M → 1 = random/well-mixed; → 0 = segregated.

    Parameters
    ----------
    snap : SimulationSnapshot
    type_a, type_b : int
        Integer type labels of the two species. Particles whose type
        is neither are ignored.
    axis : 'x' or 'y'
        Direction along which to bin.
    n_bins : int
        Number of bins along ``axis``. Should be chosen so each bin
        contains O(10) particles for the binomial baseline to be
        meaningful.

    Returns
    -------
    MixingIndex
    """
    n_a, n_total, p_global = _binned_fractions(
        snap, axis=axis, n_bins=n_bins, type_a=type_a, type_b=type_b)

    # Skip empty bins so the variance is well-defined.
    occupied = n_total > 0
    if occupied.sum() < 2:
        raise ValueError("need at least 2 occupied bins to compute Lacey index")
    p = n_a[occupied] / n_total[occupied]
    sigma2 = float(np.var(p, ddof=0))

    n_bar = float(n_total[occupied].mean())
    sigma2_random = p_global * (1.0 - p_global) / max(n_bar, 1.0)
    sigma2_max    = p_global * (1.0 - p_global)

    denom = sigma2_max - sigma2_random
    if denom <= 0.0:
        # Degenerate: only one occupied bin or single-type input.
        M = float("nan")
    else:
        M = float((sigma2_max - sigma2) / denom)

    I_S = float(sigma2 / sigma2_max) if sigma2_max > 0 else float("nan")

    return MixingIndex(
        n_bins=int(occupied.sum()),
        axis=axis,
        type_a=int(type_a),
        type_b=int(type_b),
        n_total=int(np.sum(n_total)),
        p_global=float(p_global),
        sigma2=sigma2,
        sigma2_random=float(sigma2_random),
        sigma2_max=float(sigma2_max),
        M_lacey=M,
        I_danckwerts=I_S,
    )


def danckwerts_intensity(
    snap: SimulationSnapshot,
    *,
    type_a: int = 0,
    type_b: int = 1,
    axis: str = "y",
    n_bins: int = 10,
) -> float:
    """Danckwerts (1952) intensity of segregation. I_S → 0 = mixed.

    Convenience wrapper that returns the scalar intensity from
    `lacey_index`. Use ``lacey_index(...).I_danckwerts`` if you also
    want the variance components.
    """
    return lacey_index(snap, type_a=type_a, type_b=type_b,
                       axis=axis, n_bins=n_bins).I_danckwerts


def contact_asymmetry(
    snap: SimulationSnapshot,
    *,
    type_a: int = 0,
    type_b: int = 1,
    contact_cutoff: float = 1.0,
) -> float:
    """Contact asymmetry index ∈ [-1, 1].

    Definition::

      A = (n_AA - n_BB) / (n_AA + n_BB)

    where ``n_XY`` is the number of unique (X, Y) pairs in contact
    (centre-distance < r_i + r_j + ``contact_cutoff``). Returns 0.0
    when both counts are zero.

    Hex lattice with all type-A particles → A = 1; all type-B → A = -1;
    perfectly mixed alternating arrangement → A ≈ 0.
    """
    pos = snap.positions
    rad = snap.radii
    typ = snap.types
    n   = pos.shape[0]
    if n < 2:
        return 0.0

    Lx = snap.Lx if snap.periodic_x else 0.0

    n_aa = 0
    n_bb = 0
    for i in range(n):
        ti = int(typ[i])
        if ti != type_a and ti != type_b:
            continue
        for j in range(i + 1, n):
            tj = int(typ[j])
            if tj != type_a and tj != type_b:
                continue
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            if Lx > 0.0:
                if dx >  0.5 * Lx: dx -= Lx
                elif dx < -0.5 * Lx: dx += Lx
            if dx * dx + dy * dy <= (rad[i] + rad[j] + contact_cutoff) ** 2:
                if ti == tj == type_a:
                    n_aa += 1
                elif ti == tj == type_b:
                    n_bb += 1
                # cross pairs ignored — they go in the denominator of
                # other diagnostics (per_type_contact_stats), not here.

    denom = n_aa + n_bb
    if denom == 0:
        return 0.0
    return float((n_aa - n_bb) / denom)
