"""Species-resolved radial distribution function g_AB(r).

The RDF measures the local density of B-type particles at distance r
from any A-type particle, normalised by the bulk B-density. For an
ideal gas g(r) ≡ 1; peaks at characteristic spacings indicate
structural correlations (lattice spacing, contact distance, etc.).

Definition (2-D, e.g. Allen & Tildesley §2.6)::

  g_AB(r) = (Lx · Ly / (2π r dr)) · ⟨n_B(r; r+dr)⟩_A / N_A
            ÷ N_B / (Lx · Ly)

where ``⟨n_B(r; r+dr)⟩_A`` is the average count of type-B particles in
the annular shell [r, r+dr] around a type-A particle. We compute it
by summing over all (A, B) pairs and dividing by the number of A
particles. Periodic-x is handled via minimum-image distances; the
y-direction is treated as bounded (no wrap), matching SoftFlow's
channel geometry.

For self-pairs (type_a == type_b) we count each unordered pair once
and divide by N_A; the result is the standard same-species g(r).

Implementation: pure numpy, ``scipy.spatial.cKDTree.query_pairs`` for
the neighbour search. Particle counts in the few-thousand range run
in milliseconds.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from .snapshot import SimulationSnapshot


__all__ = ["RDFResult", "radial_distribution"]


@dataclass(frozen=True)
class RDFResult:
    """Result bundle for a species-resolved RDF computation."""
    type_a:    int
    type_b:    int
    r_centres: np.ndarray   # (n_bins,)  bin centres
    g_r:       np.ndarray   # (n_bins,)  g_AB(r)
    n_a:       int
    n_b:       int
    box_area:  float


def radial_distribution(
    snap: SimulationSnapshot,
    *,
    type_a: int = 0,
    type_b: int = 1,
    r_max: float | None = None,
    n_bins: int = 50,
) -> RDFResult:
    """Compute the species-resolved RDF g_AB(r).

    Parameters
    ----------
    snap : SimulationSnapshot
    type_a, type_b : int
        Particle type labels for the two populations. Pass the same
        value for both to get the same-species g_AA(r).
    r_max : float, optional
        Maximum pair distance to bin. Defaults to ``min(Lx/2, Ly/2)``
        — beyond half the box, periodic-x makes pair counts ill-
        defined. The y-direction is finite, so the half-Ly bound is
        the conservative choice.
    n_bins : int
        Number of radial bins. Bin width is ``r_max / n_bins``.

    Returns
    -------
    RDFResult
    """
    if n_bins < 2:
        raise ValueError("n_bins must be ≥ 2")

    Lx, Ly  = snap.Lx, snap.Ly
    if r_max is None:
        r_max = 0.5 * min(Lx, Ly)
    if not (r_max > 0):
        raise ValueError("r_max must be > 0")

    pos    = snap.positions
    types  = snap.types
    mask_a = types == int(type_a)
    mask_b = types == int(type_b)
    n_a = int(mask_a.sum())
    n_b = int(mask_b.sum())
    if n_a == 0 or n_b == 0:
        # No pairs possible — return zeros, valid bins.
        edges = np.linspace(0.0, r_max, n_bins + 1)
        centres = 0.5 * (edges[:-1] + edges[1:])
        return RDFResult(
            type_a=int(type_a), type_b=int(type_b),
            r_centres=centres, g_r=np.zeros(n_bins),
            n_a=n_a, n_b=n_b, box_area=Lx * Ly)

    # Collect pair distances. We use cKDTree for speed; periodic-x is
    # handled by inserting *image* copies of B particles shifted by
    # ±Lx, then deduplicating. For r_max ≤ Lx/2 only one image per
    # side is sufficient.
    pos_a = pos[mask_a]
    pos_b = pos[mask_b]

    if snap.periodic_x and Lx > 0.0:
        b_with_images = np.vstack([
            pos_b,
            pos_b + np.array([ Lx, 0.0]),
            pos_b + np.array([-Lx, 0.0]),
        ])
    else:
        b_with_images = pos_b
    tree_b = cKDTree(b_with_images)

    same_species = (int(type_a) == int(type_b))
    pair_distances = []
    for i, p in enumerate(pos_a):
        idxs = tree_b.query_ball_point(p, r=r_max)
        for j in idxs:
            # Map image index back to canonical b-index.
            j_canon = j % n_b
            if same_species and j_canon == i:
                continue   # skip self
            d = np.linalg.norm(p - b_with_images[j])
            if d <= 0.0:
                continue
            if same_species:
                # Avoid double counting unordered pairs.
                if j_canon <= i:
                    continue
            pair_distances.append(d)
    pair_distances = np.asarray(pair_distances, dtype=np.float64)

    edges = np.linspace(0.0, r_max, n_bins + 1)
    counts, _ = np.histogram(pair_distances, bins=edges)
    centres = 0.5 * (edges[:-1] + edges[1:])
    dr      = (edges[1:] - edges[:-1])

    box_area = Lx * Ly
    rho_b    = n_b / box_area

    # 2-D shell area: 2π r dr. For same-species we counted each pair
    # once and divide by N_A; for cross-species we counted each (A, B)
    # pair once.
    shell_area = 2.0 * np.pi * centres * dr
    norm = n_a * rho_b * shell_area
    if same_species:
        # Each unique pair contributes to a single A particle's
        # neighbour count when we double back, so multiply counts by
        # 2 to recover the conventional same-species normalisation.
        counts = counts * 2

    with np.errstate(divide="ignore", invalid="ignore"):
        g_r = np.where(norm > 0, counts / norm, 0.0)

    return RDFResult(
        type_a=int(type_a), type_b=int(type_b),
        r_centres=centres, g_r=g_r,
        n_a=n_a, n_b=n_b, box_area=box_area,
    )
