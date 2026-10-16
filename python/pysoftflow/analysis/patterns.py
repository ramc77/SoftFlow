"""Pattern-detection diagnostics: lane order, Hoshen–Kopelman clustering,
cluster persistence.

Functions
---------

  lane_order(snap, axis='x')
      Φ_lane = ⟨cos(2 θ)⟩, with θ the angle between particle velocity
      and the chosen channel axis. Φ → 1 means motion is aligned
      with the axis (lanes); Φ → -1 means motion is perpendicular;
      Φ → 0 means isotropic.

  hoshen_kopelman(snap, contact_cutoff)
      Connected-component labelling on the contact graph using
      ``scipy.sparse.csgraph.connected_components`` (which gives the
      same result as Hoshen-Kopelman for this problem). Returns a
      ``ClusterLabels`` bundle with per-particle labels and the
      cluster-size distribution.

  cluster_persistence(snaps_a, snaps_b, contact_cutoff)
      Lag-1 autocorrelation of cluster membership: how much of the
      "stay together" structure is preserved between two snapshots?
      Defined here as the Jaccard similarity over particle pairs
      that share a cluster:

          P = |pairs_in_same_cluster_a ∩ pairs_in_same_cluster_b|
              / |pairs_in_same_cluster_a ∪ pairs_in_same_cluster_b|

      P → 1: structure is fully preserved.
      P → 0: clusters have completely re-formed.

References
----------
  - Hoshen & Kopelman, *Phys. Rev. B* **14**, 3438 (1976).
  - Vissers, van Blaaderen, Imhof, *Phys. Rev. Lett.* **106**, 228303
    (2011) — lane formation in driven binary mixtures.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from .snapshot import SimulationSnapshot


__all__ = [
    "lane_order",
    "ClusterLabels",
    "hoshen_kopelman",
    "cluster_persistence",
]


# ── Lane order parameter ────────────────────────────────────────────


def lane_order(
    snap: SimulationSnapshot,
    *,
    axis: str = "x",
    type_filter: int | None = None,
    speed_threshold: float = 0.0,
) -> float:
    """Lane order parameter Φ = ⟨cos(2 θ)⟩.

    Parameters
    ----------
    snap : SimulationSnapshot
    axis : 'x' or 'y'
        Reference axis. θ is the angle between the particle's velocity
        and this axis.
    type_filter : int, optional
        If given, only count particles of this type.
    speed_threshold : float
        Particles with |v| ≤ this threshold contribute zero (they
        contribute neither alignment nor anti-alignment information).
        Default 0.0 — every particle with any velocity counts.

    Returns
    -------
    float
        Φ ∈ [-1, 1]. ``nan`` if no particle clears the filter.
    """
    if axis not in ("x", "y"):
        raise ValueError(f"axis must be 'x' or 'y' (got {axis!r})")

    v = snap.velocities
    speed = np.linalg.norm(v, axis=1)
    sel = speed > speed_threshold
    if type_filter is not None:
        sel &= snap.types == int(type_filter)
    if not np.any(sel):
        return float("nan")

    vx = v[sel, 0]
    vy = v[sel, 1]
    s  = speed[sel]
    if axis == "x":
        cos_th = vx / s
    else:
        cos_th = vy / s
    cos_2th = 2.0 * cos_th * cos_th - 1.0   # cos(2θ) = 2cos²θ − 1
    return float(np.mean(cos_2th))


# ── Cluster labelling (Hoshen–Kopelman semantics) ──────────────────


@dataclass(frozen=True)
class ClusterLabels:
    """Result bundle for cluster labelling on a contact graph."""
    labels:        np.ndarray   # (N,)  cluster id per particle
    n_clusters:    int          # number of distinct clusters
    sizes:         np.ndarray   # (n_clusters,) sorted DESCENDING
    largest_size:  int


def _build_contact_graph(
    snap: SimulationSnapshot,
    *,
    contact_cutoff: float,
    use_bonds: bool,
) -> coo_matrix:
    """Return a sparse symmetric adjacency matrix of the contact graph.

    ``contact_cutoff`` is the absolute centre-distance threshold
    *added* to the sum of radii: two particles k, ℓ are in contact iff
    ``d_kℓ < r_k + r_ℓ + contact_cutoff``.

    If ``use_bonds`` is True and ``snap.bonds`` is non-empty, the
    graph is the adhesion bond graph instead — cheaper and more
    physically meaningful in adhesive runs.
    """
    n = snap.n_particles
    if n == 0:
        return coo_matrix((n, n), dtype=np.uint8)

    if use_bonds and snap.bonds.shape[0] > 0:
        edges = snap.bonds
    else:
        # Pad each particle's effective radius into the KDTree query
        # by querying with the maximum sum-of-radii + cutoff, then
        # filtering. With monodisperse + small polydispersity this is
        # tight enough; for very heterogeneous radii a per-particle
        # query loop would be tighter but slower.
        r_max = float(np.max(snap.radii))
        cutoff = 2.0 * r_max + contact_cutoff
        Lx = snap.Lx if snap.periodic_x else 0.0
        if Lx > 0.0:
            pts = np.vstack([
                snap.positions,
                snap.positions + np.array([ Lx, 0.0]),
                snap.positions + np.array([-Lx, 0.0]),
            ])
        else:
            pts = snap.positions

        tree = cKDTree(pts)
        # Use query_pairs only on the original copy; for image copies
        # we cross-query.
        pairs = []
        for i, p in enumerate(snap.positions):
            idxs = tree.query_ball_point(p, r=cutoff)
            for j in idxs:
                j_canon = j % n
                if j_canon == i:
                    continue
                d = np.linalg.norm(p - pts[j])
                threshold = snap.radii[i] + snap.radii[j_canon] + contact_cutoff
                if d <= threshold:
                    pairs.append((min(i, j_canon), max(i, j_canon)))
        edges = np.unique(np.asarray(pairs, dtype=np.int64), axis=0) \
                if pairs else np.empty((0, 2), dtype=np.int64)

    if edges.shape[0] == 0:
        return coo_matrix((n, n), dtype=np.uint8)

    rows = np.concatenate([edges[:, 0], edges[:, 1]])
    cols = np.concatenate([edges[:, 1], edges[:, 0]])
    data = np.ones(rows.size, dtype=np.uint8)
    return coo_matrix((data, (rows, cols)), shape=(n, n))


def hoshen_kopelman(
    snap: SimulationSnapshot,
    *,
    contact_cutoff: float = 0.5,
    use_bonds: bool = False,
) -> ClusterLabels:
    """Label particles by connected component of the contact graph.

    Equivalent to the Hoshen–Kopelman algorithm for this problem
    (HK is just connected-component labelling with an in-place
    union-find). We use ``scipy.sparse.csgraph.connected_components``
    for the actual labelling — same answer, dependable performance.

    Parameters
    ----------
    snap : SimulationSnapshot
    contact_cutoff : float
        Centre-distance slack ``δ_c`` so contact means
        ``d_kℓ < r_k + r_ℓ + δ_c``.
    use_bonds : bool
        If True and the snapshot carries adhesion bonds, use that
        graph instead of distance-based contacts.
    """
    n = snap.n_particles
    if n == 0:
        return ClusterLabels(labels=np.empty(0, dtype=np.int64),
                              n_clusters=0,
                              sizes=np.empty(0, dtype=np.int64),
                              largest_size=0)

    A = _build_contact_graph(snap, contact_cutoff=contact_cutoff,
                              use_bonds=use_bonds)
    n_components, labels = connected_components(
        csgraph=A, directed=False, return_labels=True)
    sizes = np.bincount(labels)
    order = np.argsort(-sizes)
    sizes_sorted = sizes[order]

    # Re-label so that cluster 0 is the largest, cluster 1 the second,
    # etc. Useful for plotting and for the persistence diagnostic.
    relabel = np.empty(n_components, dtype=np.int64)
    relabel[order] = np.arange(n_components)
    new_labels = relabel[labels]

    return ClusterLabels(
        labels=new_labels.astype(np.int64),
        n_clusters=int(n_components),
        sizes=sizes_sorted.astype(np.int64),
        largest_size=int(sizes_sorted[0]),
    )


# ── Cluster persistence (lag-1 Jaccard over same-cluster pairs) ─────


def _same_cluster_pairs(labels: np.ndarray) -> set[tuple[int, int]]:
    """Set of (i, j) pairs with i < j that share a cluster label."""
    pairs: set[tuple[int, int]] = set()
    n = labels.size
    by_cluster: dict[int, list[int]] = {}
    for idx, lab in enumerate(labels):
        by_cluster.setdefault(int(lab), []).append(idx)
    for members in by_cluster.values():
        if len(members) < 2:
            continue
        members.sort()
        m = len(members)
        for a in range(m):
            for b in range(a + 1, m):
                pairs.add((members[a], members[b]))
    return pairs


def cluster_persistence(
    labels_a: np.ndarray,
    labels_b: np.ndarray,
) -> float:
    """Jaccard similarity of same-cluster pairs across two labellings.

    Both labellings must have the same length (i.e. a fixed particle
    set across the two snapshots — no insertion / deletion in
    between). Returns 1.0 if the sets of "co-clustered" pairs are
    identical, 0.0 if they're disjoint, ``nan`` if both are empty.
    """
    if labels_a.shape != labels_b.shape:
        raise ValueError("labels_a and labels_b must have the same shape")

    pa = _same_cluster_pairs(labels_a)
    pb = _same_cluster_pairs(labels_b)

    inter = len(pa & pb)
    union = len(pa | pb)
    if union == 0:
        return float("nan")
    return float(inter / union)
