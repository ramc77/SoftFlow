"""Jamming diagnostics: packing field φ(x,y), contact number Z̄, per-type
contact statistics matrix Z_ij, force-network percolation, MSD plateau
detection, and Falk–Langer non-affine D²_min.

All functions consume one or more ``SimulationSnapshot`` objects and
return numpy arrays or simple result dataclasses. Periodic-x is
applied throughout.

References
----------
  - Falk & Langer, *Phys. Rev. E* **57**, 7192 (1998) — D²_min.
  - O'Hern et al., *Phys. Rev. E* **68**, 011306 (2003) — jamming Z.
  - Stauffer & Aharony, *Introduction to Percolation Theory* (1994).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from .snapshot import SimulationSnapshot
from .patterns import _build_contact_graph, hoshen_kopelman


__all__ = [
    "PackingField",
    "packing_field",
    "ContactStats",
    "contact_number",
    "PerTypeContactStats",
    "per_type_contact_stats",
    "PercolationResult",
    "force_percolation",
    "MSDResult",
    "mean_squared_displacement",
    "non_affine_d2min",
]


# ── Local packing-fraction field φ(x, y) ────────────────────────────


@dataclass(frozen=True)
class PackingField:
    """Coarse-grained packing-fraction field on a regular grid.

    Attributes
    ----------
    grid_x, grid_y : np.ndarray
        Cell-centre coordinates along each axis.
    phi : np.ndarray, shape (n_y, n_x)
        Coarse-grained area fraction in each cell — the sum of disk
        areas whose centre lies in the cell, divided by the cell area.
        Note this is a centre-of-mass approximation; disks straddling
        a cell boundary are assigned to a single cell. For accurate
        sub-cell averaging use a finer grid.
    cell_area : float
    """
    grid_x:    np.ndarray
    grid_y:    np.ndarray
    phi:       np.ndarray
    cell_area: float


def packing_field(
    snap: SimulationSnapshot,
    *,
    n_x: int = 30,
    n_y: int = 10,
) -> PackingField:
    """Coarse-grained packing-fraction field φ(x, y).

    The domain is partitioned into ``n_x × n_y`` rectangular cells.
    Each disk is assigned to its centroid's cell; the cell's φ is
    the sum of disk areas in it divided by the cell area.

    For uniform fills, mean(φ) ≈ N · π · ⟨r²⟩ / (Lx · Ly).
    """
    if n_x < 1 or n_y < 1:
        raise ValueError("n_x and n_y must be ≥ 1")
    Lx, Ly = snap.Lx, snap.Ly
    if not (Lx > 0 and Ly > 0):
        raise ValueError("snapshot has zero box area")

    edges_x = np.linspace(0.0, Lx, n_x + 1)
    edges_y = np.linspace(0.0, Ly, n_y + 1)
    cell_area = (Lx / n_x) * (Ly / n_y)

    ix = np.clip(np.digitize(snap.positions[:, 0], edges_x[1:-1]),
                 0, n_x - 1)
    iy = np.clip(np.digitize(snap.positions[:, 1], edges_y[1:-1]),
                 0, n_y - 1)

    disk_area = np.pi * snap.radii ** 2
    phi = np.zeros((n_y, n_x), dtype=np.float64)
    np.add.at(phi, (iy, ix), disk_area)
    phi /= cell_area

    grid_x = 0.5 * (edges_x[:-1] + edges_x[1:])
    grid_y = 0.5 * (edges_y[:-1] + edges_y[1:])
    return PackingField(grid_x=grid_x, grid_y=grid_y, phi=phi,
                         cell_area=cell_area)


# ── Contact number Z̄ ───────────────────────────────────────────────


@dataclass(frozen=True)
class ContactStats:
    """Result bundle for the bulk contact-number diagnostic."""
    n_particles:   int
    n_contacts:    int
    Z_mean:        float
    Z_per_particle: np.ndarray   # (N,)


def contact_number(
    snap: SimulationSnapshot,
    *,
    contact_cutoff: float = 0.5,
) -> ContactStats:
    """Average and per-particle contact number Z.

    Definition: two particles k, ℓ are in contact iff
    ``d_kℓ < r_k + r_ℓ + contact_cutoff``. ``Z_per_particle[k]`` is
    the number of distinct ℓ ≠ k that touch k. ``Z_mean`` is the
    average of that array.

    For a hex lattice with cutoff slightly above the lattice spacing,
    Z_mean → 6 for interior particles (less for boundary particles in
    a finite snapshot).
    """
    n = snap.n_particles
    if n == 0:
        return ContactStats(n_particles=0, n_contacts=0,
                             Z_mean=0.0,
                             Z_per_particle=np.empty(0, dtype=np.int64))

    A = _build_contact_graph(snap, contact_cutoff=contact_cutoff,
                              use_bonds=False)
    deg = np.asarray(A.sum(axis=1)).ravel().astype(np.int64)
    return ContactStats(
        n_particles=n,
        n_contacts=int(deg.sum() // 2),
        Z_mean=float(deg.mean()),
        Z_per_particle=deg,
    )


# ── Per-type contact statistics matrix Z_ij ─────────────────────────


@dataclass(frozen=True)
class PerTypeContactStats:
    """Per-type contact matrix (the project specification + user request).

    Attributes
    ----------
    types : np.ndarray, shape (T,)
        Distinct type labels present in the snapshot, sorted ascending.
    n_particles : np.ndarray, shape (T,)
        Count of particles per type.
    Z_matrix : np.ndarray, shape (T, T)
        ``Z_matrix[i, j]`` = average number of contacts a particle of
        type ``types[i]`` makes with particles of type ``types[j]``.
        Diagonal counts AA pairs *as seen from one A particle*: each
        unique AA pair contributes to two A particles, so
        ``Z_matrix[i, i] = 2 N_ii / n_particles[i]``. Off-diagonal
        ``Z_matrix[i, j] = N_ij / n_particles[i]`` (asymmetric in
        general; satisfies ``n_particles[i] * Z_matrix[i, j] ==
        n_particles[j] * Z_matrix[j, i]``).
    n_pairs : np.ndarray, shape (T, T)
        Symmetric matrix of *unique* pair counts (so ``n_pairs[i, i]
        == N_ii``, not ``2 N_ii``).
    """
    types:        np.ndarray
    n_particles:  np.ndarray
    Z_matrix:     np.ndarray
    n_pairs:      np.ndarray


def per_type_contact_stats(
    snap: SimulationSnapshot,
    *,
    contact_cutoff: float = 0.5,
) -> PerTypeContactStats:
    """Compute the per-type contact matrix.

    Uses the same KDTree-based contact graph as ``contact_number``,
    then accumulates per-type pair counts.
    """
    n = snap.n_particles
    types_present = snap.distinct_types().astype(np.int64)
    T = types_present.size
    n_per_type = np.asarray(
        [(snap.types == t).sum() for t in types_present], dtype=np.int64)

    if n == 0 or T == 0:
        return PerTypeContactStats(
            types=types_present,
            n_particles=n_per_type,
            Z_matrix=np.zeros((T, T), dtype=np.float64),
            n_pairs=np.zeros((T, T), dtype=np.int64),
        )

    # Contact graph as a sparse (n × n) adjacency matrix.
    A = _build_contact_graph(snap, contact_cutoff=contact_cutoff,
                              use_bonds=False)

    # Map each type label to its index in the result matrix.
    type_to_idx = {int(t): k for k, t in enumerate(types_present)}
    idx_of = np.fromiter(
        (type_to_idx[int(t)] for t in snap.types),
        dtype=np.int64, count=n)

    # Walk the unique edges of the symmetric graph (i < j).
    coo = A.tocoo()
    upper = coo.row < coo.col
    rows = coo.row[upper]
    cols = coo.col[upper]

    n_pairs = np.zeros((T, T), dtype=np.int64)
    for r, c in zip(rows, cols):
        ti = idx_of[r]
        tj = idx_of[c]
        n_pairs[ti, tj] += 1
        if ti != tj:
            n_pairs[tj, ti] += 1   # keep n_pairs symmetric

    Z = np.zeros((T, T), dtype=np.float64)
    for i in range(T):
        if n_per_type[i] == 0:
            continue
        for j in range(T):
            if i == j:
                Z[i, i] = 2.0 * n_pairs[i, i] / n_per_type[i]
            else:
                Z[i, j] = n_pairs[i, j] / n_per_type[i]

    return PerTypeContactStats(
        types=types_present,
        n_particles=n_per_type,
        Z_matrix=Z,
        n_pairs=n_pairs,
    )


# ── Force-network percolation ───────────────────────────────────────


@dataclass(frozen=True)
class PercolationResult:
    """Whether the contact / bond graph spans the channel cross-section.

    A contact / bond cluster is "spanning" if it contains at least one
    particle whose centroid is in the bottom band ``y < y_low_band``
    *and* at least one in the top band ``y > y_top_band``. The
    spanning cluster's size is reported separately.
    """
    spans:              bool
    spanning_size:      int
    largest_cluster:    int
    n_clusters:         int
    y_low_band:         float
    y_top_band:         float


def force_percolation(
    snap: SimulationSnapshot,
    *,
    contact_cutoff: float = 0.5,
    use_bonds: bool = False,
    band_fraction: float = 0.10,
) -> PercolationResult:
    """Detect cross-channel percolation of the contact / bond network.

    ``band_fraction`` defines the y-thickness (as a fraction of Ly) of
    the "near-wall" bands at the top and bottom that the spanning
    cluster must touch. With the default 0.10, a cluster spans the
    channel if it includes any particle with ``y < 0.10 Ly`` and any
    particle with ``y > 0.90 Ly``.
    """
    if not (0.0 < band_fraction < 0.5):
        raise ValueError("band_fraction must be in (0, 0.5)")

    out = hoshen_kopelman(snap, contact_cutoff=contact_cutoff,
                           use_bonds=use_bonds)
    if out.n_clusters == 0:
        return PercolationResult(
            spans=False, spanning_size=0, largest_cluster=0,
            n_clusters=0, y_low_band=0.0, y_top_band=snap.Ly)

    Ly = snap.Ly
    y_low = band_fraction * Ly
    y_high = (1.0 - band_fraction) * Ly

    in_low  = snap.positions[:, 1] <  y_low
    in_high = snap.positions[:, 1] >  y_high

    spanning_size = 0
    for cluster_id in np.unique(out.labels):
        members = out.labels == cluster_id
        if np.any(members & in_low) and np.any(members & in_high):
            sz = int(members.sum())
            if sz > spanning_size:
                spanning_size = sz

    return PercolationResult(
        spans=spanning_size > 0,
        spanning_size=spanning_size,
        largest_cluster=int(out.largest_size),
        n_clusters=int(out.n_clusters),
        y_low_band=y_low,
        y_top_band=y_high,
    )


# ── Mean-squared displacement and plateau detection ────────────────


@dataclass(frozen=True)
class MSDResult:
    """Time-series MSD with plateau diagnostic."""
    lag:           np.ndarray   # (T-1,) lag in time-units
    msd:           np.ndarray   # (T-1,) ⟨|Δr|²⟩
    is_plateau:    bool
    plateau_value: float        # nan if no plateau detected
    log_slope:     float        # log–log slope over the late half


def mean_squared_displacement(
    snaps: list[SimulationSnapshot],
    *,
    plateau_slope_threshold: float = 0.1,
) -> MSDResult:
    """MSD(t) from a sequence of snapshots, with plateau detection.

    The snapshot list must contain at least 3 entries with a fixed
    particle set (no insertion / deletion in between). Periodic-x is
    *not* unwrapped here — for accurate MSDs over many wraps, use the
    saved CSV trajectory (which records absolute positions) and
    construct snapshots from it.

    Plateau detection: the late-half log–log slope is compared to
    ``plateau_slope_threshold``; below it, we declare a plateau and
    report the late-half mean of MSD as ``plateau_value``.
    """
    if len(snaps) < 3:
        raise ValueError("need at least 3 snapshots for MSD")
    n = snaps[0].n_particles
    for s in snaps[1:]:
        if s.n_particles != n:
            raise ValueError(
                "MSD requires a constant particle set across snapshots")

    times = np.asarray([s.time for s in snaps], dtype=np.float64)
    pos0  = snaps[0].positions
    Lx    = snaps[0].Lx if snaps[0].periodic_x else 0.0

    msd  = np.empty(len(snaps) - 1, dtype=np.float64)
    lags = np.empty(len(snaps) - 1, dtype=np.float64)
    for k, s in enumerate(snaps[1:], start=1):
        d = s.positions - pos0
        if Lx > 0.0:
            d[:, 0] -= Lx * np.round(d[:, 0] / Lx)
        msd[k - 1]  = float(np.mean(d[:, 0] ** 2 + d[:, 1] ** 2))
        lags[k - 1] = times[k] - times[0]

    # Late-half log-log slope.
    half = max(1, len(lags) // 2)
    late_l = lags[half:]
    late_m = msd[half:]
    valid  = (late_l > 0) & (late_m > 0)
    if valid.sum() < 2:
        slope = float("nan")
    else:
        slope = float(np.polyfit(np.log(late_l[valid]),
                                  np.log(late_m[valid]), 1)[0])
    is_plateau = bool(np.isfinite(slope) and slope < plateau_slope_threshold)
    plateau_value = float(late_m.mean()) if is_plateau else float("nan")

    return MSDResult(lag=lags, msd=msd,
                      is_plateau=is_plateau,
                      plateau_value=plateau_value,
                      log_slope=slope)


# ── Non-affine displacement (Falk & Langer 1998) ────────────────────


def non_affine_d2min(
    snap_t0: SimulationSnapshot,
    snap_t1: SimulationSnapshot,
    *,
    neighbour_cutoff: float,
) -> np.ndarray:
    """Per-particle Falk–Langer D²_min between two snapshots.

    D²_min(k) is the residual of the best local affine fit to k's
    neighbours' displacements::

      F_k = argmin_F Σ_{ℓ ∈ N_k} ‖(r_ℓ(t_1) - r_k(t_1))
                                    - F (r_ℓ(t_0) - r_k(t_0))‖²
      D²_min(k) = (1 / |N_k|) Σ ‖(...)‖²    at the optimum F.

    Pure affine deformations give D²_min ≡ 0 (the local F absorbs
    them). Plastic / non-affine motion is what the metric measures.

    The two snapshots must contain the same particle set; periodic-x
    is unwrapped via minimum-image at the per-pair level.
    """
    if snap_t0.n_particles != snap_t1.n_particles:
        raise ValueError("D²_min requires identical particle sets")
    n = snap_t0.n_particles
    if n == 0:
        return np.empty(0, dtype=np.float64)

    pos0 = snap_t0.positions
    pos1 = snap_t1.positions
    Lx   = snap_t0.Lx if snap_t0.periodic_x else 0.0

    tree = cKDTree(pos0)
    d2 = np.zeros(n, dtype=np.float64)
    for k in range(n):
        idxs = tree.query_ball_point(pos0[k], r=neighbour_cutoff)
        idxs = [j for j in idxs if j != k]
        m = len(idxs)
        if m < 2:
            d2[k] = 0.0
            continue
        # Reference and current relative coordinates, with periodic-x.
        X = pos0[idxs] - pos0[k]    # (m, 2)
        Y = pos1[idxs] - pos1[k]    # (m, 2)
        if Lx > 0.0:
            X[:, 0] -= Lx * np.round(X[:, 0] / Lx)
            Y[:, 0] -= Lx * np.round(Y[:, 0] / Lx)
        # Best affine F = (Y^T X) (X^T X)^{-1} via lstsq for stability.
        # We solve X · F^T = Y in the least-squares sense, so F^T is
        # the (2 × 2) matrix returned.
        F_T, residuals, rank, _ = np.linalg.lstsq(X, Y, rcond=None)
        # residuals is empty when rank deficient — fall back to manual
        # residual computation (more robust than relying on lstsq's
        # output).
        resid = Y - X @ F_T
        d2[k] = float(np.sum(resid ** 2) / m)
    return d2
