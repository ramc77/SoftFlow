"""
Post-processing and analysis functions for SoftFlow simulations.

Import this module separately for lightweight post-processing without
loading the full simulation engine:

    from pysoftflow.postprocess import (
        compute_segregation_index,
        compute_deformation_stats,
        compute_velocity_profile,
        compute_cell_free_layer,
        compute_radial_distribution,
    )

All functions work with numpy arrays extracted from the simulation,
so they can also be used on saved data files.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Tuple


# ══════════════════════════════════════════════════════════════
# Segregation & Mixing Metrics
# ══════════════════════════════════════════════════════════════

def compute_segregation_index(
    positions_by_type: Dict[str, np.ndarray],
    ny: int,
    n_bins: int = 20,
) -> dict:
    """Compute segregation index, mixing entropy, and margination.

    Parameters
    ----------
    positions_by_type : dict
        Maps type name → (N, 2) array of (x, y) centroid positions.
    ny : int
        Channel height in lattice units.
    n_bins : int
        Number of lateral bins for entropy calculation.

    Returns
    -------
    dict with keys:
        - ``SI`` : float — Segregation index (0=mixed, 1=segregated)
        - ``entropy`` : float — Mixing entropy
        - ``max_entropy`` : float — Maximum possible entropy
        - ``margination`` : dict[str, float] — Fraction of each type
          in the near-wall region (y < 0.2*ny or y > 0.8*ny)
        - ``cfl_bottom`` : float — Cell-free layer thickness (bottom wall)
        - ``cfl_top`` : float — Cell-free layer thickness (top wall)
        - ``lateral_dist`` : dict[str, np.ndarray] — Lateral distribution
          histogram for each type
        - ``bin_centers`` : np.ndarray — Bin center y-positions
    """
    # Bin edges
    bin_edges = np.linspace(0, ny, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # Count per bin per type
    type_names = list(positions_by_type.keys())
    n_types = len(type_names)
    counts = np.zeros((n_types, n_bins))

    for t, name in enumerate(type_names):
        pos = positions_by_type[name]
        if len(pos) == 0:
            continue
        y_vals = pos[:, 1]
        hist, _ = np.histogram(y_vals, bins=bin_edges)
        counts[t, :] = hist

    total_per_bin = counts.sum(axis=0)

    # Mixing entropy: S = -sum_i p_i * ln(p_i)  per occupied bin
    entropy = 0.0
    n_occupied = 0
    for b in range(n_bins):
        if total_per_bin[b] < 1:
            continue
        n_occupied += 1
        for t in range(n_types):
            p = counts[t, b] / total_per_bin[b]
            if p > 0:
                entropy -= p * np.log(p)

    if n_occupied > 0:
        entropy /= n_occupied

    max_entropy = np.log(max(n_types, 1)) if n_types > 0 else 1.0

    # Segregation index: SI = 1 - S/S_max
    SI = 1.0 - (entropy / max_entropy) if max_entropy > 0 else 0.0

    # Margination: fraction of each type in near-wall region
    wall_zone = 0.2 * ny
    margination = {}
    for name, pos in positions_by_type.items():
        if len(pos) == 0:
            margination[name] = 0.0
            continue
        y_vals = pos[:, 1]
        near_wall = np.sum((y_vals < wall_zone) | (y_vals > ny - wall_zone))
        margination[name] = float(near_wall) / len(y_vals)

    # Cell-free layer (CFL): distance from wall to nearest capsule
    all_y = np.concatenate([p[:, 1] for p in positions_by_type.values()
                            if len(p) > 0]) if positions_by_type else np.array([])

    if len(all_y) > 0:
        cfl_bottom = float(np.min(all_y))
        cfl_top = float(ny - np.max(all_y))
    else:
        cfl_bottom = float(ny) / 2
        cfl_top = float(ny) / 2

    # Lateral distributions (normalized)
    lateral_dist = {}
    for name, pos in positions_by_type.items():
        if len(pos) == 0:
            lateral_dist[name] = np.zeros(n_bins)
            continue
        hist, _ = np.histogram(pos[:, 1], bins=bin_edges, density=True)
        lateral_dist[name] = hist

    return {
        "SI": SI,
        "entropy": entropy,
        "max_entropy": max_entropy,
        "margination": margination,
        "cfl_bottom": cfl_bottom,
        "cfl_top": cfl_top,
        "lateral_dist": lateral_dist,
        "bin_centers": bin_centers,
    }


def compute_deformation_stats(
    deformation_indices: Dict[str, np.ndarray],
) -> dict:
    """Compute deformation index statistics per capsule type.

    Parameters
    ----------
    deformation_indices : dict
        Maps type name → 1D array of deformation index values.

    Returns
    -------
    dict mapping type name → dict with ``mean``, ``std``, ``min``, ``max``.
    """
    stats = {}
    for name, D_arr in deformation_indices.items():
        D_arr = np.asarray(D_arr)
        if len(D_arr) == 0:
            stats[name] = {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
        else:
            stats[name] = {
                "mean": float(np.mean(D_arr)),
                "std": float(np.std(D_arr)),
                "min": float(np.min(D_arr)),
                "max": float(np.max(D_arr)),
            }
    return stats


# ══════════════════════════════════════════════════════════════
# Velocity & Flow Profiles
# ══════════════════════════════════════════════════════════════

def compute_velocity_profile(
    ux_field: np.ndarray,
    x_range: Optional[Tuple[int, int]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute x-averaged velocity profile ux(y).

    Parameters
    ----------
    ux_field : np.ndarray
        2D velocity field (ny, nx).
    x_range : (x0, x1), optional
        Average only over this x-range. Default: full domain.

    Returns
    -------
    y_coords : np.ndarray
        Y-coordinates.
    ux_profile : np.ndarray
        Averaged ux as a function of y.
    """
    if x_range is not None:
        ux_slice = ux_field[:, x_range[0]:x_range[1]]
    else:
        ux_slice = ux_field

    ux_profile = np.mean(ux_slice, axis=1)
    y_coords = np.arange(ux_field.shape[0])
    return y_coords, ux_profile


def compute_cell_free_layer(
    capsule_y_positions: np.ndarray,
    ny: int,
    n_samples: int = 1,
) -> Tuple[float, float]:
    """Compute cell-free layer thickness at bottom and top walls.

    Parameters
    ----------
    capsule_y_positions : np.ndarray
        1D array of capsule centroid y-positions.
    ny : int
        Channel height.
    n_samples : int
        Number of time samples for averaging (if called multiple times).

    Returns
    -------
    cfl_bottom : float
        Cell-free layer at bottom wall.
    cfl_top : float
        Cell-free layer at top wall.
    """
    if len(capsule_y_positions) == 0:
        return float(ny) / 2, float(ny) / 2

    return float(np.min(capsule_y_positions)), float(ny - np.max(capsule_y_positions))


def compute_radial_distribution(
    positions: np.ndarray,
    r_max: float = 50.0,
    dr: float = 1.0,
    domain_area: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute radial distribution function g(r) for capsule centroids.

    Parameters
    ----------
    positions : np.ndarray
        (N, 2) array of capsule centroid positions.
    r_max : float
        Maximum radius for g(r).
    dr : float
        Bin width.
    domain_area : float, optional
        Domain area for normalization. If None, uses bounding box.

    Returns
    -------
    r_centers : np.ndarray
        Bin center radii.
    g_r : np.ndarray
        Radial distribution function values.
    """
    N = len(positions)
    if N < 2:
        r_centers = np.arange(dr / 2, r_max, dr)
        return r_centers, np.zeros_like(r_centers)

    n_bins = int(r_max / dr)
    r_edges = np.linspace(0, r_max, n_bins + 1)
    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
    hist = np.zeros(n_bins)

    for i in range(N):
        for j in range(i + 1, N):
            dx = positions[j, 0] - positions[i, 0]
            dy = positions[j, 1] - positions[i, 1]
            r = np.sqrt(dx * dx + dy * dy)
            if r < r_max:
                b = int(r / dr)
                if 0 <= b < n_bins:
                    hist[b] += 2  # count both (i,j) and (j,i)

    # Normalize
    if domain_area is None:
        x_range = np.ptp(positions[:, 0])
        y_range = np.ptp(positions[:, 1])
        domain_area = max(x_range * y_range, 1.0)

    rho = N / domain_area
    for b in range(n_bins):
        shell_area = np.pi * (r_edges[b + 1] ** 2 - r_edges[b] ** 2)
        if shell_area > 0 and rho > 0:
            g_r_val = hist[b] / (N * rho * shell_area)
            hist[b] = g_r_val

    return r_centers, hist


# ══════════════════════════════════════════════════════════════
# Data I/O Helpers
# ══════════════════════════════════════════════════════════════

def save_timeseries(
    filepath: str,
    data: List[dict],
    columns: List[str],
    header: str = "",
) -> None:
    """Save a list of dicts as a column-formatted text file.

    Parameters
    ----------
    filepath : str
        Output file path.
    data : list of dict
        Each dict is one row, with keys matching ``columns``.
    columns : list of str
        Column names (in order).
    header : str
        Optional header comment.
    """
    with open(filepath, "w") as f:
        if header:
            f.write(f"# {header}\n")
        f.write("# " + "  ".join(columns) + "\n")
        for row in data:
            vals = [f"{row.get(c, 0.0)}" for c in columns]
            f.write("  ".join(vals) + "\n")


def save_positions(
    filepath: str,
    positions_by_type: Dict[str, np.ndarray],
) -> None:
    """Save capsule positions grouped by type.

    Parameters
    ----------
    filepath : str
        Output file path.
    positions_by_type : dict
        Maps type name → (N, 2) array.
    """
    with open(filepath, "w") as f:
        f.write("# type  x  y\n")
        for name, pos in positions_by_type.items():
            for i in range(len(pos)):
                f.write(f"{name}  {pos[i, 0]:.4f}  {pos[i, 1]:.4f}\n")
