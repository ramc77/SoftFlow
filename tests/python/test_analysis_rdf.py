"""Tests for the species-resolved radial distribution function."""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest


HERE = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE / "python"))

from pysoftflow.analysis import SimulationSnapshot           # noqa: E402
from pysoftflow.analysis.rdf import radial_distribution      # noqa: E402


def _hex_lattice(n_rows: int, n_cols: int, spacing: float):
    """Generate centres of a hex-packed lattice."""
    dy = 0.5 * np.sqrt(3.0) * spacing
    pts = []
    for j in range(n_rows):
        x_offset = 0.5 * spacing if (j & 1) else 0.0
        for i in range(n_cols):
            pts.append([i * spacing + x_offset, j * dy])
    return np.asarray(pts, dtype=np.float64)


def test_hex_lattice_has_first_peak_at_lattice_spacing():
    spacing = 4.0
    pts = _hex_lattice(20, 20, spacing)
    n   = pts.shape[0]
    snap = SimulationSnapshot.from_arrays(
        positions=pts, radii=np.ones(n) * 0.5,
        types=np.zeros(n, dtype=int),
        box=(80.0, 80.0), periodic_x=False,
    )
    res = radial_distribution(snap, type_a=0, type_b=0,
                               r_max=2.5 * spacing, n_bins=80)

    # The first peak should sit close to the lattice spacing.
    peak_bin = int(np.argmax(res.g_r))
    peak_r   = res.r_centres[peak_bin]
    assert abs(peak_r - spacing) < 0.10 * spacing, \
        f"expected first peak near r={spacing}, got r={peak_r}"


def test_random_uniform_recovers_g_of_one():
    rng = np.random.default_rng(7)
    n   = 4000
    Lx  = Ly = 50.0
    pts = rng.uniform(0.0, Lx, size=(n, 2))
    snap = SimulationSnapshot.from_arrays(
        positions=pts, radii=np.ones(n) * 0.3,
        types=np.zeros(n, dtype=int),
        box=(Lx, Ly), periodic_x=True,
    )
    res = radial_distribution(snap, type_a=0, type_b=0,
                               r_max=10.0, n_bins=20)

    # Drop the first bin (finite-size depletion) and the last (boundary
    # truncation in the non-wrapped y direction). The remaining bulk
    # should average to ≈ 1 within a few percent.
    bulk = res.g_r[2:-2]
    mean = float(bulk.mean())
    assert 0.85 < mean < 1.15


def test_cross_species_zeroes_when_no_pairs():
    snap = SimulationSnapshot.from_arrays(
        positions=np.array([[0, 0], [10, 10]], dtype=float),
        radii=np.array([0.5, 0.5]),
        types=np.array([0, 0]),       # both type 0
        box=(20.0, 20.0),
    )
    res = radial_distribution(snap, type_a=0, type_b=1,
                               r_max=10.0, n_bins=10)
    # No type-B particles at all → g_r is identically zero.
    assert (res.g_r == 0.0).all()
    assert res.n_a == 2 and res.n_b == 0


def test_periodic_x_wrap_finds_neighbour_across_seam():
    snap = SimulationSnapshot.from_arrays(
        positions=np.array([[1.0, 5.0], [9.0, 5.0]], dtype=float),
        radii=np.array([0.5, 0.5]),
        types=np.array([0, 0]),
        box=(10.0, 10.0),
        periodic_x=True,
    )
    res = radial_distribution(snap, type_a=0, type_b=0,
                               r_max=4.0, n_bins=20)
    # Distance under wrap = 2 → there should be a peak near r=2.
    bin_at_2 = int(np.searchsorted(res.r_centres, 2.0))
    assert res.g_r[bin_at_2 - 1: bin_at_2 + 2].sum() > 0


def test_input_validation():
    snap = SimulationSnapshot.from_arrays(
        positions=np.zeros((4, 2)), radii=np.ones(4),
        types=np.zeros(4, dtype=int), box=(10.0, 10.0))
    with pytest.raises(ValueError, match="n_bins"):
        radial_distribution(snap, n_bins=1)
    with pytest.raises(ValueError, match="r_max"):
        radial_distribution(snap, r_max=-1.0)
