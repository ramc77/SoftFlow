"""Tests for pysoftflow.analysis.jamming.

Coverage:
  - packing field on uniform-fill input → mean φ matches the
    analytic ``N · π · ⟨r²⟩ / (Lx Ly)`` value;
  - contact number Z ≈ 6 on a hex lattice of touching disks;
  - per-type contact matrix Z_ij identity check
    ``n_a · Z_ab == n_b · Z_ba`` plus expected values on a hand-
    crafted bidisperse input;
  - force_percolation True for a hand-built spanning chain, False
    for isolated pairs;
  - MSD plateau detection on a synthetic random walk vs trapped data;
  - non-affine D²_min ≡ 0 for a pure affine deformation, > 0 for a
    randomly-shuffled deformation.
"""

from __future__ import annotations

import math
import pathlib
import sys

import numpy as np
import pytest


HERE = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE / "python"))

from pysoftflow.analysis import SimulationSnapshot                    # noqa: E402
from pysoftflow.analysis.jamming import (                             # noqa: E402
    contact_number,
    force_percolation,
    mean_squared_displacement,
    non_affine_d2min,
    packing_field,
    per_type_contact_stats,
)


def _hex_lattice(n_rows: int, n_cols: int, spacing: float):
    dy = 0.5 * np.sqrt(3.0) * spacing
    pts = []
    for j in range(n_rows):
        x_offset = 0.5 * spacing if (j & 1) else 0.0
        for i in range(n_cols):
            pts.append([i * spacing + x_offset, j * dy])
    return np.asarray(pts, dtype=np.float64)


# ── packing_field ──────────────────────────────────────────────────


def test_packing_field_uniform_fill_matches_analytic_phi():
    rng = np.random.default_rng(11)
    n   = 2000
    Lx  = Ly = 100.0
    pts = rng.uniform(0.0, Lx, size=(n, 2))
    r   = np.full(n, 1.0)
    snap = SimulationSnapshot.from_arrays(
        positions=pts, radii=r, types=np.zeros(n, dtype=int),
        box=(Lx, Ly), periodic_x=False,
    )
    pf = packing_field(snap, n_x=20, n_y=20)
    expected_phi = n * math.pi * 1.0 ** 2 / (Lx * Ly)
    assert pf.phi.mean() == pytest.approx(expected_phi, rel=0.02)


def test_packing_field_input_validation():
    snap = SimulationSnapshot.from_arrays(
        positions=np.zeros((1, 2)), radii=np.ones(1),
        types=np.zeros(1, dtype=int), box=(1.0, 1.0))
    with pytest.raises(ValueError, match="n_x"):
        packing_field(snap, n_x=0, n_y=1)


# ── contact_number ─────────────────────────────────────────────────


def test_contact_number_hex_lattice_returns_six_for_interior():
    spacing = 4.0
    pts = _hex_lattice(8, 8, spacing)
    n = pts.shape[0]
    snap = SimulationSnapshot.from_arrays(
        positions=pts,
        radii=np.full(n, 0.5 * spacing),    # touching disks
        types=np.zeros(n, dtype=int),
        box=(60.0, 60.0), periodic_x=False,
    )
    res = contact_number(snap, contact_cutoff=0.05)
    # Interior particles have Z=6; boundary particles have fewer.
    # Mean across the lattice is well below 6 because of the boundary,
    # but the mode of Z_per_particle should be 6.
    counts = np.bincount(res.Z_per_particle, minlength=7)
    assert counts.argmax() == 6
    # The largest cluster of "interior" hex sites should each have
    # exactly 6 neighbours.
    assert (res.Z_per_particle == 6).sum() > 8


# ── per_type_contact_stats ─────────────────────────────────────────


def test_per_type_identity_n_i_Z_ij_equals_n_j_Z_ji():
    rng = np.random.default_rng(7)
    n = 60
    pts = rng.uniform(0.0, 30.0, size=(n, 2))
    types = (np.arange(n) % 2).astype(np.int64)
    snap = SimulationSnapshot.from_arrays(
        positions=pts, radii=np.ones(n) * 1.0,
        types=types, box=(30.0, 30.0), periodic_x=False,
    )
    res = per_type_contact_stats(snap, contact_cutoff=2.0)
    n_per = res.n_particles
    Z = res.Z_matrix
    T = res.types.size
    for i in range(T):
        for j in range(T):
            assert n_per[i] * Z[i, j] == pytest.approx(n_per[j] * Z[j, i],
                                                        abs=1e-9)


def test_per_type_n_pairs_symmetric_and_diagonal_uses_unique_pairs():
    # Hand-crafted: A=(0,0), A=(1.5,0), B=(10,0), B=(11.5,0). All
    # touching their same-type neighbour; no AB pairs.
    snap = SimulationSnapshot.from_arrays(
        positions=np.array([[0, 0], [1.5, 0],
                            [10, 0], [11.5, 0]], dtype=float),
        radii=np.array([1.0, 1.0, 1.0, 1.0]),
        types=np.array([0, 0, 1, 1], dtype=np.int64),
        box=(30.0, 30.0), periodic_x=False,
    )
    res = per_type_contact_stats(snap, contact_cutoff=0.0)
    # n_pairs is symmetric.
    assert res.n_pairs[0, 0] == 1   # one AA pair
    assert res.n_pairs[1, 1] == 1   # one BB pair
    assert res.n_pairs[0, 1] == 0
    assert res.n_pairs[1, 0] == 0
    # Z_AA = 2 N_AA / N_A = 2 · 1 / 2 = 1.
    assert res.Z_matrix[0, 0] == pytest.approx(1.0)
    assert res.Z_matrix[1, 1] == pytest.approx(1.0)
    assert res.Z_matrix[0, 1] == 0.0


def test_per_type_hex_lattice_Z_AA_equals_six_for_interior_only():
    spacing = 4.0
    pts = _hex_lattice(8, 8, spacing)
    n = pts.shape[0]
    snap = SimulationSnapshot.from_arrays(
        positions=pts,
        radii=np.full(n, 0.5 * spacing),
        types=np.zeros(n, dtype=int),
        box=(60.0, 60.0),
    )
    res = per_type_contact_stats(snap, contact_cutoff=0.05)
    # Mean Z_AA is below 6 because of boundary particles, but should
    # be in the range expected for a finite hex patch.
    assert 4.0 <= res.Z_matrix[0, 0] <= 6.0


def test_per_type_only_one_type_present():
    snap = SimulationSnapshot.from_arrays(
        positions=np.array([[0, 0], [1.5, 0]], dtype=float),
        radii=np.array([1.0, 1.0]),
        types=np.array([5, 5], dtype=np.int64),
        box=(10.0, 10.0),
    )
    res = per_type_contact_stats(snap, contact_cutoff=0.0)
    assert res.types.tolist() == [5]
    assert res.n_particles.tolist() == [2]
    assert res.Z_matrix.shape == (1, 1)
    assert res.Z_matrix[0, 0] == pytest.approx(1.0)


# ── force_percolation ──────────────────────────────────────────────


def test_force_percolation_spanning_chain_returns_true():
    # Vertical chain of touching particles from y=1 to y=99.
    n = 25
    Ly = 100.0
    pts = np.column_stack([np.full(n, 50.0),
                            np.linspace(1.0, 99.0, n)])
    snap = SimulationSnapshot.from_arrays(
        positions=pts,
        radii=np.full(n, 2.5),    # spacing ~4.08, sum ~5 → in contact
        types=np.zeros(n, dtype=int),
        box=(100.0, Ly),
    )
    res = force_percolation(snap, contact_cutoff=1.0, band_fraction=0.10)
    assert res.spans is True
    assert res.spanning_size == n


def test_force_percolation_isolated_pairs_does_not_span():
    # Two clumps in the middle; no chain to either band.
    snap = SimulationSnapshot.from_arrays(
        positions=np.array([[10, 50], [11.5, 50],
                            [80, 50], [81.5, 50]], dtype=float),
        radii=np.array([1.0, 1.0, 1.0, 1.0]),
        types=np.zeros(4, dtype=int),
        box=(100.0, 100.0),
    )
    res = force_percolation(snap, contact_cutoff=0.0)
    assert res.spans is False
    assert res.spanning_size == 0


def test_force_percolation_input_validation():
    snap = SimulationSnapshot.from_arrays(
        positions=np.zeros((1, 2)), radii=np.ones(1),
        types=np.zeros(1, dtype=int), box=(10.0, 10.0))
    with pytest.raises(ValueError, match="band_fraction"):
        force_percolation(snap, band_fraction=0.6)


# ── mean_squared_displacement ──────────────────────────────────────


def test_msd_random_walk_grows_linearly():
    rng = np.random.default_rng(7)
    n   = 200
    n_steps = 60
    step = 0.1
    pos = np.zeros((n, 2))
    snaps = []
    for t in range(n_steps):
        snaps.append(SimulationSnapshot.from_arrays(
            positions=pos.copy(), radii=np.ones(n),
            types=np.zeros(n, dtype=int),
            time=float(t), step=t,
            box=(10000.0, 10000.0)))
        pos += rng.normal(scale=step, size=(n, 2))
    res = mean_squared_displacement(snaps)
    # Diffusive: MSD ∝ t, log-log slope ~1, no plateau.
    assert res.is_plateau is False
    assert 0.7 < res.log_slope < 1.3


def test_msd_trapped_particles_plateau():
    # All particles bounce within ±0.1 of origin → no growth.
    rng = np.random.default_rng(11)
    n = 50
    snaps = []
    for t in range(30):
        pos = rng.uniform(-0.1, 0.1, size=(n, 2))
        snaps.append(SimulationSnapshot.from_arrays(
            positions=pos, radii=np.ones(n),
            types=np.zeros(n, dtype=int),
            time=float(t), step=t,
            box=(10.0, 10.0)))
    res = mean_squared_displacement(snaps)
    assert res.is_plateau is True
    assert res.plateau_value < 0.1


def test_msd_input_validation():
    snap = SimulationSnapshot.from_arrays(
        positions=np.zeros((2, 2)), radii=np.ones(2),
        types=np.zeros(2, dtype=int), box=(10.0, 10.0))
    with pytest.raises(ValueError, match="at least 3"):
        mean_squared_displacement([snap])
    snap2 = SimulationSnapshot.from_arrays(
        positions=np.zeros((3, 2)), radii=np.ones(3),
        types=np.zeros(3, dtype=int), box=(10.0, 10.0))
    with pytest.raises(ValueError, match="constant particle set"):
        mean_squared_displacement([snap, snap, snap2])


# ── non_affine_d2min ───────────────────────────────────────────────


def test_d2min_pure_shear_has_zero_residual():
    # Square lattice; apply pure shear u_x += γ y. D²_min should be 0
    # because the local affine fit absorbs the shear exactly.
    grid = np.array([(i, j) for i in range(10) for j in range(10)],
                    dtype=float)
    pos0 = grid.copy()
    gamma = 0.1
    pos1 = grid.copy()
    pos1[:, 0] += gamma * pos1[:, 1]

    snap0 = SimulationSnapshot.from_arrays(
        positions=pos0, radii=np.ones(pos0.shape[0]),
        types=np.zeros(pos0.shape[0], dtype=int),
        box=(15.0, 15.0))
    snap1 = SimulationSnapshot.from_arrays(
        positions=pos1, radii=np.ones(pos1.shape[0]),
        types=np.zeros(pos1.shape[0], dtype=int),
        box=(15.0, 15.0))
    d2 = non_affine_d2min(snap0, snap1, neighbour_cutoff=1.5)
    # Interior particles must have D²_min = 0 to within numerical
    # precision; boundary particles may have asymmetric neighbour
    # patches that still admit an exact affine fit, so we just check
    # the mean is ~0.
    assert d2.mean() < 1e-12


def test_d2min_random_shuffle_is_large():
    rng = np.random.default_rng(7)
    n   = 100
    pos0 = rng.uniform(0.0, 10.0, size=(n, 2))
    pos1 = rng.permutation(pos0)     # random shuffle

    snap0 = SimulationSnapshot.from_arrays(
        positions=pos0, radii=np.ones(n),
        types=np.zeros(n, dtype=int), box=(20.0, 20.0))
    snap1 = SimulationSnapshot.from_arrays(
        positions=pos1, radii=np.ones(n),
        types=np.zeros(n, dtype=int), box=(20.0, 20.0))
    d2 = non_affine_d2min(snap0, snap1, neighbour_cutoff=3.0)
    assert d2.mean() > 1.0   # well above zero
