"""Tests for pysoftflow.analysis.mixing — Lacey, Danckwerts, contact asymmetry.

Synthetic inputs with known mixing behaviour:
  - Perfectly random binomial → Lacey M ≈ 1, Danckwerts I_S ≈ 0.
  - Stripe-segregated (all A in left half, all B in right) →
    M ≈ 0, I_S ≈ 1.
  - Hex lattice of one type → contact asymmetry = +1.
  - 50/50 alternating → contact asymmetry ≈ 0.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest


HERE = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE / "python"))

from pysoftflow.analysis import SimulationSnapshot   # noqa: E402
from pysoftflow.analysis.mixing import (              # noqa: E402
    lacey_index,
    danckwerts_intensity,
    contact_asymmetry,
)


def _binomial_random(n_a: int, n_b: int, *, seed: int, box):
    """Random uniform positions for n_a + n_b particles, types shuffled."""
    rng = np.random.default_rng(seed)
    n = n_a + n_b
    positions = rng.uniform(0.0, 1.0, size=(n, 2))
    positions[:, 0] *= box[0]
    positions[:, 1] *= box[1]
    types = np.zeros(n, dtype=np.int64)
    types[n_a:] = 1
    rng.shuffle(types)
    return SimulationSnapshot.from_arrays(
        positions=positions, radii=np.ones(n) * 0.5,
        types=types, box=box, periodic_x=False)


def _stripe_segregated(n_each: int, *, box):
    """A-particles in left half, B-particles in right half."""
    Lx, Ly = box
    rng = np.random.default_rng(42)
    pos_a = np.column_stack([
        rng.uniform(0.0, 0.5 * Lx, size=n_each),
        rng.uniform(0.0, Ly,       size=n_each),
    ])
    pos_b = np.column_stack([
        rng.uniform(0.5 * Lx, Lx, size=n_each),
        rng.uniform(0.0, Ly,      size=n_each),
    ])
    positions = np.vstack([pos_a, pos_b])
    types     = np.concatenate([np.zeros(n_each), np.ones(n_each)]).astype(np.int64)
    return SimulationSnapshot.from_arrays(
        positions=positions, radii=np.ones(2 * n_each) * 0.5,
        types=types, box=box, periodic_x=False)


# ── Lacey / Danckwerts ─────────────────────────────────────────────


def test_random_mix_gives_high_lacey_low_danckwerts():
    snap = _binomial_random(500, 500, seed=11, box=(100.0, 100.0))
    res = lacey_index(snap, type_a=0, type_b=1, axis="y", n_bins=10)
    assert 0.7 < res.M_lacey <= 1.0 + 1e-9
    assert res.I_danckwerts < 0.10


def test_stripe_segregated_gives_low_lacey_high_danckwerts_along_x():
    snap = _stripe_segregated(500, box=(100.0, 100.0))
    res = lacey_index(snap, type_a=0, type_b=1, axis="x", n_bins=10)
    assert res.M_lacey < 0.10
    assert res.I_danckwerts > 0.90


def test_stripe_segregation_invisible_along_orthogonal_axis():
    # Same setup but binning along y: the stripe is uniform in y, so
    # Lacey should report well-mixed.
    snap = _stripe_segregated(500, box=(100.0, 100.0))
    res = lacey_index(snap, type_a=0, type_b=1, axis="y", n_bins=10)
    assert res.M_lacey > 0.5


def test_global_fraction_recovered():
    snap = _binomial_random(300, 700, seed=3, box=(50.0, 50.0))
    res = lacey_index(snap, type_a=0, type_b=1, axis="y", n_bins=10)
    assert res.p_global == pytest.approx(0.3, abs=1e-9)
    assert res.n_total == 1000


def test_danckwerts_wrapper_matches_lacey_field():
    snap = _binomial_random(400, 600, seed=7, box=(80.0, 80.0))
    res = lacey_index(snap, type_a=0, type_b=1, axis="x", n_bins=8)
    assert danckwerts_intensity(snap, type_a=0, type_b=1,
                                 axis="x", n_bins=8) == res.I_danckwerts


def test_input_validation():
    snap = _binomial_random(10, 10, seed=0, box=(10.0, 10.0))
    with pytest.raises(ValueError, match="n_bins"):
        lacey_index(snap, n_bins=1)
    with pytest.raises(ValueError, match="axis"):
        lacey_index(snap, axis="z")


# ── Contact asymmetry ──────────────────────────────────────────────


def test_contact_asymmetry_all_one_type_returns_plus_one():
    # Two type-A capsules touching, no type-B → asymmetry = +1.
    snap = SimulationSnapshot.from_arrays(
        positions=np.array([[0.0, 0.0], [1.5, 0.0]], dtype=float),
        radii=np.array([1.0, 1.0]),
        types=np.array([0, 0]),
        box=(10.0, 10.0),
    )
    assert contact_asymmetry(snap, type_a=0, type_b=1,
                              contact_cutoff=0.0) == pytest.approx(1.0)


def test_contact_asymmetry_all_other_type_returns_minus_one():
    snap = SimulationSnapshot.from_arrays(
        positions=np.array([[0.0, 0.0], [1.5, 0.0]], dtype=float),
        radii=np.array([1.0, 1.0]),
        types=np.array([1, 1]),
        box=(10.0, 10.0),
    )
    assert contact_asymmetry(snap, type_a=0, type_b=1,
                              contact_cutoff=0.0) == pytest.approx(-1.0)


def test_contact_asymmetry_balanced_pairs_returns_zero():
    # One AA pair + one BB pair → asymmetry = 0.
    snap = SimulationSnapshot.from_arrays(
        positions=np.array([[0, 0], [1.5, 0],
                            [10, 0], [11.5, 0]], dtype=float),
        radii=np.array([1.0, 1.0, 1.0, 1.0]),
        types=np.array([0, 0, 1, 1]),
        box=(50.0, 50.0),
    )
    assert contact_asymmetry(snap, type_a=0, type_b=1,
                              contact_cutoff=0.0) == pytest.approx(0.0)


def test_contact_asymmetry_no_pairs_returns_zero():
    # Particles too far apart to touch → 0/0 → return 0.
    snap = SimulationSnapshot.from_arrays(
        positions=np.array([[0, 0], [50, 0]], dtype=float),
        radii=np.array([1.0, 1.0]),
        types=np.array([0, 1]),
        box=(100.0, 100.0),
    )
    assert contact_asymmetry(snap, contact_cutoff=0.0) == 0.0


def test_contact_asymmetry_periodic_x_minimum_image():
    # Two type-A particles "close" only via periodic wrap.
    snap = SimulationSnapshot.from_arrays(
        positions=np.array([[1.0, 5.0], [9.0, 5.0]], dtype=float),
        radii=np.array([1.0, 1.0]),
        types=np.array([0, 0]),
        box=(10.0, 10.0),
        periodic_x=True,
    )
    # Distance under wrap = 2, sum-of-radii = 2 → contact (≤).
    assert contact_asymmetry(snap, contact_cutoff=0.5) == pytest.approx(1.0)
