"""Tests for pysoftflow.analysis.patterns — lane order, HK, persistence."""

from __future__ import annotations

import math
import pathlib
import sys

import numpy as np
import pytest


HERE = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE / "python"))

from pysoftflow.analysis import SimulationSnapshot                    # noqa: E402
from pysoftflow.analysis.patterns import (                            # noqa: E402
    cluster_persistence,
    hoshen_kopelman,
    lane_order,
)


# ── Lane order ─────────────────────────────────────────────────────


def test_lane_order_perfect_alignment_returns_one():
    n = 100
    snap = SimulationSnapshot.from_arrays(
        positions=np.zeros((n, 2)),
        radii=np.ones(n),
        types=np.zeros(n, dtype=int),
        velocities=np.column_stack([np.ones(n), np.zeros(n)]),
        box=(10.0, 10.0),
    )
    assert lane_order(snap, axis="x") == pytest.approx(1.0)


def test_lane_order_perpendicular_returns_minus_one():
    n = 100
    snap = SimulationSnapshot.from_arrays(
        positions=np.zeros((n, 2)),
        radii=np.ones(n),
        types=np.zeros(n, dtype=int),
        velocities=np.column_stack([np.zeros(n), np.ones(n)]),
        box=(10.0, 10.0),
    )
    assert lane_order(snap, axis="x") == pytest.approx(-1.0)


def test_lane_order_isotropic_returns_zero():
    rng = np.random.default_rng(11)
    n = 5000
    theta = rng.uniform(0.0, 2.0 * math.pi, size=n)
    velocities = np.column_stack([np.cos(theta), np.sin(theta)])
    snap = SimulationSnapshot.from_arrays(
        positions=np.zeros((n, 2)),
        radii=np.ones(n),
        types=np.zeros(n, dtype=int),
        velocities=velocities,
        box=(10.0, 10.0),
    )
    assert abs(lane_order(snap, axis="x")) < 0.05


def test_lane_order_zero_speed_skipped():
    snap = SimulationSnapshot.from_arrays(
        positions=np.zeros((4, 2)),
        radii=np.ones(4),
        types=np.zeros(4, dtype=int),
        velocities=np.zeros((4, 2)),  # all zero
        box=(10.0, 10.0),
    )
    assert math.isnan(lane_order(snap, axis="x"))


def test_lane_order_type_filter():
    n = 4
    snap = SimulationSnapshot.from_arrays(
        positions=np.zeros((n, 2)),
        radii=np.ones(n),
        types=np.array([0, 0, 1, 1]),
        velocities=np.array([
            [1, 0], [1, 0],   # type-0 along x
            [0, 1], [0, 1],   # type-1 along y
        ], dtype=float),
        box=(10.0, 10.0),
    )
    assert lane_order(snap, axis="x", type_filter=0) == pytest.approx(1.0)
    assert lane_order(snap, axis="x", type_filter=1) == pytest.approx(-1.0)


# ── Hoshen-Kopelman cluster labelling ──────────────────────────────


def test_hk_isolated_particles_each_form_their_own_cluster():
    snap = SimulationSnapshot.from_arrays(
        positions=np.array([[0, 0], [10, 0], [20, 0], [30, 0]], dtype=float),
        radii=np.array([1.0, 1.0, 1.0, 1.0]),
        types=np.zeros(4, dtype=int),
        box=(50.0, 50.0),
    )
    out = hoshen_kopelman(snap, contact_cutoff=0.5)
    assert out.n_clusters == 4
    assert (out.sizes == np.array([1, 1, 1, 1])).all()
    assert out.largest_size == 1


def test_hk_chain_of_three_one_cluster_size_three():
    # Three particles in contact: 0—1—2 with 0 and 2 not directly
    # touching, but transitively connected through 1.
    snap = SimulationSnapshot.from_arrays(
        positions=np.array([[0, 0], [2.4, 0], [4.8, 0]], dtype=float),
        radii=np.array([1.2, 1.2, 1.2]),
        types=np.zeros(3, dtype=int),
        box=(50.0, 50.0),
    )
    out = hoshen_kopelman(snap, contact_cutoff=0.1)
    assert out.n_clusters == 1
    assert out.largest_size == 3
    # Same cluster label for all three.
    assert (out.labels == out.labels[0]).all()


def test_hk_two_clusters_returns_correct_size_distribution():
    snap = SimulationSnapshot.from_arrays(
        positions=np.array([
            [0, 0], [2.0, 0],            # cluster A (size 2)
            [50, 50], [52.0, 50], [52.0, 52.0],   # cluster B (size 3)
        ], dtype=float),
        radii=np.array([1.0, 1.0, 1.0, 1.0, 1.0]),
        types=np.zeros(5, dtype=int),
        box=(100.0, 100.0),
    )
    out = hoshen_kopelman(snap, contact_cutoff=0.5)
    assert out.n_clusters == 2
    assert (out.sizes == np.array([3, 2])).all()
    # Labels are sorted by cluster size descending.
    assert out.labels[0] == 1   # A is the smaller cluster → label 1
    assert out.labels[2] == 0   # B is the larger cluster  → label 0


def test_hk_uses_bonds_when_available():
    # Two particles far apart geometrically but bonded by adhesion →
    # one cluster.
    snap = SimulationSnapshot.from_arrays(
        positions=np.array([[0, 0], [100, 100]], dtype=float),
        radii=np.array([1.0, 1.0]),
        types=np.zeros(2, dtype=int),
        box=(200.0, 200.0),
        bonds=np.array([[0, 1]], dtype=np.int64),
    )
    out = hoshen_kopelman(snap, use_bonds=True)
    assert out.n_clusters == 1


# ── Cluster persistence ────────────────────────────────────────────


def test_persistence_identical_labels_returns_one():
    labs = np.array([0, 0, 1, 1, 2])
    assert cluster_persistence(labs, labs) == 1.0


def test_persistence_random_relabel_returns_low_value():
    rng = np.random.default_rng(42)
    n = 60
    labs_a = rng.integers(0, 4, size=n)
    labs_b = rng.integers(0, 4, size=n)
    # Two random labellings should have very low Jaccard similarity.
    assert cluster_persistence(labs_a, labs_b) < 0.40


def test_persistence_no_pairs_returns_nan():
    labs = np.arange(5)         # every particle its own cluster → no pairs
    assert math.isnan(cluster_persistence(labs, labs))


def test_persistence_partial_relabel():
    # Three pairs in (0,0)+(1,1) labelling; reshuffle one particle out.
    labs_a = np.array([0, 0, 0, 1, 1, 1])
    labs_b = np.array([0, 0, 2, 1, 1, 1])
    # Same-cluster pairs in A: {(0,1),(0,2),(1,2),(3,4),(3,5),(4,5)} = 6
    # Same-cluster pairs in B: {(0,1),(3,4),(3,5),(4,5)} = 4
    # Intersection: {(0,1),(3,4),(3,5),(4,5)} = 4; union = 6.
    assert cluster_persistence(labs_a, labs_b) == pytest.approx(4 / 6)
