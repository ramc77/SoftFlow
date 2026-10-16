"""Tests for pysoftflow.tumor_growth.daughters.DaughterPlacer."""

from __future__ import annotations

import math
import pathlib
import sys

import numpy as np
import pytest


HERE = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE / "python"))

from pysoftflow.tumor_growth import DaughterPlacement, DaughterPlacer  # noqa: E402


def _placer(**kw):
    return DaughterPlacer(**kw)


# ── Free-space placement always succeeds ──────────────────────────


def test_isolated_parent_in_free_space_gets_a_daughter():
    p = _placer()
    rng = np.random.default_rng(7)
    result = p.propose(
        parent_pos=(50.0, 30.0), parent_radius=2.0,
        existing_centers=np.array([[50.0, 30.0]], dtype=float),
        existing_radii=np.array([2.0], dtype=float),
        wall_y_bottom=0.5, wall_y_top=59.5,
        rng=rng,
    )
    assert result is not None
    # Daughter radius matches default factor (1.0).
    assert result.radius == pytest.approx(2.0)
    # Daughter centre lies on the ring of expected radius (which
    # includes min_gap to clear the parent's exclusion envelope).
    dx = result.center[0] - 50.0
    dy = result.center[1] - 30.0
    expected_ring = 1.05 * (2.0 + 2.0 + 0.5)   # default min_gap=0.5
    assert math.hypot(dx, dy) == pytest.approx(expected_ring, rel=1e-9)


def test_smaller_daughter_when_factor_below_one():
    p = _placer(daughter_radius_factor=0.7937)   # 2-D volume halved
    rng = np.random.default_rng(0)
    result = p.propose(
        parent_pos=(50.0, 30.0), parent_radius=2.0,
        existing_centers=np.array([[50.0, 30.0]]),
        existing_radii=np.array([2.0]),
        wall_y_bottom=0.5, wall_y_top=59.5,
        rng=rng,
    )
    assert result is not None
    assert result.radius == pytest.approx(2.0 * 0.7937)


# ── Wall envelope ─────────────────────────────────────────────────


def test_parent_pinned_against_bottom_wall_skips_below_wall_angles():
    p = _placer(min_gap=0.5)
    rng = np.random.default_rng(0)
    # Parent close to bottom wall; daughter ring radius is 4.2 →
    # angles pointing down would push daughter into wall envelope.
    # The placer should sample a valid angle eventually.
    result = p.propose(
        parent_pos=(50.0, 5.0), parent_radius=2.0,
        existing_centers=np.array([[50.0, 5.0]]),
        existing_radii=np.array([2.0]),
        wall_y_bottom=0.5, wall_y_top=59.5,
        rng=rng,
    )
    assert result is not None
    cy = result.center[1]
    daughter_r = result.radius
    # Daughter clears the bottom wall by min_gap.
    assert cy - daughter_r >= 0.5 + 0.5 - 1e-9


def test_channel_narrower_than_daughter_envelope_returns_none():
    """Channel height = 5 < 2·(daughter_r + min_gap) = 6.

    No daughter centre can clear *both* walls with min_gap=1 and
    daughter_r=2: y must be in [0.5+1+2, 5.5-1-2] = [3.5, 2.5] which
    is empty. The placer must report None regardless of angle.
    """
    p = _placer(min_gap=1.0, max_attempts=24)
    rng = np.random.default_rng(0)
    result = p.propose(
        parent_pos=(50.0, 3.0), parent_radius=2.0,
        existing_centers=np.array([[50.0, 3.0]]),
        existing_radii=np.array([2.0]),
        wall_y_bottom=0.5, wall_y_top=5.5,
        rng=rng,
    )
    assert result is None


# ── Existing-capsule overlap ──────────────────────────────────────


def test_dense_neighbourhood_returns_none():
    """Surround the parent with a tight ring of capsules at exactly
    the expected daughter ring radius. Every angle is blocked."""
    p = _placer(max_attempts=24, min_gap=0.5)
    rng = np.random.default_rng(0)

    parent_pos = np.array([50.0, 30.0])
    parent_r = 2.0
    # New formula includes min_gap.
    ring_r = 1.05 * (parent_r + parent_r + 0.5)

    # Place 24 blockers densely on the daughter ring.
    angles = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    blockers = parent_pos + ring_r * np.column_stack([np.cos(angles),
                                                       np.sin(angles)])
    centers = np.vstack([[parent_pos], blockers])
    radii   = np.array([parent_r] + [parent_r] * 24)

    result = p.propose(
        parent_pos=tuple(parent_pos), parent_radius=parent_r,
        existing_centers=centers, existing_radii=radii,
        wall_y_bottom=0.5, wall_y_top=59.5,
        rng=rng,
    )
    assert result is None


def test_existing_capsules_far_away_do_not_block():
    p = _placer()
    rng = np.random.default_rng(7)
    centers = np.array([[50.0, 30.0],   # parent
                         [200.0, 30.0]], # far blocker, no effect
                         dtype=float)
    radii = np.array([2.0, 2.0])
    result = p.propose(
        parent_pos=(50.0, 30.0), parent_radius=2.0,
        existing_centers=centers, existing_radii=radii,
        wall_y_bottom=0.5, wall_y_top=59.5,
        rng=rng,
    )
    assert result is not None


# ── Periodic-x wrap ───────────────────────────────────────────────


def test_periodic_x_wrap_brings_daughter_into_box():
    """Parent near x=Lx; right-side angles push daughter past Lx,
    which gets wrapped back to small x."""
    p = _placer()
    rng = np.random.default_rng(42)
    Lx = 50.0
    # Force angle 0 (rightward) by patching the RNG's first uniform.
    # Simpler: just try many seeds and check at least one gives wrap.
    found_wrap = False
    for seed in range(40):
        rng = np.random.default_rng(seed)
        result = p.propose(
            parent_pos=(48.0, 30.0), parent_radius=2.0,
            existing_centers=np.array([[48.0, 30.0]]),
            existing_radii=np.array([2.0]),
            wall_y_bottom=0.5, wall_y_top=59.5,
            rng=rng,
            Lx=Lx, periodic_x=True,
        )
        if result is not None and result.center[0] < 5.0:
            found_wrap = True
            break
    assert found_wrap


# ── Obstacles ─────────────────────────────────────────────────────


def test_obstacle_blocks_overlapping_angles():
    """A signed-distance function that reports `r ≤ 0` near (60, 30)
    should block any daughter that lands near (60, 30)."""
    p = _placer(max_attempts=24)
    rng = np.random.default_rng(7)

    def sd(x, y):
        # Disk obstacle of radius 8 at (60, 30); +ve outside.
        return np.hypot(x - 60.0, y - 30.0) - 8.0

    # Parent close enough that some angles point into the obstacle.
    result = p.propose(
        parent_pos=(50.0, 30.0), parent_radius=2.0,
        existing_centers=np.array([[50.0, 30.0]]),
        existing_radii=np.array([2.0]),
        wall_y_bottom=0.5, wall_y_top=59.5,
        rng=rng,
        obstacle_signed_distances=[sd],
    )
    if result is not None:
        # Daughter must be at distance > 8 + r + min_gap from
        # obstacle centre.
        d = math.hypot(result.center[0] - 60.0,
                        result.center[1] - 30.0)
        assert d > 8.0 + result.radius + 0.5


# ── Reproducibility ───────────────────────────────────────────────


def test_seeded_rng_reproduces_bit_exact():
    p = _placer()
    a = np.random.default_rng(99)
    b = np.random.default_rng(99)
    centers = np.array([[50.0, 30.0]])
    radii   = np.array([2.0])
    ra = p.propose((50.0, 30.0), 2.0, centers, radii,
                    0.5, 59.5, rng=a)
    rb = p.propose((50.0, 30.0), 2.0, centers, radii,
                    0.5, 59.5, rng=b)
    assert ra is not None and rb is not None
    assert ra.center == rb.center
    assert ra.radius == rb.radius
    assert ra.angle == rb.angle


# ── Input validation ──────────────────────────────────────────────


def test_validates_inputs():
    with pytest.raises(ValueError, match="ring_radius_factor"):
        DaughterPlacer(ring_radius_factor=0.0)
    with pytest.raises(ValueError, match="max_attempts"):
        DaughterPlacer(max_attempts=0)
    with pytest.raises(ValueError, match="min_gap"):
        DaughterPlacer(min_gap=-0.1)
    with pytest.raises(ValueError, match="daughter_radius_factor"):
        DaughterPlacer(daughter_radius_factor=0.0)
