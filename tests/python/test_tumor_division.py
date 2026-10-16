"""Tests for pysoftflow.tumor_growth.division."""

from __future__ import annotations

import math
import pathlib
import sys

import numpy as np
import pytest


HERE = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE / "python"))

from pysoftflow.tumor_growth import (                  # noqa: E402
    ParentState, StressNutrientDivision,
)


def _parent(cid=0, n=0):
    return ParentState(capsule_id=cid, n_divisions=n)


# ── Threshold gates ───────────────────────────────────────────────


def test_high_shear_blocks_division_even_when_nutrient_high():
    k = StressNutrientDivision(k_div=10.0, stress_max=1e-3,
                                 nutrient_min=0.1)
    rng = np.random.default_rng(7)
    fired = sum(k.can_divide(_parent(), shear_rate=0.1,
                                nutrient_C=1.0, dt=1.0, rng=rng)
                for _ in range(1000))
    assert fired == 0


def test_low_nutrient_blocks_division_even_when_shear_low():
    k = StressNutrientDivision(k_div=10.0, stress_max=1e-3,
                                 nutrient_min=0.1)
    rng = np.random.default_rng(7)
    fired = sum(k.can_divide(_parent(), shear_rate=1e-4,
                                nutrient_C=0.01, dt=1.0, rng=rng)
                for _ in range(1000))
    assert fired == 0


def test_both_thresholds_satisfied_fires_at_expected_rate():
    """At low stress + high nutrient, firing rate = 1 − exp(−k·dt)."""
    k_div = 0.1
    dt = 1.0
    k = StressNutrientDivision(k_div=k_div, stress_max=1.0,
                                 nutrient_min=0.0)
    rng = np.random.default_rng(11)
    n_trials = 20000
    fired = sum(k.can_divide(_parent(), shear_rate=0.0,
                                nutrient_C=1.0, dt=dt, rng=rng)
                for _ in range(n_trials))
    expected_p = 1.0 - math.exp(-k_div * dt)
    expected = expected_p * n_trials
    sigma = math.sqrt(expected_p * (1 - expected_p) * n_trials)
    # 4σ envelope.
    assert abs(fired - expected) < 4.0 * sigma


def test_zero_rate_never_fires():
    k = StressNutrientDivision(k_div=0.0, stress_max=1.0,
                                 nutrient_min=0.0)
    rng = np.random.default_rng(0)
    for _ in range(500):
        assert k.can_divide(_parent(), 0.0, 1.0, 1.0, rng) is False


# ── Lineage cap ───────────────────────────────────────────────────


def test_max_divisions_caps_lineage_depth():
    k = StressNutrientDivision(k_div=10.0, stress_max=1.0,
                                 nutrient_min=0.0,
                                 max_divisions=3)
    rng = np.random.default_rng(0)
    # Parent at depth 3 → blocked.
    assert k.can_divide(_parent(n=3), 0.0, 1.0, 1.0, rng) is False
    # Parent at depth 2 → permitted (probabilistically).
    fired = sum(k.can_divide(_parent(n=2), 0.0, 1.0, 1.0, rng)
                for _ in range(20))
    assert fired > 0


def test_max_divisions_none_means_unlimited():
    k = StressNutrientDivision(k_div=1.0, stress_max=1.0,
                                 nutrient_min=0.0,
                                 max_divisions=None)
    rng = np.random.default_rng(0)
    # Parent claiming 1000 prior divisions still permitted.
    fired = sum(k.can_divide(_parent(n=1000), 0.0, 1.0, 1.0, rng)
                for _ in range(20))
    assert fired > 0


# ── Reproducibility ───────────────────────────────────────────────


def test_seeded_rng_reproduces_bit_exact():
    k = StressNutrientDivision(k_div=0.5, stress_max=1.0,
                                 nutrient_min=0.0)
    a, b = np.random.default_rng(42), np.random.default_rng(42)
    for _ in range(200):
        assert (k.can_divide(_parent(), 0.0, 1.0, 1.0, a)
                == k.can_divide(_parent(), 0.0, 1.0, 1.0, b))


# ── Boundary cases ────────────────────────────────────────────────


def test_shear_exactly_at_threshold_permits_division():
    k = StressNutrientDivision(k_div=1.0, stress_max=0.5,
                                 nutrient_min=0.0)
    rng = np.random.default_rng(0)
    # γ̇ == stress_max should *not* be blocked (we use strict >).
    fired = sum(k.can_divide(_parent(), 0.5, 1.0, 1.0, rng)
                for _ in range(20))
    assert fired > 0


def test_nutrient_exactly_at_threshold_permits_division():
    k = StressNutrientDivision(k_div=1.0, stress_max=1.0,
                                 nutrient_min=0.5)
    rng = np.random.default_rng(0)
    fired = sum(k.can_divide(_parent(), 0.0, 0.5, 1.0, rng)
                for _ in range(20))
    assert fired > 0


# ── Input validation ──────────────────────────────────────────────


def test_validates_inputs():
    with pytest.raises(ValueError, match="k_div"):
        StressNutrientDivision(k_div=-0.1, stress_max=1.0,
                                 nutrient_min=0.0)
    with pytest.raises(ValueError, match="stress_max"):
        StressNutrientDivision(k_div=0.1, stress_max=-1.0,
                                 nutrient_min=0.0)
    with pytest.raises(ValueError, match="nutrient_min"):
        StressNutrientDivision(k_div=0.1, stress_max=1.0,
                                 nutrient_min=-0.1)
    with pytest.raises(ValueError, match="max_divisions"):
        StressNutrientDivision(k_div=0.1, stress_max=1.0,
                                 nutrient_min=0.0,
                                 max_divisions=-1)
