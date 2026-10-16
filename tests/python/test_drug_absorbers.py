"""Tests for pysoftflow.drug_delivery.absorbers — WallAbsorber.

Coverage:
  - First-order patch uptake matches the analytic exponential decay.
  - Cells outside the patch are untouched.
  - Cumulative-absorbed counter equals what was removed from the field.
  - Michaelis-Menten saturates at high C and behaves linearly at low C.
  - Cells cannot be over-depleted (ΔC clamped at C).
  - Constructor rejects malformed input.
"""

from __future__ import annotations

import math
import pathlib
import sys

import numpy as np
import pytest


HERE = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE / "python"))

from pysoftflow.drug_delivery import WallAbsorber               # noqa: E402


# ── First-order kinetics ──────────────────────────────────────────


def test_first_order_uniform_field_decays_exponentially():
    nx, ny = 30, 10
    C = np.full((ny, nx), 1.0)
    abs_ = WallAbsorber(i_range=(0, nx), j_range=(0, ny),
                         k=0.05, mode="first_order")
    dt = 1.0
    for _ in range(50):
        abs_.step(C, dt)
    # Each cell evolves as C(t+dt) = C(t) - min(C, k*C*dt) = C(t)*(1 - k*dt)
    # over 50 steps → (1 - 0.05)^50.
    expected = (1.0 - 0.05) ** 50
    assert C[5, 15] == pytest.approx(expected, rel=1e-9)


def test_first_order_cumulative_matches_field_loss():
    nx, ny = 20, 5
    C = np.full((ny, nx), 1.0)
    initial_total = float(C.sum())
    abs_ = WallAbsorber(i_range=(5, 15), j_range=(0, ny),
                         k=0.1, mode="first_order")
    for _ in range(20):
        abs_.step(C, 1.0)
    # Mass conservation: cumulative absorbed == initial mass in patch
    # − final mass in patch.
    patch_now = C[0:ny, 5:15].sum()
    patch_initial = 10 * 5 * 1.0
    assert abs_.cumulative_absorbed == pytest.approx(
        patch_initial - patch_now, rel=1e-9)
    # Cells outside the patch untouched.
    assert (C[:, :5] == 1.0).all()
    assert (C[:, 15:] == 1.0).all()


def test_first_order_step_returns_correct_record():
    nx, ny = 5, 5
    C = np.full((ny, nx), 2.0)
    abs_ = WallAbsorber(i_range=(0, nx), j_range=(0, ny),
                         k=0.1, mode="first_order")
    rec = abs_.step(C, 1.0)
    # Each cell loses 0.1 * 2.0 * 1.0 = 0.2; 25 cells → 5.0 absorbed.
    assert rec.absorbed == pytest.approx(5.0)
    assert rec.cumulative == pytest.approx(5.0)
    assert (C == 1.8).all()


# ── Michaelis-Menten kinetics ──────────────────────────────────────


def test_mm_low_C_approaches_first_order():
    """At C ≪ K_M: J ≈ (k/K_M) · C, indistinguishable from first-order."""
    nx, ny = 20, 5
    C = np.full((ny, nx), 0.001)        # well below K_M = 1.0
    abs_ = WallAbsorber(i_range=(0, nx), j_range=(0, ny),
                         mode="michaelis_menten",
                         k=0.5, K_M=1.0)
    abs_.step(C, 1.0)
    # Effective rate at low C ≈ 0.5 → ΔC/C ≈ 0.5.
    expected = 0.001 * (1.0 - 0.5)
    assert C[2, 10] == pytest.approx(expected, rel=1e-3)


def test_mm_high_C_saturates_at_k_cat():
    """At C ≫ K_M: J ≈ k_cat (constant), ΔC ≈ k_cat · dt per cell."""
    nx, ny = 5, 5
    C = np.full((ny, nx), 1000.0)        # well above K_M = 1.0
    k_cat = 0.5
    abs_ = WallAbsorber(i_range=(0, nx), j_range=(0, ny),
                         mode="michaelis_menten",
                         k=k_cat, K_M=1.0)
    abs_.step(C, 1.0)
    # Each cell loses ≈ k_cat * dt = 0.5 (saturated rate).
    assert C[2, 2] == pytest.approx(1000.0 - k_cat, rel=1e-3)


def test_mm_validates_K_M():
    with pytest.raises(ValueError, match="K_M"):
        WallAbsorber(i_range=(0, 5), j_range=(0, 5),
                     mode="michaelis_menten", k=0.1, K_M=0.0)


# ── Over-depletion clamp ─────────────────────────────────────────


def test_step_does_not_overdraw_cell():
    # k * dt = 10, C = 0.5 → naive ΔC = 5, but clamp keeps it at 0.5.
    nx, ny = 5, 5
    C = np.full((ny, nx), 0.5)
    abs_ = WallAbsorber(i_range=(0, nx), j_range=(0, ny),
                         k=10.0, mode="first_order")
    rec = abs_.step(C, 1.0)
    assert (C == 0.0).all()
    # Cumulative absorbed = total initial mass in patch.
    assert rec.absorbed == pytest.approx(0.5 * 25)


# ── History accessor ─────────────────────────────────────────────


def test_history_records_per_step():
    nx, ny = 5, 5
    abs_ = WallAbsorber(i_range=(0, nx), j_range=(0, ny),
                         k=0.1, mode="first_order")
    for k in range(7):
        C = np.full((ny, nx), float(k + 1))
        abs_.step(C, 1.0)
    h = abs_.history
    assert h.shape == (7,)
    # Expected per-step absorbed = k * dt * total_C; total_C[k] = (k+1)*25.
    for k in range(7):
        assert h[k] == pytest.approx(0.1 * (k + 1) * 25)


# ── Input validation ─────────────────────────────────────────────


def test_validates_empty_ranges():
    with pytest.raises(ValueError, match="i_range"):
        WallAbsorber(i_range=(5, 5), j_range=(0, 5))


def test_validates_mode_string():
    with pytest.raises(ValueError, match="mode"):
        WallAbsorber(i_range=(0, 5), j_range=(0, 5), mode="zeroth")


def test_validates_negative_k():
    with pytest.raises(ValueError, match="k"):
        WallAbsorber(i_range=(0, 5), j_range=(0, 5), k=-0.1)


def test_validates_species_index():
    with pytest.raises(ValueError, match="species"):
        WallAbsorber(i_range=(0, 5), j_range=(0, 5), species=-1)


def test_n_cells_property():
    abs_ = WallAbsorber(i_range=(10, 30), j_range=(0, 5))
    assert abs_.n_cells == 100
