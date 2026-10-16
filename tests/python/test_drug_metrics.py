"""Tests for pysoftflow.drug_delivery.metrics."""

from __future__ import annotations

import math
import pathlib
import sys

import numpy as np
import pytest


HERE = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE / "python"))

from pysoftflow.analysis import SimulationSnapshot               # noqa: E402
from pysoftflow.drug_delivery import (                           # noqa: E402
    CarrierState, WallAbsorber,
    delivery_efficiency, off_target_fraction,
    residence_time_distribution, spatial_dose_map,
)


def _carrier(M0=1.0, cid=0):
    return CarrierState(capsule_id=cid, M_p=M0, M_p_initial=M0)


# ── delivery_efficiency / off_target_fraction ─────────────────────


def test_efficiency_full_target_returns_one():
    target = WallAbsorber(i_range=(0, 10), j_range=(0, 5),
                           cumulative_absorbed=4.0, label="target")
    carriers = [_carrier(M0=1.0, cid=i) for i in range(4)]
    assert delivery_efficiency(target, carriers) == pytest.approx(1.0)


def test_efficiency_no_payload_returns_nan():
    target = WallAbsorber(i_range=(0, 10), j_range=(0, 5),
                           cumulative_absorbed=0.0)
    assert math.isnan(delivery_efficiency(target, []))


def test_efficiency_partial():
    target = WallAbsorber(i_range=(0, 10), j_range=(0, 5),
                           cumulative_absorbed=0.7)
    carriers = [_carrier(M0=1.0, cid=i) for i in range(3)]
    assert delivery_efficiency(target, carriers) == pytest.approx(0.7 / 3)


def test_off_target_fraction_sums_multiple_absorbers():
    a = WallAbsorber(i_range=(0, 5), j_range=(0, 5),
                      cumulative_absorbed=0.2, label="off1")
    b = WallAbsorber(i_range=(10, 20), j_range=(0, 5),
                      cumulative_absorbed=0.3, label="off2")
    carriers = [_carrier(M0=1.0, cid=i) for i in range(2)]
    assert off_target_fraction([a, b], carriers) == pytest.approx(0.25)


def test_efficiency_plus_off_target_plus_remaining_sums_to_one_idealised():
    """Idealised mass-balance: η + OTF + (M_p / loaded) ≈ 1
    when there are no other sinks and the scalar is conservative."""
    target = WallAbsorber(i_range=(0, 5), j_range=(0, 5),
                           cumulative_absorbed=0.4)
    off    = WallAbsorber(i_range=(10, 20), j_range=(0, 5),
                           cumulative_absorbed=0.3)
    carriers = [_carrier(M0=1.0)]
    carriers[0].M_p = 0.3   # remaining payload
    eta = delivery_efficiency(target, carriers)
    otf = off_target_fraction([off], carriers)
    remaining = sum(c.M_p for c in carriers) / sum(c.M_p_initial for c in carriers)
    assert eta + otf + remaining == pytest.approx(1.0)


# ── residence_time_distribution ────────────────────────────────────


def _snap(t, positions, types=None):
    n = positions.shape[0]
    if types is None:
        types = np.zeros(n, dtype=int)
    return SimulationSnapshot.from_arrays(
        positions=positions, radii=np.ones(n),
        types=types, time=t, step=int(t),
        box=(100.0, 100.0))


def test_rtd_static_carrier_in_band_returns_full_duration():
    pos = np.array([[10.0, 5.0]], dtype=float)   # always at y=5, in band
    snaps = [_snap(t=float(t), positions=pos) for t in range(11)]
    res = residence_time_distribution(
        snaps, target_band=(0.0, 10.0), n_bins=5)
    # 10 intervals each of dt=1 → cumulative = 10.
    assert res.per_carrier[0] == pytest.approx(10.0)
    assert res.mean == pytest.approx(10.0)


def test_rtd_carrier_outside_band_returns_zero():
    pos = np.array([[10.0, 50.0]], dtype=float)
    snaps = [_snap(t=float(t), positions=pos) for t in range(11)]
    res = residence_time_distribution(
        snaps, target_band=(0.0, 10.0))
    assert res.per_carrier[0] == 0.0


def test_rtd_carrier_partially_inside():
    # First half inside, second half outside.
    snaps = []
    for t in range(11):
        y = 5.0 if t < 5 else 50.0
        snaps.append(_snap(t=float(t),
                            positions=np.array([[10.0, y]], dtype=float)))
    res = residence_time_distribution(
        snaps, target_band=(0.0, 10.0))
    # Carrier was inside at t=0,1,2,3,4 → 5 intervals counted.
    assert res.per_carrier[0] == pytest.approx(5.0)


def test_rtd_type_filter():
    pos = np.array([[10.0, 5.0],     # type 0 — always in band
                    [10.0, 50.0]],   # type 1 — never in band
                    dtype=float)
    types = np.array([0, 1])
    snaps = [_snap(t=float(t), positions=pos, types=types)
              for t in range(6)]
    res0 = residence_time_distribution(
        snaps, target_band=(0.0, 10.0), type_filter=0)
    res1 = residence_time_distribution(
        snaps, target_band=(0.0, 10.0), type_filter=1)
    assert res0.per_carrier[0] == pytest.approx(5.0)
    assert res1.per_carrier[0] == 0.0
    assert res0.per_carrier.size == 1
    assert res1.per_carrier.size == 1


def test_rtd_input_validation():
    pos = np.zeros((1, 2))
    with pytest.raises(ValueError, match="at least 2"):
        residence_time_distribution(
            [_snap(0.0, pos)], target_band=(0.0, 10.0))


# ── spatial_dose_map ──────────────────────────────────────────────


def test_dose_map_zero_when_no_absorbers():
    dose = spatial_dose_map([], nx=10, ny=5)
    assert dose.shape == (5, 10)
    assert (dose == 0.0).all()


def test_dose_map_distributes_uniformly_across_patch():
    a = WallAbsorber(i_range=(2, 6), j_range=(0, 2),
                      cumulative_absorbed=8.0)
    dose = spatial_dose_map([a], nx=10, ny=5)
    # Patch is 4×2 = 8 cells, total dose 8 → 1.0 per cell.
    assert (dose[0:2, 2:6] == 1.0).all()
    # Outside the patch is zero.
    assert (dose[2:, :] == 0.0).all()
    assert (dose[:, :2] == 0.0).all()


def test_dose_map_sums_overlapping_absorbers():
    a = WallAbsorber(i_range=(0, 4), j_range=(0, 4),
                      cumulative_absorbed=4.0)
    b = WallAbsorber(i_range=(2, 6), j_range=(0, 4),
                      cumulative_absorbed=4.0)
    dose = spatial_dose_map([a, b], nx=10, ny=4)
    # Overlap region (i ∈ [2, 4), j ∈ [0, 4)) gets contributions
    # from both absorbers: 4/16 from a, 4/16 from b → 0.5 per cell.
    overlap = dose[0:4, 2:4]
    assert np.allclose(overlap, 0.5)
    # a-only region: 0.25; b-only region: 0.25.
    assert np.allclose(dose[0:4, 0:2], 0.25)
    assert np.allclose(dose[0:4, 4:6], 0.25)


def test_dose_map_skips_unabsorbed_absorber():
    a = WallAbsorber(i_range=(0, 4), j_range=(0, 4),
                      cumulative_absorbed=0.0)
    dose = spatial_dose_map([a], nx=10, ny=4)
    assert (dose == 0.0).all()
