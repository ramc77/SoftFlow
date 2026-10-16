"""Tests for pysoftflow.drug_delivery.kinetics — the five release models.

Each model is checked against its analytic solution under a synthetic
fluid probe, plus boundary cases (depleted carrier, threshold sigmoids,
burst fraction).
"""

from __future__ import annotations

import math
import pathlib
import sys

import numpy as np
import pytest


HERE = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE / "python"))

from pysoftflow.drug_delivery import (                        # noqa: E402
    Burst, CarrierState, DiffusionControlled, FirstOrder,
    FluidProbe, PhTriggered, ShearTriggered,
)


def _carrier(M0: float = 1.0, cid: int = 0) -> CarrierState:
    return CarrierState(capsule_id=cid, M_p=M0, M_p_initial=M0)


def _probe(t: float = 0.0, *, conc: float = 0.0,
           shear: float = 0.0, species=()) -> FluidProbe:
    return FluidProbe(time=t, centroid=(0.0, 0.0),
                       concentration=conc, shear_rate=shear,
                       species_concs=tuple(species))


# ── 1. DiffusionControlled — adapter is a no-op on the Python side ─


def test_diffusion_controlled_is_passthrough():
    k = DiffusionControlled(k_leach=0.05, C_eq=1.0)
    car = _carrier()
    delta = k.update(car, _probe(t=0.0, conc=0.0), dt=1.0)
    assert delta == 0.0
    assert car.M_p == 1.0     # the C++ side handles M_p drift, not us


def test_diffusion_controlled_validates_inputs():
    with pytest.raises(ValueError, match="k_leach"):
        DiffusionControlled(k_leach=-1.0, C_eq=1.0)


# ── 2. FirstOrder — exact closed-form decay ───────────────────────


def test_first_order_matches_exponential_decay_over_long_run():
    k_rel = 0.1
    fo = FirstOrder(k_rel=k_rel)
    car = _carrier(M0=10.0)
    dt = 1.0
    for _ in range(50):
        fo.update(car, _probe(), dt)
    # Analytic: M(t) = M0 e^{-k_rel * t}, so after 50 dt → M0 e^{-5}.
    expected = 10.0 * math.exp(-k_rel * 50.0 * dt)
    assert car.M_p == pytest.approx(expected, rel=1e-9)
    # Cumulative released = M0 - M_p.
    assert car.cumulative_released == pytest.approx(10.0 - expected,
                                                       rel=1e-9)


def test_first_order_zero_carrier_does_nothing():
    fo = FirstOrder(k_rel=0.5)
    car = _carrier(M0=0.0)
    assert fo.update(car, _probe(), dt=1.0) == 0.0


def test_first_order_validates_inputs():
    with pytest.raises(ValueError, match="k_rel"):
        FirstOrder(k_rel=-0.1)


# ── 3. ShearTriggered — sigmoid in shear rate ────────────────────


def test_shear_triggered_below_threshold_essentially_off():
    k = ShearTriggered(k_max=1.0, gamma_thresh=10.0, sharpness=50.0)
    car = _carrier(M0=1.0)
    # γ̇ = 1, threshold = 10 → sigmoid ~= 0 → minimal release.
    delta = k.update(car, _probe(shear=1.0), dt=1.0)
    assert delta < 1e-150
    assert car.M_p == pytest.approx(1.0, abs=1e-150)


def test_shear_triggered_above_threshold_releases_at_max_rate():
    k_max = 0.1
    k = ShearTriggered(k_max=k_max, gamma_thresh=1.0, sharpness=50.0)
    car = _carrier(M0=1.0)
    # γ̇ ≫ threshold → sigmoid ~= 1 → full first-order at k_max.
    delta = k.update(car, _probe(shear=10.0), dt=1.0)
    expected_decay = 1.0 - math.exp(-k_max)
    assert delta == pytest.approx(expected_decay, rel=1e-3)


def test_shear_triggered_at_threshold_half_rate():
    k = ShearTriggered(k_max=0.1, gamma_thresh=5.0, sharpness=10.0)
    car = _carrier(M0=1.0)
    # γ̇ = threshold → sigmoid = 0.5 → effective k = k_max/2.
    delta = k.update(car, _probe(shear=5.0), dt=1.0)
    expected = 1.0 - math.exp(-0.05)
    assert delta == pytest.approx(expected, rel=1e-9)


def test_shear_triggered_validates_inputs():
    with pytest.raises(ValueError, match="k_max"):
        ShearTriggered(k_max=-1.0, gamma_thresh=1.0)
    with pytest.raises(ValueError, match="sharpness"):
        ShearTriggered(k_max=1.0, gamma_thresh=1.0, sharpness=0.0)


# ── 4. PhTriggered — sigmoid in trigger species ───────────────────


def test_ph_triggered_uses_concentration_field_when_species_zero():
    k = PhTriggered(k_max=0.1, C_thresh=1.0, species=0, sharpness=20.0)
    car = _carrier(M0=1.0)
    # C = 0 ≪ threshold → sigmoid ≈ 0 → essentially no release.
    delta_low  = k.update(_carrier(M0=1.0), _probe(conc=0.0), dt=1.0)
    delta_high = k.update(car, _probe(conc=10.0), dt=1.0)
    assert delta_low < 1e-9
    assert delta_high > 0.5 * (1.0 - math.exp(-0.1))


def test_ph_triggered_uses_correct_species_index():
    k = PhTriggered(k_max=0.1, C_thresh=0.5, species=1, sharpness=50.0)
    car = _carrier(M0=1.0)
    # species index 1 contains 1.0; species 0 = 0.0. Sigmoid uses
    # species 1 → sees 1.0 > threshold 0.5 → releases.
    delta = k.update(car, _probe(species=(0.0, 1.0)), dt=1.0)
    assert delta > 0.5 * (1.0 - math.exp(-0.1))


def test_ph_triggered_validates_species():
    with pytest.raises(ValueError, match="species"):
        PhTriggered(k_max=0.1, C_thresh=1.0, species=-1)


# ── 5. Burst — one-shot release ───────────────────────────────────


def test_burst_fires_once_at_release_time():
    burst = Burst(release_time=5.0, fraction=0.6)
    car = _carrier(M0=1.0, cid=0)

    # Before t = release_time: nothing.
    assert burst.update(car, _probe(t=4.5), dt=1.0) == 0.0
    assert car.M_p == 1.0

    # At t ≥ release_time: drop by fraction.
    delta = burst.update(car, _probe(t=5.0), dt=1.0)
    assert delta == pytest.approx(0.6)
    assert car.M_p == pytest.approx(0.4)
    assert car.cumulative_released == pytest.approx(0.6)

    # Subsequent calls do nothing (already fired).
    assert burst.update(car, _probe(t=10.0), dt=1.0) == 0.0
    assert car.M_p == pytest.approx(0.4)


def test_burst_clamps_when_fraction_overdrains():
    # Carrier already partially depleted; fraction × M_initial > M_p.
    burst = Burst(release_time=0.0, fraction=1.0)
    car = CarrierState(capsule_id=0, M_p=0.3, M_p_initial=1.0)
    delta = burst.update(car, _probe(t=0.0), dt=1.0)
    assert delta == pytest.approx(0.3)   # not 1.0 — clamped
    assert car.M_p == pytest.approx(0.0)


def test_burst_each_carrier_fires_independently():
    burst = Burst(release_time=0.0, fraction=0.5)
    car1 = _carrier(cid=1)
    car2 = _carrier(cid=2)
    burst.update(car1, _probe(t=0.0), 1.0)
    # car2 hasn't fired yet — the per-id _fired set keeps them
    # independent.
    delta2 = burst.update(car2, _probe(t=0.0), 1.0)
    assert delta2 == pytest.approx(0.5)


def test_burst_validates_inputs():
    with pytest.raises(ValueError, match="release_time"):
        Burst(release_time=-1.0)
    with pytest.raises(ValueError, match="fraction"):
        Burst(release_time=0.0, fraction=0.0)
    with pytest.raises(ValueError, match="fraction"):
        Burst(release_time=0.0, fraction=1.5)


# ── Integration: a long FirstOrder run conserves cumulative + M_p ─


def test_first_order_conservation():
    k = FirstOrder(k_rel=0.05)
    car = _carrier(M0=2.5)
    for _ in range(200):
        k.update(car, _probe(), dt=1.0)
    # M_p + cumulative_released should always equal M_p_initial.
    assert car.M_p + car.cumulative_released == pytest.approx(2.5)
