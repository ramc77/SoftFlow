"""Tests for pysoftflow.tumor_growth.embolization.EmbolizationDetector.

The detector takes a ``sim`` and a ``SimulationSnapshot``. We use
hand-rolled stub objects for both — the only pieces of the engine
the detector touches are ``lbmSolver().field().getUx()`` /
``getNx()`` / ``getNy()``, which are easy to mock.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest


HERE = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE / "python"))

from pysoftflow.analysis import SimulationSnapshot              # noqa: E402
from pysoftflow.tumor_growth import (                           # noqa: E402
    EmbolizationDetector, EmbolizationEvent,
)


# ── Stub Sim / Field ──────────────────────────────────────────────


class _FakeField:
    def __init__(self, nx, ny, ux_profile):
        self._nx = nx
        self._ny = ny
        # ux_profile: callable (i, j) -> float, OR a 2-D array (ny, nx).
        self._ux = ux_profile

    def getNx(self): return self._nx
    def getNy(self): return self._ny

    def getUx(self, i, j):
        if callable(self._ux):
            return float(self._ux(i, j))
        return float(self._ux[j, i])


class _FakeLBM:
    def __init__(self, field): self._field = field
    def field(self): return self._field


class _FakeSim:
    def __init__(self, lbm): self._lbm = lbm
    def lbmSolver(self): return self._lbm


def _make_sim(nx=200, ny=40, ux=0.01):
    """Build a fake sim with uniform u_x profile."""
    field = _FakeField(nx, ny, ux_profile=lambda i, j: ux)
    return _FakeSim(_FakeLBM(field))


def _spanning_chain(*, nx=200, ny=40, x=100):
    """Vertical chain of touching capsules from y=1 to y=ny-1."""
    n = 25
    pts = np.column_stack([np.full(n, float(x)),
                            np.linspace(1.0, ny - 1.0, n)])
    return SimulationSnapshot.from_arrays(
        positions=pts, radii=np.full(n, 2.5),
        types=np.zeros(n, dtype=int),
        time=0.0, step=0,
        box=(float(nx), float(ny)),
        periodic_x=True,
    )


def _isolated_pair(*, nx=200, ny=40):
    pts = np.array([[100.0, 20.0], [105.0, 20.0]], dtype=float)
    return SimulationSnapshot.from_arrays(
        positions=pts, radii=np.array([1.0, 1.0]),
        types=np.zeros(2, dtype=int),
        box=(float(nx), float(ny)), periodic_x=True,
    )


# ── Q(t) measurement ──────────────────────────────────────────────


def test_baseline_captures_initial_flow_rate():
    sim = _make_sim(nx=200, ny=40, ux=0.005)
    det = EmbolizationDetector(x_section=100)
    Q0 = det.baseline(sim)
    # Σ u_x · Δy = 40 · 0.005 = 0.20.
    assert Q0 == pytest.approx(0.20)


def test_measure_flow_rate_classmethod():
    sim = _make_sim(nx=50, ny=20, ux=0.01)
    Q = EmbolizationDetector.measure_flow_rate(sim, x_section=10)
    assert Q == pytest.approx(0.20)


def test_measure_flow_rate_validates_x_section():
    sim = _make_sim(nx=50, ny=10, ux=0.01)
    with pytest.raises(ValueError, match="x_section"):
        EmbolizationDetector.measure_flow_rate(sim, x_section=99)


# ── Event detection: full gates ───────────────────────────────────


def test_spanning_chain_with_flow_drop_fires_event():
    sim = _make_sim(nx=200, ny=40, ux=0.01)
    det = EmbolizationDetector(
        x_section=100, flow_drop_threshold=0.5, contact_cutoff=1.0)
    det.baseline(sim)

    # Now drop the flow to half: simulate by re-creating sim with
    # half-strength velocity.
    sim_drop = _make_sim(nx=200, ny=40, ux=0.0049)   # 0.49 of baseline

    snap = _spanning_chain()
    event = det.step(sim_drop, snap)
    assert event is not None
    assert event.flow_rate_ratio < 0.5
    assert event.spanning_size == 25
    # Cluster spans roughly y=1 to y=39 → span ~ 38.
    assert event.cluster_y_span > 30.0


def test_spanning_chain_without_flow_drop_does_not_fire():
    sim = _make_sim(nx=200, ny=40, ux=0.01)
    det = EmbolizationDetector(
        x_section=100, flow_drop_threshold=0.5, contact_cutoff=1.0)
    det.baseline(sim)

    # Same flow as baseline → ratio ≈ 1, no event despite spanning.
    snap = _spanning_chain()
    event = det.step(sim, snap)
    assert event is None
    # But the percolation gate did fire — cluster_span_history has a
    # nonzero entry.
    assert det.cluster_span_history[0] > 0.0


def test_flow_drop_without_spanning_chain_does_not_fire():
    sim = _make_sim(nx=200, ny=40, ux=0.01)
    det = EmbolizationDetector(
        x_section=100, flow_drop_threshold=0.5)
    det.baseline(sim)

    sim_drop = _make_sim(nx=200, ny=40, ux=0.001)
    snap = _isolated_pair()
    event = det.step(sim_drop, snap)
    assert event is None


# ── History accumulation ──────────────────────────────────────────


def test_history_grows_per_step():
    sim = _make_sim()
    det = EmbolizationDetector(x_section=50, flow_drop_threshold=0.5)
    det.baseline(sim)
    snap = _isolated_pair()
    for _ in range(5):
        det.step(sim, snap)
    assert det.flow_rate_history.shape == (5,)
    assert det.cluster_span_history.shape == (5,)
    # No spanning cluster was ever present → spans are all zero.
    assert (det.cluster_span_history == 0.0).all()


def test_event_list_records_only_firings():
    sim = _make_sim()
    det = EmbolizationDetector(
        x_section=50, flow_drop_threshold=0.5, contact_cutoff=1.0)
    det.baseline(sim)

    # Step 1: chain + drop → event.
    sim_drop = _make_sim(ux=0.001)
    det.step(sim_drop, _spanning_chain())
    # Step 2: chain + no drop → no event.
    det.step(sim, _spanning_chain())
    # Step 3: pair + drop → no event.
    det.step(sim_drop, _isolated_pair())

    assert len(det.events) == 1
    e = det.events[0]
    assert isinstance(e, EmbolizationEvent)
    assert e.spanning_size == 25


# ── Input validation ──────────────────────────────────────────────


def test_validates_inputs():
    with pytest.raises(ValueError, match="x_section"):
        EmbolizationDetector(x_section=-1)
    with pytest.raises(ValueError, match="flow_drop_threshold"):
        EmbolizationDetector(x_section=10, flow_drop_threshold=0.0)
    with pytest.raises(ValueError, match="flow_drop_threshold"):
        EmbolizationDetector(x_section=10, flow_drop_threshold=1.5)
    with pytest.raises(ValueError, match="band_fraction"):
        EmbolizationDetector(x_section=10, band_fraction=0.7)
