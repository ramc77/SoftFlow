"""Embolization detection for tumour-growth / aggregate-formation runs.

**Coarse-grained mechano-chemical proxy. Not a validated cancer
model.** Phase-5 strong-language constraint repeated here.

We declare an "embolization event" at step k when both:

  1. A spanning cluster exists at the cross-section of interest
     (Phase-3 ``force_percolation`` returns ``spans=True``), and
  2. The streamwise volumetric flow rate
     ``Q(t) = Σ u_x(x_section, j) · Δy``
     has dropped below ``flow_drop_threshold`` × ``Q(0)``.

Both gates must fire simultaneously to count — a transient flow drop
without a cluster doesn't count, and a spanning cluster that hasn't
slowed the flow yet doesn't count.

Per-step time series of ``Q/Q_0`` and the spanning-cluster y-span are
recorded for plotting.

References
----------
  - Au et al., *Proc. Natl. Acad. Sci. USA* **113**, 4947 (2016) —
    circulating tumour-cell embolic events.
  - Stauffer & Aharony, *Introduction to Percolation Theory* (1994).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from pysoftflow.analysis.jamming import force_percolation
from pysoftflow.analysis.snapshot import SimulationSnapshot


__all__ = ["EmbolizationEvent", "EmbolizationDetector"]


@dataclass(frozen=True)
class EmbolizationEvent:
    """One snapshot that simultaneously satisfies both gates."""
    step:               int
    time:               float
    flow_rate_ratio:    float
    spanning_size:      int
    cluster_y_span:     float
    spanning_x_centre:  float


@dataclass
class EmbolizationDetector:
    """Watches a single cross-section for spanning + flow-drop events.

    Construct → call ``baseline(sim)`` once at t=0 → call ``step(sim,
    snap)`` every timestep. Returns an :class:`EmbolizationEvent` (or
    None) and accumulates the per-step time series in
    ``flow_rate_history`` / ``cluster_span_history``.

    Parameters
    ----------
    x_section : int
        Lattice column at which to integrate u_x for Q(t). Choose a
        column where you expect plug formation (e.g. just downstream
        of the seeded aggregate).
    flow_drop_threshold : float
        Event fires when Q(t)/Q_0 drops below this fraction. Default
        0.5 (Q at half the baseline).
    contact_cutoff : float
        Centre-distance slack passed through to ``force_percolation``.
    band_fraction : float
        Top/bottom band thickness (as a fraction of Ly) used by
        ``force_percolation`` to decide "spanning".
    """
    x_section:           int
    flow_drop_threshold: float = 0.5
    contact_cutoff:      float = 0.5
    band_fraction:       float = 0.10

    Q_baseline:          float = 0.0
    _flow_rate_history:  list[float] = field(
        default_factory=list, init=False, repr=False, compare=False)
    _cluster_span_history: list[float] = field(
        default_factory=list, init=False, repr=False, compare=False)
    _events:             list[EmbolizationEvent] = field(
        default_factory=list, init=False, repr=False, compare=False)

    def __post_init__(self):
        if self.x_section < 0:
            raise ValueError("x_section must be ≥ 0")
        if not (0.0 < self.flow_drop_threshold <= 1.0):
            raise ValueError(
                "flow_drop_threshold must be in (0, 1]")
        if not (0.0 < self.band_fraction < 0.5):
            raise ValueError("band_fraction must be in (0, 0.5)")

    # ── Q(t) measurement ─────────────────────────────────────────

    @staticmethod
    def measure_flow_rate(sim, x_section: int) -> float:
        """Σ u_x(x_section, j) · Δy. Δy = 1 in lattice units."""
        field = sim.lbmSolver().field()
        ny    = field.getNy()
        nx    = field.getNx()
        if not (0 <= x_section < nx):
            raise ValueError(
                f"x_section {x_section} out of range [0, {nx})")
        Q = 0.0
        for j in range(ny):
            Q += float(field.getUx(x_section, j))
        return Q

    def baseline(self, sim) -> float:
        """Sample Q(0) and store as the reference. Returns it."""
        self.Q_baseline = self.measure_flow_rate(sim, self.x_section)
        return self.Q_baseline

    # ── Per-step driver ──────────────────────────────────────────

    def step(
        self,
        sim,
        snap: SimulationSnapshot,
    ) -> EmbolizationEvent | None:
        """Sample Q, run percolation, append to history, return event.

        ``baseline(sim)`` should already have been called once. If
        ``Q_baseline`` is still 0 we calibrate on the first call.
        """
        Q = self.measure_flow_rate(sim, self.x_section)
        if self.Q_baseline == 0.0:
            self.Q_baseline = Q if Q > 0.0 else 1e-30
        ratio = Q / self.Q_baseline if self.Q_baseline != 0.0 else 0.0

        perc = force_percolation(
            snap,
            contact_cutoff=self.contact_cutoff,
            use_bonds=(snap.bonds.shape[0] > 0),
            band_fraction=self.band_fraction,
        )

        # Cluster y-span: 0.0 if no spanning cluster.
        cluster_y_span = 0.0
        spanning_x_centre = 0.0
        if perc.spans:
            # Re-derive the spanning cluster's y-extent from the
            # snapshot. force_percolation already verified that some
            # cluster touches both bands; we now find that cluster's
            # member positions and compute the span.
            from pysoftflow.analysis.patterns import hoshen_kopelman
            hk = hoshen_kopelman(
                snap, contact_cutoff=self.contact_cutoff,
                use_bonds=(snap.bonds.shape[0] > 0))
            cluster_y_span, spanning_x_centre = self._spanning_cluster_extent(
                snap, hk, perc.y_low_band, perc.y_top_band)

        self._flow_rate_history.append(float(ratio))
        self._cluster_span_history.append(float(cluster_y_span))

        if perc.spans and ratio < self.flow_drop_threshold:
            event = EmbolizationEvent(
                step=int(snap.step),
                time=float(snap.time),
                flow_rate_ratio=float(ratio),
                spanning_size=int(perc.spanning_size),
                cluster_y_span=float(cluster_y_span),
                spanning_x_centre=float(spanning_x_centre),
            )
            self._events.append(event)
            return event
        return None

    # ── Internal helpers ─────────────────────────────────────────

    @staticmethod
    def _spanning_cluster_extent(
        snap: SimulationSnapshot,
        hk,
        y_low: float, y_top: float,
    ) -> tuple[float, float]:
        """y-span and x-centre of the largest spanning cluster."""
        labels = hk.labels
        best_span = 0.0
        best_x_centre = 0.0
        for cid in np.unique(labels):
            members = labels == cid
            ys = snap.positions[members, 1]
            if not (np.any(ys < y_low) and np.any(ys > y_top)):
                continue
            span = float(ys.max() - ys.min())
            if span > best_span:
                best_span = span
                best_x_centre = float(snap.positions[members, 0].mean())
        return best_span, best_x_centre

    # ── History accessors ────────────────────────────────────────

    @property
    def flow_rate_history(self) -> np.ndarray:
        return np.asarray(self._flow_rate_history, dtype=np.float64)

    @property
    def cluster_span_history(self) -> np.ndarray:
        return np.asarray(self._cluster_span_history, dtype=np.float64)

    @property
    def events(self) -> list[EmbolizationEvent]:
        return list(self._events)
