"""Division kinetics for tumour-growth / aggregate-formation studies.

**Coarse-grained mechano-chemical proxy. Not a validated cancer
model.** This module models circulating-cell-style aggregation under
flow for methodological exploration and student training only — every
class docstring repeats this caveat per the project specification.

The canonical model (``StressNutrientDivision``) fires when:

  - Local fluid shear rate γ̇ is **below** ``stress_max`` (high shear
    fragments aggregates faster than they can replicate), and
  - Local nutrient concentration C is **above** ``nutrient_min``
    (the cell needs food to divide), and
  - A Poisson clock fires: ``rng() < k_div · dt``.

γ̇ is the magnitude of the local rate-of-strain tensor at the
parent's centroid (same probe used by Phase-4
``ShearTriggered`` kinetics). "Nutrient" is the concentration of a
designated scalar species — we don't simulate nutrient chemistry,
we simulate sigmoidal sensitivity to a designated trigger species
(same honest naming convention as Phase-4 ``PhTriggered``).

Custom division rules subclass ``DivisionKinetic`` and override
``can_divide``. The orchestrator never inspects subclass internals.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


__all__ = [
    "DivisionKinetic",
    "ParentState",
    "StressNutrientDivision",
]


@dataclass
class ParentState:
    """Per-capsule state passed into ``DivisionKinetic.can_divide``.

    Attributes
    ----------
    capsule_id : int
        Index into the simulation capsule system.
    n_divisions : int
        Number of times this capsule has divided since registration.
        Used to enforce per-capsule division caps and to feed into
        future kinetics that depend on lineage depth.
    """
    capsule_id:   int
    n_divisions:  int = 0


class DivisionKinetic(ABC):
    """Strategy interface for division logic.

    ``can_divide`` returns True if a daughter should be proposed this
    step. The orchestrator handles the actual placement and increments
    ``parent.n_divisions`` on success.

    Implementations must be deterministic in the supplied RNG so
    seeded runs reproduce bit-exact, and must run in O(1) per call
    (no allocation, no neighbour search — those happen in the
    daughter-placement layer).
    """

    @abstractmethod
    def can_divide(
        self,
        parent: ParentState,
        shear_rate: float,
        nutrient_C: float,
        dt: float,
        rng: np.random.Generator,
    ) -> bool:
        ...


class StressNutrientDivision(DivisionKinetic):
    """Division gated by stress (γ̇) AND nutrient concentration.

    *Coarse-grained mechano-chemical proxy.* The thresholds are
    phenomenological knobs; they don't represent any specific
    cell-cycle checkpoint or molecular signal.

    Parameters
    ----------
    k_div : float
        Poisson rate per unit time. After both thresholds are met,
        firing probability per step is ``1 − exp(−k_div · dt)``
        (closed-form Poisson over the timestep — equivalent to
        ``k_div · dt`` for small dt, exact for any dt).
    stress_max : float
        Maximum local shear rate at which division is permitted.
        Above this value, the carrier is being mechanically damaged
        and cannot divide. Set to a large value (e.g. 1e9) to
        disable the stress filter.
    nutrient_min : float
        Minimum local nutrient concentration for division. Below
        this, the cell starves. Set to 0 to disable the filter.
    max_divisions : int, optional
        Per-capsule cap on lineage depth. ``None`` (default) → no cap.
    """

    def __init__(
        self,
        k_div: float,
        stress_max: float,
        nutrient_min: float,
        max_divisions: int | None = None,
    ):
        if not (k_div >= 0.0):
            raise ValueError("k_div must be ≥ 0")
        if not (stress_max >= 0.0):
            raise ValueError("stress_max must be ≥ 0")
        if not (nutrient_min >= 0.0):
            raise ValueError("nutrient_min must be ≥ 0")
        if max_divisions is not None and max_divisions < 0:
            raise ValueError("max_divisions must be ≥ 0 or None")
        self.k_div         = float(k_div)
        self.stress_max    = float(stress_max)
        self.nutrient_min  = float(nutrient_min)
        self.max_divisions = max_divisions

    def can_divide(self, parent, shear_rate, nutrient_C, dt, rng):
        if self.k_div == 0.0:
            return False
        if (self.max_divisions is not None
                and parent.n_divisions >= self.max_divisions):
            return False
        if shear_rate > self.stress_max:
            return False
        if nutrient_C < self.nutrient_min:
            return False
        # Closed-form Poisson firing probability over the timestep.
        # Accurate for any dt; reduces to k_div·dt for small dt.
        p_fire = 1.0 - float(np.exp(-self.k_div * dt))
        return bool(rng.random() < p_fire)
