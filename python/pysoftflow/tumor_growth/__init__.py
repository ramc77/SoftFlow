"""Phase-5 tumour-growth / aggregate-formation extensions.

the project specification strong language constraint, repeated in every module:

    *This is a coarse-grained mechano-chemical proxy for circulating-cell
    aggregation under flow, not a validated cancer model. The pipeline
    is for methodological exploration and student training, not
    clinical claims.*

The C++ adhesion machinery (Bell + catch/slip kinetics, cluster
detection via union-find) and the Phase-3 ``hoshen_kopelman`` /
``force_percolation`` diagnostics already do most of the heavy
lifting. Phase 5 adds three small Python pieces:

    division     — stochastic division of "tumour" particles when
                   local stress and nutrient thresholds are met
    daughters    — daughter placement that respects volume exclusion
                   (uses Phase-2 isPlacementValid)
    embolization — quantification: flow-rate drop time series +
                   spanning-cluster span time series + event detector
    runner       — TumorGrowthRun orchestrator (single per-step callback)
"""

from __future__ import annotations

from .division import (
    DivisionKinetic,
    ParentState,
    StressNutrientDivision,
)
from .daughters import DaughterPlacement, DaughterPlacer
from .embolization import EmbolizationDetector, EmbolizationEvent
from .runner import RunStepRecord, TumorGrowthRun

__all__ = [
    "DivisionKinetic",
    "ParentState",
    "StressNutrientDivision",
    "DaughterPlacement",
    "DaughterPlacer",
    "EmbolizationDetector",
    "EmbolizationEvent",
    "TumorGrowthRun",
    "RunStepRecord",
]
