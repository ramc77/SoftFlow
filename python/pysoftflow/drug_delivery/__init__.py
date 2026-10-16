"""Phase-4 drug-delivery extensions.

Five release-kinetic models, wall-region absorbers, and metrics
(delivery efficiency, off-target fraction, residence-time
distribution, spatial dose map). All Python-side; the C++ chemistry
machinery (Fick leaching, finite reservoir M_p, Langmuir adsorption,
Peskin-spread fluxes) does the work, this layer schedules it.

Strong language constraint (the project specification spirit / the project plan)
— this is a coarse-grained methodological-exploration toolkit, not a
validated drug-carrier or tissue-uptake model. Caveats live in
``examples/04_drug_delivery/README.md`` and ``docs/drug_delivery.md``.

Submodules:

    kinetics   — DiffusionControlled, FirstOrder, ShearTriggered,
                 PhTriggered, Burst
    absorbers  — WallAbsorber (first-order or Michaelis-Menten)
    metrics    — delivery_efficiency, off_target_fraction,
                 residence_time_distribution, spatial_dose_map
    runner     — DrugDeliveryRun orchestrator
"""

from __future__ import annotations

from .kinetics import (
    Burst,
    CarrierState,
    DiffusionControlled,
    FirstOrder,
    FluidProbe,
    PhTriggered,
    ReleaseKinetic,
    ShearTriggered,
)
from .absorbers import AbsorberStep, WallAbsorber
from .metrics import (
    ResidenceTimeResult,
    delivery_efficiency,
    off_target_fraction,
    residence_time_distribution,
    spatial_dose_map,
)
from .runner import DrugDeliveryRun, RunStepRecord

__all__ = [
    "ReleaseKinetic",
    "CarrierState",
    "FluidProbe",
    "DiffusionControlled",
    "FirstOrder",
    "ShearTriggered",
    "PhTriggered",
    "Burst",
    "WallAbsorber",
    "AbsorberStep",
    "ResidenceTimeResult",
    "delivery_efficiency",
    "off_target_fraction",
    "residence_time_distribution",
    "spatial_dose_map",
    "DrugDeliveryRun",
    "RunStepRecord",
]
