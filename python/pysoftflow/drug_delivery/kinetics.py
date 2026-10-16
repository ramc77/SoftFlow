"""Release kinetics for drug-carrying capsules.

Each carrier holds a finite payload mass ``M_p(t)`` (lattice units of
scalar). At every timestep, a ``ReleaseKinetic`` reduces M_p according
to its model and returns the released mass — which the orchestrator
adds to the local scalar field via the existing ``Simulation::set-
LeachingParams`` / ``setParticleMass`` infrastructure (or, for non-
Fickian models, directly subtracts from ``getCapsuleMp`` and Peskin-
spreads the released mass to the lattice).

Five models, all cleanly separable:

  - ``DiffusionControlled(k_leach, C_eq)`` — wraps the existing
    Higuchi-style Fick law J = k · (C_eq − C_surface). The C++ side
    already implements this; the Python class is a no-op pass-through
    so the user-facing API is uniform across models.

  - ``FirstOrder(k_rel)`` — dM_p/dt = −k_rel · M_p, independent of
    local concentration. Closed-form solution M_p(t) = M_0 e^{−k_rel t}.

  - ``ShearTriggered(k_max, gamma_thresh, sharpness)`` — first-order
    rate gated by a sigmoid in local fluid shear rate
    γ̇ = √(2 D : D), with smooth onset around the threshold.

  - ``PhTriggered(k_max, C_thresh, species, sharpness)`` — first-order
    rate gated by a sigmoid in the local concentration of a
    *second* scalar species (the "pH proxy"). Honest naming: we
    don't simulate pH chemistry, we simulate sigmoidal sensitivity
    to a designated trigger species.

  - ``Burst(release_time, fraction)`` — one-shot release: at
    ``t ≥ release_time``, drop M_p by ``fraction · M_0`` once.

Each class implements ``update(carrier_state, fluid_probe, dt) ->
released_mass``. The orchestrator iterates over all carriers each
step, calls ``update`` per carrier, and Peskin-spreads the released
mass into the scalar field.

References
----------
  - Higuchi, *J. Pharm. Sci.* **50**, 874 (1961) — diffusion-controlled.
  - Schmaljohann, *Adv. Drug Deliv. Rev.* **58**, 1655 (2006) —
    stimuli-responsive carriers.
  - Bao & Suresh, *Nat. Mater.* **2**, 715 (2003) — mechanosensitive.

Strong language note: every class docstring repeats "this is a
coarse-grained model, not a validated drug-carrier mechanism".
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


__all__ = [
    "ReleaseKinetic",
    "CarrierState",
    "FluidProbe",
    "DiffusionControlled",
    "FirstOrder",
    "ShearTriggered",
    "PhTriggered",
    "Burst",
]


# ── Lightweight state containers ───────────────────────────────────


@dataclass
class CarrierState:
    """Per-carrier mutable state passed into ``ReleaseKinetic.update``.

    The orchestrator owns one of these per capsule and updates
    ``M_p`` after each kinetic call.

    Attributes
    ----------
    capsule_id : int
        Index into the capsule system.
    M_p : float
        Current remaining payload mass.
    M_p_initial : float
        Initial payload at registration time. Used by ``Burst`` for
        the fractional-drop reference and by metrics for normalisation.
    cumulative_released : float
        Total mass released since registration.
    """
    capsule_id:           int
    M_p:                  float
    M_p_initial:          float
    cumulative_released:  float = 0.0


@dataclass(frozen=True)
class FluidProbe:
    """Local fluid state at a carrier's centroid.

    Filled by the orchestrator before each kinetic call. Fields are
    optional so a kinetic that only needs ``concentration`` doesn't
    pay the cost of a shear-rate evaluation.
    """
    time:           float
    centroid:       tuple[float, float]
    concentration:  float = 0.0          # of species 0 by default
    shear_rate:     float = 0.0          # γ̇ at the centroid
    species_concs:  tuple[float, ...] = ()   # full per-species probe


# ── Abstract base ──────────────────────────────────────────────────


class ReleaseKinetic(ABC):
    """Strategy interface for payload release.

    Implementations consume a carrier's state and a fluid probe at the
    carrier's location, return the mass released this timestep, and
    are expected to leave their internal state coherent with that
    return value (so the orchestrator can call them N times without
    surprise).
    """

    @abstractmethod
    def update(
        self,
        carrier: CarrierState,
        fluid: FluidProbe,
        dt: float,
    ) -> float:
        """Return the mass released this timestep (≥ 0)."""

    # Optional hook: a kinetic may want to set per-type ScalarParams
    # at registration time so the C++ Fick path picks it up. Default
    # is a no-op; only DiffusionControlled overrides this.
    def configure_simulation(self, sim, type_id: int) -> None:   # noqa: B027
        return None


# ── 1. Diffusion-controlled (wraps existing C++ Fick path) ────────


class DiffusionControlled(ReleaseKinetic):
    """Higuchi-style diffusion-controlled release.

    The flux into the fluid is ``J = k_leach · (C_eq − C_surface)``;
    when C_surface < C_eq the carrier releases, when > it absorbs.
    The C++ ``AdvectionDiffusion::applyChemistry`` already implements
    this exactly via ``Simulation::setLeachingParams``. This Python
    class is a thin adapter so the user API is uniform across models.

    Limitation: this is a *coarse-grained Fick-type* model, not a
    validated dissolution model for any specific carrier chemistry.
    """

    def __init__(self, k_leach: float, C_eq: float):
        if not (k_leach >= 0.0):
            raise ValueError("k_leach must be ≥ 0")
        self.k_leach = float(k_leach)
        self.C_eq    = float(C_eq)

    def configure_simulation(self, sim, type_id: int) -> None:
        sim.setLeachingParams(int(type_id),
                               float(self.k_leach),
                               float(self.C_eq))

    def update(self, carrier, fluid, dt):
        # The C++ chemistry path drives M_p; we don't double-count
        # here. The orchestrator's per-step refresh of carrier.M_p
        # from ``getCapsuleMp(id)`` keeps Python state in sync.
        return 0.0


# ── 2. First-order release ────────────────────────────────────────


class FirstOrder(ReleaseKinetic):
    """First-order release: dM_p/dt = −k_rel · M_p.

    Direct integration, independent of local concentration. We use
    the exact closed-form ``ΔM_p = M_p (1 − exp(−k_rel · dt))`` to
    avoid drift over long runs.

    Limitation: the model assumes infinite fluid sink and no
    saturation. For high local C, real carriers slow down; the
    diffusion-controlled model captures that.
    """

    def __init__(self, k_rel: float):
        if not (k_rel >= 0.0):
            raise ValueError("k_rel must be ≥ 0")
        self.k_rel = float(k_rel)

    def update(self, carrier, fluid, dt):
        if carrier.M_p <= 0.0:
            return 0.0
        decay  = 1.0 - float(np.exp(-self.k_rel * dt))
        delta  = carrier.M_p * decay
        carrier.M_p -= delta
        carrier.cumulative_released += delta
        return delta


# ── 3. Shear-triggered ────────────────────────────────────────────


class ShearTriggered(ReleaseKinetic):
    """First-order release gated by a sigmoid in local shear rate.

    Effective rate::

        k_eff(γ̇) = k_max / (1 + exp(−sharpness · (γ̇ − γ_thresh)))

    γ̇ is the local fluid shear rate at the carrier's centroid,
    computed by the orchestrator as ``√(2 ∂_i u_j ∂_i u_j)``.

    Limitation: this is a sigmoid in the magnitude of the
    rate-of-strain tensor; it does not represent any specific
    molecular shear-sensitive bond. Use as a phenomenological knob.
    """

    def __init__(self, k_max: float, gamma_thresh: float,
                 sharpness: float = 50.0):
        if not (k_max >= 0.0):
            raise ValueError("k_max must be ≥ 0")
        if not (sharpness > 0.0):
            raise ValueError("sharpness must be > 0")
        self.k_max        = float(k_max)
        self.gamma_thresh = float(gamma_thresh)
        self.sharpness    = float(sharpness)

    def update(self, carrier, fluid, dt):
        if carrier.M_p <= 0.0:
            return 0.0
        x = self.sharpness * (fluid.shear_rate - self.gamma_thresh)
        # Numerically stable sigmoid.
        if x >= 0.0:
            sig = 1.0 / (1.0 + float(np.exp(-x)))
        else:
            ex  = float(np.exp(x))
            sig = ex / (1.0 + ex)
        k_eff = self.k_max * sig
        decay = 1.0 - float(np.exp(-k_eff * dt))
        delta = carrier.M_p * decay
        carrier.M_p -= delta
        carrier.cumulative_released += delta
        return delta


# ── 4. pH- / chemical-triggered ───────────────────────────────────


class PhTriggered(ReleaseKinetic):
    """First-order release gated by a sigmoid in a trigger species.

    Effective rate::

        k_eff(C) = k_max / (1 + exp(−sharpness · (C − C_thresh)))

    where C is the local concentration of species ``species`` at the
    carrier's centroid. Set ``species=0`` to use the same scalar that
    the carrier releases into; use ``species=1`` to use a separate
    "pH-proxy" species seeded at a target tissue patch.

    Limitation: we don't simulate pH chemistry. We simulate a sigmoid
    sensitivity to a designated scalar that the user has labelled
    "pH proxy". For real pH-responsive carriers, the trigger species
    should encode a calibrated proxy (e.g. H+ equivalent
    concentration mapped to lattice units).
    """

    def __init__(self, k_max: float, C_thresh: float,
                 species: int = 0, sharpness: float = 50.0):
        if not (k_max >= 0.0):
            raise ValueError("k_max must be ≥ 0")
        if not (sharpness > 0.0):
            raise ValueError("sharpness must be > 0")
        if species < 0:
            raise ValueError("species must be ≥ 0")
        self.k_max     = float(k_max)
        self.C_thresh  = float(C_thresh)
        self.species   = int(species)
        self.sharpness = float(sharpness)

    def update(self, carrier, fluid, dt):
        if carrier.M_p <= 0.0:
            return 0.0
        if self.species < len(fluid.species_concs):
            C = fluid.species_concs[self.species]
        else:
            C = fluid.concentration
        x = self.sharpness * (C - self.C_thresh)
        if x >= 0.0:
            sig = 1.0 / (1.0 + float(np.exp(-x)))
        else:
            ex  = float(np.exp(x))
            sig = ex / (1.0 + ex)
        k_eff = self.k_max * sig
        decay = 1.0 - float(np.exp(-k_eff * dt))
        delta = carrier.M_p * decay
        carrier.M_p -= delta
        carrier.cumulative_released += delta
        return delta


# ── 5. Burst ─────────────────────────────────────────────────────


class Burst(ReleaseKinetic):
    """One-shot burst release at a fixed time.

    At the first ``update`` call with ``fluid.time ≥ release_time``,
    drops ``M_p`` by ``fraction · M_p_initial`` (or to zero if the
    fraction would over-deplete). All subsequent calls return 0.

    Limitation: an idealised step-function release. Real burst
    release is sharp but not instantaneous; combine with FirstOrder
    if you want a "fast then slow" two-phase profile.
    """

    def __init__(self, release_time: float, fraction: float = 1.0):
        if not (release_time >= 0.0):
            raise ValueError("release_time must be ≥ 0")
        if not (0.0 < fraction <= 1.0):
            raise ValueError("fraction must be in (0, 1]")
        self.release_time = float(release_time)
        self.fraction     = float(fraction)
        self._fired: set[int] = set()

    def update(self, carrier, fluid, dt):
        if carrier.capsule_id in self._fired:
            return 0.0
        if fluid.time < self.release_time:
            return 0.0
        delta = min(carrier.M_p,
                    self.fraction * carrier.M_p_initial)
        carrier.M_p -= delta
        carrier.cumulative_released += delta
        self._fired.add(carrier.capsule_id)
        return delta
