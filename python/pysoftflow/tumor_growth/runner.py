"""TumorGrowthRun — Phase-5 orchestrator.

**Coarse-grained mechano-chemical proxy. Not a validated cancer
model.** This is the same caveat that lives in every Phase-5 module:
methodological exploration only.

Wires up:

  - one or more (type_id → DivisionKinetic + DaughterPlacer + MembraneParams)
    registrations, and
  - zero or more EmbolizationDetector instances,

via a single per-step callback registered on ``Simulation::set-
StepCallback``. Same shape as Phase-4's ``DrugDeliveryRun``.

The callback (in registration order):

  1. Sample shear rate γ̇ and local nutrient C at every registered
     capsule's centroid.
  2. Call ``DivisionKinetic.can_divide(...)``; on True, ask the
     ``DaughterPlacer`` for a valid offset; on success, append the
     daughter via ``CapsuleSystem.addCapsule(...)``. Daughters are
     tracked from the next step onward so they can't divide in the
     same step they were created.
  3. Build one ``SimulationSnapshot`` per step and feed it to every
     embolization detector.
  4. Append a ``RunStepRecord`` to ``history``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from pysoftflow.analysis.snapshot import SimulationSnapshot

from .daughters import DaughterPlacer
from .division import DivisionKinetic, ParentState
from .embolization import EmbolizationDetector


__all__ = ["TumorGrowthRun", "RunStepRecord"]


@dataclass
class RunStepRecord:
    """One row of the per-step history."""
    step:               int
    time:               float
    n_capsules:         int
    n_divisions_total:  int
    n_divisions_step:   int
    largest_cluster:    int
    n_events_step:      int


@dataclass
class _Registration:
    type_id:       int
    kinetic:       DivisionKinetic
    placer:        DaughterPlacer
    mparams:       object
    nutrient_species: int = 0
    num_nodes:     int = 0


@dataclass
class TumorGrowthRun:
    """Per-simulation orchestrator. Construct, register, ``attach()``."""

    sim: object

    # Internal registries; users hit them through add_* methods.
    _registrations: dict[int, _Registration] = field(
        default_factory=dict, repr=False)
    _parent_states: dict[int, ParentState] = field(
        default_factory=dict, repr=False)
    _detectors:     list[EmbolizationDetector] = field(
        default_factory=list, repr=False)

    history:        list[RunStepRecord] = field(
        default_factory=list, repr=False)
    rng:            np.random.Generator = field(
        default_factory=lambda: np.random.default_rng(0xBEEFCAFE))

    _attached: bool = field(default=False, repr=False)

    # ── Registration API ───────────────────────────────────────────

    def add_division_kinetic(
        self,
        type_id: int,
        kinetic: DivisionKinetic,
        mparams,
        *,
        placer: Optional[DaughterPlacer] = None,
        nutrient_species: int = 0,
        num_nodes: int = 0,
    ) -> None:
        """Register a division kinetic for every capsule of ``type_id``.

        Parameters
        ----------
        type_id : int
            Capsule type whose members are eligible to divide.
        kinetic : DivisionKinetic
        mparams : MembraneParams
            Daughter capsules inherit these. Phase-5 design decision:
            we don't read parent mparams from the C++ side (not bound)
            — the user's registration is the source of truth.
        placer : DaughterPlacer, optional
            Default: ``DaughterPlacer()`` with stock parameters.
        nutrient_species : int
            Which scalar species the kinetic should sample as
            "nutrient". Default 0.
        num_nodes : int
            Daughter capsule node count. 0 (default) → auto via
            ``Capsule::computeOptimalNodes(r)``.
        """
        self._registrations[int(type_id)] = _Registration(
            type_id=int(type_id), kinetic=kinetic,
            placer=placer or DaughterPlacer(),
            mparams=mparams,
            nutrient_species=int(nutrient_species),
            num_nodes=int(num_nodes),
        )

    def add_embolization_detector(self,
                                    detector: EmbolizationDetector) -> None:
        self._detectors.append(detector)

    def set_seed(self, seed: int) -> None:
        """Reset the orchestrator's RNG."""
        self.rng = np.random.default_rng(int(seed))

    # ── Hook into Simulation ──────────────────────────────────────

    def attach(self) -> None:
        """Bind parent states to existing eligible capsules + register
        the per-step callback. Idempotent."""
        if self._attached:
            return
        self._sync_parent_states()
        # Calibrate detectors at t=0.
        for d in self._detectors:
            d.baseline(self.sim)
        self.sim.setStepCallback(self._step_callback)
        self._attached = True

    def _sync_parent_states(self) -> None:
        """Allocate a ``ParentState`` for every existing capsule whose
        type has a registered kinetic. Idempotent — called both at
        attach() and after each daughter insert."""
        caps = self.sim.capsules()
        n = caps.numCapsules()
        for cid in range(n):
            if cid in self._parent_states:
                continue
            t = caps[cid].getType()
            if t in self._registrations:
                self._parent_states[cid] = ParentState(
                    capsule_id=cid, n_divisions=0)

    # ── Per-step driver ────────────────────────────────────────────

    def _step_callback(self, sim, step):
        params = sim.params()
        dt = float(params.dt)
        nx = int(params.nx)
        ny = int(params.ny)

        from softflow_core import BoundaryType
        periodic_x = (params.fluid.boundary_type == BoundaryType.PERIODIC)
        Lx = float(nx)

        # Snapshot for percolation diagnostics + a stable centroid view
        # (we re-derive shear / nutrient on the fly — just one read).
        snap = SimulationSnapshot.from_simulation(sim)

        adr = sim.advectionDiffusion()
        if adr is not None:
            scalar = {s: adr.concentration(s)
                      for s in range(adr.getNumSpecies())}
        else:
            scalar = {}

        # Snapshot the eligible capsule indices BEFORE we insert any
        # daughters this step — daughters added now should not divide
        # in the same step.
        eligible_ids = [cid for cid in self._parent_states.keys()]
        caps = sim.capsules()
        wall_y_bottom = 0.5
        wall_y_top    = float(ny) - 1.5

        # Existing capsule centres + radii (vectorised — cached).
        n_existing = caps.numCapsules()
        existing_pos = np.empty((n_existing, 2), dtype=np.float64)
        existing_r   = np.empty(n_existing, dtype=np.float64)
        for i in range(n_existing):
            c = caps[i]
            ct = c.centroid()
            existing_pos[i, 0] = ct.x
            existing_pos[i, 1] = ct.y
            existing_r[i]      = c.effectiveRadius()

        # 1 + 2. Per-eligible-parent: probe → can_divide → place daughter.
        n_divisions_step = 0
        new_daughters: list[tuple[int, "object"]] = []   # (type, registration)
        for cid in eligible_ids:
            reg = self._registrations[caps[cid].getType()]
            kinetic = reg.kinetic

            ct = caps[cid].centroid()
            ix = int(np.clip(ct.x, 0.0, nx - 1.0))
            iy = int(np.clip(ct.y, 0.0, ny - 1.0))
            shear = self._estimate_shear(sim, ix, iy)
            nutrient_C = (float(scalar[reg.nutrient_species][iy, ix])
                           if reg.nutrient_species in scalar else 0.0)

            parent = self._parent_states[cid]
            if not kinetic.can_divide(parent, shear, nutrient_C, dt, self.rng):
                continue

            # Try to find a non-overlapping daughter position.
            placement = reg.placer.propose(
                parent_pos=(ct.x, ct.y),
                parent_radius=float(caps[cid].effectiveRadius()),
                existing_centers=existing_pos,
                existing_radii=existing_r,
                wall_y_bottom=wall_y_bottom,
                wall_y_top=wall_y_top,
                Lx=Lx, periodic_x=periodic_x,
                rng=self.rng,
            )
            if placement is None:
                continue

            # Add the daughter. addCapsule signature:
            #   addCapsule(center, radius, num_nodes, params, type)
            from softflow_core import Vec2d
            caps.addCapsule(
                Vec2d(*placement.center),
                float(placement.radius),
                int(reg.num_nodes),
                reg.mparams,
                int(reg.type_id),
            )
            parent.n_divisions += 1
            n_divisions_step += 1

            # Append the daughter to our local view so subsequent
            # parents in this step see it for overlap.
            existing_pos = np.vstack([existing_pos,
                                       np.asarray(placement.center)])
            existing_r   = np.append(existing_r, placement.radius)
            new_daughters.append((reg.type_id, reg))

        # Sync parent_states with the new capsule count.
        if n_divisions_step > 0:
            self._sync_parent_states()

        # 3. Embolization detectors.
        n_events_step = 0
        for d in self._detectors:
            event = d.step(sim, snap)
            if event is not None:
                n_events_step += 1

        # 4. Bookkeeping.
        from pysoftflow.analysis.patterns import hoshen_kopelman
        hk = hoshen_kopelman(snap, contact_cutoff=0.5)
        self.history.append(RunStepRecord(
            step=int(step),
            time=float(step) * dt,
            n_capsules=int(caps.numCapsules()),
            n_divisions_total=sum(p.n_divisions
                                   for p in self._parent_states.values()),
            n_divisions_step=int(n_divisions_step),
            largest_cluster=int(hk.largest_size),
            n_events_step=int(n_events_step),
        ))

    # ── Helpers ────────────────────────────────────────────────────

    @staticmethod
    def _estimate_shear(sim, ix: int, iy: int) -> float:
        """Magnitude of local rate-of-strain tensor at (ix, iy).

        Identical to the Phase-4 helper. We don't import it to keep
        the two phases independent.
        """
        try:
            field = sim.lbmSolver().field()
        except AttributeError:
            return 0.0
        nx = field.getNx() if hasattr(field, "getNx") else None
        ny = field.getNy() if hasattr(field, "getNy") else None
        if nx is None or ny is None:
            return 0.0
        ip = min(nx - 1, ix + 1); im = max(0, ix - 1)
        jp = min(ny - 1, iy + 1); jm = max(0, iy - 1)
        denom_x = max(1, ip - im)
        denom_y = max(1, jp - jm)
        dux_dx = (field.getUx(ip, iy) - field.getUx(im, iy)) / denom_x
        duy_dy = (field.getUy(ix, jp) - field.getUy(ix, jm)) / denom_y
        dux_dy = (field.getUx(ix, jp) - field.getUx(ix, jm)) / denom_y
        duy_dx = (field.getUy(ip, iy) - field.getUy(im, iy)) / denom_x
        s11 = dux_dx
        s22 = duy_dy
        s12 = 0.5 * (dux_dy + duy_dx)
        return float(np.sqrt(2.0 * (s11 * s11 + s22 * s22 + 2.0 * s12 * s12)))

    # ── Summary ───────────────────────────────────────────────────

    def summary(self) -> dict:
        """Headline numbers for HDF5 export / printing."""
        n_total_div = sum(p.n_divisions
                           for p in self._parent_states.values())
        n_events = sum(len(d.events) for d in self._detectors)
        return {
            "n_capsules_final":  self.sim.capsules().numCapsules(),
            "n_divisions_total": int(n_total_div),
            "n_embolization_events": int(n_events),
            "n_steps_recorded":  len(self.history),
        }
