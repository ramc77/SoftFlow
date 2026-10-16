"""DrugDeliveryRun — Phase-4 orchestrator.

Wires up release kinetics + wall absorbers to a live ``Simulation``
via a single per-step callback. The callback runs in this order:

  1. Refresh per-carrier ``M_p`` from the C++ side
     (``AdvectionDiffusion.getCapsuleMp(id)``).
  2. For each carrier, build a ``FluidProbe`` (centroid concentration
     + shear rate at the centroid + multi-species probe).
  3. For each carrier, call ``kinetic.update(carrier_state, probe, dt)``;
     for non-Fickian models, the released mass is Peskin-spread into
     the local scalar field by writing into a 4×4 cell window around
     the centroid.
  4. Apply each ``WallAbsorber`` to the scalar field in place.
  5. Append a snapshot of the metrics to ``history``.

This single-callback design is easier to debug than scattered hooks
and matches the existing ``Simulation::setStepCallback`` API.

Limitations (per the strong-language constraint, repeated here so
users running the showcase don't miss them):
  - Carriers are 2-D coarse-grained spring-network capsules, not
    realistic liposomes / micelles / RBC ghosts.
  - Wall absorbers are boundary-condition sinks; not validated tissue
    uptake models.
  - "pH-triggered" is shorthand for "second-scalar-triggered"; we
    don't simulate pH chemistry.
  - "Shear-triggered" is a sigmoid in the magnitude of the local
    rate-of-strain tensor; not any specific molecular bond.
  - Non-Fickian release uses uniform 4-cell spreading, not the full
    Peskin 4-pt kernel that the C++ chemistry path uses; this is fine
    for relative comparisons but not for cell-by-cell quantitative
    deposition profiles.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from .absorbers import WallAbsorber
from .kinetics import (
    CarrierState, DiffusionControlled, FluidProbe, ReleaseKinetic,
)


__all__ = ["DrugDeliveryRun", "RunStepRecord"]


@dataclass
class RunStepRecord:
    """One row of the orchestrator's per-step history."""
    step:                int
    time:                float
    total_M_p:           float
    total_released:      float
    target_absorbed:     float
    off_target_absorbed: float


@dataclass
class DrugDeliveryRun:
    """Per-simulation orchestrator. Construct, register, ``attach()``."""
    sim: object
    target: WallAbsorber | None = None
    off_targets: list[WallAbsorber] = field(default_factory=list)

    # carrier_type → ReleaseKinetic
    _kinetics_by_type: dict[int, ReleaseKinetic] = field(
        default_factory=dict, repr=False)
    # carrier capsule_id → CarrierState
    _carriers: dict[int, CarrierState] = field(
        default_factory=dict, repr=False)
    # carrier capsule_id → type
    _types_by_id: dict[int, int] = field(
        default_factory=dict, repr=False)

    history: list[RunStepRecord] = field(default_factory=list, repr=False)
    _attached: bool = field(default=False, repr=False)

    # ── Registration API ───────────────────────────────────────────

    def add_carrier_type(self, type_id: int,
                          kinetic: ReleaseKinetic,
                          initial_mass: float) -> None:
        """Register a release kinetic for every capsule of ``type_id``.

        Must be called before ``attach()``. The orchestrator looks up
        every capsule with this type when ``attach()`` runs and binds
        a ``CarrierState`` to it.

        Diffusion-controlled kinetics call ``kinetic.configure_simulation``
        which forwards ``setLeachingParams`` and ``setParticleMass``
        to the C++ Fick path.
        """
        if not (initial_mass > 0.0):
            raise ValueError("initial_mass must be > 0")
        self._kinetics_by_type[int(type_id)] = kinetic
        # Diffusion-controlled gets routed through the C++ chemistry.
        kinetic.configure_simulation(self.sim, int(type_id))
        # Also tell the engine the per-type initial reservoir so
        # ``getCapsuleMp`` returns sensible values from t=0.
        self.sim.setParticleMass(int(type_id), float(initial_mass))
        # Cache the initial mass on the kinetic for later carrier
        # registration.
        kinetic._initial_mass = float(initial_mass)   # type: ignore[attr-defined]

    def add_target(self, absorber: WallAbsorber) -> None:
        absorber.label = absorber.label or "target"
        self.target = absorber

    def add_off_target(self, absorber: WallAbsorber) -> None:
        if not absorber.label or absorber.label == "absorber":
            absorber.label = f"off_target_{len(self.off_targets)}"
        self.off_targets.append(absorber)

    # ── Hook into Simulation ──────────────────────────────────────

    def attach(self) -> None:
        """Bind carriers to existing capsules and register the callback.

        Walks every capsule in ``sim.capsules()``; for each capsule
        whose type has a registered kinetic, allocates a
        ``CarrierState`` with ``M_p = M_p_initial`` from the kinetic.
        """
        if self._attached:
            return
        caps = self.sim.capsules()
        n = caps.numCapsules()
        for cid in range(n):
            cap = caps[cid]
            t   = cap.getType()
            if t not in self._kinetics_by_type:
                continue
            M0 = float(getattr(self._kinetics_by_type[t],
                                "_initial_mass", 1.0))
            self._carriers[cid] = CarrierState(
                capsule_id=cid, M_p=M0, M_p_initial=M0)
            self._types_by_id[cid] = t
        self.sim.setStepCallback(self._step_callback)
        self._attached = True

    # ── Per-step driver ────────────────────────────────────────────

    def _step_callback(self, sim, step):
        params = sim.params()
        dt = float(params.dt)
        time = float(step) * dt

        adr = sim.advectionDiffusion()
        if adr is None:
            return
        scalar = adr.concentration(0)        # writable view (ny, nx)
        nx = sim.params().nx
        ny = sim.params().ny

        caps = sim.capsules()

        # 1. Refresh M_p from the C++ side (only meaningful for
        #    DiffusionControlled — the others manage M_p themselves).
        for cid, st in self._carriers.items():
            t = self._types_by_id[cid]
            if isinstance(self._kinetics_by_type[t], DiffusionControlled):
                Mp_cpp = adr.getCapsuleMp(cid)
                if Mp_cpp >= 0.0:
                    st.M_p = float(Mp_cpp)

        # 2 + 3. Probe + kinetic update + spread.
        total_released_this_step = 0.0
        for cid, st in self._carriers.items():
            kinetic = self._kinetics_by_type[self._types_by_id[cid]]
            cap = caps[cid]
            c   = cap.centroid()

            # Concentration probe (clamped sample at nearest cell).
            ix = int(np.clip(c.x, 0.0, nx - 1.0))
            iy = int(np.clip(c.y, 0.0, ny - 1.0))
            C0 = float(scalar[iy, ix])
            shear = self._estimate_shear(sim, ix, iy)

            probe = FluidProbe(
                time=time, centroid=(c.x, c.y),
                concentration=C0, shear_rate=shear,
                species_concs=(C0,))
            released = kinetic.update(st, probe, dt)

            if released > 0.0 and not isinstance(kinetic, DiffusionControlled):
                # Spread released mass across a 3×3 window centred on
                # the carrier. This is a *coarse* spreader, not the
                # Peskin 4-pt kernel; sufficient for relative
                # comparisons. See Limitations in the module docstring.
                self._spread(scalar, ix, iy, released, nx, ny)
                total_released_this_step += released

        # 4. Apply absorbers.
        if self.target is not None:
            self.target.step(scalar, dt)
        for a in self.off_targets:
            a.step(scalar, dt)

        # 5. Bookkeeping.
        total_M_p = sum(st.M_p for st in self._carriers.values())
        total_released = sum(st.cumulative_released
                              for st in self._carriers.values())
        self.history.append(RunStepRecord(
            step=int(step), time=time,
            total_M_p=float(total_M_p),
            total_released=float(total_released),
            target_absorbed=(self.target.cumulative_absorbed
                              if self.target else 0.0),
            off_target_absorbed=sum(a.cumulative_absorbed
                                     for a in self.off_targets),
        ))

    # ── Helpers ────────────────────────────────────────────────────

    @staticmethod
    def _estimate_shear(sim, ix: int, iy: int) -> float:
        """Magnitude of the local rate-of-strain tensor at (ix, iy).

        γ̇ = √(2 D : D), D = ½ (∇u + ∇u^T). We use centred finite
        differences on the LBM velocity field; near the boundary we
        fall back to one-sided differences via clamping.
        """
        try:
            field = sim.lbmSolver().field()
        except AttributeError:
            return 0.0
        nx = field.getNx() if hasattr(field, "getNx") else None
        ny = field.getNy() if hasattr(field, "getNy") else None
        if nx is None or ny is None:
            return 0.0
        ip = min(nx - 1, ix + 1)
        im = max(0,        ix - 1)
        jp = min(ny - 1, iy + 1)
        jm = max(0,        iy - 1)
        denom_x = max(1, ip - im)
        denom_y = max(1, jp - jm)
        # field.getUx / getUy / getRho are bound on LatticeField.
        dux_dx = (field.getUx(ip, iy) - field.getUx(im, iy)) / denom_x
        duy_dy = (field.getUy(ix, jp) - field.getUy(ix, jm)) / denom_y
        dux_dy = (field.getUx(ix, jp) - field.getUx(ix, jm)) / denom_y
        duy_dx = (field.getUy(ip, iy) - field.getUy(im, iy)) / denom_x
        s11 = dux_dx
        s22 = duy_dy
        s12 = 0.5 * (dux_dy + duy_dx)
        return float(np.sqrt(2.0 * (s11 * s11 + s22 * s22 + 2.0 * s12 * s12)))

    @staticmethod
    def _spread(scalar: np.ndarray, ix: int, iy: int,
                mass: float, nx: int, ny: int) -> None:
        """Distribute ``mass`` uniformly across a 3×3 cell window.

        Bounded by the lattice; cells outside the domain receive no
        mass. The actual mass added equals ``mass`` only when the
        whole 3×3 window is in-domain — at corners / edges, less is
        deposited. This is a feature: it prevents over-injecting
        mass into reflected images that the LBM doesn't see.
        """
        i_lo = max(0, ix - 1); i_hi = min(nx, ix + 2)
        j_lo = max(0, iy - 1); j_hi = min(ny, iy + 2)
        n_cells = (i_hi - i_lo) * (j_hi - j_lo)
        if n_cells == 0:
            return
        scalar[j_lo:j_hi, i_lo:i_hi] += mass / n_cells

    # ── Summary at end of run ──────────────────────────────────────

    def summary(self) -> dict:
        """Return a dict of headline numbers for HDF5 export / printing."""
        from .metrics import (
            delivery_efficiency, off_target_fraction,
        )
        carriers = list(self._carriers.values())
        eta = (delivery_efficiency(self.target, carriers)
               if self.target is not None else float("nan"))
        otf = off_target_fraction(self.off_targets, carriers)
        total_loaded = sum(c.M_p_initial for c in carriers)
        total_remaining = sum(c.M_p for c in carriers)
        return {
            "n_carriers":         len(carriers),
            "total_loaded":       float(total_loaded),
            "total_remaining":    float(total_remaining),
            "delivery_efficiency": float(eta),
            "off_target_fraction": float(otf),
            "n_steps_recorded":   len(self.history),
        }
