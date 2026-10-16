"""SimulationSnapshot — universal data layer for Phase-3 diagnostics.

A ``SimulationSnapshot`` is a frozen view of a simulation at one
timestep, containing only the per-particle quantities the diagnostics
need (positions, velocities, radii, types, optional adhesion bond
list). Every diagnostic in this package takes a ``SimulationSnapshot``
(or an iterable of them, for time-series quantities) and returns plain
numpy arrays or simple dataclasses — no diagnostic mutates the
simulation or carries hidden state.

Two extractors:

  - ``SimulationSnapshot.from_simulation(sim)`` — call directly during
    a Python-driven run.
  - ``SimulationSnapshot.from_arrays(...)`` — build from numpy arrays
    you already have (e.g. parsed from a CSV trajectory file).

Centroid velocities are computed by averaging the LBM-interpolated
node velocities exposed by ``Capsule.nodeVelocity``. This is the same
quantity the engine uses to advect each capsule.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class SimulationSnapshot:
    """Frozen per-particle view of a simulation at one timestep.

    Attributes
    ----------
    step : int
        Discrete simulation step that produced this snapshot.
    time : float
        Physical time in lattice units (= step * dt, in the canonical
        run loop).
    positions : np.ndarray, shape (N, 2)
        Capsule centroids in lattice coordinates.
    velocities : np.ndarray, shape (N, 2)
        Centroid velocities, averaged from LBM-interpolated node
        velocities. Filled with NaN if the engine is not currently
        coupled to the LBM (e.g. immediately after a fresh insert).
    radii : np.ndarray, shape (N,)
        Effective radii.
    types : np.ndarray, shape (N,)
        Integer type labels (matches ``Capsule.getType()``).
    box : tuple[float, float]
        Lattice box dimensions ``(Lx, Ly)``. The streamwise length
        equals ``params.nx``; the cross-stream length equals
        ``params.ny``.
    periodic_x : bool
        True if ``params.fluid.boundary_type == PERIODIC``. Diagnostics
        use this to apply minimum-image distance under streamwise wrap.
    bonds : np.ndarray, shape (M, 2)
        Adhesion bond pair indices into the capsule list. Wall bonds
        (``capsule_j == -1`` or ``-2``) are excluded — only inter-
        capsule bonds appear here. Empty array if no adhesion model
        is registered.
    """

    step: int
    time: float
    positions: np.ndarray
    velocities: np.ndarray
    radii: np.ndarray
    types: np.ndarray
    box: tuple[float, float]
    periodic_x: bool
    bonds: np.ndarray = field(default_factory=lambda: np.empty((0, 2), dtype=np.int64))

    # ── Construction helpers ────────────────────────────────────────

    @classmethod
    def from_simulation(cls, sim) -> "SimulationSnapshot":
        """Pull a snapshot from a live ``softflow_core.Simulation``.

        Reads centroid positions, type labels, and effective radii
        from the capsule system; computes centroid velocities by
        averaging node velocities; copies adhesion bond pairs (skip
        wall bonds) if ``sim.adhesion()`` returns a non-null model.
        """
        params = sim.params()
        caps   = sim.capsules()
        n      = caps.numCapsules()

        positions  = np.empty((n, 2), dtype=np.float64)
        velocities = np.empty((n, 2), dtype=np.float64)
        radii      = np.empty(n,      dtype=np.float64)
        types      = np.empty(n,      dtype=np.int64)

        for i in range(n):
            cap = caps[i]
            c   = cap.centroid()
            positions[i, 0] = c.x
            positions[i, 1] = c.y
            radii[i]        = cap.effectiveRadius()
            types[i]        = cap.getType()
            # Centroid velocity = mean of node velocities.
            nn = cap.numNodes()
            if nn > 0:
                vx = vy = 0.0
                for k in range(nn):
                    v = cap.nodeVelocity(k)
                    vx += v.x
                    vy += v.y
                velocities[i, 0] = vx / nn
                velocities[i, 1] = vy / nn
            else:
                velocities[i] = np.nan

        # Adhesion bonds (capsule indices only, wall bonds dropped).
        bonds = np.empty((0, 2), dtype=np.int64)
        adh   = sim.adhesion() if hasattr(sim, "adhesion") else None
        if adh is not None:
            raw = adh.getBonds()
            pairs = []
            for b in raw:
                if b.capsule_j < 0:
                    continue   # wall bond
                if b.capsule_i == b.capsule_j:
                    continue   # self-bond (defensive; should not occur)
                pairs.append((b.capsule_i, b.capsule_j))
            if pairs:
                bonds = np.asarray(pairs, dtype=np.int64)

        # Box / periodicity info.
        from softflow_core import BoundaryType
        periodic_x = (params.fluid.boundary_type == BoundaryType.PERIODIC)
        box = (float(params.nx), float(params.ny))

        return cls(
            step=int(sim.currentStep()),
            time=float(sim.currentStep()) * float(params.dt),
            positions=positions,
            velocities=velocities,
            radii=radii,
            types=types,
            box=box,
            periodic_x=periodic_x,
            bonds=bonds,
        )

    @classmethod
    def from_arrays(
        cls,
        *,
        step: int = 0,
        time: float = 0.0,
        positions: np.ndarray,
        radii: np.ndarray,
        types: np.ndarray,
        velocities: Optional[np.ndarray] = None,
        box: tuple[float, float] = (0.0, 0.0),
        periodic_x: bool = False,
        bonds: Optional[np.ndarray] = None,
    ) -> "SimulationSnapshot":
        """Build a snapshot directly from numpy arrays.

        Useful for unit tests and for replaying CSV trajectories.
        Validates shapes; coerces dtypes; defaults ``velocities`` to
        zeros and ``bonds`` to an empty (0, 2) array.
        """
        positions = np.ascontiguousarray(positions, dtype=np.float64)
        radii     = np.ascontiguousarray(radii,     dtype=np.float64)
        types     = np.ascontiguousarray(types,     dtype=np.int64)
        n = positions.shape[0]
        if positions.ndim != 2 or positions.shape[1] != 2:
            raise ValueError("positions must have shape (N, 2)")
        if radii.shape != (n,):
            raise ValueError("radii must have shape (N,)")
        if types.shape != (n,):
            raise ValueError("types must have shape (N,)")
        if velocities is None:
            velocities = np.zeros((n, 2), dtype=np.float64)
        else:
            velocities = np.ascontiguousarray(velocities, dtype=np.float64)
            if velocities.shape != (n, 2):
                raise ValueError("velocities must have shape (N, 2)")
        if bonds is None:
            bonds = np.empty((0, 2), dtype=np.int64)
        else:
            bonds = np.ascontiguousarray(bonds, dtype=np.int64)
            if bonds.ndim != 2 or bonds.shape[1] != 2:
                raise ValueError("bonds must have shape (M, 2)")

        return cls(
            step=int(step), time=float(time),
            positions=positions, velocities=velocities,
            radii=radii, types=types,
            box=(float(box[0]), float(box[1])),
            periodic_x=bool(periodic_x),
            bonds=bonds,
        )

    # ── Convenience views ──────────────────────────────────────────

    @property
    def n_particles(self) -> int:
        return int(self.positions.shape[0])

    @property
    def Lx(self) -> float:
        return float(self.box[0])

    @property
    def Ly(self) -> float:
        return float(self.box[1])

    def by_type(self, t: int) -> np.ndarray:
        """Boolean mask selecting particles of type ``t``."""
        return self.types == int(t)

    def distinct_types(self) -> np.ndarray:
        """Sorted array of distinct type labels present in the snapshot."""
        return np.unique(self.types)

    # ── Internal: shared distance helpers ──────────────────────────

    def pairwise_distance(self, i: int, j: int) -> float:
        """Distance between particles i and j with streamwise minimum-image
        wrap when ``periodic_x``. Used by diagnostics that don't justify
        building a full KDTree (small synthetic tests).
        """
        dx = self.positions[i, 0] - self.positions[j, 0]
        dy = self.positions[i, 1] - self.positions[j, 1]
        if self.periodic_x and self.Lx > 0.0:
            if dx >  0.5 * self.Lx:
                dx -= self.Lx
            elif dx < -0.5 * self.Lx:
                dx += self.Lx
        return float(np.sqrt(dx * dx + dy * dy))
