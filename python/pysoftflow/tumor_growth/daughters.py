"""Daughter placement for dividing capsules.

**Coarse-grained mechano-chemical proxy. Not a validated cancer
model.** Phase-5 strong-language constraint repeated here.

When a parent capsule divides, ``DaughterPlacer.propose(...)`` looks
for a non-overlapping spot in a ring around the parent. We try
``max_attempts`` random angles at the same ring radius; the first
candidate that clears walls, obstacles, and the existing capsule
field by ``min_gap`` is returned. If every angle fails, we return
``None`` and the orchestrator skips this division this step.

The overlap logic is intentionally a small Python copy of Phase-2's
``isPlacementValid`` — exposing the C++ ``InsertionContext`` to
Python would require a new pybind11 wrapper for marginal benefit, and
the Python version composes naturally with the
``SimulationSnapshot`` data layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


__all__ = ["DaughterPlacement", "DaughterPlacer"]


@dataclass(frozen=True)
class DaughterPlacement:
    """One accepted daughter position."""
    center: tuple[float, float]
    radius: float
    angle:  float    # radians, for diagnostics


class DaughterPlacer:
    """Try to find a non-overlapping daughter position around a parent.

    Parameters
    ----------
    ring_radius_factor : float
        Multiplier on ``parent_radius + daughter_radius`` for the
        candidate ring distance. Default 1.05 → daughter centre sits
        just outside contact distance, leaving a small gap that the
        Phase-1 repulsion + adhesion forces resolve over the next
        few timesteps.
    max_attempts : int
        Number of random angles to try before giving up. Default 12.
    min_gap : float
        Required clearance between daughter and any existing capsule,
        wall, or obstacle. Default 0.5 lattice units.
    daughter_radius_factor : float
        Multiplier on parent radius for daughter size. Default 1.0
        (same size). Some biological models use 0.79 (volume halved
        in 2-D), set ``daughter_radius_factor=0.7937`` to match.
    """

    def __init__(
        self,
        ring_radius_factor: float = 1.05,
        max_attempts: int = 12,
        min_gap: float = 0.5,
        daughter_radius_factor: float = 1.0,
    ):
        if not (ring_radius_factor > 0.0):
            raise ValueError("ring_radius_factor must be > 0")
        if max_attempts <= 0:
            raise ValueError("max_attempts must be > 0")
        if not (min_gap >= 0.0):
            raise ValueError("min_gap must be ≥ 0")
        if not (daughter_radius_factor > 0.0):
            raise ValueError("daughter_radius_factor must be > 0")
        self.ring_radius_factor     = float(ring_radius_factor)
        self.max_attempts           = int(max_attempts)
        self.min_gap                = float(min_gap)
        self.daughter_radius_factor = float(daughter_radius_factor)

    # ── Public API ─────────────────────────────────────────────────

    def propose(
        self,
        parent_pos: tuple[float, float],
        parent_radius: float,
        existing_centers: np.ndarray,
        existing_radii: np.ndarray,
        wall_y_bottom: float,
        wall_y_top: float,
        rng: np.random.Generator,
        *,
        Lx: float = 0.0,
        periodic_x: bool = False,
        obstacle_signed_distances=None,
    ) -> Optional[DaughterPlacement]:
        """Try to find a valid daughter centre. Return None on failure.

        Parameters
        ----------
        parent_pos : (px, py)
            Parent centroid in lattice units.
        parent_radius : float
        existing_centers : (N, 2) array
            Centroids of all existing capsules in the simulation
            (including the parent — the proposer skips self-collision
            implicitly because the offset > 0).
        existing_radii : (N,) array
            Effective radii.
        wall_y_bottom, wall_y_top : float
            Channel wall y-coordinates.
        rng : np.random.Generator
            Caller-owned RNG for reproducibility.
        Lx : float
            Lattice nx; only used when ``periodic_x``.
        periodic_x : bool
            Streamwise wrap.
        obstacle_signed_distances : callable or None
            Optional callable ``(x, y) -> signed_distance`` (positive
            outside the obstacle). One per registered obstacle. Pass
            ``None`` (default) when no obstacles need consideration.
        """
        daughter_r = self.daughter_radius_factor * parent_radius
        # Ring distance accounts for min_gap so the daughter clears
        # the parent's exclusion envelope on placement; otherwise a
        # naive `factor * (r1 + r2)` puts the daughter exactly at the
        # contact boundary and the overlap check rejects it.
        ring_r     = self.ring_radius_factor * (parent_radius + daughter_r
                                                  + self.min_gap)

        for _ in range(self.max_attempts):
            theta = rng.uniform(0.0, 2.0 * np.pi)
            cx = parent_pos[0] + ring_r * np.cos(theta)
            cy = parent_pos[1] + ring_r * np.sin(theta)
            if periodic_x and Lx > 0.0:
                cx = (cx % Lx + Lx) % Lx
            if self._is_valid(
                cx, cy, daughter_r,
                existing_centers, existing_radii,
                wall_y_bottom, wall_y_top,
                Lx, periodic_x,
                obstacle_signed_distances,
            ):
                return DaughterPlacement(center=(float(cx), float(cy)),
                                          radius=float(daughter_r),
                                          angle=float(theta))
        return None

    # ── Internal: overlap check ────────────────────────────────────

    def _is_valid(
        self,
        cx: float, cy: float, r: float,
        existing_centers: np.ndarray,
        existing_radii: np.ndarray,
        wall_y_bottom: float, wall_y_top: float,
        Lx: float, periodic_x: bool,
        obstacle_signed_distances,
    ) -> bool:
        # Walls.
        if cy - r < wall_y_bottom + self.min_gap:
            return False
        if cy + r > wall_y_top - self.min_gap:
            return False

        # Obstacles.
        if obstacle_signed_distances:
            for sd in obstacle_signed_distances:
                if sd(cx, cy) < r + self.min_gap:
                    return False

        # Existing capsules — vectorised distance with periodic-x.
        if existing_centers.size > 0:
            dx = existing_centers[:, 0] - cx
            dy = existing_centers[:, 1] - cy
            if periodic_x and Lx > 0.0:
                dx -= Lx * np.round(dx / Lx)
            d2 = dx * dx + dy * dy
            min_d = r + existing_radii + self.min_gap
            if np.any(d2 < min_d * min_d):
                return False
        return True
