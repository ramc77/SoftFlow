"""Pythonic facade for the SoftFlow particle-insertion module.

The C++ side lives at ``softflow_core.insertion``. This file re-exports
the concrete types under their canonical names and provides factory
helpers that match the API sketch from the project specification::

    from pysoftflow.insertion import (
        Inserter, DynamicInserter,
        RectRegion, CircleRegion, PolygonRegion, ImageMaskRegion,
        SizeDistribution,
    )

    inserter = Inserter.hex_lattice(
        region  = RectRegion(x=(0, 200), y=(10, 90)),
        spacing = 8.0,
        sizes   = SizeDistribution.bidisperse(r_small=2.0, r_large=4.0,
                                              fraction_small=0.5),
        jitter  = 0.1,
    )
    sim.insertCapsules(inserter, mparams=rbc, type=0)

The factory entry points (``Inserter.hex_lattice`` etc.) are thin
wrappers; you can equivalently construct the C++ classes directly.
The factories exist because the project-specification sketch uses keyword tuples
(``x=(0, 200)``) that the raw C++ constructors don't accept.
"""

from __future__ import annotations

import os
import sys
from typing import Sequence, Tuple, Union


def _load_core_insertion():
    """Import the C++ extension and return the insertion submodule.

    Mirrors `pysoftflow.__init__._load_core` so the heavy import
    happens lazily — pure Python users that only need post-processing
    don't pay the cost.
    """
    import softflow_core  # type: ignore
    return softflow_core.insertion


_ins = _load_core_insertion()


# ── Type aliases (re-exports) ───────────────────────────────────────
# These are the C++ classes; users can isinstance-check against them.
Region            = _ins.Region
RectRegion_C      = _ins.RectRegion
CircleRegion_C    = _ins.CircleRegion
PolygonRegion     = _ins.PolygonRegion
ImageMaskRegion_C = _ins.ImageMaskRegion

SizeDistributionBase = _ins.SizeDistribution
Monodisperse_C       = _ins.Monodisperse
Bidisperse_C         = _ins.Bidisperse
Lognormal_C          = _ins.Lognormal
UserDiscrete_C       = _ins.UserDiscrete

InserterBase                = _ins.Inserter
SquareLatticeInserter_C     = _ins.SquareLatticeInserter
HexagonalLatticeInserter_C  = _ins.HexagonalLatticeInserter
RSAInserter_C               = _ins.RSAInserter
PoissonDiskInserter_C       = _ins.PoissonDiskInserter

DynamicInserterBase           = _ins.DynamicInserter
PoissonStochasticInserter_C   = _ins.PoissonStochasticInserter
ConstantFluxInserter_C        = _ins.ConstantFluxInserter
ConveyorInserter_C            = _ins.ConveyorInserter


# ── Pythonic constructors that accept the project specification syntax ────

PairF = Tuple[float, float]


def RectRegion(*, x: PairF, y: PairF):
    """``RectRegion(x=(x0, x1), y=(y0, y1))`` — keyword form preferred."""
    return RectRegion_C(x[0], x[1], y[0], y[1])


def CircleRegion(*, center: Sequence[float], radius: float):
    """``CircleRegion(center=(cx, cy), radius=r)``."""
    from softflow_core import Vec2d
    return CircleRegion_C(Vec2d(float(center[0]), float(center[1])),
                          float(radius))


def ImageMaskRegion_fromPGM(path: str, *, origin: Sequence[float],
                             scale: float, threshold: int = 127):
    """Read a PGM file and wrap it as an ImageMaskRegion."""
    from softflow_core import Vec2d
    if not os.path.isfile(path):
        raise FileNotFoundError(f"PGM file not found: {path}")
    return ImageMaskRegion_C.fromPGM(
        path, Vec2d(float(origin[0]), float(origin[1])),
        float(scale), int(threshold))


# ── SizeDistribution facade ────────────────────────────────────────

class SizeDistribution:
    """Factory namespace for size-distribution constructors.

    Use as ``SizeDistribution.bidisperse(...)`` / ``.lognormal(...)``
    / ``.monodisperse(...)`` / ``.user(...)``. Every factory returns a
    C++ ``ISizeDistribution`` ready to plug into an inserter.
    """

    @staticmethod
    def monodisperse(radius: float):
        return Monodisperse_C(float(radius))

    @staticmethod
    def bidisperse(*, r_small: float, r_large: float,
                   fraction_small: float):
        return Bidisperse_C(float(r_small), float(r_large),
                            float(fraction_small))

    @staticmethod
    def lognormal(*, mu_log: float, sigma_log: float,
                  r_min: float, r_max: float):
        return Lognormal_C(float(mu_log), float(sigma_log),
                           float(r_min), float(r_max))

    @staticmethod
    def user(*, radii: Sequence[float], weights: Sequence[float]):
        return UserDiscrete_C(list(map(float, radii)),
                              list(map(float, weights)))


# ── Inserter facade ────────────────────────────────────────────────

class Inserter:
    """Factory namespace for static-fill inserters.

    Each method returns a C++ inserter object. Pass it to
    ``Simulation.insertCapsules(inserter, mparams, type, …)``.
    """

    @staticmethod
    def square_lattice(*, region, spacing: Union[float, PairF],
                       sizes, jitter: float = 0.0):
        if isinstance(spacing, (int, float)):
            sx = sy = float(spacing)
        else:
            sx, sy = float(spacing[0]), float(spacing[1])
        return SquareLatticeInserter_C(region, sx, sy, sizes, float(jitter))

    @staticmethod
    def hex_lattice(*, region, spacing: float, sizes,
                    jitter: float = 0.0):
        return HexagonalLatticeInserter_C(region, float(spacing), sizes,
                                          float(jitter))

    @staticmethod
    def rsa(*, region, target_count: int, sizes,
            max_attempts: int = 0):
        return RSAInserter_C(region, int(target_count), sizes,
                             int(max_attempts))

    @staticmethod
    def poisson_disk(*, region, r_min: float, sizes, k: int = 30):
        return PoissonDiskInserter_C(region, float(r_min), sizes, int(k))


# ── DynamicInserter facade ─────────────────────────────────────────

class DynamicInserter:
    """Factory namespace for per-step dynamic inserters."""

    @staticmethod
    def poisson(*, region, rate: float, sizes,
                attempts_per_event: int = 16):
        return PoissonStochasticInserter_C(region, float(rate), sizes,
                                            int(attempts_per_event))

    @staticmethod
    def constant_flux(*, region, target_phi: float, sizes,
                      max_per_step: int = 4,
                      attempts_per_event: int = 32):
        return ConstantFluxInserter_C(region, float(target_phi), sizes,
                                       int(max_per_step),
                                       int(attempts_per_event))

    @staticmethod
    def conveyor(*, region, target_count: int, sizes,
                 max_per_step: int = 4,
                 attempts_per_event: int = 32):
        return ConveyorInserter_C(region, int(target_count), sizes,
                                   int(max_per_step),
                                   int(attempts_per_event))


__all__ = [
    "Region",
    "RectRegion", "CircleRegion", "PolygonRegion",
    "ImageMaskRegion_C", "ImageMaskRegion_fromPGM",
    "SizeDistribution", "SizeDistributionBase",
    "Inserter", "InserterBase",
    "DynamicInserter", "DynamicInserterBase",
]
