"""Active learning — let the surrogate choose the next simulation to run.

A SoftFlow cell costs ~1 hour of compute, so which point to simulate next
matters. Given a fitted :class:`~pysoftflow.ml.surrogate.GaussianProcessSurrogate`,
these acquisition functions score candidate points so the most *informative*
(or most *promising*) one can be run next. Iterating this is how a 16-point
sweep grows into the hundreds of points that deeper models would need.

Two acquisitions are provided:

    max_variance        — pure exploration: pick where the model is least sure
                          (best for building an accurate global surrogate).
    expected_improvement — exploration/exploitation trade-off: pick where the
                          target is likely to beat the current best (best when
                          the goal is to *find an optimum*, i.e. inverse design).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from .surrogate import GaussianProcessSurrogate

__all__ = ["candidate_grid", "max_variance", "expected_improvement", "suggest_next"]


def candidate_grid(bounds: np.ndarray, n_per_dim: int = 40) -> np.ndarray:
    """Build a dense regular grid of candidate points over a box.

    Parameters
    ----------
    bounds:
        Per-feature ``[min, max]``, shape ``(d, 2)``.
    n_per_dim:
        Number of grid points per dimension.

    Returns
    -------
    numpy.ndarray
        Candidate points, shape ``(n_per_dim ** d, d)``.
    """
    bounds = np.asarray(bounds, dtype=float)
    axes = [np.linspace(lo, hi, n_per_dim) for lo, hi in bounds]
    mesh = np.meshgrid(*axes, indexing="ij")
    return np.column_stack([m.ravel() for m in mesh])


def max_variance(
    surrogate: GaussianProcessSurrogate, candidates: np.ndarray
) -> tuple[int, np.ndarray, float]:
    """Pick the candidate with the largest predictive standard deviation.

    Returns
    -------
    (int, numpy.ndarray, float)
        Index into ``candidates``, the chosen point, and its predictive std.
    """
    _, std = surrogate.predict(candidates, return_std=True)
    idx = int(np.argmax(std))
    return idx, candidates[idx], float(std[idx])


def expected_improvement(
    surrogate: GaussianProcessSurrogate,
    candidates: np.ndarray,
    y_best: float,
    maximise: bool = True,
    xi: float = 0.01,
) -> np.ndarray:
    """Expected-improvement acquisition over candidate points.

    Parameters
    ----------
    surrogate:
        Fitted GP surrogate.
    candidates:
        Points to score, shape ``(m, d)``.
    y_best:
        Best target observed so far.
    maximise:
        Whether the target is being maximised (True) or minimised (False).
    xi:
        Exploration margin (larger -> more exploratory).

    Returns
    -------
    numpy.ndarray
        EI value at each candidate (>= 0), shape ``(m,)``.
    """
    mean, std = surrogate.predict(candidates, return_std=True)
    std = np.clip(std, 1e-12, None)
    improvement = (mean - y_best - xi) if maximise else (y_best - mean - xi)
    z = improvement / std
    ei = improvement * norm.cdf(z) + std * norm.pdf(z)
    return np.clip(ei, 0.0, None)


def suggest_next(
    surrogate: GaussianProcessSurrogate,
    bounds: np.ndarray,
    mode: str = "explore",
    y_best: float | None = None,
    maximise: bool = True,
    n_per_dim: int = 60,
) -> dict:
    """Suggest the next point to simulate.

    Parameters
    ----------
    surrogate:
        Fitted GP surrogate.
    bounds:
        Per-feature ``[min, max]`` box, shape ``(d, 2)``.
    mode:
        ``"explore"`` (max variance) or ``"optimise"`` (expected improvement).
    y_best:
        Required for ``"optimise"`` mode: best observed target so far.
    maximise:
        Direction of optimisation for ``"optimise"`` mode.
    n_per_dim:
        Candidate-grid resolution per dimension.

    Returns
    -------
    dict
        ``{"point", "score", "mode"}`` where ``point`` is in feature space.
    """
    candidates = candidate_grid(bounds, n_per_dim=n_per_dim)
    if mode == "explore":
        idx, point, score = max_variance(surrogate, candidates)
        return {"point": point, "score": score, "mode": "explore (max-variance std)"}
    if mode == "optimise":
        if y_best is None:
            raise ValueError("mode='optimise' requires y_best")
        ei = expected_improvement(surrogate, candidates, y_best, maximise=maximise)
        idx = int(np.argmax(ei))
        return {
            "point": candidates[idx],
            "score": float(ei[idx]),
            "mode": "optimise (expected improvement)",
        }
    raise ValueError(f"unknown mode {mode!r}; use 'explore' or 'optimise'")
