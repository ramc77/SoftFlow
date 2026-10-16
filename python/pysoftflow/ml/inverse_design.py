"""Inverse design — optimise the surrogate to find the best operating point.

Once a surrogate maps inputs -> outcome, the design question inverts: *which
input maximises (or minimises) the outcome?* This module runs a global search
(SciPy differential evolution, then a local polish) over the surrogate's
predictive mean. Because the surrogate also gives uncertainty, the optimum is
returned with a predictive standard deviation, and the caller is expected to
*confirm the optimum with one full simulation* before trusting it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import differential_evolution, minimize

from .surrogate import GaussianProcessSurrogate

__all__ = ["InverseDesignResult", "optimise_target"]


@dataclass
class InverseDesignResult:
    """Outcome of an inverse-design optimisation.

    Attributes
    ----------
    x:
        Optimal point in (transformed) feature space.
    predicted_value:
        Surrogate mean at the optimum.
    predicted_std:
        Surrogate standard deviation at the optimum (model uncertainty there).
    maximise:
        Whether the target was maximised.
    """

    x: np.ndarray
    predicted_value: float
    predicted_std: float
    maximise: bool


def optimise_target(
    surrogate: GaussianProcessSurrogate,
    bounds: np.ndarray,
    maximise: bool = True,
    seed: int | None = 0,
) -> InverseDesignResult:
    """Find the input that optimises the surrogate's predicted target.

    Parameters
    ----------
    surrogate:
        Fitted GP surrogate.
    bounds:
        Per-feature ``[min, max]`` box (e.g. ``SweepDataset.bounds()``),
        shape ``(d, 2)``.
    maximise:
        Maximise (True) or minimise (False) the predicted target.
    seed:
        Seed for the differential-evolution search (reproducibility).

    Returns
    -------
    InverseDesignResult
    """
    bounds = np.asarray(bounds, dtype=float)
    sign = -1.0 if maximise else 1.0

    def objective(x: np.ndarray) -> float:
        mean = surrogate.predict(np.atleast_2d(x), return_std=False)
        return sign * float(mean[0])

    de = differential_evolution(
        objective,
        bounds=[tuple(b) for b in bounds],
        seed=seed,
        tol=1e-7,
        polish=False,
        updating="deferred",
    )
    # Local polish from the DE optimum.
    local = minimize(objective, de.x, method="L-BFGS-B", bounds=[tuple(b) for b in bounds])
    x_opt = local.x if local.fun <= de.fun else de.x

    mean, std = surrogate.predict(np.atleast_2d(x_opt), return_std=True)
    return InverseDesignResult(
        x=np.asarray(x_opt, dtype=float),
        predicted_value=float(mean[0]),
        predicted_std=float(std[0]),
        maximise=maximise,
    )
