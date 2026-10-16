"""A from-scratch Gaussian-process surrogate (NumPy + SciPy only).

Why a Gaussian process here
---------------------------
A SoftFlow sweep gives O(10) labelled points. A GP is the right tool at that
scale: it interpolates smoothly, returns a *calibrated* predictive variance
(which `active_learning` uses to choose the next simulation), and its
automatic-relevance-determination (ARD) length-scales provide an interpretable
feature-importance measure for free.

Model
-----
Zero-mean GP with an ARD squared-exponential (RBF) kernel plus i.i.d. noise:

    k(x, x') = sigma_f^2 * exp( -0.5 * sum_d (x_d - x'_d)^2 / l_d^2 )
    K = k(X, X) + sigma_n^2 I

Inputs are z-scored and the target is standardised before fitting, so the
hyperparameters live on a comparable scale across features. Hyperparameters
(l_d, sigma_f, sigma_n) are fit by maximising the log marginal likelihood with
L-BFGS-B over log-parameters, with a few random restarts.

References
----------
  - Rasmussen & Williams, *Gaussian Processes for Machine Learning*
    (MIT Press, 2006): kernel and marginal likelihood (ch. 2), ARD (ch. 5.1),
    closed-form leave-one-out CV (eqs 5.10-5.12).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize

__all__ = ["GaussianProcessSurrogate", "LOOResult"]


@dataclass
class LOOResult:
    """Leave-one-out cross-validation summary (in physical target units).

    Attributes
    ----------
    predictions:
        LOO predictive mean for each training point.
    std:
        LOO predictive standard deviation for each training point.
    rmse:
        Root-mean-square LOO error.
    mae:
        Mean absolute LOO error.
    r2:
        LOO coefficient of determination.
    standardised_residuals:
        ``(y - pred) / std`` per point; should be ~N(0, 1) if the model's
        uncertainty is well calibrated (most magnitudes < 2).
    """

    predictions: np.ndarray
    std: np.ndarray
    rmse: float
    mae: float
    r2: float
    standardised_residuals: np.ndarray


class GaussianProcessSurrogate:
    """ARD-RBF Gaussian-process regressor with marginal-likelihood fitting.

    Parameters
    ----------
    jitter:
        Diagonal added to the kernel for numerical stability.
    n_restarts:
        Number of random restarts for the hyperparameter optimisation.
    random_state:
        Seed for restart sampling (reproducibility).
    """

    def __init__(
        self,
        jitter: float = 1e-8,
        n_restarts: int = 8,
        random_state: int | None = 0,
    ) -> None:
        self.jitter = float(jitter)
        self.n_restarts = int(n_restarts)
        self._rng = np.random.default_rng(random_state)
        self._fitted = False

    # ── public API ──────────────────────────────────────────────────────

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GaussianProcessSurrogate":
        """Fit hyperparameters by maximising the log marginal likelihood.

        Parameters
        ----------
        X:
            Feature matrix, shape ``(n, d)``.
        y:
            Target vector, shape ``(n,)``.

        Returns
        -------
        GaussianProcessSurrogate
            ``self`` (fitted).
        """
        X = np.atleast_2d(np.asarray(X, dtype=float))
        y = np.asarray(y, dtype=float).ravel()
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y have inconsistent number of samples")

        self._x_mean = X.mean(axis=0)
        self._x_std = X.std(axis=0)
        self._x_std[self._x_std == 0.0] = 1.0
        self._y_mean = float(y.mean())
        self._y_std = float(y.std())
        if self._y_std == 0.0:
            self._y_std = 1.0

        self._Xs = (X - self._x_mean) / self._x_std
        self._ys = (y - self._y_mean) / self._y_std
        self._n, self._d = self._Xs.shape

        # theta = [log l_1..l_d, log sigma_f, log sigma_n]
        best_theta, best_nll = None, np.inf
        for theta0 in self._initial_thetas():
            res = minimize(
                self._neg_log_marginal_likelihood,
                theta0,
                method="L-BFGS-B",
                bounds=self._theta_bounds(),
            )
            if res.fun < best_nll:
                best_nll, best_theta = float(res.fun), res.x

        self._theta = best_theta
        self._lengthscales = np.exp(best_theta[: self._d])
        self._signal_var = float(np.exp(best_theta[self._d]) ** 2)
        self._noise_var = float(np.exp(best_theta[self._d + 1]) ** 2)
        self.log_marginal_likelihood = -best_nll

        # Cache the factorisation at the optimum for prediction / LOO.
        K = self._kernel(self._Xs, self._Xs, self._lengthscales, self._signal_var)
        K[np.diag_indices_from(K)] += self._noise_var + self.jitter
        self._chol = cho_factor(K, lower=True)
        self._alpha = cho_solve(self._chol, self._ys)
        self._K = K
        self._fitted = True
        return self

    def predict(
        self, X: np.ndarray, return_std: bool = True
    ) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
        """Predict the target at new inputs (physical units).

        Parameters
        ----------
        X:
            Query points, shape ``(m, d)``.
        return_std:
            If True, also return the predictive standard deviation.

        Returns
        -------
        numpy.ndarray or (numpy.ndarray, numpy.ndarray)
            Predictive mean (and standard deviation if requested).
        """
        self._check_fitted()
        X = np.atleast_2d(np.asarray(X, dtype=float))
        Xs = (X - self._x_mean) / self._x_std

        Ks = self._kernel(Xs, self._Xs, self._lengthscales, self._signal_var)
        mean_s = Ks @ self._alpha
        mean = mean_s * self._y_std + self._y_mean
        if not return_std:
            return mean

        v = cho_solve(self._chol, Ks.T)  # (n, m)
        var_s = self._signal_var - np.einsum("ij,ji->i", Ks, v)
        var_s = np.clip(var_s, 0.0, None)
        std = np.sqrt(var_s) * self._y_std
        return mean, std

    def leave_one_out(self) -> LOOResult:
        """Closed-form leave-one-out cross-validation at the fitted optimum.

        Uses the standard GP-LOO identities (Rasmussen & Williams eqs 5.10-5.12)
        which reuse the cached factorisation, so all ``n`` folds cost one solve.

        Returns
        -------
        LOOResult
        """
        self._check_fitted()
        K_inv = cho_solve(self._chol, np.eye(self._n))
        diag = np.diag(K_inv)
        mu_loo_s = self._ys - self._alpha / diag
        var_loo_s = 1.0 / diag

        pred = mu_loo_s * self._y_std + self._y_mean
        std = np.sqrt(np.clip(var_loo_s, 0.0, None)) * self._y_std
        y_true = self._ys * self._y_std + self._y_mean

        err = y_true - pred
        rmse = float(np.sqrt(np.mean(err**2)))
        mae = float(np.mean(np.abs(err)))
        ss_res = float(np.sum(err**2))
        ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        with np.errstate(divide="ignore", invalid="ignore"):
            zres = np.where(std > 0, err / std, np.nan)

        return LOOResult(
            predictions=pred,
            std=std,
            rmse=rmse,
            mae=mae,
            r2=r2,
            standardised_residuals=zres,
        )

    @property
    def lengthscales(self) -> np.ndarray:
        """Fitted ARD length-scales (on z-scored inputs), shape ``(d,)``."""
        self._check_fitted()
        return self._lengthscales.copy()

    @property
    def signal_var(self) -> float:
        """Fitted signal variance ``sigma_f^2`` (standardised target units)."""
        self._check_fitted()
        return self._signal_var

    @property
    def noise_var(self) -> float:
        """Fitted noise variance ``sigma_n^2`` (standardised target units)."""
        self._check_fitted()
        return self._noise_var

    def feature_importance(self, feature_names: list[str] | None = None) -> dict[str, float]:
        """ARD-based relevance per feature (sums to 1; larger = more important).

        Relevance is the inverse length-scale on z-scored inputs: a short
        length-scale means the output changes quickly along that feature, so it
        is more important.

        Parameters
        ----------
        feature_names:
            Optional names; defaults to ``x0, x1, ...``.

        Returns
        -------
        dict[str, float]
        """
        self._check_fitted()
        relevance = 1.0 / self._lengthscales
        relevance = relevance / relevance.sum()
        names = feature_names or [f"x{i}" for i in range(self._d)]
        return {name: float(r) for name, r in zip(names, relevance)}

    # ── internals ───────────────────────────────────────────────────────

    @staticmethod
    def _kernel(
        A: np.ndarray, B: np.ndarray, lengthscales: np.ndarray, signal_var: float
    ) -> np.ndarray:
        """ARD squared-exponential kernel between rows of A and B."""
        Aw = A / lengthscales
        Bw = B / lengthscales
        sq = (
            np.sum(Aw**2, axis=1)[:, None]
            + np.sum(Bw**2, axis=1)[None, :]
            - 2.0 * Aw @ Bw.T
        )
        sq = np.clip(sq, 0.0, None)
        return signal_var * np.exp(-0.5 * sq)

    def _neg_log_marginal_likelihood(self, theta: np.ndarray) -> float:
        lengthscales = np.exp(theta[: self._d])
        signal_var = np.exp(theta[self._d]) ** 2
        noise_var = np.exp(theta[self._d + 1]) ** 2
        K = self._kernel(self._Xs, self._Xs, lengthscales, signal_var)
        K[np.diag_indices_from(K)] += noise_var + self.jitter
        try:
            chol = cho_factor(K, lower=True)
        except np.linalg.LinAlgError:
            return 1e25
        alpha = cho_solve(chol, self._ys)
        log_det = 2.0 * np.sum(np.log(np.diag(chol[0])))
        nll = 0.5 * self._ys @ alpha + 0.5 * log_det + 0.5 * self._n * np.log(2 * np.pi)
        return float(nll)

    def _theta_bounds(self) -> list[tuple[float, float]]:
        # Bounds in log space; inputs/targets are standardised so O(1) scales.
        ls = [(np.log(0.05), np.log(20.0))] * self._d
        sf = [(np.log(1e-2), np.log(10.0))]
        sn = [(np.log(1e-4), np.log(2.0))]
        return ls + sf + sn

    def _initial_thetas(self) -> list[np.ndarray]:
        thetas = [np.array([0.0] * self._d + [0.0, np.log(0.1)])]  # sensible default
        bounds = np.array(self._theta_bounds())
        for _ in range(self.n_restarts):
            lo, hi = bounds[:, 0], bounds[:, 1]
            thetas.append(lo + (hi - lo) * self._rng.random(len(lo)))
        return thetas

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("surrogate is not fitted; call .fit(X, y) first")
