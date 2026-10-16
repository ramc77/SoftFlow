"""Load a SoftFlow parameter sweep into a feature/target table for the surrogate.

A sweep is stored as a flat ``sweep_results.npz`` with one 1-D array per
column (e.g. ``severity``, ``gamma_th``, ``eta_deposit``). This module packs a
chosen set of input columns into a feature matrix ``X`` and one output column
into a target vector ``y``, applying per-feature transforms (e.g. ``log10`` for
a threshold that spans decades) so that the Gaussian process sees
well-conditioned inputs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["SweepDataset", "load_sweep"]


@dataclass
class SweepDataset:
    """A packed (feature, target) view of one sweep.

    Attributes
    ----------
    X:
        Feature matrix, shape ``(n_samples, n_features)``, after transforms.
    y:
        Target vector, shape ``(n_samples,)``.
    feature_names:
        Human-readable name of each feature column (post-transform).
    target_name:
        Name of the target column.
    raw_features:
        Untransformed feature matrix, same shape as ``X`` (handy for plotting
        and for reporting an optimum in physical units).
    transforms:
        Mapping ``feature_name -> "log10" | "identity"`` recording what was
        applied, so an optimum found in feature space can be mapped back.
    """

    X: np.ndarray
    y: np.ndarray
    feature_names: list[str]
    target_name: str
    raw_features: np.ndarray
    transforms: dict[str, str]

    @property
    def n_samples(self) -> int:
        return int(self.X.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.X.shape[1])

    def bounds(self) -> np.ndarray:
        """Per-feature ``[min, max]`` box in (transformed) feature space.

        Returns
        -------
        numpy.ndarray
            Shape ``(n_features, 2)``.
        """
        lo = self.X.min(axis=0)
        hi = self.X.max(axis=0)
        return np.column_stack([lo, hi])

    def invert_features(self, x: np.ndarray) -> np.ndarray:
        """Map a point from transformed feature space back to physical units.

        Parameters
        ----------
        x:
            A point (or array of points) in transformed feature space.

        Returns
        -------
        numpy.ndarray
            The same point(s) with each ``log10`` feature exponentiated back.
        """
        x = np.atleast_2d(np.asarray(x, dtype=float)).copy()
        for j, name in enumerate(self.feature_names):
            if self.transforms.get(name) == "log10":
                x[:, j] = 10.0 ** x[:, j]
        return x.squeeze()


def load_sweep(
    npz_path: str,
    features: tuple[str, ...] = ("severity", "gamma_th"),
    target: str = "eta_deposit",
    log_features: tuple[str, ...] = ("gamma_th",),
    drop_nan_target: bool = True,
) -> SweepDataset:
    """Load a ``sweep_results.npz`` into a :class:`SweepDataset`.

    Parameters
    ----------
    npz_path:
        Path to the ``sweep_results.npz`` produced by a sweep driver.
    features:
        Input column names to use as features, in order.
    target:
        Output column name to predict.
    log_features:
        Subset of ``features`` to transform with ``log10`` (use for quantities
        that span decades, e.g. an activation threshold). A ``"log10_"`` prefix
        is added to the corresponding feature name.
    drop_nan_target:
        If True, rows whose target is NaN are dropped (some diagnostics such as
        a time-to-half metric are undefined for control cells).

    Returns
    -------
    SweepDataset

    Raises
    ------
    KeyError
        If a requested feature or target column is absent from the file.
    """
    data = np.load(npz_path, allow_pickle=True)
    available = set(data.files)
    for col in (*features, target):
        if col not in available:
            raise KeyError(
                f"column {col!r} not in {npz_path} (have: {sorted(available)})"
            )

    raw_cols = [np.asarray(data[name], dtype=float) for name in features]
    raw_features = np.column_stack(raw_cols)

    feat_cols = []
    feature_names: list[str] = []
    transforms: dict[str, str] = {}
    for name, col in zip(features, raw_cols):
        if name in log_features:
            if np.any(col <= 0.0):
                raise ValueError(f"cannot log10-transform {name!r}: non-positive values")
            feat_cols.append(np.log10(col))
            new_name = f"log10_{name}"
            transforms[new_name] = "log10"
            feature_names.append(new_name)
        else:
            feat_cols.append(col)
            transforms[name] = "identity"
            feature_names.append(name)

    X = np.column_stack(feat_cols)
    y = np.asarray(data[target], dtype=float)

    if drop_nan_target:
        keep = ~np.isnan(y)
        X, y, raw_features = X[keep], y[keep], raw_features[keep]

    return SweepDataset(
        X=X,
        y=y,
        feature_names=feature_names,
        target_name=target,
        raw_features=raw_features,
        transforms=transforms,
    )
