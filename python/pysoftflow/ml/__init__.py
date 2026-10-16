"""Reduced-order ML layer — surrogate modelling, active learning, inverse design.

This package turns a SoftFlow parameter *sweep* (e.g. a 4x4 stenosis grid) into a cheap, queryable model of
the input -> outcome map. It is intentionally **dependency-light**: only
NumPy and SciPy (both already required by SoftFlow). No PyTorch / scikit-learn.

What it provides
----------------
    dataset          — load a ``sweep_results.npz`` into a feature/target table
    surrogate        — a from-scratch Gaussian-process regressor with an
                       ARD-RBF kernel, marginal-likelihood hyperparameter fit,
                       closed-form leave-one-out CV, and ARD feature importance
    active_learning  — acquisition functions (max-variance / expected
                       improvement) that suggest the next simulation to run
    inverse_design   — optimise the surrogate to find the input that maximises
                       (or minimises) a chosen outcome

Design intent
-------------
With only ~16 labelled sweep cells, the right model is a Gaussian process,
not a deep network: it is data-efficient, gives calibrated uncertainty (which
drives active learning), and its ARD length-scales double as an interpretable
feature-importance measure. Deep models (CNN / LSTM / GNN) become feasible
only after active learning has grown the dataset.

References
----------
  - Rasmussen & Williams, *Gaussian Processes for Machine Learning*
    (MIT Press, 2006) — kernel, marginal likelihood, LOO (eqs 5.10-5.12).
"""

from __future__ import annotations

from .dataset import SweepDataset, load_sweep
from .surrogate import GaussianProcessSurrogate, LOOResult
from .active_learning import expected_improvement, max_variance, suggest_next
from .inverse_design import InverseDesignResult, optimise_target
from .graph_extraction import (
    ParticleGraph,
    build_graph_dataset,
    contact_pairs,
    load_graph_dataset,
    load_particle_csv,
    save_graph_dataset,
    snapshot_to_graph,
)

__all__ = [
    # ── surrogate / active learning / inverse design (Stage 1) ──
    "SweepDataset",
    "load_sweep",
    "GaussianProcessSurrogate",
    "LOOResult",
    "expected_improvement",
    "max_variance",
    "suggest_next",
    "InverseDesignResult",
    "optimise_target",
    # ── graph extraction for the Flavor-B GNN (Stage 3 data layer) ──
    "ParticleGraph",
    "snapshot_to_graph",
    "contact_pairs",
    "load_particle_csv",
    "build_graph_dataset",
    "save_graph_dataset",
    "load_graph_dataset",
]
