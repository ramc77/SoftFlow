"""Flavor-B graph neural network: predict a collective outcome from a
suspension configuration graph (nodes = capsules, edges = near-contacts).

This is the *graph-property prediction* GNN: each :class:`ParticleGraph`
(from :mod:`pysoftflow.ml.graph_extraction`) is classified/regressed to a
graph-level label such as clog / no-clog or a segregation index. Unlike a
dynamical learned simulator it does no autoregressive rollout, so it is
data-efficient and stable.

This module is the *only* part of :mod:`pysoftflow.ml` that requires PyTorch
and PyTorch-Geometric, so it is imported explicitly rather than from the
package ``__init__`` (keeping the surrogate / graph-extraction layers
dependency-light):

    from pysoftflow.ml.gnn import train_clog_gnn

The message-passing layer consumes the **edge features** (surface gap,
relative speed, approach rate) — these carry the arch-formation signal, so
they matter for clog prediction.

References
----------
  - Gilmer et al., *Neural Message Passing for Quantum Chemistry*, ICML 2017.
  - Hu et al., *Strategies for Pre-training GNNs* (GINE edge-aware conv), 2020.
  - Ma et al., *HIGNN* (suspensions as graphs), CMAME 400, 115496 (2022).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Hard dependency: this module is GNN-only. Importing it without torch /
# torch_geometric raises a clear error (the rest of pysoftflow.ml does not
# import this module, so it stays optional).
import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing, global_max_pool, global_mean_pool

from .graph_extraction import ParticleGraph

__all__ = ["ClogGNN", "graphs_to_pyg", "train_clog_gnn", "fit_and_score", "GNNTrainResult"]


# ---------------------------------------------------------------------------
# Data conversion
# ---------------------------------------------------------------------------

def graphs_to_pyg(graphs: list[ParticleGraph]) -> list[Data]:
    """Convert :class:`ParticleGraph` objects to PyG ``Data`` (drops unlabeled)."""
    data = []
    for g in graphs:
        if g.graph_label is None:
            continue
        d = g.to_pyg()
        if d.edge_index.numel() == 0:
            # Self-loops keep message passing well-defined for contact-free
            # frames (which legitimately occur in the no-clog class).
            n = d.x.size(0)
            d.edge_index = torch.arange(n).repeat(2, 1)
            d.edge_attr = torch.zeros((n, len(g.edge_feature_names)), dtype=torch.float32)
        data.append(d)
    return data


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class _EdgeMLPConv(MessagePassing):
    """Edge-conditioned message passing: message = MLP([x_i, x_j, edge_attr])."""

    def __init__(self, hidden: int, edge_dim: int) -> None:
        super().__init__(aggr="mean")
        self.msg_mlp = nn.Sequential(
            nn.Linear(2 * hidden + edge_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),
        )
        self.upd_mlp = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),
        )

    def forward(self, x, edge_index, edge_attr):
        agg = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        return self.upd_mlp(torch.cat([x, agg], dim=-1))

    def message(self, x_i, x_j, edge_attr):
        return self.msg_mlp(torch.cat([x_i, x_j, edge_attr], dim=-1))


class ClogGNN(nn.Module):
    """Edge-aware GNN for graph-level classification of suspension configs.

    Parameters
    ----------
    node_dim, edge_dim:
        Per-node / per-edge feature dimensions (9 and 6 for the default
        :func:`pysoftflow.ml.graph_extraction.snapshot_to_graph`).
    hidden:
        Hidden width.
    n_layers:
        Number of message-passing rounds.
    """

    def __init__(self, node_dim: int = 9, edge_dim: int = 6,
                 hidden: int = 48, n_layers: int = 3) -> None:
        super().__init__()
        self.node_encoder = nn.Linear(node_dim, hidden)
        self.convs = nn.ModuleList(_EdgeMLPConv(hidden, edge_dim) for _ in range(n_layers))
        self.norms = nn.ModuleList(nn.LayerNorm(hidden) for _ in range(n_layers))
        # Readout: [mean-pool || max-pool] -> logit.
        self.head = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden, 1),
        )

    def forward(self, data: Data) -> torch.Tensor:
        x = self.node_encoder(data.x)
        for conv, norm in zip(self.convs, self.norms):
            x = x + norm(torch.relu(conv(x, data.edge_index, data.edge_attr)))
        pooled = torch.cat(
            [global_mean_pool(x, data.batch), global_max_pool(x, data.batch)], dim=-1
        )
        return self.head(pooled).squeeze(-1)  # logits, shape (batch,)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

@dataclass
class GNNTrainResult:
    """Outcome of a GNN training run."""

    model: ClogGNN
    train_acc: float
    val_acc: float
    val_balanced_acc: float  # mean per-class recall (imbalance-robust)
    val_auc: float
    history: list[tuple[int, float, float]]  # (epoch, train_loss, val_acc)


def _auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Area under the ROC curve via rank statistic (no sklearn dependency)."""
    pos = scores[y_true == 1]
    neg = scores[y_true == 0]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, scores.size + 1)
    r_pos = ranks[y_true == 1].sum()
    return float((r_pos - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size))


def train_clog_gnn(
    graphs: list[ParticleGraph],
    *,
    epochs: int = 120,
    hidden: int = 48,
    n_layers: int = 3,
    lr: float = 3e-3,
    batch_size: int = 16,
    val_fraction: float = 0.25,
    seed: int = 0,
) -> GNNTrainResult:
    """Train :class:`ClogGNN` to classify graphs by their binary label.

    Parameters
    ----------
    graphs:
        Labeled :class:`ParticleGraph` objects (label 0/1, e.g. no-clog/clog).
    epochs, hidden, n_layers, lr, batch_size, val_fraction, seed:
        Training hyperparameters.

    Returns
    -------
    GNNTrainResult
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    data = graphs_to_pyg(graphs)
    if len(data) < 4:
        raise ValueError(f"need at least 4 labeled graphs, got {len(data)}")
    node_dim = data[0].x.size(1)
    edge_dim = data[0].edge_attr.size(1)

    idx = rng.permutation(len(data))
    n_val = max(1, int(len(data) * val_fraction))
    val_idx, train_idx = set(idx[:n_val].tolist()), idx[n_val:].tolist()
    train_set = [data[i] for i in train_idx]
    val_set = [data[i] for i in sorted(val_idx)]

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)

    model = ClogGNN(node_dim, edge_dim, hidden=hidden, n_layers=n_layers)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    # Class weighting for imbalance: BCEWithLogitsLoss up-weights the positive
    # (clog) class by pos_weight = n_neg / n_pos, so the minority class is not
    # ignored (otherwise the model collapses to predicting the majority class).
    y_train = np.array([float(d.y) for d in train_set])
    n_pos = max(1.0, float((y_train == 1).sum()))
    n_neg = max(1.0, float((y_train == 0).sum()))
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    history: list[tuple[int, float, float]] = []
    for epoch in range(epochs):
        model.train()
        total = 0.0
        for batch in train_loader:
            opt.zero_grad()
            logits = model(batch)
            loss = loss_fn(logits, batch.y.float().view(-1))
            loss.backward()
            opt.step()
            total += float(loss) * batch.num_graphs
        train_loss = total / max(1, len(train_set))
        if epoch % 10 == 0 or epoch == epochs - 1:
            history.append((epoch, train_loss, _accuracy(model, val_loader)))

    train_acc = _accuracy(model, DataLoader(train_set, batch_size=batch_size))
    val_acc = _accuracy(model, val_loader)
    val_bal = _balanced_accuracy(model, val_loader)
    y_true, scores = _scores(model, val_loader)
    return GNNTrainResult(model, train_acc, val_acc, val_bal, _auc(y_true, scores), history)


def fit_and_score(
    train_graphs: list[ParticleGraph],
    val_graphs: list[ParticleGraph],
    *,
    epochs: int = 120,
    hidden: int = 48,
    n_layers: int = 3,
    lr: float = 3e-3,
    batch_size: int = 16,
    seed: int = 0,
) -> dict:
    """Train on ``train_graphs`` and evaluate on a held-out ``val_graphs`` set.

    Used for *group-level* (e.g. leave-one-run-out) cross-validation, where the
    validation graphs come from runs entirely absent from training — the honest
    generalization test, free of the frame-correlation leakage that a random
    split over pooled frames suffers.

    Returns ``{"acc", "balanced_acc", "auc"}`` on the validation set.
    """
    torch.manual_seed(seed)
    tr = graphs_to_pyg(train_graphs)
    va = graphs_to_pyg(val_graphs)
    nan = {"acc": float("nan"), "balanced_acc": float("nan"), "auc": float("nan")}
    if len(tr) < 2 or len(va) < 1:
        return nan

    node_dim, edge_dim = tr[0].x.size(1), tr[0].edge_attr.size(1)
    y_tr = np.array([float(d.y) for d in tr])
    n_pos = max(1.0, float((y_tr == 1).sum()))
    n_neg = max(1.0, float((y_tr == 0).sum()))

    model = ClogGNN(node_dim, edge_dim, hidden=hidden, n_layers=n_layers)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([n_neg / n_pos], dtype=torch.float32))
    loader = DataLoader(tr, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(va, batch_size=batch_size)

    for _ in range(epochs):
        model.train()
        for batch in loader:
            opt.zero_grad()
            loss = loss_fn(model(batch), batch.y.float().view(-1))
            loss.backward()
            opt.step()

    y_true, scores = _scores(model, val_loader)
    return {
        "acc": _accuracy(model, val_loader),
        "balanced_acc": _balanced_accuracy(model, val_loader),
        "auc": _auc(y_true, scores),
    }


@torch.no_grad()
def _scores(model: ClogGNN, loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    ys, ss = [], []
    for batch in loader:
        ss.append(torch.sigmoid(model(batch)).cpu().numpy())
        ys.append(batch.y.float().view(-1).cpu().numpy())
    return np.concatenate(ys), np.concatenate(ss)


def _accuracy(model: ClogGNN, loader: DataLoader) -> float:
    y, s = _scores(model, loader)
    if y.size == 0:
        return float("nan")
    return float(((s > 0.5).astype(float) == y).mean())


def _balanced_accuracy(model: ClogGNN, loader: DataLoader) -> float:
    """Mean of per-class recall — robust to class imbalance."""
    y, s = _scores(model, loader)
    if y.size == 0:
        return float("nan")
    pred = (s > 0.5).astype(float)
    recalls = [float((pred[y == c] == c).mean()) for c in (0.0, 1.0) if (y == c).any()]
    return float(np.mean(recalls)) if recalls else float("nan")
