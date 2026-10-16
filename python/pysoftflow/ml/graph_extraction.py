"""Graph extraction — turn a capsule trajectory into a dataset of graphs.

This is the data layer for "Flavor B": a graph neural network that predicts a
*collective* outcome (clogging, segregation, arch formation) from a suspension
configuration, with **particles as nodes and near-contacts as edges**.

Each simulation frame becomes one labelled graph:

    nodes  = capsules, with features (position, velocity, wall distance,
             type, effective radius, contact degree)
    edges  = pairs within contact range, with features (centre distance,
             surface gap, unit direction, relative speed, approach rate)
    label  = a graph-level target supplied by the caller (e.g. a clog flag,
             a segregation index, or the cell's deposition efficiency)

The contact criterion matches the analysis package convention used by the
jamming diagnostics (two capsules touch iff
``d_ij < r_i + r_j + contact_cutoff``) with streamwise minimum-image wrap when
periodic, so the graph topology is consistent with ``analysis.jamming``.

Dependencies: NumPy + SciPy only. ``ParticleGraph.to_pyg`` lazily imports
PyTorch-Geometric so the rest of the module stays dependency-free; the GNN
model itself (a later step) is the only piece that will require torch.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from ..analysis.snapshot import SimulationSnapshot

__all__ = [
    "ParticleGraph",
    "contact_pairs",
    "snapshot_to_graph",
    "load_particle_csv",
    "build_graph_dataset",
    "save_graph_dataset",
    "load_graph_dataset",
]


@dataclass
class ParticleGraph:
    """One configuration graph: nodes = capsules, edges = near-contacts.

    Attributes
    ----------
    node_features:
        Shape ``(N, F)`` array of per-capsule features.
    node_feature_names:
        Names of the ``F`` node-feature columns.
    edge_index:
        Shape ``(2, E)`` array of directed edges (each undirected contact
        appears as both ``i->j`` and ``j->i``), as expected by message-passing
        GNN libraries.
    edge_features:
        Shape ``(E, Fe)`` array of per-edge features (aligned with
        ``edge_index`` columns).
    edge_feature_names:
        Names of the ``Fe`` edge-feature columns.
    graph_label:
        Optional graph-level target (clog flag, segregation index, ...).
    step:
        Simulation step that produced this frame.
    time:
        Simulation time of this frame (lattice units).
    """

    node_features: np.ndarray
    node_feature_names: list[str]
    edge_index: np.ndarray
    edge_features: np.ndarray
    edge_feature_names: list[str]
    graph_label: float | None
    step: int
    time: float

    @property
    def n_nodes(self) -> int:
        return int(self.node_features.shape[0])

    @property
    def n_edges(self) -> int:
        return int(self.edge_index.shape[1])

    @property
    def mean_degree(self) -> float:
        """Average node degree (twice the undirected contacts / N)."""
        if self.n_nodes == 0:
            return 0.0
        return float(self.edge_index.shape[1] / self.n_nodes)

    def to_pyg(self):
        """Convert to a ``torch_geometric.data.Data`` object (lazy import).

        Raises
        ------
        ImportError
            If PyTorch / PyTorch-Geometric are not installed. This keeps the
            extractor dependency-free; only the eventual GNN needs torch.
        """
        try:
            import torch
            from torch_geometric.data import Data
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "ParticleGraph.to_pyg() needs torch + torch_geometric; "
                "install them only when you build the GNN model."
            ) from exc

        y = None if self.graph_label is None else torch.tensor(
            [float(self.graph_label)], dtype=torch.float32
        )
        return Data(
            x=torch.tensor(self.node_features, dtype=torch.float32),
            edge_index=torch.tensor(self.edge_index, dtype=torch.long),
            edge_attr=torch.tensor(self.edge_features, dtype=torch.float32),
            y=y,
        )


def _min_image_delta(
    p_i: np.ndarray, p_j: np.ndarray, Lx: float, periodic_x: bool
) -> np.ndarray:
    """Vector ``p_j - p_i`` with streamwise minimum-image wrap if periodic."""
    d = p_j - p_i
    if periodic_x and Lx > 0.0:
        if d[0] > 0.5 * Lx:
            d[0] -= Lx
        elif d[0] < -0.5 * Lx:
            d[0] += Lx
    return d


def contact_pairs(
    snap: SimulationSnapshot, contact_cutoff: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """Find contacting capsule pairs and their separation vectors.

    A pair (i, j) is in contact iff ``d_ij <= r_i + r_j + contact_cutoff``
    (the same rule as ``analysis.patterns._build_contact_graph``), with
    streamwise minimum-image distance when ``snap.periodic_x``.

    Parameters
    ----------
    snap:
        Frame to extract contacts from.
    contact_cutoff:
        Surface gap (in lattice units) below which two capsules count as in
        contact. ``~1`` LU matches the lubrication near-contact scale.

    Returns
    -------
    (numpy.ndarray, numpy.ndarray)
        ``edges`` of shape ``(M, 2)`` (i < j, undirected) and ``deltas`` of
        shape ``(M, 2)`` giving the minimum-image vector ``p_j - p_i`` for each
        pair.
    """
    n = snap.n_particles
    if n < 2:
        return np.empty((0, 2), dtype=np.int64), np.empty((0, 2), dtype=np.float64)

    r_max = float(np.max(snap.radii))
    query_radius = 2.0 * r_max + contact_cutoff
    Lx = snap.Lx if snap.periodic_x else 0.0

    if Lx > 0.0:
        pts = np.vstack(
            [
                snap.positions,
                snap.positions + np.array([Lx, 0.0]),
                snap.positions + np.array([-Lx, 0.0]),
            ]
        )
    else:
        pts = snap.positions

    tree = cKDTree(pts)
    seen: set[tuple[int, int]] = set()
    edges: list[tuple[int, int]] = []
    deltas: list[np.ndarray] = []
    for i, p in enumerate(snap.positions):
        for j_raw in tree.query_ball_point(p, r=query_radius):
            j = j_raw % n
            if j == i:
                continue
            a, b = (i, j) if i < j else (j, i)
            if (a, b) in seen:
                continue
            delta = _min_image_delta(snap.positions[a], snap.positions[b], Lx, Lx > 0.0)
            dist = float(np.hypot(delta[0], delta[1]))
            if dist <= snap.radii[a] + snap.radii[b] + contact_cutoff:
                seen.add((a, b))
                edges.append((a, b))
                deltas.append(delta)

    if not edges:
        return np.empty((0, 2), dtype=np.int64), np.empty((0, 2), dtype=np.float64)
    return np.asarray(edges, dtype=np.int64), np.asarray(deltas, dtype=np.float64)


def snapshot_to_graph(
    snap: SimulationSnapshot,
    *,
    contact_cutoff: float = 1.0,
    label: float | None = None,
) -> ParticleGraph:
    """Build a :class:`ParticleGraph` from a single frame.

    Parameters
    ----------
    snap:
        The frame.
    contact_cutoff:
        Surface-gap threshold for an edge (lattice units).
    label:
        Optional graph-level target to attach.

    Returns
    -------
    ParticleGraph
    """
    n = snap.n_particles
    edges, deltas = contact_pairs(snap, contact_cutoff=contact_cutoff)

    # Directed edge list (both orientations) for message passing.
    if edges.shape[0] > 0:
        src = np.concatenate([edges[:, 0], edges[:, 1]])
        dst = np.concatenate([edges[:, 1], edges[:, 0]])
        edge_index = np.vstack([src, dst])
        # Delta for the reverse direction is negated.
        delta_dir = np.vstack([deltas, -deltas])
    else:
        edge_index = np.empty((2, 0), dtype=np.int64)
        delta_dir = np.empty((0, 2), dtype=np.float64)

    degree = np.bincount(edge_index[0], minlength=n).astype(np.float64) if n else np.zeros(0)

    # ── Node features ───────────────────────────────────────────────
    Lx, Ly = snap.Lx, snap.Ly
    cx, cy = snap.positions[:, 0], snap.positions[:, 1]
    vx, vy = snap.velocities[:, 0], snap.velocities[:, 1]
    speed = np.hypot(vx, vy)
    wall_dist_norm = (
        np.minimum(cy, Ly - cy) / (0.5 * Ly) if Ly > 0 else np.zeros(n)
    )
    node_features = np.column_stack(
        [
            cx / Lx if Lx > 0 else cx,
            cy / Ly if Ly > 0 else cy,
            vx,
            vy,
            speed,
            wall_dist_norm,
            snap.types.astype(np.float64),
            snap.radii,
            degree,
        ]
    )
    node_feature_names = [
        "x_norm",
        "y_norm",
        "vx",
        "vy",
        "speed",
        "wall_dist_norm",
        "type",
        "radius",
        "contact_degree",
    ]

    # ── Edge features ───────────────────────────────────────────────
    if edge_index.shape[1] > 0:
        dist = np.hypot(delta_dir[:, 0], delta_dir[:, 1])
        ri = snap.radii[edge_index[0]]
        rj = snap.radii[edge_index[1]]
        gap = dist - (ri + rj)
        with np.errstate(invalid="ignore", divide="ignore"):
            ux = np.where(dist > 0, delta_dir[:, 0] / dist, 0.0)
            uy = np.where(dist > 0, delta_dir[:, 1] / dist, 0.0)
        dvx = snap.velocities[edge_index[1], 0] - snap.velocities[edge_index[0], 0]
        dvy = snap.velocities[edge_index[1], 1] - snap.velocities[edge_index[0], 1]
        rel_speed = np.hypot(dvx, dvy)
        # Positive approach rate => the pair is closing (relevant to arching).
        approach_rate = -(dvx * ux + dvy * uy)
        edge_features = np.column_stack([dist, gap, ux, uy, rel_speed, approach_rate])
    else:
        edge_features = np.empty((0, 6), dtype=np.float64)
    edge_feature_names = ["distance", "gap", "dir_x", "dir_y", "rel_speed", "approach_rate"]

    return ParticleGraph(
        node_features=node_features,
        node_feature_names=node_feature_names,
        edge_index=edge_index,
        edge_features=edge_features,
        edge_feature_names=edge_feature_names,
        graph_label=label,
        step=int(snap.step),
        time=float(snap.time),
    )


def load_particle_csv(
    path: str,
    *,
    radius: float | Mapping[int, float],
    box: tuple[float, float],
    periodic_x: bool,
    frame_stride: int = 1,
    max_frames: int | None = None,
) -> list[SimulationSnapshot]:
    """Load a ``particle_data.csv`` into a list of :class:`SimulationSnapshot`.

    The CSV layout is the SoftFlow per-capsule trajectory dump with columns
    ``timestep, time, capsule_id, type, cx, cy, vx, vy``. Rows are grouped by
    ``timestep`` into frames.

    Parameters
    ----------
    path:
        Path to ``particle_data.csv``.
    radius:
        Effective capsule radius (not stored in the CSV). Either a scalar
        applied to all capsules, or a mapping ``type -> radius``.
    box:
        Domain size ``(Lx, Ly)`` in lattice units.
    periodic_x:
        Whether the streamwise direction is periodic.
    frame_stride:
        Keep every ``frame_stride``-th distinct timestep (1 = all frames).
    max_frames:
        Optional cap on the number of frames returned.

    Returns
    -------
    list[SimulationSnapshot]
    """
    table = np.genfromtxt(path, delimiter=",", names=True)
    steps = table["timestep"].astype(np.int64)
    unique_steps = np.unique(steps)[::frame_stride]
    if max_frames is not None:
        unique_steps = unique_steps[:max_frames]

    def radius_for(types: np.ndarray) -> np.ndarray:
        if isinstance(radius, Mapping):
            return np.array([float(radius[int(t)]) for t in types], dtype=np.float64)
        return np.full(types.shape[0], float(radius), dtype=np.float64)

    snapshots: list[SimulationSnapshot] = []
    for s in unique_steps:
        m = steps == s
        types = table["type"][m].astype(np.int64)
        snap = SimulationSnapshot.from_arrays(
            step=int(s),
            time=float(table["time"][m][0]),
            positions=np.column_stack([table["cx"][m], table["cy"][m]]),
            velocities=np.column_stack([table["vx"][m], table["vy"][m]]),
            radii=radius_for(types),
            types=types,
            box=box,
            periodic_x=periodic_x,
        )
        snapshots.append(snap)
    return snapshots


def build_graph_dataset(
    csv_path: str,
    *,
    radius: float | Mapping[int, float],
    box: tuple[float, float],
    periodic_x: bool,
    contact_cutoff: float = 1.0,
    frame_stride: int = 1,
    max_frames: int | None = None,
    label=None,
) -> list[ParticleGraph]:
    """Build a list of :class:`ParticleGraph` from one trajectory CSV.

    Parameters
    ----------
    csv_path:
        Path to a ``particle_data.csv``.
    radius, box, periodic_x:
        Passed to :func:`load_particle_csv`.
    contact_cutoff:
        Surface-gap threshold for edges.
    frame_stride, max_frames:
        Frame subsampling controls.
    label:
        Graph-level target. Either ``None``, a scalar applied to every frame,
        or a callable ``label(snapshot) -> float`` evaluated per frame (use
        this to attach a per-frame diagnostic such as a jamming index).

    Returns
    -------
    list[ParticleGraph]
    """
    snaps = load_particle_csv(
        csv_path,
        radius=radius,
        box=box,
        periodic_x=periodic_x,
        frame_stride=frame_stride,
        max_frames=max_frames,
    )
    graphs: list[ParticleGraph] = []
    for snap in snaps:
        if callable(label):
            lab = float(label(snap))
        else:
            lab = label
        graphs.append(snapshot_to_graph(snap, contact_cutoff=contact_cutoff, label=lab))
    return graphs


def save_graph_dataset(graphs: list[ParticleGraph], path: str) -> None:
    """Save a list of graphs to a single ``.npz`` (ragged, object arrays).

    The format is dependency-free (NumPy only). Reload with
    :func:`load_graph_dataset`.
    """
    if not graphs:
        raise ValueError("nothing to save: empty graph list")

    def _ragged(values: list) -> np.ndarray:
        # Build a 1-D object array element-by-element. ``np.array(list, object)``
        # mis-broadcasts when the per-graph arrays share a leading dimension but
        # differ in the trailing one (e.g. edge_index (2, n_edges) with varying
        # n_edges), so assign explicitly.
        arr = np.empty(len(values), dtype=object)
        for i, v in enumerate(values):
            arr[i] = v
        return arr

    np.savez(
        path,
        node_features=_ragged([g.node_features for g in graphs]),
        edge_index=_ragged([g.edge_index for g in graphs]),
        edge_features=_ragged([g.edge_features for g in graphs]),
        graph_label=_ragged([g.graph_label for g in graphs]),
        step=np.array([g.step for g in graphs], dtype=np.int64),
        time=np.array([g.time for g in graphs], dtype=np.float64),
        node_feature_names=np.array(graphs[0].node_feature_names, dtype=object),
        edge_feature_names=np.array(graphs[0].edge_feature_names, dtype=object),
    )


def load_graph_dataset(path: str) -> list[ParticleGraph]:
    """Reload a dataset written by :func:`save_graph_dataset`."""
    d = np.load(path, allow_pickle=True)
    node_names = list(d["node_feature_names"])
    edge_names = list(d["edge_feature_names"])
    graphs: list[ParticleGraph] = []
    for k in range(d["step"].shape[0]):
        lab = d["graph_label"][k]
        graphs.append(
            ParticleGraph(
                node_features=d["node_features"][k],
                node_feature_names=node_names,
                edge_index=d["edge_index"][k],
                edge_features=d["edge_features"][k],
                edge_feature_names=edge_names,
                graph_label=None if lab is None else float(lab),
                step=int(d["step"][k]),
                time=float(d["time"][k]),
            )
        )
    return graphs
