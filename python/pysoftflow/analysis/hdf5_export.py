"""HDF5 export for Phase-3 diagnostics.

Layout
------

A single HDF5 file holds one or many diagnostics for one simulation
run, plus a copy of the run's ``run_manifest.json`` for traceability::

    /
    ├── manifest                (group; attrs copied from run_manifest.json)
    ├── snapshots               (group; one subgroup per saved snapshot)
    │   └── step_NNNNNN
    │       ├── positions       (dataset, (N, 2))
    │       ├── velocities      (dataset, (N, 2))
    │       ├── radii           (dataset, (N,))
    │       ├── types           (dataset, (N,))
    │       └── attrs: step, time, Lx, Ly, periodic_x
    └── diagnostics             (group; one subgroup per diagnostic)
        ├── lacey_index/...
        ├── per_type_contact_stats/...
        └── ...

Each diagnostic subgroup carries an ``attrs["definition"]`` short
text label and any scalar attributes; numpy arrays go in datasets.

This module does not depend on the C++ extension — it only consumes
``SimulationSnapshot`` and result dataclasses from sibling submodules.
"""

from __future__ import annotations

import dataclasses
import json
import os
import pathlib
from typing import Any, Iterable, Mapping

import numpy as np

try:
    import h5py
except ImportError as exc:    # pragma: no cover
    raise ImportError(
        "pysoftflow.analysis.hdf5_export requires h5py "
        "(install with `pip install h5py`)"
    ) from exc

from .snapshot import SimulationSnapshot


__all__ = [
    "write_diagnostics_h5",
    "load_diagnostics_h5",
    "load_run_manifest",
]


# ── manifest helpers ───────────────────────────────────────────────


def load_run_manifest(output_dir: str | os.PathLike) -> dict | None:
    """Locate and parse a ``run_manifest.json`` next to a simulation's
    output directory. Returns the parsed dict or None if not found.

    Searches both the modern layout (``output_dir/config/run_manifest.json``)
    and the legacy flat layout (``output_dir/run_manifest.json``).
    """
    p = pathlib.Path(output_dir)
    for candidate in (p / "config" / "run_manifest.json",
                      p / "run_manifest.json"):
        if candidate.is_file():
            with candidate.open() as f:
                return json.load(f)
    return None


def _flatten_for_attrs(d: Mapping[str, Any], prefix: str = "") -> dict:
    """Flatten a nested manifest into ``"a.b.c"`` keyed scalars / strings.

    HDF5 attrs cannot hold nested dicts; we serialise leaf scalars
    directly and JSON-encode anything else (lists, etc.). Strings are
    stored as Python str.
    """
    out: dict = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(_flatten_for_attrs(v, key))
        elif isinstance(v, (int, float, bool, str)):
            out[key] = v
        elif isinstance(v, list):
            # JSON-encode (HDF5 has no first-class list-of-mixed type).
            out[key] = json.dumps(v)
        else:
            out[key] = repr(v)
    return out


# ── snapshot writer ────────────────────────────────────────────────


def _write_snapshot_group(group: "h5py.Group",
                          snap: SimulationSnapshot) -> None:
    group.create_dataset("positions",  data=snap.positions)
    group.create_dataset("velocities", data=snap.velocities)
    group.create_dataset("radii",      data=snap.radii)
    group.create_dataset("types",      data=snap.types)
    if snap.bonds.shape[0] > 0:
        group.create_dataset("bonds",  data=snap.bonds)
    group.attrs["step"]       = int(snap.step)
    group.attrs["time"]       = float(snap.time)
    group.attrs["Lx"]         = float(snap.Lx)
    group.attrs["Ly"]         = float(snap.Ly)
    group.attrs["periodic_x"] = bool(snap.periodic_x)


# ── diagnostic writer ──────────────────────────────────────────────


def _write_diagnostic_group(group: "h5py.Group",
                            name: str,
                            value: Any,
                            *,
                            definition: str | None = None) -> None:
    """Write one diagnostic to its own subgroup.

    Three accepted value shapes:
      - dataclass instance   → each field becomes a dataset or attr.
      - np.ndarray           → single dataset named ``"data"``.
      - dict[str, ...]       → recurse: each item becomes a dataset
                                or scalar attr.
    """
    sub = group.create_group(name)
    if definition is not None:
        sub.attrs["definition"] = definition

    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        for field in dataclasses.fields(value):
            v = getattr(value, field.name)
            _store_field(sub, field.name, v)
    elif isinstance(value, np.ndarray):
        sub.create_dataset("data", data=value)
    elif isinstance(value, Mapping):
        for k, v in value.items():
            _store_field(sub, str(k), v)
    else:
        # Best-effort: store as a scalar attr.
        sub.attrs["value"] = value


def _store_field(parent: "h5py.Group", name: str, value: Any) -> None:
    if isinstance(value, np.ndarray):
        parent.create_dataset(name, data=value)
    elif isinstance(value, (list, tuple)):
        parent.create_dataset(name, data=np.asarray(value))
    elif isinstance(value, (int, float, bool, str)):
        parent.attrs[name] = value
    elif value is None:
        parent.attrs[name] = "None"
    elif dataclasses.is_dataclass(value) and not isinstance(value, type):
        # Nested dataclass (e.g. MixingIndex inside a snap-level dict)
        # → recurse, creating a subgroup. Same shape as a top-level
        # diagnostic so loaders see a consistent tree.
        _write_diagnostic_group(parent, name, value)
    elif isinstance(value, Mapping):
        # Nested dict → recurse as a subgroup.
        _write_diagnostic_group(parent, name, value)
    else:
        # Fall back to repr for unknown types.
        parent.attrs[name] = repr(value)


# ── public API ─────────────────────────────────────────────────────


def write_diagnostics_h5(
    path: str | os.PathLike,
    diagnostics: Mapping[str, Any],
    *,
    snapshots: Iterable[SimulationSnapshot] | None = None,
    manifest: Mapping[str, Any] | None = None,
    overwrite: bool = False,
) -> None:
    """Write diagnostics + optional snapshots + optional manifest to HDF5.

    Parameters
    ----------
    path : path-like
        Output HDF5 file. Refuses to overwrite an existing file unless
        ``overwrite=True``.
    diagnostics : mapping name → (dataclass | ndarray | mapping)
        Each entry lands under ``/diagnostics/<name>``.
    snapshots : iterable of SimulationSnapshot, optional
        One subgroup per snapshot under ``/snapshots/step_NNNNNN``.
    manifest : mapping, optional
        The run manifest (typically loaded via ``load_run_manifest``).
        Stored as flattened attrs under ``/manifest`` for traceability.
    overwrite : bool
        Replace an existing file if True.
    """
    p = pathlib.Path(path)
    if p.exists() and not overwrite:
        raise FileExistsError(
            f"{p} already exists; pass overwrite=True to replace it.")

    with h5py.File(p, "w") as f:
        if manifest is not None:
            mg = f.create_group("manifest")
            for k, v in _flatten_for_attrs(manifest).items():
                mg.attrs[k] = v

        if snapshots is not None:
            sg = f.create_group("snapshots")
            for snap in snapshots:
                gname = f"step_{int(snap.step):06d}"
                _write_snapshot_group(sg.create_group(gname), snap)

        dg = f.create_group("diagnostics")
        for name, value in diagnostics.items():
            _write_diagnostic_group(dg, name, value)


def load_diagnostics_h5(path: str | os.PathLike) -> dict:
    """Inverse of ``write_diagnostics_h5``: returns a nested dict tree.

    Datasets become numpy arrays; group attrs become a sibling
    ``"_attrs"`` dict in their parent. The schema is intentionally
    loose — this is for inspection, not strict round-trip parity with
    the dataclasses (which would require re-importing each one).
    """
    out: dict = {}
    with h5py.File(path, "r") as f:
        out = _h5_group_to_dict(f)
    return out


def _h5_group_to_dict(group: "h5py.Group") -> dict:
    d: dict = {}
    if len(group.attrs) > 0:
        d["_attrs"] = {k: v for k, v in group.attrs.items()}
    for key in group:
        item = group[key]
        if isinstance(item, h5py.Group):
            d[key] = _h5_group_to_dict(item)
        else:
            d[key] = np.asarray(item[()])
    return d
