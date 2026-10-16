"""Tests for pysoftflow.analysis.hdf5_export."""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pytest


HERE = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE / "python"))

from pysoftflow.analysis import SimulationSnapshot                    # noqa: E402
from pysoftflow.analysis.hdf5_export import (                         # noqa: E402
    load_diagnostics_h5,
    load_run_manifest,
    write_diagnostics_h5,
)
from pysoftflow.analysis.mixing import lacey_index                    # noqa: E402
from pysoftflow.analysis.jamming import per_type_contact_stats        # noqa: E402


pytest.importorskip("h5py")


def _make_snap():
    rng = np.random.default_rng(0)
    n = 100
    pts = rng.uniform(0.0, 50.0, size=(n, 2))
    types = (np.arange(n) % 2).astype(np.int64)
    return SimulationSnapshot.from_arrays(
        positions=pts, radii=np.ones(n) * 0.8,
        types=types, step=42, time=4.2,
        box=(50.0, 50.0), periodic_x=True)


def test_round_trip_snapshots_and_diagnostics(tmp_path):
    snap = _make_snap()
    diags = {
        "lacey":  lacey_index(snap, type_a=0, type_b=1, axis="y", n_bins=5),
        "ptypes": per_type_contact_stats(snap, contact_cutoff=2.0),
        "raw_arr": np.arange(10, dtype=np.float64),
    }
    out_path = tmp_path / "diags.h5"

    write_diagnostics_h5(out_path, diags, snapshots=[snap])

    loaded = load_diagnostics_h5(out_path)
    assert "snapshots" in loaded and "diagnostics" in loaded

    # Snapshot fidelity.
    snap_group = loaded["snapshots"]["step_000042"]
    assert np.allclose(snap_group["positions"], snap.positions)
    assert snap_group["_attrs"]["step"] == 42
    assert snap_group["_attrs"]["periodic_x"] in (True, np.True_)

    # Diagnostic group includes both dataclass-derived datasets and
    # the raw array.
    assert "lacey" in loaded["diagnostics"]
    assert "ptypes" in loaded["diagnostics"]
    raw = loaded["diagnostics"]["raw_arr"]
    # raw_arr was an ndarray → wrapped as {"data": ...}.
    assert "data" in raw
    assert np.allclose(raw["data"], np.arange(10))

    # ptypes carries the Z_matrix dataset.
    pt = loaded["diagnostics"]["ptypes"]
    assert "Z_matrix" in pt
    assert pt["Z_matrix"].shape == (2, 2)


def test_overwrite_guard(tmp_path):
    p = tmp_path / "x.h5"
    write_diagnostics_h5(p, {"a": np.zeros(3)})
    with pytest.raises(FileExistsError):
        write_diagnostics_h5(p, {"a": np.zeros(3)})
    write_diagnostics_h5(p, {"a": np.ones(3)}, overwrite=True)
    loaded = load_diagnostics_h5(p)
    assert (loaded["diagnostics"]["a"]["data"] == 1.0).all()


def test_manifest_flattening_and_round_trip(tmp_path):
    manifest = {
        "build": {"git_sha": "abc123", "compiler_id": "AppleClang"},
        "rng_seed": 42,
        "fluid": {"tau": 0.8, "boundary_type": "PERIODIC"},
    }
    p = tmp_path / "with_manifest.h5"
    write_diagnostics_h5(p, {}, manifest=manifest)

    loaded = load_diagnostics_h5(p)
    attrs = loaded["manifest"]["_attrs"]
    assert attrs["build.git_sha"] == "abc123"
    assert attrs["build.compiler_id"] == "AppleClang"
    assert attrs["rng_seed"] == 42
    assert attrs["fluid.tau"] == pytest.approx(0.8)
    assert attrs["fluid.boundary_type"] == "PERIODIC"


def test_load_run_manifest_finds_modern_layout(tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    manifest = {"build": {"git_sha": "xyz"}, "rng_seed": 7}
    (config_dir / "run_manifest.json").write_text(json.dumps(manifest))

    loaded = load_run_manifest(tmp_path)
    assert loaded == manifest


def test_load_run_manifest_finds_legacy_flat_layout(tmp_path):
    manifest = {"hello": "world"}
    (tmp_path / "run_manifest.json").write_text(json.dumps(manifest))
    assert load_run_manifest(tmp_path) == manifest


def test_load_run_manifest_returns_none_when_missing(tmp_path):
    assert load_run_manifest(tmp_path) is None
