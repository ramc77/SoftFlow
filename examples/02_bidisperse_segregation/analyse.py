"""Phase-3 diagnostic walk over the bidisperse-segregation example.

Builds the same simulation as `run.py`, runs a short trajectory,
samples snapshots periodically, computes every Phase-3 diagnostic
on each snapshot, and writes the lot to a single HDF5 file alongside
a copy of the run's `run_manifest.json` for traceability.

Use this as the template for your own analysis scripts: build sim,
collect snapshots, call diagnostics, write HDF5.

Usage::

    python examples/02_bidisperse_segregation/analyse.py [--smoke]

`--smoke` runs ~30 steps (5 snapshots) for CI; otherwise ~2000 steps
(11 snapshots) for a meaningful trajectory.
"""

from __future__ import annotations

import importlib.util
import os
import pathlib
import sys


HERE = pathlib.Path(__file__).resolve()
PROJECT = HERE.parents[2]

sys.path.insert(0, str(PROJECT / "python"))
for build_dir in ["build_phase2", "build", "build_phase1"]:
    cand = PROJECT / build_dir / "python"
    if cand.is_dir():
        sys.path.insert(0, str(cand))
        break

from pysoftflow.analysis import SimulationSnapshot                # noqa: E402
from pysoftflow.analysis.mixing import (                          # noqa: E402
    contact_asymmetry, danckwerts_intensity, lacey_index,
)
from pysoftflow.analysis.rdf import radial_distribution           # noqa: E402
from pysoftflow.analysis.patterns import (                        # noqa: E402
    cluster_persistence, hoshen_kopelman, lane_order,
)
from pysoftflow.analysis.jamming import (                         # noqa: E402
    contact_number, force_percolation, mean_squared_displacement,
    non_affine_d2min, packing_field, per_type_contact_stats,
)
from pysoftflow.analysis.hdf5_export import (                     # noqa: E402
    load_run_manifest, write_diagnostics_h5,
)


def _load_example():
    spec = importlib.util.spec_from_file_location(
        "bidisperse_run",
        PROJECT / "examples" / "02_bidisperse_segregation" / "run.py",
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def diagnostics_on(snap: SimulationSnapshot) -> dict:
    """Run every Phase-3 single-snapshot diagnostic. Time-series ones
    (MSD, persistence, D²_min) are run separately on the snapshot list.
    """
    return {
        "lacey_y": lacey_index(snap, type_a=0, type_b=1, axis="y", n_bins=8),
        "lacey_x": lacey_index(snap, type_a=0, type_b=1, axis="x", n_bins=8),
        "danckwerts_y": danckwerts_intensity(
            snap, type_a=0, type_b=1, axis="y", n_bins=8),
        "contact_asymmetry": contact_asymmetry(
            snap, type_a=0, type_b=1, contact_cutoff=0.5),
        "rdf_AA": radial_distribution(snap, type_a=0, type_b=0, n_bins=40),
        "rdf_AB": radial_distribution(snap, type_a=0, type_b=1, n_bins=40),
        "rdf_BB": radial_distribution(snap, type_a=1, type_b=1, n_bins=40),
        "lane_order_x": lane_order(snap, axis="x"),
        "clusters": hoshen_kopelman(snap, contact_cutoff=0.5),
        "packing_field": packing_field(snap, n_x=20, n_y=10),
        "contact_number": contact_number(snap, contact_cutoff=0.5),
        "per_type_contact_stats": per_type_contact_stats(
            snap, contact_cutoff=0.5),
        "force_percolation": force_percolation(
            snap, contact_cutoff=0.5, band_fraction=0.1),
    }


def main(argv=None):
    argv = argv or sys.argv[1:]
    smoke = "--smoke" in argv

    n_steps      = 30 if smoke else 2000
    n_snapshots  = 5  if smoke else 11
    sample_every = max(1, n_steps // (n_snapshots - 1))

    out_dir = (str(pathlib.Path(os.getenv("TMPDIR", "/tmp"))
                   / "softflow_phase3_smoke")
               if smoke else
               str(PROJECT / "output" / "02_bidisperse_segregation"))
    h5_path = pathlib.Path(out_dir) / "diagnostics.h5"
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)

    print(f"Building simulation (steps = {n_steps}, sampling every "
          f"{sample_every} steps, output = {out_dir}) …")
    ex = _load_example()
    sim, n_small, n_large = ex.build_simulation(
        num_steps=n_steps, output_dir=out_dir, rng_seed=0xBEEFCAFE)
    sim.initialize()

    snapshots: list[SimulationSnapshot] = []
    for s in range(n_steps):
        sim.step()
        if (s + 1) % sample_every == 0 or s == n_steps - 1:
            snap = SimulationSnapshot.from_simulation(sim)
            snapshots.append(snap)
            print(f"  step {s + 1:5d}: snapshot {len(snapshots)}, "
                  f"|u|_max={sim.getMaxSpeed():.4g}")

    sim.finalize()
    print(f"Collected {len(snapshots)} snapshots; computing diagnostics …")

    # Per-snapshot diagnostics.
    diagnostics: dict = {}
    for k, snap in enumerate(snapshots):
        diagnostics[f"snap_{k:03d}_step_{snap.step:06d}"] = diagnostics_on(snap)

    # Time-series diagnostics across the whole snapshot list.
    if len(snapshots) >= 3:
        diagnostics["msd"] = mean_squared_displacement(snapshots)

    if len(snapshots) >= 2:
        a = hoshen_kopelman(snapshots[0], contact_cutoff=0.5).labels
        b = hoshen_kopelman(snapshots[-1], contact_cutoff=0.5).labels
        if a.shape == b.shape:
            diagnostics["cluster_persistence_first_to_last"] = (
                cluster_persistence(a, b))

        diagnostics["d2min_first_to_last"] = non_affine_d2min(
            snapshots[0], snapshots[-1], neighbour_cutoff=8.0)

    manifest = load_run_manifest(out_dir)

    write_diagnostics_h5(h5_path, diagnostics,
                          snapshots=snapshots,
                          manifest=manifest,
                          overwrite=True)
    print(f"Wrote diagnostics → {h5_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
