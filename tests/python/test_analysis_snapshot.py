"""Tests for pysoftflow.analysis.SimulationSnapshot."""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest


HERE = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE / "python"))
for build_dir in ["build_phase2", "build", "build_phase1"]:
    cand = HERE / build_dir / "python"
    if cand.is_dir():
        sys.path.insert(0, str(cand))
        break

from pysoftflow.analysis import SimulationSnapshot   # noqa: E402


def test_from_arrays_validates_shapes():
    SimulationSnapshot.from_arrays(
        positions=np.array([[0, 0], [1, 0], [0, 1]], dtype=float),
        radii=np.array([1.0, 1.0, 1.0]),
        types=np.array([0, 0, 1]),
        box=(10.0, 10.0),
    )
    with pytest.raises(ValueError, match="positions"):
        SimulationSnapshot.from_arrays(
            positions=np.array([0, 0, 1], dtype=float),
            radii=np.array([1.0]), types=np.array([0]),
        )
    with pytest.raises(ValueError, match="radii"):
        SimulationSnapshot.from_arrays(
            positions=np.array([[0, 0]], dtype=float),
            radii=np.array([1.0, 2.0]),
            types=np.array([0]),
        )


def test_from_arrays_default_velocities_and_bonds():
    s = SimulationSnapshot.from_arrays(
        positions=np.array([[0, 0], [1, 0]], dtype=float),
        radii=np.array([1.0, 1.0]),
        types=np.array([0, 1]),
    )
    assert s.velocities.shape == (2, 2)
    assert (s.velocities == 0.0).all()
    assert s.bonds.shape == (0, 2)
    assert s.bonds.dtype == np.int64


def test_distinct_types_and_by_type_mask():
    s = SimulationSnapshot.from_arrays(
        positions=np.zeros((5, 2)),
        radii=np.ones(5),
        types=np.array([1, 0, 1, 1, 0]),
    )
    assert (s.distinct_types() == np.array([0, 1])).all()
    assert s.by_type(0).tolist() == [False, True, False, False, True]
    assert s.by_type(1).tolist() == [True, False, True, True, False]


def test_pairwise_distance_periodic_x_minimum_image():
    s = SimulationSnapshot.from_arrays(
        positions=np.array([[1.0, 5.0], [9.0, 5.0]], dtype=float),
        radii=np.ones(2),
        types=np.zeros(2, dtype=int),
        box=(10.0, 10.0),
        periodic_x=True,
    )
    # Without wrap, distance is 8. With wrap, 2.
    assert s.pairwise_distance(0, 1) == pytest.approx(2.0)


def test_pairwise_distance_non_periodic():
    s = SimulationSnapshot.from_arrays(
        positions=np.array([[1.0, 0.0], [4.0, 4.0]], dtype=float),
        radii=np.ones(2),
        types=np.zeros(2, dtype=int),
        box=(10.0, 10.0),
        periodic_x=False,
    )
    assert s.pairwise_distance(0, 1) == pytest.approx(5.0)


def test_from_simulation_extracts_positions_radii_types():
    pytest.importorskip("softflow_core")
    import softflow_core as core
    from pysoftflow import insertion as ins

    p = core.SimulationParams()
    p.nx, p.ny = 200, 60
    p.fluid.boundary_type = core.BoundaryType.PERIODIC
    p.enable_stability_checks = False
    p.enable_profiling = False
    p.metrics_interval = 0
    p.vtk_dump_every = p.csv_dump_every = 0
    p.probe_dump_every = p.stats_dump_every = 0
    p.rng_seed = 0xC0FFEE
    p.output_dir = "/tmp/softflow_phase3_snapshot"
    p.vtk_format = "ascii"
    sim = core.Simulation(p)

    region = ins.RectRegion(x=(20.0, 180.0), y=(15.0, 45.0))
    sizes  = ins.SizeDistribution.bidisperse(
        r_small=2.0, r_large=4.0, fraction_small=0.5)
    inserter = ins.Inserter.rsa(region=region, target_count=20,
                                 sizes=sizes, max_attempts=10000)
    sim.insertCapsules(inserter, core.MembraneParams(),
                        type=0, min_gap=1.0, num_nodes=12,
                        seed_tag="snapshot_test")

    snap = SimulationSnapshot.from_simulation(sim)
    assert snap.n_particles >= 5
    assert snap.positions.shape == (snap.n_particles, 2)
    assert snap.radii.shape == (snap.n_particles,)
    assert snap.types.shape == (snap.n_particles,)
    assert snap.box == (200.0, 60.0)
    assert snap.periodic_x is True
    # Radii were drawn from {2.0, 4.0} but effectiveRadius() returns
    # sqrt(area/π) of the discretised polygon, slightly less than the
    # construction radius. Floating-point noise in centroid() makes
    # exact-uniqueness brittle, so we just check the min/max bracket
    # and that all values cluster near one of the two expected radii.
    rmin, rmax = snap.radii.min(), snap.radii.max()
    assert 1.7 < rmin < 2.0
    assert 3.5 < rmax < 4.0
    # Each radius is within 1% of either 1.954 or 3.909.
    near_small = np.abs(snap.radii - 1.954) < 0.02
    near_large = np.abs(snap.radii - 3.909) < 0.04
    assert (near_small | near_large).all()
    # type labels all 0 (we only registered one type)
    assert (snap.types == 0).all()
