"""End-to-end test for pysoftflow.insertion.

Builds against the compiled ``softflow_core`` extension produced by
``cmake -DBUILD_PYTHON=ON``. The test path appends both the Python
package source and the build's extension directory so we can run
without ``pip install``.
"""

from __future__ import annotations

import math
import os
import pathlib
import sys

import pytest


HERE = pathlib.Path(__file__).resolve().parents[2]

# Push the python/ source dir for `pysoftflow` and one of the build
# directories for `softflow_core`. We try both build_phase1 and
# build_phase2 (the second is the build that has BUILD_PYTHON=ON).
sys.path.insert(0, str(HERE / "python"))
for build_dir in ["build_phase2", "build", "build_phase1"]:
    candidate = HERE / build_dir / "python"
    if candidate.is_dir():
        sys.path.insert(0, str(candidate))
        break


pytest.importorskip("softflow_core",
    reason="softflow_core extension not built — run `cmake -DBUILD_PYTHON=ON`")


from pysoftflow import insertion as ins  # noqa: E402
import softflow_core as core              # noqa: E402


# ── Region facade ──────────────────────────────────────────────────


def test_rect_region_keyword_form():
    r = ins.RectRegion(x=(0.0, 200.0), y=(10.0, 70.0))
    assert r.area() == pytest.approx(200.0 * 60.0)
    assert r.contains(core.Vec2d(100.0, 40.0))
    assert not r.contains(core.Vec2d(-1.0, 40.0))


def test_circle_region_keyword_form():
    c = ins.CircleRegion(center=(50.0, 40.0), radius=10.0)
    assert c.area() == pytest.approx(math.pi * 100.0)
    assert c.contains(core.Vec2d(55.0, 45.0))
    assert not c.contains(core.Vec2d(70.0, 40.0))


def test_polygon_region_orientation_independent():
    ccw = ins.PolygonRegion([
        core.Vec2d(0, 0), core.Vec2d(10, 0),
        core.Vec2d(10, 10), core.Vec2d(0, 10),
    ])
    cw = ins.PolygonRegion([
        core.Vec2d(0, 0), core.Vec2d(0, 10),
        core.Vec2d(10, 10), core.Vec2d(10, 0),
    ])
    assert ccw.area() == pytest.approx(100.0)
    assert cw.area()  == pytest.approx(100.0)


# ── Size distribution facade ──────────────────────────────────────


def test_size_distribution_factories_return_concrete():
    m = ins.SizeDistribution.monodisperse(2.0)
    assert m.minRadius() == m.maxRadius() == 2.0

    b = ins.SizeDistribution.bidisperse(r_small=2.0, r_large=4.0,
                                         fraction_small=0.5)
    assert b.minRadius() == 2.0 and b.maxRadius() == 4.0

    ln = ins.SizeDistribution.lognormal(mu_log=0.0, sigma_log=0.4,
                                         r_min=0.5, r_max=2.0)
    assert ln.minRadius() == 0.5 and ln.maxRadius() == 2.0

    u = ins.SizeDistribution.user(radii=[1.0, 2.0, 3.0],
                                   weights=[1.0, 2.0, 3.0])
    assert u.minRadius() == 1.0 and u.maxRadius() == 3.0


# ── Inserter facade + end-to-end Simulation ────────────────────────


def _make_sim(rng_seed=0xCAFEBABE):
    p = core.SimulationParams()
    p.nx = 200
    p.ny = 60
    p.fluid.boundary_type = core.BoundaryType.PERIODIC
    p.enable_stability_checks = False
    p.enable_profiling        = False
    p.metrics_interval        = 0
    p.vtk_dump_every = p.csv_dump_every = 0
    p.probe_dump_every = p.stats_dump_every = 0
    p.rng_seed   = rng_seed
    p.output_dir = "/tmp/softflow_phase2_pyfacade"
    p.vtk_format = "ascii"
    return core.Simulation(p), p


def test_hex_lattice_via_facade_places_capsules():
    region = ins.RectRegion(x=(20.0, 180.0), y=(15.0, 45.0))
    sizes  = ins.SizeDistribution.monodisperse(2.0)
    inserter = ins.Inserter.hex_lattice(region=region, spacing=8.0,
                                         sizes=sizes, jitter=0.0)

    sim, p = _make_sim()
    n = sim.insertCapsules(inserter, core.MembraneParams(),
                           type=0, min_gap=1.0, num_nodes=12,
                           seed_tag="hex_facade_test")
    # Hex lattice in a 160 × 30 region with spacing 8 gives a healthy
    # but smallish count; we just need it nonzero and finite.
    assert 5 < n < 200
    assert sim.capsules().numCapsules() == n


def test_dynamic_constant_flux_via_facade():
    sim, p = _make_sim()
    region = ins.RectRegion(x=(0.0, 60.0), y=(15.0, 45.0))
    sizes  = ins.SizeDistribution.monodisperse(2.0)

    flux = ins.DynamicInserter.constant_flux(
        region=region, target_phi=0.05, sizes=sizes,
        max_per_step=2, attempts_per_event=64)
    sim.registerDynamicInserter(flux, core.MembraneParams(),
                                 type=0, min_gap=1.0, num_nodes=12,
                                 seed_tag="flux_facade_test")
    assert sim.numDynamicInserters() == 1

    sim.initialize()
    for _ in range(200):
        sim.step()
    sim.finalize()

    # 60 × 30 region, π·2² per capsule → each capsule contributes
    # ~0.007 to φ; target 0.05 → ~7 capsules expected.
    n = sim.capsules().numCapsules()
    assert 3 < n < 30


def test_seed_tag_drives_independent_streams():
    sim_a, _ = _make_sim(rng_seed=42)
    sim_b, _ = _make_sim(rng_seed=42)

    region = ins.RectRegion(x=(20.0, 180.0), y=(15.0, 45.0))
    sizes  = ins.SizeDistribution.bidisperse(
        r_small=2.0, r_large=4.0, fraction_small=0.5)

    rsa = ins.Inserter.rsa(region=region, target_count=30, sizes=sizes,
                            max_attempts=3000)

    n1 = sim_a.insertCapsules(rsa, core.MembraneParams(), type=0,
                               min_gap=1.0, num_nodes=12,
                               seed_tag="reproducible")
    n2 = sim_b.insertCapsules(rsa, core.MembraneParams(), type=0,
                               min_gap=1.0, num_nodes=12,
                               seed_tag="reproducible")
    assert n1 == n2

    # Centres should be bit-exact across the two sims.
    for i in range(sim_a.capsules().numCapsules()):
        c_a = sim_a.capsules()[i].centroid()
        c_b = sim_b.capsules()[i].centroid()
        assert c_a.x == pytest.approx(c_b.x)
        assert c_a.y == pytest.approx(c_b.y)
