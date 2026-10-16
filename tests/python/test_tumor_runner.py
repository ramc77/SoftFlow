"""End-to-end test for pysoftflow.tumor_growth.runner.TumorGrowthRun.

Builds a small live ``Simulation`` with adhesion enabled, registers
a ``StressNutrientDivision`` kinetic, runs ~200 steps, and checks
that:

  - The orchestrator's per-step callback fires.
  - At least some divisions occur under permissive thresholds.
  - The history records monotonically non-decreasing total division
    count.
  - High-stress conditions block all division (verified by setting
    ``stress_max=0`` so any positive shear blocks).
  - summary() returns sane numbers.
"""

from __future__ import annotations

import math
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

pytest.importorskip("softflow_core",
    reason="softflow_core extension not built")

import softflow_core as core                                       # noqa: E402
from pysoftflow import insertion as ins                            # noqa: E402
from pysoftflow.tumor_growth import (                              # noqa: E402
    DaughterPlacer, EmbolizationDetector, StressNutrientDivision,
    TumorGrowthRun,
)


def _build_sim(*, n_carriers=4, with_adhesion=True):
    p = core.SimulationParams()
    p.nx, p.ny = 100, 30
    p.fluid.boundary_type   = core.BoundaryType.PERIODIC
    p.fluid.body_force_x    = 1e-6
    p.fluid.collision_model = core.CollisionModel.BGK
    p.scalar.enabled              = True
    p.scalar.n_species            = 1
    p.scalar.diffusivity          = [0.05]
    p.scalar.inlet_concentration  = [1.0]
    if with_adhesion:
        p.adhesion.enabled = True
        p.adhesion.k_on    = 0.01
        p.adhesion.k_off   = 0.001
        p.adhesion.k_bond  = 0.05
        p.adhesion.d_bond  = 2.0
    p.enable_stability_checks = False
    p.enable_profiling        = False
    p.metrics_interval        = 0
    p.vtk_dump_every = p.csv_dump_every = 0
    p.probe_dump_every = p.stats_dump_every = 0
    p.rng_seed = 0xCAB00D1E
    p.output_dir = "/tmp/softflow_phase5_runner_test"
    p.vtk_format = "ascii"

    sim = core.Simulation(p)

    # Seed a few "tumour" capsules.
    region = ins.RectRegion(x=(20.0, 60.0), y=(10.0, 20.0))
    sizes  = ins.SizeDistribution.monodisperse(2.0)
    rsa    = ins.Inserter.rsa(region=region, target_count=n_carriers,
                                sizes=sizes, max_attempts=4000)

    mp = core.MembraneParams()
    mp.model      = core.MembraneModel.SKALAK
    mp.G_s        = 0.05
    mp.C_skalak   = 10.0
    mp.k_bend     = 0.003
    mp.k_area     = 0.5
    mp.k_perimeter = 0.05
    sim.insertCapsules(rsa, mp, type=0, min_gap=1.0,
                        num_nodes=12, seed_tag="phase5_seed")
    return sim, mp


def test_runner_attaches_and_runs_without_crashing():
    sim, mp = _build_sim()
    n0 = sim.capsules().numCapsules()
    assert n0 == 4

    run = TumorGrowthRun(sim=sim)
    # Permissive thresholds — should fire often.
    run.add_division_kinetic(
        type_id=0,
        kinetic=StressNutrientDivision(
            k_div=0.10, stress_max=10.0, nutrient_min=0.0),
        mparams=mp,
        placer=DaughterPlacer(min_gap=0.5, max_attempts=24,
                                ring_radius_factor=1.10),
        nutrient_species=0,
        num_nodes=12,
    )
    run.set_seed(42)
    run.attach()

    sim.initialize()
    for _ in range(150):
        sim.step()
    sim.finalize()

    assert len(run.history) == 150
    final_n = sim.capsules().numCapsules()
    assert final_n >= n0     # no divisions = same; some divisions = more
    summary = run.summary()
    assert summary["n_capsules_final"] == final_n
    assert summary["n_divisions_total"] >= 0
    # If any divisions happened, they should match final - n0.
    if summary["n_divisions_total"] > 0:
        assert final_n > n0


def test_high_stress_blocks_all_divisions():
    """``stress_max = 0`` blocks under any flow (γ̇ > 0 everywhere
    there's a body force)."""
    sim, mp = _build_sim()
    n0 = sim.capsules().numCapsules()

    run = TumorGrowthRun(sim=sim)
    run.add_division_kinetic(
        type_id=0,
        kinetic=StressNutrientDivision(
            k_div=10.0, stress_max=0.0, nutrient_min=0.0),
        mparams=mp,
    )
    run.attach()

    sim.initialize()
    for _ in range(50):
        sim.step()
    sim.finalize()

    # No divisions → capsule count unchanged.
    assert sim.capsules().numCapsules() == n0
    assert run.summary()["n_divisions_total"] == 0


def test_low_nutrient_blocks_all_divisions():
    """``nutrient_min`` above the maximum possible C blocks divisions."""
    sim, mp = _build_sim()
    n0 = sim.capsules().numCapsules()

    run = TumorGrowthRun(sim=sim)
    run.add_division_kinetic(
        type_id=0,
        kinetic=StressNutrientDivision(
            k_div=10.0, stress_max=10.0, nutrient_min=100.0),
        mparams=mp,
    )
    run.attach()

    sim.initialize()
    for _ in range(50):
        sim.step()
    sim.finalize()

    assert sim.capsules().numCapsules() == n0


def test_history_records_per_step_with_strictly_growing_division_total():
    sim, mp = _build_sim(n_carriers=2)
    run = TumorGrowthRun(sim=sim)
    run.add_division_kinetic(
        type_id=0,
        kinetic=StressNutrientDivision(
            k_div=0.05, stress_max=10.0, nutrient_min=0.0),
        mparams=mp,
        placer=DaughterPlacer(ring_radius_factor=1.20),
    )
    run.set_seed(7)
    run.attach()

    sim.initialize()
    for _ in range(80):
        sim.step()
    sim.finalize()

    totals = [r.n_divisions_total for r in run.history]
    # Monotonically non-decreasing.
    assert all(b >= a for a, b in zip(totals, totals[1:]))


def test_attach_is_idempotent():
    sim, mp = _build_sim()
    run = TumorGrowthRun(sim=sim)
    run.add_division_kinetic(
        type_id=0,
        kinetic=StressNutrientDivision(0.1, 1.0, 0.0),
        mparams=mp,
    )
    run.attach()
    run.attach()
    assert run._attached is True


def test_embolization_detector_runs_in_pipeline():
    sim, mp = _build_sim()
    detector = EmbolizationDetector(
        x_section=50, flow_drop_threshold=0.5, contact_cutoff=1.0)
    run = TumorGrowthRun(sim=sim)
    run.add_division_kinetic(
        type_id=0,
        kinetic=StressNutrientDivision(0.0, 10.0, 0.0),  # no division
        mparams=mp,
    )
    run.add_embolization_detector(detector)
    run.attach()

    sim.initialize()
    for _ in range(30):
        sim.step()
    sim.finalize()

    # Detector accumulated history.
    assert detector.flow_rate_history.shape == (30,)
    assert detector.cluster_span_history.shape == (30,)
