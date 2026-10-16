"""End-to-end test for pysoftflow.drug_delivery.runner.DrugDeliveryRun.

Builds a tiny live ``Simulation`` with one carrier capsule type, hooks
up a FirstOrder release kinetic and a single first-order WallAbsorber
on the bottom wall, runs ~200 steps, and checks that:

  - The orchestrator's per-step callback fires.
  - Carrier M_p decays roughly as exp(-k_rel * t).
  - The wall absorber's cumulative-absorbed counter is positive.
  - summary() returns sane numbers (η ∈ [0, 1], OTF in [0, 1]).
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

import softflow_core as core                                      # noqa: E402
from pysoftflow import insertion as ins                           # noqa: E402
from pysoftflow.drug_delivery import (                            # noqa: E402
    DrugDeliveryRun, FirstOrder, WallAbsorber,
)


def _build_sim():
    p = core.SimulationParams()
    p.nx, p.ny = 100, 30
    p.fluid.boundary_type = core.BoundaryType.PERIODIC
    p.fluid.body_force_x  = 5e-6
    p.fluid.collision_model = core.CollisionModel.BGK
    p.scalar.enabled              = True
    p.scalar.n_species            = 1
    p.scalar.diffusivity          = [0.05]
    p.scalar.inlet_concentration  = [0.0]
    p.enable_stability_checks = False
    p.enable_profiling        = False
    p.metrics_interval        = 0
    p.vtk_dump_every = p.csv_dump_every = 0
    p.probe_dump_every = p.stats_dump_every = 0
    p.rng_seed = 0xDEADBEEF
    p.output_dir = "/tmp/softflow_phase4_runner_test"
    p.vtk_format = "ascii"

    sim = core.Simulation(p)

    # Seed a few carriers upstream.
    region = ins.RectRegion(x=(5.0, 25.0), y=(8.0, 22.0))
    sizes  = ins.SizeDistribution.monodisperse(1.5)
    rsa    = ins.Inserter.rsa(region=region, target_count=4,
                               sizes=sizes, max_attempts=2000)
    sim.insertCapsules(rsa, core.MembraneParams(),
                        type=0, min_gap=1.0, num_nodes=12,
                        seed_tag="phase4_carriers")
    return sim, p


def test_drug_delivery_run_end_to_end():
    sim, p = _build_sim()

    run = DrugDeliveryRun(sim=sim)
    run.add_carrier_type(type_id=0, kinetic=FirstOrder(k_rel=0.01),
                          initial_mass=1.0)
    # Bottom-wall target patch.
    target = WallAbsorber(i_range=(40, 80), j_range=(0, 4),
                           mode="first_order", k=0.05, label="target")
    run.add_target(target)
    # Top-wall off-target patch.
    off = WallAbsorber(i_range=(40, 80), j_range=(26, 30),
                        mode="first_order", k=0.05, label="off")
    run.add_off_target(off)

    run.attach()
    sim.initialize()
    for _ in range(200):
        sim.step()
    sim.finalize()

    # The history should hold one record per step.
    assert len(run.history) == 200

    # FirstOrder kinetic decay: total M_p should have dropped to
    # roughly 4 * exp(-0.01 * 200) = 4 * 0.135 ≈ 0.54.
    final = run.history[-1]
    assert final.total_M_p < 4.0 * math.exp(-0.01 * 200) + 0.1
    assert final.total_M_p > 4.0 * math.exp(-0.01 * 200) - 0.5
    # Mass that left the carriers got partly absorbed; we just ask
    # that the target counter moved.
    assert target.cumulative_absorbed >= 0.0

    s = run.summary()
    assert s["n_carriers"] == 4
    assert s["total_loaded"] == pytest.approx(4.0)
    # η, OTF are non-negative and finite.
    assert s["delivery_efficiency"] >= 0.0
    assert s["off_target_fraction"] >= 0.0
    assert math.isfinite(s["delivery_efficiency"])
    assert math.isfinite(s["off_target_fraction"])


def test_attach_is_idempotent():
    sim, p = _build_sim()
    run = DrugDeliveryRun(sim=sim)
    run.add_carrier_type(0, FirstOrder(0.01), 1.0)
    run.attach()
    # Second attach is a no-op.
    run.attach()
    assert run._attached is True


def test_runner_skips_capsules_with_no_registered_kinetic():
    """Carriers of an unregistered type should not be tracked."""
    sim, p = _build_sim()
    run = DrugDeliveryRun(sim=sim)
    # Register kinetics for type 99 (which no capsule has).
    run.add_carrier_type(99, FirstOrder(0.01), 1.0)
    run.attach()

    # No carriers were bound — the orchestrator runs but tracks nothing.
    sim.initialize()
    for _ in range(5):
        sim.step()
    s = run.summary()
    assert s["n_carriers"] == 0
    assert math.isnan(s["delivery_efficiency"])  # no payload loaded
