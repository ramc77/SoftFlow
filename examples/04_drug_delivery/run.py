"""
Drug-delivery showcase (Phase 4)
================================
Carrier capsules released upstream, target absorber on the bottom wall
just downstream, off-target absorber on the top wall opposite. The
``RELEASE_MODE`` flag at the top selects the kinetic:

  diffusion      Higuchi-style Fick law           (existing C++ chemistry)
  first_order    M(t) = M_0 exp(-k t)
  shear          sigmoid in local fluid shear rate
  ph             sigmoid in a designated trigger species
  burst          one-shot release at t = release_time

This is methodological — see ../../docs/drug_delivery.md for the
strong-language Limitations block.

Output:
  - VTK files: vtk_drug/{fluid,particles}/*.{vti,vtp}
  - PVD time-series: vtk_drug/{fluid,particles}.pvd
  - history.npz: per-step (M_p, target_absorbed, off_target_absorbed)
  - run_manifest.json under vtk_drug/config/

Usage:
    # Edit RELEASE_MODE below, then:
    python 04_drug_delivery/run.py
"""

import os, sys
import numpy as np

# -- Path setup (find the build) --
script_dir  = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(os.path.dirname(script_dir))
sys.path.insert(0, os.path.join(project_dir, "python"))
for build_dir in ("build", "build_phase1", "build_phase2"):
    cand = os.path.join(project_dir, build_dir, "python")
    if os.path.isdir(cand):
        sys.path.insert(0, cand)

from pysoftflow import SoftFlowSimulation
from pysoftflow.drug_delivery import (
    Burst, DiffusionControlled, DrugDeliveryRun,
    FirstOrder, PhTriggered, ShearTriggered, WallAbsorber,
)


# =================================================================
# Physical parameters (lattice units)
# =================================================================

NX, NY      = 300, 60
N_STEPS     = 10000             # quick smoke run; raise to 1500+ for production
TAU         = 0.8
BODY_FX     = 5e-6

N_CARRIERS  = 12
R_CARRIER   = 2.5

# Pick one: "diffusion" / "first_order" / "shear" / "ph" / "burst"
RELEASE_MODE = "first_order"
K_RELEASE    = 0.005          # rate constant for first_order/shear/ph
INITIAL_MASS = 1.0            # payload per carrier

# Target patch on bottom wall, off-target on top — both downstream.
TARGET_X     = (180, 240)
TARGET_J     = (1, 6)         # bottom wall band
OFF_J        = (54, 59)       # top    wall band


# =================================================================
# Simulation setup
# =================================================================

sim = SoftFlowSimulation()

sim.domain(nx=NX, ny=NY)
sim.boundary(x="periodic", y="wall")
sim.fluid(tau=TAU)
sim.body_force(BODY_FX, 0.0)

sim.ibm(iterations=2)

# Scalar transport carries the released chemical.
sim.scalar_transport(enabled=True,
                     diffusivity=0.05,
                     n_species=1,
                     initial_concentration=0.0)

sim.output(format="vtk",
           directory=os.path.join(script_dir, "vtk_drug"),
           interval=200)

# Carrier capsules — soft Skalak (RBC-like).
sim.particle_type("carrier",
                  model="skalak",
                  G_s=0.06, C_skalak=10.0,
                  k_bend=0.003, k_area=0.6, k_perimeter=0.06)

# Seed carriers upstream via random fill.
sim.region("inlet", x=(10, 60), y=(20, 40))
sim.generate("carrier", count=N_CARRIERS, region="inlet",
             radius=R_CARRIER, num_nodes=18,
             method="random", seed=42, min_gap=1.0)

sim.initialize()


# =================================================================
# Phase-4 release kinetic + wall absorbers
# =================================================================

if RELEASE_MODE == "diffusion":
    kinetic = DiffusionControlled(k_leach=K_RELEASE, C_eq=1.0)
elif RELEASE_MODE == "first_order":
    kinetic = FirstOrder(k_rel=K_RELEASE)
elif RELEASE_MODE == "shear":
    kinetic = ShearTriggered(k_max=K_RELEASE,
                              gamma_thresh=2e-4, sharpness=5e4)
elif RELEASE_MODE == "ph":
    kinetic = PhTriggered(k_max=K_RELEASE,
                           C_thresh=0.1, species=0, sharpness=20.0)
elif RELEASE_MODE == "burst":
    kinetic = Burst(release_time=200.0, fraction=0.6)
else:
    raise ValueError(f"unknown RELEASE_MODE: {RELEASE_MODE}")

run = DrugDeliveryRun(sim=sim.core)
run.add_carrier_type(type_id=0, kinetic=kinetic, initial_mass=INITIAL_MASS)
run.add_target(WallAbsorber(i_range=TARGET_X, j_range=TARGET_J,
                              mode="first_order", k=0.08, label="target"))
run.add_off_target(WallAbsorber(i_range=TARGET_X, j_range=OFF_J,
                                  mode="first_order", k=0.08,
                                  label="off_target"))
run.attach()


# =================================================================
# Run
# =================================================================

sim.thermo(interval=200)
sim.run(N_STEPS)


# =================================================================
# Save delivery time-series + summary
# =================================================================

print(f"\nRelease mode: {RELEASE_MODE}")
print(f"Summary: {run.summary()}")

h = np.array(
    [(r.step, r.time, r.total_M_p, r.total_released,
      r.target_absorbed, r.off_target_absorbed)
     for r in run.history],
    dtype=[("step", "i8"), ("time", "f8"),
            ("total_M_p", "f8"), ("total_released", "f8"),
            ("target_absorbed", "f8"),
            ("off_target_absorbed", "f8")])

s = run.summary()
np.savez(os.path.join(script_dir, "history.npz"), history=h,
          summary=np.array([(s["delivery_efficiency"],
                              s["off_target_fraction"],
                              s["total_remaining"])],
                            dtype=[("eta", "f8"),
                                   ("off_target", "f8"),
                                   ("remaining", "f8")]))
