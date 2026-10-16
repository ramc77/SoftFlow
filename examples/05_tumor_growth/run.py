"""
Aggregate growth in flow with embolization detection (Phase 5)
==============================================================
**Coarse-grained mechano-chemical proxy. NOT a validated cancer model.**
Read the strong-language Limitations block in
[README.md](README.md) and ../../docs/tumor_growth.md before drawing
conclusions.

Five adhesive Skalak capsules in a deliberately narrow 200×40 channel.
Adhesion is the Phase-0 Bell model (toggle the catch/slip variant via
``USE_CATCH_SLIP`` below). Capsules of type "tumour" divide
stochastically when the local fluid shear is below ``STRESS_MAX`` and
the local nutrient concentration is above ``NUTRIENT_MIN``. An
EmbolizationDetector at x=140 watches for spanning-cluster +
flow-rate-drop events.

Output:
  - VTK files: vtk_tumor/{fluid,particles}/*.{vti,vtp}
  - PVD time-series: vtk_tumor/{fluid,particles}.pvd
  - history.npz: per-step capsule count, divisions, cluster size,
                  flow rate, cluster span, embolization events
  - run_manifest.json under vtk_tumor/config/

Usage:
    python 05_tumor_growth/run.py
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
from pysoftflow.tumor_growth import (
    DaughterPlacer, EmbolizationDetector, StressNutrientDivision,
    TumorGrowthRun,
)


# =================================================================
# Physical parameters (lattice units)
# =================================================================

NX, NY      = 200, 40         # deliberately narrow → embolization possible
N_STEPS     = 100000             # quick smoke run; raise to 5000+ for production
TAU         = 0.8
BODY_FX     = 3e-6

N_SEED      = 5               # initial adhesive capsules
R_CAPSULE   = 2.5

# Division kinetics
K_DIV         = 0.005         # Poisson rate per unit lattice time
STRESS_MAX    = 8e-4          # block division above this γ̇
NUTRIENT_MIN  = 0.05          # block division below this C

# Adhesion: Bell model by default; flip to True for catch/slip variant.
USE_CATCH_SLIP = False

# Embolization: watch this cross-section.
EMBOL_X         = 140
FLOW_DROP_THRESH = 0.5


# =================================================================
# Simulation setup
# =================================================================

sim = SoftFlowSimulation()

sim.domain(nx=NX, ny=NY)
sim.boundary(x="periodic", y="wall")
sim.fluid(tau=TAU)
sim.body_force(BODY_FX, 0.0)

sim.ibm(iterations=2)

# Adhesion: Bell model (or catch/slip if enabled). The wrapper auto-
# fills the type-pair adhesion matrix so single-type bonds form.
sim.adhesion(enabled=True,
             k_on=0.05, k_off=0.001, k_bond=0.05, d_bond=2.0,
             F_crit=0.01, max_bonds_per_node=3,
             bond_model="catch_slip" if USE_CATCH_SLIP else "bell")

# Nutrient field: prefilled uniformly so divisions aren't trivially
# gated by nutrient depletion in the first few steps.
sim.scalar_transport(enabled=True,
                     diffusivity=0.05, n_species=1,
                     initial_concentration=1.0)

sim.output(format="vtk",
           directory=os.path.join(script_dir, "vtk_tumor"),
           interval=200)


# =================================================================
# Particle types
# =================================================================

# Type 0: aggregate-forming "tumour" capsule (Skalak, soft)
sim.particle_type("tumour",
                  model="skalak",
                  G_s=0.06, C_skalak=10.0,
                  k_bend=0.003, k_area=0.6, k_perimeter=0.06)


# =================================================================
# Particle placement
# =================================================================

sim.region("seed", x=(20, 80), y=(12, 28))
sim.generate("tumour", count=N_SEED, region="seed",
             radius=R_CAPSULE, num_nodes=18,
             method="random", seed=42, min_gap=1.0)

sim.initialize()


# =================================================================
# Phase-5 division + embolization detection
# =================================================================

run = TumorGrowthRun(sim=sim.core)

# Daughter capsules inherit the parent's membrane params.
mp_inherit = sim.params.membrane

run.add_division_kinetic(
    type_id=0,
    kinetic=StressNutrientDivision(k_div=K_DIV,
                                     stress_max=STRESS_MAX,
                                     nutrient_min=NUTRIENT_MIN),
    mparams=mp_inherit,
    placer=DaughterPlacer(min_gap=0.5, max_attempts=24,
                            ring_radius_factor=1.10),
    nutrient_species=0,
    num_nodes=18,
)
run.set_seed(0xA5A5A5)

run.add_embolization_detector(
    EmbolizationDetector(x_section=EMBOL_X,
                          flow_drop_threshold=FLOW_DROP_THRESH,
                          contact_cutoff=1.0,
                          band_fraction=0.10))

run.attach()


# =================================================================
# Run
# =================================================================

sim.thermo(interval=200)
sim.run(N_STEPS)


# =================================================================
# Save Phase-5 time-series + events
# =================================================================

print(f"\nSummary: {run.summary()}")

h = np.array([(r.step, r.time, r.n_capsules,
                r.n_divisions_total, r.n_divisions_step,
                r.largest_cluster, r.n_events_step)
               for r in run.history],
              dtype=[("step", "i8"), ("time", "f8"),
                     ("n_capsules", "i8"),
                     ("n_divisions_total", "i8"),
                     ("n_divisions_step", "i8"),
                     ("largest_cluster", "i8"),
                     ("n_events_step", "i8")])
detector = run._detectors[0]
np.savez(os.path.join(script_dir, "history.npz"),
          history=h,
          flow_rate=detector.flow_rate_history,
          cluster_span=detector.cluster_span_history)

if detector.events:
    ev = np.array([(e.step, e.time, e.flow_rate_ratio,
                     e.spanning_size, e.cluster_y_span,
                     e.spanning_x_centre)
                    for e in detector.events],
                   dtype=[("step", "i8"), ("time", "f8"),
                          ("flow_rate_ratio", "f8"),
                          ("spanning_size", "i8"),
                          ("cluster_y_span", "f8"),
                          ("spanning_x_centre", "f8")])
    np.savez(os.path.join(script_dir, "events.npz"), events=ev)
    print(f"  recorded {len(ev)} embolization event(s)")
