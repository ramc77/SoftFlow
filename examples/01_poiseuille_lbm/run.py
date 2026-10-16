"""
Poiseuille channel with deformable capsules (Phase-1 showcase)
==============================================================
A periodic 2-D microchannel driven by a constant body force, with six
soft Skalak capsules immersed in the flow. The body-force-driven
Poiseuille profile is the principal V&V case (see the project specification):
the streamwise velocity should reach a clean parabola in y once the
flow has relaxed.

This example also demonstrates the Phase-1 reproducibility infrastructure:
every run writes ``output/01_poiseuille_lbm/config/run_manifest.json``
containing the git SHA, compiler ID and resolved flags, OpenMP thread
count, the canonical RNG seed, and the fully resolved parameters.

Output:
  - VTK files: vtk_poiseuille/fluid/*.vti (rho, u_x, u_y) +
               vtk_poiseuille/particles/*.vtp (capsule polygons)
  - PVD time-series: vtk_poiseuille/{fluid,particles}.pvd
  - CSV trajectories under data_poiseuille/
  - run_manifest.json under vtk_poiseuille/config/

Usage:
    python 01_poiseuille_lbm/run.py

Open in ParaView:
    File → Open → vtk_poiseuille/fluid/fluid.pvd
                  vtk_poiseuille/particles/particles.pvd
"""

import os, sys

# -- Path setup (find the build). The Python package is in python/;
#    the C++ extension lands in <build_dir>/python/. Prepend in
#    reverse priority order so the most-recent build directory wins.
script_dir  = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(os.path.dirname(script_dir))
sys.path.insert(0, os.path.join(project_dir, "python"))
for build_dir in ("build", "build_phase1", "build_phase2"):
    cand = os.path.join(project_dir, build_dir, "python")
    if os.path.isdir(cand):
        sys.path.insert(0, cand)

from pysoftflow import SoftFlowSimulation


# =================================================================
# Physical parameters (lattice units)
# =================================================================

NX, NY     = 200, 60          # channel length × diameter
N_STEPS    = 200              # quick smoke run; raise to 4000+ for production
TAU        = 0.8              # → kinematic viscosity ν = (τ−½)/3 = 0.1
BODY_FX    = 4e-6             # body force ~ pressure gradient

N_CAPSULES = 6                # soft Skalak capsules
R_CAPSULE  = 3.5              # effective radius


# =================================================================
# Simulation setup
# =================================================================

sim = SoftFlowSimulation()

# -- Domain: periodic channel with top/bottom no-slip walls --
sim.domain(nx=NX, ny=NY)
sim.boundary(x="periodic", y="wall")
sim.fluid(tau=TAU)
sim.body_force(BODY_FX, 0.0)

# -- IBM coupling (multi-direct forcing for accuracy) --
sim.ibm(iterations=2)

# -- Output: VTK for ParaView, CSV for trajectories --
sim.output(format="vtk",
           directory=os.path.join(script_dir, "vtk_poiseuille"),
           interval=200)


# =================================================================
# Particle types
# =================================================================

# Type 0: Red-blood-cell-like soft capsule (Skalak membrane)
sim.particle_type("rbc",
                  model="skalak",
                  G_s=0.05,
                  C_skalak=10.0,
                  k_bend=0.003,
                  k_area=0.6,
                  k_perimeter=0.06)


# =================================================================
# Particle placement
# =================================================================

sim.region("vessel", x=(20, NX - 20), y=(15, NY - 15))
sim.generate("rbc", count=N_CAPSULES, region="vessel",
             radius=R_CAPSULE, num_nodes=20,
             seed=42, min_gap=2.0)


# =================================================================
# Run
# =================================================================

sim.thermo(interval=200)
sim.run(N_STEPS)
