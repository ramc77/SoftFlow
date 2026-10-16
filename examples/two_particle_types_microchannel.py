"""
Example: Three Types of Soft Particles in a Microfluidic Channel
=================================================================
Simulates red blood cells (RBCs), circulating tumor cells (CTCs),
and microplastics flowing through a periodic microfluidic channel
driven by a body force, with adhesion between microplastics and walls.

Physical setup:
  - Periodic microchannel with top/bottom walls
  - Body-force driven flow (pressure gradient equivalent)
  - RBCs:  soft, deformable (Skalak membrane, low G_s)
  - CTCs:  stiffer, larger  (Neo-Hookean membrane, high G_s)
  - Microplastics: small, rigid-ish (Hookean, very high G_s)
  - Adhesion: microplastics stick to walls and to each other
  - Lubrication corrections for close cell-cell interactions

Output:
  - VTK files for ParaView visualization (fluid + particles)
  - CSV trajectory data in separate directory (only if requested)

Usage:
    python two_particle_types_microchannel.py
"""

import sys, os
import numpy as np

# -- Path setup (find the build) --
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
sys.path.insert(0, os.path.join(project_dir, "build", "python"))
sys.path.insert(0, os.path.join(project_dir, "python"))

from pysoftflow import SoftFlowSimulation

# ═══════════════════════════════════════════════════════════════
# Parameters
# ═══════════════════════════════════════════════════════════════

NX, NY    = 300, 60           # channel length x width (lattice units)
N_STEPS   = 100000            # total simulation timesteps
TAU       = 0.8               # relaxation time → nu = (tau-0.5)/3 = 0.1
BODY_FX   = 8e-6              # body force driving the flow

# RBC properties
N_RBC     = 25                # number of red blood cells
R_RBC     = 5.0               # RBC radius

# CTC properties
N_CTC     = 10                # number of circulating tumor cells
R_CTC     = 8.0               # CTC radius (larger than RBCs)

# Microplastic properties
N_MP      = 8                 # number of microplastic particles
R_MP      = 3.0               # microplastic radius (small)

# ═══════════════════════════════════════════════════════════════
# Simulation setup
# ═══════════════════════════════════════════════════════════════

sim = SoftFlowSimulation()

# -- Domain and boundaries --
sim.domain(nx=NX, ny=NY)
sim.boundary(x="periodic", y="wall")
sim.fluid(tau=TAU, density=1.0)
sim.body_force(BODY_FX, 0.0)

# -- Obstacle in the center --
sim.obstacle("circle", center=(NX // 2, NY // 2), radius=6.0)

# -- Coupling --
sim.ibm(iterations=2)
sim.lubrication(enabled=True)

# -- Adhesion (microplastics stick to walls and to each other) --
sim.adhesion(enabled=True,
             k_on=0.0003, k_off=0.008,
             k_bond=0.008, d_bond=1.5, F_crit=0.008,
             max_bonds_per_node=1,
             wall_adhesion=True,
             wall_k_on=0.0005, wall_k_off=0.005,
             wall_k_bond=0.01, wall_receptor_spacing=4.0)

# -- Output (VTK only — flat directory, no subdirectories) --
sim.output(format="vtk_legacy",
           directory=os.path.join(script_dir, "output_vtk"),
           interval=1000)

# -- CSV data output (only when explicitly requested, separate directory) --
sim.data_output(trajectory=True, interval=2000, format="csv",
                directory=os.path.join(script_dir, "output_csv"))

# ═══════════════════════════════════════════════════════════════
# Particle types
# ═══════════════════════════════════════════════════════════════

# Type 0: Red Blood Cells (Skalak membrane)
#   - Highly deformable, biconcave-like shape
#   - Low shear modulus G_s → soft under flow
#   - C_skalak controls area dilation resistance
sim.particle_type("rbc",
                  model="skalak",
                  G_s=0.03,
                  C_skalak=5.0,
                  k_bend=0.002,
                  k_area=0.3,
                  k_perimeter=0.03,
                  gamma_visc=0.02)

# Type 1: Circulating Tumor Cells (Neo-Hookean membrane)
#   - Stiffer than RBCs (higher G_s)
#   - Larger radius → more flow disturbance
#   - Neo-Hookean: isotropic elastic response
sim.particle_type("ctc",
                  model="neo_hookean",
                  G_s=0.10,
                  k_bend=0.008,
                  k_area=0.6,
                  k_perimeter=0.06,
                  gamma_visc=0.02)

# Type 2: Microplastics (Hookean membrane)
#   - Small, nearly rigid particles
#   - Very high G_s → minimal deformation
#   - Adhesion makes them stick to walls and form aggregates
sim.particle_type("microplastic",
                  model="hookean",
                  G_s=0.50,
                  k_bend=0.02,
                  k_area=1.0,
                  k_perimeter=0.10,
                  gamma_visc=0.01)

# ═══════════════════════════════════════════════════════════════
# Particle placement
# ═══════════════════════════════════════════════════════════════

# Seeding region: away from walls
margin = max(R_RBC, R_CTC, R_MP) + 2.0
sim.region("channel",
           x=(10, NX - 10),
           y=(margin, NY - margin))

# Place RBCs
sim.generate("rbc", count=N_RBC, region="channel",
             radius=(R_RBC - 0.3, R_RBC + 0.3),
             seed=42, min_gap=2.0)

# Place CTCs
sim.generate("ctc", count=N_CTC, region="channel",
             radius=(R_CTC - 0.3, R_CTC + 0.3),
             seed=123, min_gap=2.0)

# Place microplastics
sim.generate("microplastic", count=N_MP, region="channel",
             radius=(R_MP - 0.2, R_MP + 0.2),
             seed=456, min_gap=1.5)

# ═══════════════════════════════════════════════════════════════
# Checkpoint / Restart
# ═══════════════════════════════════════════════════════════════

ckpt_file = os.path.join(script_dir, "vtk_two_types", "checkpoint.sfck")
if os.path.exists(ckpt_file):
    sim.restart(ckpt_file)
    print(f"Restarted from step {sim._core.currentStep()}")
else:
    sim.warmup(steps=0, ramp_steps=500)

# ═══════════════════════════════════════════════════════════════
# Run
# ═══════════════════════════════════════════════════════════════

#sim.checkpoint(interval=50000)     # auto-save every 50k steps
sim.thermo(interval=500)
sim.run(N_STEPS)
sim.save_checkpoint(ckpt_file)     # final save
