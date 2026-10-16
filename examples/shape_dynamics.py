"""
Shape-Dependent Flow Dynamics in a Microchannel
================================================
Demonstrates all three new capsule shape features:

  1. Biconcave RBCs  — realistic RBC cross-section (indent_depth=0.4)
  2. Rigid spherical microplastics — is_rigid=True, translate + rotate, no deformation
  3. Fiber-shaped microplastics  — shape="fiber", aspect_ratio=0.35 (L/D ≈ 2.9)

All three types flow together in a Poiseuille channel.
Their different shapes produce visibly different dynamics:

  - RBCs (biconcave):  tank-tread and deform; migrate toward channel center
  - Rigid spheres:     tumble rigidly without deforming; marginate toward walls
  - Fibers:            tumble end-over-end (Jeffery orbits); elongated shape visible

Physical scales (arteriole):
  dx = 0.5 µm/lu  →  channel = 150 x 50 µm
  dt = 0.32 µs/ts →  Re ≈ 0.05  (Stokes regime)

Usage:
    OMP_NUM_THREADS=8 python shape_dynamics.py

ParaView:
    Open vtk_shape_dynamics/fluid_*.vtk  — color by velocity magnitude
    Open vtk_shape_dynamics/capsules_*.vtk — color by type_id:
        0 = RBC (biconcave)   → blue
        1 = rigid sphere MP   → red
        2 = fiber MP          → green
"""

import sys, os
import numpy as np

script_dir  = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
sys.path.insert(0, os.path.join(project_dir, "build", "python"))
sys.path.insert(0, os.path.join(project_dir, "python"))

from pysoftflow import SoftFlowSimulation

# IBM resolution rule: the semi-minor axis of any particle must be >= 1.0 lu.
# For a fiber: b = R_FIBER * aspect_ratio  >=  1.0
# Violating this causes sub-lattice nodes that share the same IBM support cell
# → force cancellation → velocity explosion within a few thousand steps.
IBM_MIN_SEMIAXIS = 1.0   # lu

# ─── Domain ───────────────────────────────────────────────────────────────
NX, NY   = 300, 100       # 150 µm × 50 µm at dx = 0.5 µm/lu
TAU      = 0.8            # nu = (tau-0.5)/3 = 0.1 lu²/ts
BODY_FX  = 2e-6           # body force → u_max ≈ 0.015 lu/ts  (Ma ≈ 0.026)
N_STEPS  = 100_000

# ─── Particle counts ──────────────────────────────────────────────────────
N_RBC    = 8              # biconcave, Skalak
N_RIGID  = 8              # rigid spheres (PE microplastic)
N_FIBER  = 8              # fiber-shaped (PET microplastic)

# ─── Radii ────────────────────────────────────────────────────────────────
R_RBC    = 5.0            # effective radius
R_RIGID  = 3.0            # rigid sphere radius
R_FIBER  = 5.0            # fiber semi-major axis a
FIBER_AR = 0.35           # aspect ratio b/a → semi-minor b = 5*0.35 = 1.75 lu  ✓
assert R_FIBER * FIBER_AR >= IBM_MIN_SEMIAXIS, \
    f"Fiber semi-minor {R_FIBER*FIBER_AR:.2f} lu < IBM limit {IBM_MIN_SEMIAXIS} lu — increase R_FIBER or aspect_ratio"

# ─── Simulation setup ─────────────────────────────────────────────────────
sim = SoftFlowSimulation()

sim.domain(nx=NX, ny=NY)
sim.boundary(x="periodic", y="wall")
sim.fluid(tau=TAU, density=1.0, collision="mrt")  # MRT: more stable under large IBM forces
sim.body_force(BODY_FX, 0.0)
sim.ibm(iterations=1)
sim.lubrication(enabled=True, h_threshold=3.0, h_min=0.5)  # earlier onset, gentler floor

# ─── Particle types ────────────────────────────────────────────────────────

# Type 0: Biconcave RBC
#   shape="biconcave": r(θ) = R*(1 - 0.4*sin²θ) — dimpled disc cross-section
#   Skalak membrane: strain-hardening + strong area conservation
#   viscosity_ratio=5: interior 5× more viscous than plasma (literature value)
sim.particle_type("rbc",
                  model="skalak",
                  G_s=0.04,
                  C_skalak=10.0,
                  k_bend=0.003,
                  k_area=0.5,
                  k_perimeter=0.05,
                  gamma_visc=0.02,
                  viscosity_ratio=5.0,
                  shape="biconcave",
                  indent_depth=0.4)       # 0=circle, 0.4=realistic RBC dimple

# Type 1: Rigid spherical microplastic (PE, neutrally buoyant)
#   is_rigid=True: IBM velocities projected to rigid translation + rotation
#   No membrane deformation — shape is perfectly preserved at all times
#   neutrally buoyant (density=1.0): shape effect only, no gravity drift
sim.particle_type("rigid_mp",
                  model="hookean",
                  k_stretch=0.5,          # softer: collision force less sharp
                  k_bend=0.02,
                  k_area=1.0,
                  k_perimeter=0.1,
                  gamma_visc=0.05,        # higher damping: absorbs collision energy
                  density=1.0,
                  is_rigid=True)          # rigid body — no deformation

# Type 2: Fiber-shaped microplastic
#   shape="fiber": x=R*cos, y=R*aspect_ratio*sin
#   aspect_ratio=0.35 → semi-minor b = 5*0.35 = 1.75 lu  (above IBM limit of ~1 lu)
#   L/D ≈ 1/0.35 ≈ 2.9  (moderately elongated — safe for IBM)
#   High k_bend: fiber resists folding along its axis
sim.particle_type("fiber_mp",
                  model="neo_hookean",
                  G_s=0.5,
                  k_bend=0.08,            # resists bending / folding
                  k_area=1.0,
                  k_perimeter=0.3,
                  gamma_visc=0.01,
                  density=1.0,
                  shape="fiber",
                  aspect_ratio=FIBER_AR)

# ─── Particle placement ───────────────────────────────────────────────────
margin = 12.0  # keep particles well away from walls (fiber semi-major=5 lu)

sim.region("channel", x=(10, NX-10), y=(margin, NY-margin))

#sim.generate("rbc", count=N_RBC,   region="channel", radius=R_RBC,   seed=42,  min_gap=4.0)

sim.generate("rigid_mp", count=N_RIGID, region="channel",
             radius=R_RIGID, seed=77,  min_gap=4.0)

sim.generate("fiber_mp", count=N_FIBER, region="channel",
             radius=R_FIBER, seed=99,  min_gap=4.0)

# ─── Output ───────────────────────────────────────────────────────────────
out_dir = os.path.join(script_dir, "vtk_shape_dynamics")
sim.output(format="vtk_legacy", directory=out_dir, interval=500)

# ─── Segregation metrics ──────────────────────────────────────────────────
#sim.metrics(interval=10000)

# ─── Run ──────────────────────────────────────────────────────────────────
sim.thermo(interval=2000)
sim.warmup(steps=2000, ramp_steps=1000)
sim.run(N_STEPS)