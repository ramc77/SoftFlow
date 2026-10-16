"""
Density-Driven Particle Segregation
=====================================
Demonstrates the new capsule density + buoyancy feature.

Three particle types with different densities flow in a periodic channel
under gravity:

  - **Light** (PE plastic, ρ = 0.927) — buoyant, migrate UPWARD
  - **Neutral** (PS plastic, ρ ≈ 1.0)  — neutrally buoyant, stay mid-channel
  - **Heavy** (PET plastic, ρ = 1.346) — sink DOWNWARD

All particles are driven by a body force (Poiseuille flow) with periodic
boundaries. Gravity acts on both fluid and capsules.

Expected behavior:
  - Light particles accumulate near the TOP wall
  - Heavy particles accumulate near the BOTTOM wall
  - Neutral particles distribute throughout the channel

Run:
    python density_driven_particles.py

View in ParaView:
    Open vtk_density_particles/particles_*.vtk
    Color by "particle_type" (0=light, 1=neutral, 2=heavy)
    Or open fluid_*.vtk → Color by "density" to see fluid field
"""

import sys, os

# ── Path setup ──────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(PROJECT_DIR, "build", "python"))
sys.path.insert(0, os.path.join(PROJECT_DIR, "python"))

from pysoftflow import SoftFlowSimulation

# ══════════════════════════════════════════════════════════════════
# Parameters
# ══════════════════════════════════════════════════════════════════
NX, NY     = 300, 80           # channel dimensions (lattice units)
TAU        = 0.8               # relaxation time (nu = 0.1)
BODY_FX    = 5e-6              # body force driving Poiseuille flow
GRAVITY_Y  = -3e-5             # downward gravity (lattice units)
N_STEPS    = 100_000           # total timesteps
VTK_EVERY  = 1000              # VTK output frequency
DATA_EVERY = 500               # CSV data output frequency

# Particle counts
N_LIGHT    = 10                # PE-like (buoyant)
N_NEUTRAL  = 10                # PS-like (neutrally buoyant)
N_HEAVY    = 10                # PET-like (sinks)

# Particle radii
R_PARTICLE = 3.0               # same size for fair comparison

# Densities (lattice units, normalized to fluid rho0 = 1.0)
# Physical reference: plasma = 1025 kg/m³
RHO_LIGHT   = 0.927            # PE: 950/1025 — lighter than blood
RHO_NEUTRAL = 1.015            # PS: 1040/1025 — nearly neutral
RHO_HEAVY   = 1.346            # PET: 1380/1025 — much heavier

# Membrane stiffness (all same — we isolate density effect)
G_S        = 0.5               # moderately stiff
K_BEND     = 0.02
K_AREA     = 1.0
K_PERIM    = 0.1
GAMMA      = 0.02

# Stenosis geometry
STENOSIS_X      = NX // 2     # stenosis at channel center
STENOSIS_RADIUS = 15.0        # obstacle radius creating the narrowing

# Output
OUT_VTK  = os.path.join(SCRIPT_DIR, "vtk_density_particles")
OUT_DATA = os.path.join(SCRIPT_DIR, "data_density_particles")

# ══════════════════════════════════════════════════════════════════
# Build simulation
# ══════════════════════════════════════════════════════════════════
sim = SoftFlowSimulation()

# Domain: periodic in x, walls in y
sim.domain(nx=NX, ny=NY)
sim.boundary(x="periodic", y="wall")
sim.fluid(tau=TAU, density=1.0, collision="bgk")

# Body force drives flow from left to right
sim.body_force(BODY_FX, 0.0)

# Gravity acts on BOTH fluid and capsules
sim.gravity(gx=0.0, gy=GRAVITY_Y, apply_to="all")

sim.obstacle("circle",
             center=(STENOSIS_X, 8),
             radius=STENOSIS_RADIUS)
sim.obstacle("circle",
             center=(STENOSIS_X, NY - 8),
             radius=STENOSIS_RADIUS)

# IBM coupling
sim.ibm(iterations=2)
sim.lubrication(enabled=True)

# ── Particle types (same stiffness, different densities) ──

# Type 0: Light (PE plastic) — should float UP
sim.particle_type("light_PE",
                  model="neo_hookean",
                  G_s=G_S, k_bend=K_BEND,
                  k_area=K_AREA, k_perimeter=K_PERIM,
                  gamma_visc=GAMMA,
                  density=RHO_LIGHT)

# Type 1: Neutral (PS plastic) — stays mid-channel
sim.particle_type("neutral_PS",
                  model="neo_hookean",
                  G_s=G_S, k_bend=K_BEND,
                  k_area=K_AREA, k_perimeter=K_PERIM,
                  gamma_visc=GAMMA,
                  density=RHO_NEUTRAL)

# Type 2: Heavy (PET plastic) — should sink DOWN
sim.particle_type("heavy_PET",
                  model="neo_hookean",
                  G_s=G_S, k_bend=K_BEND,
                  k_area=K_AREA, k_perimeter=K_PERIM,
                  gamma_visc=GAMMA,
                  density=RHO_HEAVY)

# ── Regions ──
margin = R_PARTICLE + 2.0
sim.region("channel", x=(10, NX - 10), y=(margin, NY - margin))

# ── Generate particles (all in same region, will segregate by density) ──
sim.generate("light_PE",   count=N_LIGHT,   region="channel",
             radius=R_PARTICLE, seed=42, min_gap=2.0)
sim.generate("neutral_PS", count=N_NEUTRAL, region="channel",
             radius=R_PARTICLE, seed=100, min_gap=2.0)
sim.generate("heavy_PET",  count=N_HEAVY,   region="channel",
             radius=R_PARTICLE, seed=200, min_gap=2.0)

# ── Segregation metrics ──
sim.metrics(interval=5000)

# ── Output ──
sim.output(format="vtk_legacy", directory=OUT_VTK, interval=VTK_EVERY)
sim.data_output(
    trajectory=True, timeseries=True,
    interval=DATA_EVERY,
    format="csv",
    directory=OUT_DATA,
)

# =================================================================
# Checkpoint / Restart
# =================================================================
ckpt_file = os.path.join(SCRIPT_DIR, "vtk_density_particles", "checkpoint.sfck")
if os.path.exists(ckpt_file):
    sim.restart(ckpt_file)
    print(f"Restarted from step {sim._core.currentStep()}")
else:
    sim.initialize()
    sim.warmup(steps=2000, ramp_steps=1000)

# =================================================================
# Run
# =================================================================
sim.thermo(interval=1000)
#sim.checkpoint(interval=100000)       # auto-save every 100k steps
sim.run(N_STEPS)
sim.save_checkpoint(ckpt_file)       # final save