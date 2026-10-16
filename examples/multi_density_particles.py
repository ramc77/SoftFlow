"""
Multi-Density Particles in Periodic Channel Flow
==================================================
Demonstrates density-driven segregation of deformable particles (capsules)
in a Poiseuille flow channel under gravity.

Three microplastic types with DIFFERENT densities:
  • PE  (polyethylene)   — density = 0.93  → LIGHTER than plasma → rises (buoyant)
  • PS  (polystyrene)    — density = 1.00  → SAME as plasma      → stays mid-channel
  • PET (polyethylene-T) — density = 1.35  → HEAVIER than plasma → sinks

Physics:
  - Horizontal Poiseuille flow driven by body force (periodic x-boundaries)
  - Gravity pulls particles downward according to density mismatch with fluid
  - Buoyancy force per node: F = (ρ_cap - ρ_fluid) × A × g / N_nodes
  - Lubrication prevents particle-wall/particle-particle overlap

Literature densities:
  PE:  920–970 kg/m³  (Leslie et al. 2022)
  PS:  1040–1060 kg/m³ (≈ plasma 1025 kg/m³)
  PET: 1380–1400 kg/m³ (Leslie et al. 2022)

Run:
    python multi_density_particles.py

View in ParaView:
    Open vtk_multi_density/capsules_*.vtk
    Color by "type_id"  →  Discrete colormap
    Type 0 (blue)  = PE  → near top wall
    Type 1 (green) = PS  → mid-channel
    Type 2 (red)   = PET → near bottom wall
"""

import sys, os

# ── Path setup ────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(PROJECT_DIR, "build", "python"))
sys.path.insert(0, os.path.join(PROJECT_DIR, "python"))

from pysoftflow import SoftFlowSimulation

# ══════════════════════════════════════════════════════════════════════
# Parameters
# ══════════════════════════════════════════════════════════════════════

# Domain
NX, NY      = 400, 120      # 400×120 landscape channel

# Fluid
TAU         = 0.9           # relaxation time (nu = 0.1333 lu²/ts)
BODY_FX     = 3e-6          # body force → max velocity ≈ fx×ny²/(8ν) ≈ 0.027 lu/ts
                            #           → Re ≈ u_max × ny / ν ≈ 0.027×120/0.133 ≈ 0.24

# Gravity (downward) — applied to BOTH fluid and capsules
# Buoyancy drives density-based segregation
# Ma check: over 80000 steps, gravity adds u ≈ gy × few-hundred = 2e-5 × 300 = 0.006 ✓
GRAVITY_Y   = -2e-5

# Particle geometry
R_CAPSULE   = 6.0           # radius [lu] ≈ 1.5 µm physical (dx = 0.5 µm)
N_PE        = 12            # PE particles (buoyant, ρ < fluid)
N_PS        = 12            # PS particles (neutral, ρ ≈ fluid)
N_PET       = 12            # PET particles (heavy, ρ > fluid)

# Membrane mechanics (same stiffness, only density differs)
# G_s chosen so Ca ≈ 0.05-0.1 (moderate deformation)
G_S         = 0.6
K_BEND      = 0.01
K_AREA      = 0.10

# Simulation
N_WARMUP    = 5_000
N_STEPS     = 80_000
VTK_EVERY   = 1_000
DATA_EVERY  = 2_000

OUT_DIR  = os.path.join(SCRIPT_DIR, "vtk_multi_density")
DATA_DIR = os.path.join(SCRIPT_DIR, "data_multi_density")

# ══════════════════════════════════════════════════════════════════════
# Build simulation
# ══════════════════════════════════════════════════════════════════════
sim = SoftFlowSimulation()

# Domain: periodic in x, wall-bounded in y
sim.domain(nx=NX, ny=NY)
sim.boundary(x="periodic", y="wall")
sim.fluid(tau=TAU, density=1.0, collision="bgk")
sim.body_force(fx=BODY_FX, fy=0.0)

# Gravity applied to BOTH fluid (ρ × g) and capsules (buoyancy = Δρ × A × g)
sim.gravity(gx=0.0, gy=GRAVITY_Y, apply_to="all")

# IBM coupling (2 iterations for better no-slip enforcement)
sim.ibm(iterations=2)

# Lubrication: prevent particle-wall and particle-particle overlap
sim.lubrication(enabled=True, h_threshold=1.5, h_min=0.1)

# ── Particle types ─────────────────────────────────────────────────────
# All three types have identical elastic properties — only density differs.
# This isolates the density effect from stiffness effects.

# Type 0: PE microplastic — buoyant (lighter than plasma → rises)
# Physical: ρ_PE ≈ 940 kg/m³, ρ_plasma = 1025 kg/m³
# Lattice: ρ_fluid = 1.0, ρ_PE = 940/1025 ≈ 0.917
sim.particle_type(
    "PE_plastic",
    model="neo_hookean",
    G_s=G_S,
    k_bend=K_BEND,
    k_area=K_AREA,
    density=0.917,          # ← lighter than fluid → buoyancy UP
)

# Type 1: PS microplastic — neutrally buoyant (≈ plasma density → stays mid-channel)
# Physical: ρ_PS ≈ 1050 kg/m³ ≈ ρ_plasma
# Lattice: ρ_PS / ρ_plasma ≈ 1.024 ≈ 1.0 (effectively neutral)
sim.particle_type(
    "PS_plastic",
    model="neo_hookean",
    G_s=G_S,
    k_bend=K_BEND,
    k_area=K_AREA,
    density=1.0,            # ← same as fluid → no net buoyancy (standard massless IBM)
)

# Type 2: PET microplastic — heavy (heavier than plasma → sinks)
# Physical: ρ_PET ≈ 1390 kg/m³, ρ_plasma = 1025 kg/m³
# Lattice: ρ_PET = 1390/1025 ≈ 1.356
sim.particle_type(
    "PET_plastic",
    model="neo_hookean",
    G_s=G_S,
    k_bend=K_BEND,
    k_area=K_AREA,
    density=1.356,          # ← heavier than fluid → sinks DOWN
)

# ── Placement regions ───────────────────────────────────────────────────
# Distribute all particles across the full channel length
sim.region("full_channel", x=(10, NX-10), y=(int(R_CAPSULE)+2, NY-int(R_CAPSULE)-2))

sim.generate("PE_plastic",  count=N_PE,  region="full_channel",
             radius=R_CAPSULE, seed=42,  min_gap=0.5)
sim.generate("PS_plastic",  count=N_PS,  region="full_channel",
             radius=R_CAPSULE, seed=100, min_gap=0.5)
sim.generate("PET_plastic", count=N_PET, region="full_channel",
             radius=R_CAPSULE, seed=200, min_gap=0.5)

# =================================================================
# Checkpoint / Restart
# =================================================================

ckpt_file = os.path.join(script_dir, "vtk_mp_blood", "checkpoint.sfck")
if os.path.exists(ckpt_file):
    sim.restart(ckpt_file)
    print(f"Restarted from step {sim._core.currentStep()}")
else:
    sim.initialize()
    sim.warmup(steps=1000, ramp_steps=2000)


# ── Output ──────────────────────────────────────────────────────────────
sim.output(format="vtk_legacy", directory=OUT_DIR, interval=VTK_EVERY)
sim.data_output(
    trajectory=True,
    timeseries=True,
    interval=DATA_EVERY,
    format="csv",
    directory=DATA_DIR,
)

# =================================================================
# Run
# =================================================================
sim.thermo(interval=1000)
#sim.checkpoint(interval=100000)       # auto-save every 100k steps
sim.run(N_STEPS)
sim.save_checkpoint(ckpt_file)       # final save
