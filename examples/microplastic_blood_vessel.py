"""
Microplastic Contamination in Blood Microcirculation
=====================================================
Simulates microplastic particles (MPs) interacting with red blood cells
(RBCs) and white blood cells (WBCs) in a stenosed microvessel, with
physics-based chemical leaching and Langmuir adsorption.

This is a major public health concern: studies in 2024-2026 detected
microplastics in human blood, brain tissue, and placentas.  Understanding
how MPs marginate, adhere to vessel walls, and release toxic chemicals
(plasticizers, BPA, phthalates) in microcirculation is critical.

Physical setup:
  - Periodic microchannel with a stenosis (narrowing) from an obstacle
  - Body-force driven flow (pressure gradient equivalent)
  - RBCs:   soft, deformable (Skalak membrane) with viscosity contrast
  - WBCs:   stiffer, larger (Neo-Hookean), slower margination
  - MPs:    rigid-ish (Hookean, very high G_s), various sizes
  - Adhesion:  MPs stick to vessel walls (receptor-mediated)
  - Scalar transport:  physics-based Fick leaching from MPs;
                       Langmuir adsorption on RBC/WBC surfaces
  - Lubrication + viscosity contrast for realistic cell interactions
  - Segregation metrics to quantify margination (CFL thickness)

Chemistry model:
  MP leaching (Fick-type):
      J_leach = k_L * (C_eq - C_surface)          [concentration/step]
      dM_p/dt = -J_leach * A_node                 [mass conservation]

  Cell surface adsorption (Langmuir):
      dΓ/dt = k_a * C_surface * (1 - Γ/Γ_max) - k_d * Γ
      Langmuir equilibrium: Γ_eq = C * k_a / (k_d + k_a * C)

  Dimensionless numbers (estimated):
      Pe  = U * L / D  ≈ 0.04 * 400 / 0.005 ≈ 3200   (advection-dominated)
      Da  = k_L * L/U  ≈ 0.002 * 400/0.04  ≈  20     (leaching-dominated)
      Bi  = k_L * R/D  ≈ 0.002 * 3/0.005   ≈   1.2   (moderate resistance)

Key questions:
  1. Do smaller MPs marginate toward the vessel wall faster?
  2. How does stenosis affect MP trapping and wall adhesion?
  3. What is the spatial distribution of leached chemicals?
  4. Does the cell-free layer (CFL) act as a highway for MP transport?
  5. When does MP mass reservoir deplete — uniform or downstream-shifted?
  6. How does RBC surface coverage Γ evolve with time?

Output:
  - VTK files: fluid (velocity, density, concentration) + particles + Γ field
  - CSV trajectory & concentration timeseries
  - global_stats.csv: total_dissolved_mass, total_particle_mass, total_adsorbed_mass
  - Segregation metrics (margination, CFL, mixing entropy)

Usage:
    python microplastic_blood_vessel.py

References:
    Leslie et al., Environment International (2022) — MPs in human blood
    Ragusa et al., Environment International (2021) — MPs in human placenta
    Campen et al., Toxicological Sciences (2023) — MPs cross blood-brain barrier
"""

import sys, os
import numpy as np

# -- Path setup (find the build) --
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
sys.path.insert(0, os.path.join(project_dir, "build", "python"))
sys.path.insert(0, os.path.join(project_dir, "python"))

from pysoftflow import SoftFlowSimulation

# =================================================================
# Physical parameters (lattice units)
# =================================================================

NX, NY    = 400, 80           # vessel length x diameter (lattice units)
N_STEPS   = 200000            # total simulation timesteps
TAU       = 0.8               # relaxation time -> nu = (tau-0.5)/3 = 0.1
BODY_FX   = 5e-6              # body force ~ pressure gradient

# Chemistry starts only after flow + cells have fully developed.
# ~12 000 steps ≈ 30 flow-through times (L/U = 400/0.04 = 10 000 / ~0.04 ≈ 400)
WARMUP_CHEM_STEP = 12000

# Red blood cells
N_RBC     = 30                # ~40% hematocrit (realistic for 400x80)
R_RBC     = 6.0               # RBC effective radius

# White blood cells (larger, stiffer)
N_WBC     = 12                # ~1 WBC per 600 RBCs (scaled down)
R_WBC     = 12.0               # WBC radius (larger than RBC)

# Microplastics (polydisperse: small fragments + larger beads)
N_MP_SMALL = 12               # small fragments (< RBC size)
N_MP_LARGE = 10                # larger beads  (~ RBC size)
R_MP_SMALL = 2.5              # small MP radius
R_MP_LARGE = 5.0              # large MP radius

# Stenosis geometry
STENOSIS_X      = NX // 2     # stenosis at channel center
STENOSIS_RADIUS = 15.0        # obstacle radius creating the narrowing

# =================================================================
# Simulation setup
# =================================================================

sim = SoftFlowSimulation()

# -- Domain: periodic vessel with top/bottom walls --
sim.domain(nx=NX, ny=NY)
sim.boundary(x="periodic", y="wall")
sim.fluid(tau=TAU, density=1.0, collision="regularized")
sim.body_force(BODY_FX, 0.0)

# -- Stenosis: two circular obstacles creating a symmetric constriction --
#    This mimics an atherosclerotic plaque or vessel narrowing where
#    MPs are known to preferentially accumulate.
sim.obstacle("circle",
             center=(STENOSIS_X, 8),
             radius=STENOSIS_RADIUS)
sim.obstacle("circle",
             center=(STENOSIS_X, NY - 8),
             radius=STENOSIS_RADIUS)

# -- IBM coupling (multi-direct forcing for accuracy) --
sim.ibm(iterations=2)

# -- Lubrication corrections (critical for dense RBC suspensions) --
sim.lubrication(enabled=True)

# -- Adhesion: MPs stick to the vessel wall --
#    This models the observed tendency of microplastics to adhere to
#    endothelial surfaces via non-specific (van der Waals, electrostatic)
#    and receptor-mediated interactions.
sim.adhesion(enabled=True,
             k_on=0.0002,              # bond formation rate
             k_off=0.005,              # bond dissociation rate
             k_bond=0.006,             # bond stiffness
             d_bond=1.5,               # equilibrium bond length
             F_crit=0.010,             # critical force for slip
             max_bonds_per_node=2,
             wall_adhesion=True,       # key: MPs adhere to wall
             wall_k_on=0.0004,         # higher wall affinity
             wall_k_off=0.003,
             wall_k_bond=0.008,
             wall_receptor_spacing=3.0)

# -- Scalar transport: chemical leaching from microplastics --
#    MPs continuously release plasticizers (BPA, phthalates, PFAS)
#    into the blood.  RBCs and WBCs absorb these toxins.
sim.scalar_transport(enabled=True,
                     diffusivity=0.005,    # chemical diffusion coefficient
                     n_species=1,
                     initial_concentration=0.0)

# -- Viscosity contrast: RBC interior is more viscous than plasma --
sim.viscosity_contrast(enabled=True, update_interval=50)

# -- Segregation metrics: measure margination and CFL --
#sim.metrics(interval=10000)

# -- Output: VTK for ParaView (flat directory) --
sim.output(format="vtk_legacy",
           directory=os.path.join(script_dir, "vtk_mp_blood"),
           interval=2000)

# -- CSV data: trajectory + concentration to separate directory --
sim.data_output(trajectory=True,
                timeseries=True,
                positions=True,
                bonds=True,
                interval=5000,
                format="csv",
                directory=os.path.join(script_dir, "data_mp_blood"))

# =================================================================
# Particle types
# =================================================================

# Type 0: Red Blood Cells (Skalak membrane)
#   - Biconcave disc shape, highly deformable
#   - Interior viscosity ~5x plasma (viscosity_ratio)
#   - Skalak model captures area-dilation resistance
sim.particle_type("rbc",
                  model="skalak",
                  G_s=0.04,              # shear modulus (soft)
                  C_skalak=10.0,         # area dilation resistance
                  k_bend=0.003,          # bending stiffness
                  k_area=0.5,            # area conservation
                  k_perimeter=0.05,      # perimeter conservation
                  gamma_visc=0.02,       # membrane viscous damping
                  viscosity_ratio=5.0)   # lambda = eta_in / eta_out

# Type 1: White Blood Cells (Neo-Hookean)
#   - Larger, stiffer than RBCs
#   - Known to marginate toward the wall
#   - Neo-Hookean: nonlinear hyperelastic
sim.particle_type("wbc",
                  model="neo_hookean",
                  G_s=0.12,              # stiffer than RBC
                  k_bend=0.010,
                  k_area=0.8,
                  k_perimeter=0.08,
                  gamma_visc=0.03,
                  viscosity_ratio=3.0)

# Type 2: Small Microplastic Fragments (Hookean, very stiff)
#   - Small rigid particles (PE, PP, PS fragments)
#   - Nearly rigid: very high shear modulus
#   - Leach chemicals into surrounding fluid
sim.particle_type("mp_small",
                  model="hookean",
                  G_s=0.80,
                  k_bend=0.05,
                  k_area=2.0,
                  k_perimeter=0.20,
                  gamma_visc=0.01,
                  is_rigid=True)         # PE/PP fragments: E~GPa → zero deformation

# Type 3: Large Microplastic Beads (Hookean, truly rigid)
#   - Larger spherical microbeads (from cosmetics, textiles)
sim.particle_type("mp_large",
                  model="hookean",
                  G_s=1.00,
                  k_bend=0.08,
                  k_area=3.0,
                  k_perimeter=0.30,
                  gamma_visc=0.01,
                  is_rigid=True)         # rigid body: translates + rotates, never deforms

# -- Physics-based chemical transport --
# Chemistry parameters are applied in Phase 2 (after flow equilibration).
# See the run block below.

# =================================================================
# Particle placement
# =================================================================

# Seeding region: full channel away from walls and stenosis
margin = R_WBC + 3.0
sim.region("vessel",
           x=(10, NX - 10),
           y=(margin, NY - margin))

# Also define a pre-stenosis region for initial MP placement
# (MPs entering the stenosis from upstream)
sim.region("upstream",
           x=(10, STENOSIS_X - 30),
           y=(margin, NY - margin))

# Place RBCs throughout the vessel
sim.generate("rbc", count=N_RBC, region="vessel",
             radius=(R_RBC - 0.3, R_RBC + 0.3),
             seed=42, min_gap=2.0)

# Place WBCs (scattered)
sim.generate("wbc", count=N_WBC, region="vessel",
             radius=(R_WBC - 0.3, R_WBC + 0.3),
             seed=100, min_gap=3.0)

# Place small MP fragments upstream of stenosis
sim.generate("mp_small", count=N_MP_SMALL, region="upstream",
             radius=(R_MP_SMALL - 0.2, R_MP_SMALL + 0.2),
             seed=200, min_gap=1.5)

# Place large MP beads upstream
sim.generate("mp_large", count=N_MP_LARGE, region="upstream",
             radius=(R_MP_LARGE - 0.3, R_MP_LARGE + 0.3),
             seed=300, min_gap=2.0)

# =================================================================
# Checkpoint / Restart + Two-phase run
# =================================================================

ckpt_file = os.path.join(script_dir, "vtk_mp_blood", "checkpoint.sfck")
current_step = 0
if os.path.exists(ckpt_file):
    sim.restart(ckpt_file)
    current_step = sim._core.currentStep()
    print(f"Restarted from step {current_step}")
else:
    sim.warmup(steps=1000, ramp_steps=2000)

# ── Phase 1: flow + cell distribution, NO chemistry ──────────────
# Allow Poiseuille profile and cell distribution to reach steady state
# before any leaching starts; prevents the t=0 concentration burst.
if current_step < WARMUP_CHEM_STEP:
    remaining = WARMUP_CHEM_STEP - current_step
    print(f"\nPhase 1: flow equilibration ({remaining} steps, no chemistry)...")
    sim.thermo(interval=1000)
    sim.checkpoint(interval=50000)
    sim.run(remaining)
    current_step = WARMUP_CHEM_STEP

# ── Phase 2: slow physics-based chemistry ────────────────────────
# Da = k_L × L/U ≈ 0.0002 × 400/0.04 ≈ 2  →  diffusion-limited regime
# → steady monotonic rise in dissolved C; no burst then drop.
print("\nPhase 2: chemistry active (slow Fick leaching + gradual Langmuir adsorption)")

# Small MPs: slow steady leaching; large reservoir for long-term release
sim.scalar_source("mp_small",
                  k_leach=0.0002,         # 10× slower than before → Da ≈ 2
                  C_eq=1.0,
                  M_p_initial=300.0)

# Large MPs: slower leaching (lower surface-to-volume ratio), bigger reservoir
sim.scalar_source("mp_large",
                  k_leach=0.00015,
                  C_eq=1.0,
                  M_p_initial=600.0)

# RBCs: gradual Langmuir adsorption — Γ saturates over ~50 000 steps at C~0.1
#   k_a/k_d = 10 → Γ_eq ≈ 0.91 at C=1
sim.scalar_source("rbc",
                  k_adsorb=0.0005,
                  k_desorb=0.00005,
                  Gamma_max=1.0)

# WBCs: stronger adsorption (immune cells trap xenobiotics more efficiently)
#   k_a/k_d = 27 → Γ_eq ≈ 0.96 at C=1
sim.scalar_source("wbc",
                  k_adsorb=0.0008,
                  k_desorb=0.00003,
                  Gamma_max=1.0)

sim.thermo(interval=1000)
sim.checkpoint(interval=100000)
sim.run(N_STEPS - current_step)
sim.save_checkpoint(ckpt_file)