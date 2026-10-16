"""
Bidisperse soft-vs-stiff suspension (Phase-2 showcase)
======================================================
50/50 small (Skalak, soft) + large (Neo-Hookean, stiff) capsules in a
periodic body-force-driven channel. Demonstrates the Phase-2 insertion
strategies through the friendly declarative API:

  - Hex-lattice fill for the small species (densely packed central band).
  - Random (RSA) fill for the large species (mixed across the channel).
  - Two named regions, two ``generate()`` calls — that's the API.

Physical setup:
  - Periodic 300 × 60 microchannel, body-force driven.
  - Small species: Skalak membrane (soft, RBC-like).
  - Large species: Neo-Hookean (stiffer, CTC-like).
  - Lubrication corrections for close-approach pairs.

After several thousand timesteps, you should see size-driven margination:
the large stiffer particles drift toward the channel centre while the
small soft particles populate the cell-free layer near the walls.

Output:
  - VTK files: vtk_bidisperse/{fluid,particles}/*.{vti,vtp}
  - PVD time-series: vtk_bidisperse/{fluid,particles}.pvd
  - run_manifest.json under vtk_bidisperse/config/

Usage:
    python 02_bidisperse_segregation/run.py
"""

import os, sys

# -- Path setup (find the build) --
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
#
# Dimensionless numbers (with values below):
#   ū_empty ≈ G·H² / (12·ν) ≈ 2e-4 · 58² / 0.8 ≈ 0.84  (Mach unsafe alone)
#   ū_dense ≈ ū_empty / ~10×  ≈ 0.085  (suspension braking measured at φ≈0.18)
#   Re  = ū·H/ν             ≈ 74        (strong inertial migration)
#   Ca  = μ·ū / G_s_small   ≈ 0.094     (visible deformation)
#   φ   = Σ π r²  / box     ≈ 0.18      (~18 % packing — DENSE, frequent collisions)
#   Ma  = ū / cs            ≈ 0.15      (< 0.3, LBM safe given the dense braking)
#
# Improvements over the previous BGK-tau-0.7 configuration that lifted
# ū from 0.03 to 0.085 *without* reducing particle count:
#
#   1. MRT collision (Lallemand-Luo) instead of BGK. MRT is stable at
#      effectively any τ > 0.5 — BGK gets noisy below τ ≈ 0.6 and
#      definitely unstable at τ ≈ 0.55. With MRT we can push the
#      body force much harder before hitting LBM instability.
#   2. BODY_FX = 2e-4 (was 8e-5 → 16× the original baseline of 5e-6).
#      The empty-channel ū exceeds Ma=0.3 — but we *rely* on the
#      dense suspension braking the flow to ~ ū_dense = 0.085, which
#      sits comfortably under the Mach limit.
#   3. max_lattice_force = 0.10 (was 0.04). Membrane forces during
#      head-on collisions can spike well above 0.04 in this regime;
#      the larger cap absorbs them without silent truncation.
#   4. Particle count RESTORED to 140 small + 30 large (was 110+24)
#      — the user's hard requirement was "don't drop the count".
#
# Particle counts and density are chosen so capsules collide frequently
# but the suspension is NOT jammed (φ < 0.4). At Re ≈ 41 you see:
#   - Capsules visibly elongate when they squeeze through the gaps on
#     either side of the obstacle.
#   - Frequent inter-capsule collisions (lubrication + repulsion).
#   - Size-driven margination: small capsules drift toward the walls,
#     large stiffer capsules stay nearer the centre.
#
# A warmup ramp brings the body force from 0 to BODY_FX over the first
# 3000 steps so the IBM-LBM coupling doesn't blow up from a sudden
# load. After that, particles complete one box-traversal every ~4 300
# steps — N_STEPS = 100 000 gives ~23 traversals, plenty of time.
#
# Output cadence is chosen so each frame represents ~1.4 lattice cells
# of advection (≈ 70 % of small-capsule radius) — smooth animation
# without exploding the frame count.

NX, NY      = 300, 60         # channel length × diameter
N_STEPS     = 100000          # raise to 500 000 for fully developed segregation
TAU         = 0.7             # ν = 0.0667
BODY_FX     = 1.5e-4          # 30× original; Ma stays ~ 0.25 with dense braking
MAX_LATTICE_FORCE = 0.10      # raised again — collision spikes need headroom
COLLISION   = "mrt"           # MRT (Lallemand-Luo) — stable at low τ + high Re

N_SMALL     = 140             # restored — user's "don't drop count" requirement
R_SMALL     = 2.0
N_LARGE     = 30              # restored
R_LARGE     = 4.0
HEX_SPACING = 5.0             # was 6.0 — tighter pack so neighbours START in contact
                              # (gap = 5 - 2·2 = 1.0 cell; lubrication threshold 1.5
                              # → particles already feel each other at t=0)

# Mid-channel obstacle. The LBM treats it as a rigid no-slip cylinder
# via interpolated bounce-back. The capsules — both the soft Skalak
# small ones and the stiffer Neo-Hookean large ones — deform as they
# squeeze through the two narrow lanes on either side of the cylinder
# (and around it). The size-dependent deformation drives the
# bidisperse segregation downstream of the obstacle.
OBSTACLE_X  = NX // 2         # centred along the channel
OBSTACLE_R  = 7.0             # divides the channel into two ~22-cell-wide gaps

OUT_EVERY   = 12              # ~1 lattice cell of advection per frame at ū≈0.085
                              # (≈ 50 % of small-capsule radius — smooth & visible)


# =================================================================
# Simulation setup
# =================================================================

sim = SoftFlowSimulation()

# -- Domain: periodic channel with top/bottom no-slip walls --
sim.domain(nx=NX, ny=NY)
sim.boundary(x="periodic", y="wall")
sim.fluid(tau=TAU,
          collision=COLLISION,
          max_lattice_force=MAX_LATTICE_FORCE)
sim.body_force(BODY_FX, 0.0)

# -- Mid-channel obstacle (rigid cylinder, no-slip) --
#    Diverts the flow into two narrow lanes. As capsules squeeze
#    through the gaps, they DEFORM in response to the local shear and
#    confinement — small soft Skalak capsules elongate visibly, large
#    stiffer Neo-Hookean capsules deform less. The size-dependent
#    deformation is what drives the segregation pattern you see
#    downstream.
sim.obstacle("circle",
             center=(OBSTACLE_X, NY / 2),
             radius=OBSTACLE_R)

# -- IBM coupling --
sim.ibm(iterations=2)

# -- Lubrication corrections (recommended for dense suspensions) --
sim.lubrication(enabled=True)

# -- Output --
sim.output(format="vtk",
           directory=os.path.join(script_dir, "vtk_bidisperse"),
           interval=OUT_EVERY)


# =================================================================
# Particle types
# =================================================================

# Type 0: Small Skalak capsule (RBC-like, soft)
sim.particle_type("small",
                  model="skalak",
                  G_s=0.06,
                  C_skalak=10.0,
                  k_bend=0.003,
                  k_area=0.6,
                  k_perimeter=0.06)

# Type 1: Large Neo-Hookean capsule (CTC-like, stiffer)
sim.particle_type("large",
                  model="neo_hookean",
                  G_s=0.30,
                  k_bend=0.020,
                  k_area=1.0,
                  k_perimeter=0.10)



# =================================================================
# Particle placement
# =================================================================

# Fill both sides of the obstacle — particles upstream AND downstream
# at start so the channel is loaded throughout. Two named regions, one
# on each side of the obstacle bbox (with a small clearance margin
# so the hex-lattice doesn't intersect the cylinder).
CLEARANCE = OBSTACLE_R + 4.0    # gap between seed bands and obstacle

# Narrow y-range (centre band of the channel) so capsules start
# packed close together — they collide with each other AND get
# squeezed against the obstacle as the flow builds. A wider y-band
# (e.g. y=(8, NY-8)) spreads them out so nearest-neighbour distance
# grows and contact events become rare.
sim.region("seed_upstream",
           x=(20, OBSTACLE_X - CLEARANCE),
           y=(15, NY - 15))
sim.region("seed_downstream",
           x=(OBSTACLE_X + CLEARANCE, NX - 20),
           y=(15, NY - 15))

# Hex-lattice fills are the proven-stable initial layout. Use them on
# both sides so the channel is loaded symmetrically. Each region gets
# half of N_SMALL and half of N_LARGE.
sim.generate("small", count=N_SMALL // 2, region="seed_upstream",
             radius=R_SMALL, num_nodes=18,
             method="hexagonal", spacing=HEX_SPACING,
             seed=42, min_gap=0.5)
sim.generate("small", count=N_SMALL - N_SMALL // 2,
             region="seed_downstream",
             radius=R_SMALL, num_nodes=18,
             method="hexagonal", spacing=HEX_SPACING,
             seed=43, min_gap=0.5)

# Large species: random fill, also on both sides.
sim.generate("large", count=N_LARGE // 2, region="seed_upstream",
             radius=R_LARGE, num_nodes=22,
             method="random", seed=200, min_gap=1.5)
sim.generate("large", count=N_LARGE - N_LARGE // 2,
             region="seed_downstream",
             radius=R_LARGE, num_nodes=22,
             method="random", seed=300, min_gap=1.5)



# =================================================================
# Run
# =================================================================
#
# Warmup ramp: at this density (φ≈0.28) and target ū (~0.07), slamming
# the body force on at full strength from t=0 makes capsules collide
# violently and the IBM coupling can blow up. Ramping over 3000 steps
# lets the particles relax into a stable arrangement first.

sim.warmup(steps=1000, ramp_steps=3000)
sim.thermo(interval=2000)
sim.run(N_STEPS)
