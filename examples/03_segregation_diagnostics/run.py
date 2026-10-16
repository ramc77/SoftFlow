"""
Bidisperse suspension with Phase-3 diagnostics overlay
=======================================================
Same physical setup as the Phase-2 bidisperse example, but additionally
samples the simulation periodically and writes Phase-3 diagnostic
fields as their own ParaView-readable datasets.

What you get in ParaView:
  - vtk_diagnostics/{fluid,particles}.pvd     — fluid + capsules (as Phase 2)
  - diagnostics/packing_field.pvd             — coarse-grained φ(x,y) animation
  - diagnostics/scalar_history.csv            — Lacey index, Z̄, lane order
                                                  vs time (open as Spreadsheet)

Diagnostics computed at each snapshot:
  - Local packing-fraction field  φ(x, y)
  - Bulk contact number           Z̄
  - Largest connected cluster size (Hoshen–Kopelman)
  - Lacey mixing index            M ∈ [0, 1]
  - Lane order parameter          Φ ∈ [-1, 1]

Usage:
    python 03_segregation_diagnostics/run.py
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
from pysoftflow.analysis import SimulationSnapshot
from pysoftflow.analysis.jamming import contact_number, packing_field
from pysoftflow.analysis.mixing  import lacey_index
from pysoftflow.analysis.patterns import hoshen_kopelman, lane_order


# =================================================================
# Physical parameters (lattice units)
# =================================================================

NX, NY      = 300, 60
N_STEPS     = 200             # quick smoke run; raise to 3000+ for production
N_SNAPSHOTS = 10              # how many diagnostic frames to save
TAU         = 0.8
BODY_FX     = 5e-6

N_SMALL = 170; R_SMALL = 2.0
N_LARGE =  55; R_LARGE = 4.0


# =================================================================
# Simulation setup (identical to Phase 2)
# =================================================================

sim = SoftFlowSimulation()

sim.domain(nx=NX, ny=NY)
sim.boundary(x="periodic", y="wall")
sim.fluid(tau=TAU)
sim.body_force(BODY_FX, 0.0)
sim.ibm(iterations=2)
sim.lubrication(enabled=True)

sim.output(format="vtk",
           directory=os.path.join(script_dir, "vtk_diagnostics"),
           interval=200)

sim.particle_type("small",
                  model="skalak",
                  G_s=0.06, C_skalak=10.0,
                  k_bend=0.003, k_area=0.6, k_perimeter=0.06)
sim.particle_type("large",
                  model="neo_hookean",
                  G_s=0.30, k_bend=0.020,
                  k_area=1.0, k_perimeter=0.10)

sim.region("small_band", x=(20, NX - 20), y=(15, 35))
sim.generate("small", count=N_SMALL, region="small_band",
             radius=R_SMALL, num_nodes=18,
             method="hexagonal", spacing=6.0,
             seed=42, min_gap=0.5)

sim.region("large_band", x=(20, NX - 20), y=(8, NY - 8))
sim.generate("large", count=N_LARGE, region="large_band",
             radius=R_LARGE, num_nodes=22,
             method="random", seed=200, min_gap=1.5)

sim.initialize()


# =================================================================
# Phase-3 diagnostic harvest (per-step callback)
# =================================================================

diag_dir = os.path.join(script_dir, "diagnostics")
os.makedirs(diag_dir, exist_ok=True)

pvd_entries = []   # list of (time, filename) for the .pvd time-series
csv_rows    = []   # list of dicts for the scalar history


def _write_vti(path, *, values, name="phi", spacing=(1.0, 1.0, 1.0)):
    """Minimal hand-rolled .vti writer (no third-party VTK dependency)."""
    ny, nx = values.shape
    extent = f"0 {nx-1} 0 {ny-1} 0 0"
    flat = " ".join(f"{v:.6g}" for v in values.flatten())
    sp = " ".join(map(str, spacing))
    with open(path, "w") as f:
        f.write(f"""<?xml version="1.0"?>
<VTKFile type="ImageData" version="0.1" byte_order="LittleEndian">
  <ImageData WholeExtent="{extent}" Origin="0 0 0" Spacing="{sp}">
    <Piece Extent="{extent}">
      <PointData Scalars="{name}">
        <DataArray type="Float64" Name="{name}" format="ascii">{flat}</DataArray>
      </PointData>
    </Piece>
  </ImageData>
</VTKFile>
""")


def harvest(core_sim, step):
    """Called from the C++ step() loop after each timestep."""
    sample_every = max(1, N_STEPS // (N_SNAPSHOTS - 1))
    if (step + 1) % sample_every != 0 and step != N_STEPS - 1:
        return

    snap = SimulationSnapshot.from_simulation(core_sim)
    pf   = packing_field(snap, n_x=60, n_y=20)
    fname = f"packing_field_{len(pvd_entries):06d}.vti"
    _write_vti(os.path.join(diag_dir, fname),
               values=pf.phi, name="packing_fraction",
               spacing=(NX / pf.phi.shape[1], NY / pf.phi.shape[0], 1.0))
    pvd_entries.append((float(snap.time), fname))

    Z   = contact_number(snap, contact_cutoff=0.5)
    hk  = hoshen_kopelman(snap, contact_cutoff=0.5)
    lac = lacey_index(snap, type_a=0, type_b=1, axis="y", n_bins=8)
    Phi = lane_order(snap, axis="x")
    csv_rows.append({
        "step": int(snap.step), "time": float(snap.time),
        "n_capsules": int(snap.n_particles),
        "Z_mean": float(Z.Z_mean),
        "largest_cluster": int(hk.largest_size),
        "n_clusters": int(hk.n_clusters),
        "lacey_M": float(lac.M_lacey),
        "lane_order": float(Phi) if not np.isnan(Phi) else 0.0,
    })


sim.core.setStepCallback(harvest)


# =================================================================
# Run
# =================================================================

sim.thermo(interval=200)
sim.run(N_STEPS)


# =================================================================
# Finalise diagnostic outputs
# =================================================================

# .pvd collection so ParaView animates the per-step .vti files together.
blocks = "\n".join(
    f'    <DataSet timestep="{t:.6g}" file="{f}" group="" part="0"/>'
    for t, f in pvd_entries)
with open(os.path.join(diag_dir, "packing_field.pvd"), "w") as f:
    f.write(f"""<?xml version="1.0"?>
<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">
  <Collection>
{blocks}
  </Collection>
</VTKFile>
""")

# Scalar time-series CSV.
if csv_rows:
    cols = list(csv_rows[0].keys())
    with open(os.path.join(diag_dir, "scalar_history.csv"), "w") as f:
        f.write(",".join(cols) + "\n")
        for r in csv_rows:
            f.write(",".join(f"{r[c]}" for c in cols) + "\n")
