"""
Publication-quality plots for microplastic_blood_vessel.py
===========================================================
Reads:
  data_mp_blood/global_stats.csv         — mass conservation timeseries
  data_mp_blood/trajectory_*.csv         — per-capsule trajectories
  vtk_mp_blood/fluid/fluid_*.vti         — concentration snapshots (via meshio)

Produces:
  figures/fig1_mass_conservation.pdf     — total mass budget over time
  figures/fig2_concentration_snapshots.pdf — 4-panel C field at t=[0,25,50,100]%
  figures/fig3_margination.pdf           — particle y-position vs time
  figures/fig4_reservoir_depletion.pdf   — per-MP M_p depletion timeseries
  figures/fig5_gamma_timeseries.pdf      — mean RBC surface coverage Γ(t)

Requirements:
  pip install matplotlib numpy pandas meshio  (meshio needed only for VTK panels)

Usage:
  python plot_microplastic_blood.py
  python plot_microplastic_blood.py --data_dir /path/to/data_mp_blood \
                                    --vtk_dir  /path/to/vtk_mp_blood  \
                                    --out_dir  /path/to/figures
"""

import os
import sys
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                   # headless — safe on HPC/server
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

# ── Publication style ──────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "serif",
    "font.size":         9,
    "axes.labelsize":    10,
    "axes.titlesize":    10,
    "legend.fontsize":   8,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "lines.linewidth":   1.4,
    "axes.linewidth":    0.8,
    "figure.dpi":        300,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "savefig.pad_inches": 0.02,
})

COLORS = {
    "rbc":      "#E8423B",    # red
    "wbc":      "#4C78C4",    # blue
    "mp_small": "#F28E2B",    # orange
    "mp_large": "#59A14F",    # green
    "dissolved":"#4E79A7",
    "particle": "#F28E2B",
    "adsorbed": "#E15759",
    "total":    "#333333",
}

# ── CLI arguments ──────────────────────────────────────────────────────────────
def parse_args():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default=os.path.join(script_dir, "data_mp_blood"))
    p.add_argument("--vtk_dir",  default=os.path.join(script_dir, "vtk_mp_blood"))
    p.add_argument("--out_dir",  default=os.path.join(script_dir, "figures"))
    return p.parse_args()


def ensure_dir(d):
    os.makedirs(d, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# Figure 1 — Mass conservation
# ══════════════════════════════════════════════════════════════════════════════
def fig_mass_conservation(stats: pd.DataFrame, out_dir: str):
    """Stacked area plot of dissolved + particle + adsorbed mass over time."""
    if not {"total_dissolved_mass", "total_particle_mass",
            "total_adsorbed_mass"}.issubset(stats.columns):
        print("  [skip] mass conservation columns not found in global_stats.csv")
        return

    t   = stats["time"].values
    dis = stats["total_dissolved_mass"].values
    par = stats["total_particle_mass"].values
    ads = stats["total_adsorbed_mass"].values
    tot = dis + par + ads
    tot0 = tot[0] if tot[0] > 0 else 1.0

    fig, axes = plt.subplots(2, 1, figsize=(5.5, 4.5), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1]})

    ax = axes[0]
    ax.stackplot(t, par, dis, ads,
                 labels=["Particle reservoir $M_p$",
                         "Dissolved (fluid) $C_{total}$",
                         "Adsorbed on cells $\\Sigma\\Gamma$"],
                 colors=[COLORS["particle"], COLORS["dissolved"], COLORS["adsorbed"]],
                 alpha=0.82)
    ax.plot(t, tot, "k--", lw=0.9, label="Total (should be const)")
    ax.set_ylabel("Mass (lattice units)")
    ax.legend(loc="upper right", framealpha=0.85)
    ax.set_title("Chemical mass budget")

    ax2 = axes[1]
    err = (tot - tot0) / tot0 * 100.0
    ax2.plot(t, err, color=COLORS["total"])
    ax2.axhline(0, lw=0.6, color="gray", ls="--")
    ax2.set_xlabel("Time (LBM steps)")
    ax2.set_ylabel("Error (%)")
    ax2.set_title("Conservation error")

    fig.tight_layout()
    path = os.path.join(out_dir, "fig1_mass_conservation.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 2 — Concentration snapshots
# ══════════════════════════════════════════════════════════════════════════════
def fig_concentration_snapshots(vtk_dir: str, out_dir: str):
    """4-panel spatial concentration field at 4 evenly-spaced timesteps."""
    try:
        import meshio
    except ImportError:
        print("  [skip] meshio not installed — pip install meshio")
        return

    fluid_dir = os.path.join(vtk_dir, "fluid")
    if not os.path.isdir(fluid_dir):
        fluid_dir = vtk_dir           # legacy flat layout

    vti_files = sorted(glob.glob(os.path.join(fluid_dir, "*.vti")) +
                       glob.glob(os.path.join(fluid_dir, "*.vtk")))
    if len(vti_files) < 2:
        print(f"  [skip] not enough VTK fluid files in {fluid_dir}")
        return

    # Pick 4 evenly-spaced frames
    n = len(vti_files)
    idxs = [0, n // 3, 2 * n // 3, n - 1]
    frames = [vti_files[i] for i in idxs]

    fig, axes = plt.subplots(1, 4, figsize=(10, 2.8))
    for col, fpath in enumerate(frames):
        mesh = meshio.read(fpath)
        # concentration may be named "concentration" or "concentration_0"
        C = None
        for key in ["concentration", "concentration_0"]:
            if key in mesh.point_data:
                C = mesh.point_data[key]
                break
        if C is None:
            axes[col].set_visible(False)
            continue

        # Infer grid shape from points
        pts = mesh.points
        xs = np.unique(pts[:, 0])
        ys = np.unique(pts[:, 1])
        nx, ny = len(xs), len(ys)
        C2d = C.reshape(ny, nx)

        ax = axes[col]
        im = ax.imshow(C2d, origin="lower", aspect="auto",
                       extent=[xs[0], xs[-1], ys[0], ys[-1]],
                       cmap="plasma", vmin=0, vmax=C2d.max() or 1)
        step = int(os.path.splitext(os.path.basename(fpath))[0].split("_")[-1])
        ax.set_title(f"step {step:,}")
        ax.set_xlabel("x (lu)")
        if col == 0:
            ax.set_ylabel("y (lu)")

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="4%", pad=0.04)
        fig.colorbar(im, cax=cax, label="$C$" if col == 3 else "")

    fig.suptitle("Concentration field $C(x,y,t)$", y=1.01)
    fig.tight_layout()
    path = os.path.join(out_dir, "fig2_concentration_snapshots.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 3 — Margination: y-position vs time
# ══════════════════════════════════════════════════════════════════════════════
def fig_margination(data_dir: str, out_dir: str, NY: int = 120):
    """
    Reads trajectory_*.csv files (one per capsule type).
    Plots mean y-position (normalised by NY/2) vs time.

    Expected CSV columns: timestep, capsule_id, type_id, x, y, ...
    """
    traj_files = sorted(glob.glob(os.path.join(data_dir, "trajectory*.csv")))
    if not traj_files:
        print(f"  [skip] no trajectory CSV files in {data_dir}")
        return

    # Try to load a combined file or multiple files
    dfs = []
    for f in traj_files:
        try:
            dfs.append(pd.read_csv(f))
        except Exception:
            pass
    if not dfs:
        print("  [skip] could not read any trajectory CSV")
        return

    df = pd.concat(dfs, ignore_index=True)
    if "y" not in df.columns or "type_id" not in df.columns:
        print("  [skip] trajectory CSV missing y or type_id columns")
        return

    # Map type_id → name (type IDs match particle_type() registration order)
    type_names = {0: "rbc", 1: "wbc", 2: "mp_small", 3: "mp_large"}

    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    half_width = NY / 2.0
    for tid, name in type_names.items():
        sub = df[df["type_id"] == tid]
        if sub.empty:
            continue
        grp = sub.groupby("timestep")["y"].mean()
        # Normalise: distance from centreline (0 = centre, 1 = wall)
        y_norm = np.abs(grp.values - half_width) / half_width
        ax.plot(grp.index, y_norm, label=name.upper(),
                color=COLORS.get(name, "gray"))

    ax.axhline(1.0, color="black", lw=0.7, ls="--", label="Wall")
    ax.set_xlabel("Time (LBM steps)")
    ax.set_ylabel("Mean lateral position $|y - L_y/2| / (L_y/2)$")
    ax.set_title("Margination: lateral drift toward vessel wall")
    ax.set_ylim(0, 1.1)
    ax.legend()
    fig.tight_layout()
    path = os.path.join(out_dir, "fig3_margination.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 4 — Reservoir depletion
# ══════════════════════════════════════════════════════════════════════════════
def fig_reservoir_depletion(stats: pd.DataFrame, out_dir: str):
    """Plot M_p(t) and fraction depleted."""
    if "total_particle_mass" not in stats.columns:
        print("  [skip] total_particle_mass not in global_stats.csv")
        return

    t   = stats["time"].values
    Mp  = stats["total_particle_mass"].values
    Mp0 = Mp[0] if Mp[0] > 0 else 1.0
    frac = 1.0 - Mp / Mp0          # fraction depleted

    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    ax2 = ax.twinx()

    ax.plot(t, Mp, color=COLORS["particle"], label="$M_p$ remaining")
    ax2.plot(t, frac * 100, color="gray", ls="--", lw=1.0,
             label="Depletion (%)")

    ax.set_xlabel("Time (LBM steps)")
    ax.set_ylabel("Total particle mass $M_p$ (lu)")
    ax2.set_ylabel("Fraction depleted (%)")
    ax.set_title("Microplastic chemical reservoir depletion")

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="center right")

    fig.tight_layout()
    path = os.path.join(out_dir, "fig4_reservoir_depletion.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 5 — Mean surface coverage Γ(t)
# ══════════════════════════════════════════════════════════════════════════════
def fig_gamma_timeseries(stats: pd.DataFrame, out_dir: str):
    """Plot total adsorbed mass Σ Γ(t) from global_stats.csv."""
    if "total_adsorbed_mass" not in stats.columns:
        print("  [skip] total_adsorbed_mass not in global_stats.csv")
        return

    t   = stats["time"].values
    ads = stats["total_adsorbed_mass"].values

    # Also load dissolved for the ratio Γ/(Γ+C)
    dis = stats.get("total_dissolved_mass", pd.Series(np.ones(len(t)))).values
    frac_ads = ads / np.maximum(ads + dis, 1e-12)

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))

    axes[0].plot(t, ads, color=COLORS["adsorbed"])
    axes[0].set_xlabel("Time (LBM steps)")
    axes[0].set_ylabel("Total adsorbed mass $\\sum \\Gamma$")
    axes[0].set_title("Langmuir adsorption: cell surface coverage")

    axes[1].plot(t, frac_ads * 100, color=COLORS["adsorbed"])
    axes[1].set_xlabel("Time (LBM steps)")
    axes[1].set_ylabel("Fraction adsorbed (%)")
    axes[1].set_title("Adsorbed / (adsorbed + dissolved)")

    fig.tight_layout()
    path = os.path.join(out_dir, "fig5_gamma_timeseries.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 6 — Velocity and concentration profiles (lateral)
# ══════════════════════════════════════════════════════════════════════════════
def fig_lateral_profiles(vtk_dir: str, out_dir: str):
    """
    Time-averaged lateral profiles of u_x(y) and C(y) from the last VTK frame.
    """
    try:
        import meshio
    except ImportError:
        return

    fluid_dir = os.path.join(vtk_dir, "fluid")
    if not os.path.isdir(fluid_dir):
        fluid_dir = vtk_dir

    vti_files = sorted(glob.glob(os.path.join(fluid_dir, "*.vti")) +
                       glob.glob(os.path.join(fluid_dir, "*.vtk")))
    if not vti_files:
        return

    mesh = meshio.read(vti_files[-1])        # last frame
    pts  = mesh.points
    xs   = np.unique(pts[:, 0])
    ys   = np.unique(pts[:, 1])
    nx, ny = len(xs), len(ys)

    # Extract u_x and C; average over x (streamwise direction)
    ux_avg = np.zeros(ny)
    C_avg  = np.zeros(ny)

    if "velocity" in mesh.point_data:
        vel = mesh.point_data["velocity"]
        ux  = vel[:, 0].reshape(ny, nx)
        ux_avg = ux.mean(axis=1)

    for key in ["concentration", "concentration_0"]:
        if key in mesh.point_data:
            C  = mesh.point_data[key].reshape(ny, nx)
            C_avg = C.mean(axis=1)
            break

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3.5))

    ax1.plot(ux_avg, ys, color=COLORS["dissolved"])
    ax1.set_xlabel("$\\langle u_x \\rangle_x$ (lu/step)")
    ax1.set_ylabel("$y$ (lu)")
    ax1.set_title("Streamwise velocity profile")

    ax2.plot(C_avg, ys, color=COLORS["adsorbed"])
    ax2.set_xlabel("$\\langle C \\rangle_x$ (lu)")
    ax2.set_ylabel("$y$ (lu)")
    ax2.set_title("Lateral concentration profile")

    fig.suptitle("Lateral profiles (last frame, averaged over $x$)", y=1.01)
    fig.tight_layout()
    path = os.path.join(out_dir, "fig6_lateral_profiles.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    args = parse_args()
    ensure_dir(args.out_dir)

    # Load global_stats.csv
    stats_path = os.path.join(args.data_dir, "global_stats.csv")
    stats = None
    if os.path.isfile(stats_path):
        try:
            stats = pd.read_csv(stats_path)
            print(f"Loaded global_stats.csv: {len(stats)} rows, "
                  f"columns: {list(stats.columns)}")
        except Exception as e:
            print(f"  [warn] could not read {stats_path}: {e}")
    else:
        print(f"  [warn] {stats_path} not found — skipping timeseries plots")

    print("\nGenerating publication figures:")

    if stats is not None:
        fig_mass_conservation(stats, args.out_dir)
        fig_reservoir_depletion(stats, args.out_dir)
        fig_gamma_timeseries(stats, args.out_dir)

    fig_concentration_snapshots(args.vtk_dir, args.out_dir)
    fig_margination(args.data_dir, args.out_dir)
    fig_lateral_profiles(args.vtk_dir, args.out_dir)

    print(f"\nAll figures written to: {args.out_dir}/")


if __name__ == "__main__":
    main()
