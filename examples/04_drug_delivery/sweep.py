"""Phase-4 parameter sweep over carrier stiffness × release rate.

For each (G_s, k_release) cell of the grid, builds a fresh simulation,
runs it, and records η + off-target fraction. Outputs:

  output/04_drug_delivery/sweep/results.npz
  output/04_drug_delivery/sweep/dose_vs_parameter.png  (matplotlib)

Usage::

    python examples/04_drug_delivery/sweep.py [--smoke] [--full]

  default: 3×3 grid (~3 minutes on a laptop)
  --full:  5×5 grid
  --smoke: 2×2 grid, ~30 steps each (CI-grade)
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import pathlib
import sys
from typing import Iterable

import numpy as np


HERE = pathlib.Path(__file__).resolve()
PROJECT = HERE.parents[2]


def _load_run_module():
    spec = importlib.util.spec_from_file_location(
        "phase4_run",
        PROJECT / "examples" / "04_drug_delivery" / "run.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _grid(smoke: bool, full: bool) -> tuple[Iterable[float], Iterable[float], int]:
    if smoke:
        return ([0.04, 0.20], [0.005, 0.020], 30)
    if full:
        return (np.linspace(0.02, 0.30, 5),
                np.linspace(0.002, 0.030, 5),
                1500)
    return (np.linspace(0.04, 0.20, 3),
            np.linspace(0.005, 0.020, 3),
            1500)


def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--smoke", action="store_true",
                    help="2×2 grid × 30 steps for CI")
    p.add_argument("--full",  action="store_true",
                    help="5×5 grid × 1500 steps")
    p.add_argument("--mode", default="first_order",
                    choices=["diffusion", "first_order",
                             "shear", "ph", "burst"],
                    help="release-kinetic mode (default: first_order)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv or sys.argv[1:])
    G_s_grid, k_grid, n_steps = _grid(args.smoke, args.full)
    G_s_grid = np.asarray(list(G_s_grid))
    k_grid   = np.asarray(list(k_grid))

    out_root = (str(pathlib.Path(os.getenv("TMPDIR", "/tmp"))
                    / "softflow_phase4_sweep_smoke")
                if args.smoke else
                str(PROJECT / "output" / "04_drug_delivery" / "sweep"))
    pathlib.Path(out_root).mkdir(parents=True, exist_ok=True)

    run_mod = _load_run_module()

    eta_grid = np.zeros((G_s_grid.size, k_grid.size), dtype=np.float64)
    otf_grid = np.zeros_like(eta_grid)
    rem_grid = np.zeros_like(eta_grid)

    print(f"Sweep: mode={args.mode}, "
           f"|G_s|={G_s_grid.size}, |k|={k_grid.size}, steps={n_steps}")
    for i, G_s in enumerate(G_s_grid):
        for j, k_rel in enumerate(k_grid):
            cell_dir = pathlib.Path(out_root) / f"G_{G_s:.4f}_k_{k_rel:.5f}"
            cell_dir.mkdir(parents=True, exist_ok=True)
            print(f"  [{i+1}/{G_s_grid.size}, {j+1}/{k_grid.size}] "
                   f"G_s={G_s:.4f}, k={k_rel:.5f} → {cell_dir.name}")
            sim, run = run_mod.build_simulation(
                mode=args.mode, num_steps=n_steps,
                output_dir=str(cell_dir),
                G_s=float(G_s), k_release=float(k_rel))
            sim.initialize()
            for _ in range(n_steps):
                sim.step()
            sim.finalize()
            s = run.summary()
            eta_grid[i, j] = s["delivery_efficiency"]
            otf_grid[i, j] = s["off_target_fraction"]
            rem_grid[i, j] = s["total_remaining"] / s["total_loaded"]

    out_path = pathlib.Path(out_root) / "results.npz"
    np.savez(out_path,
              G_s=G_s_grid, k=k_grid,
              eta=eta_grid, off_target=otf_grid, remaining=rem_grid,
              mode=np.array([args.mode]))
    print(f"Wrote {out_path}")

    # Optional: matplotlib heatmap. We import lazily so the script
    # still runs in environments without matplotlib.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        im0 = ax[0].imshow(eta_grid, origin="lower",
            extent=[k_grid[0], k_grid[-1], G_s_grid[0], G_s_grid[-1]],
            aspect="auto", cmap="viridis")
        ax[0].set_xlabel("k_release")
        ax[0].set_ylabel("G_s")
        ax[0].set_title(f"η (mode={args.mode})")
        fig.colorbar(im0, ax=ax[0])
        im1 = ax[1].imshow(otf_grid, origin="lower",
            extent=[k_grid[0], k_grid[-1], G_s_grid[0], G_s_grid[-1]],
            aspect="auto", cmap="magma")
        ax[1].set_xlabel("k_release")
        ax[1].set_ylabel("G_s")
        ax[1].set_title(f"off-target fraction")
        fig.colorbar(im1, ax=ax[1])
        png_path = pathlib.Path(out_root) / "dose_vs_parameter.png"
        fig.tight_layout()
        fig.savefig(png_path, dpi=150)
        plt.close(fig)
        print(f"Wrote {png_path}")
    except ImportError:
        print("matplotlib not available; skipped plot")
    return 0


if __name__ == "__main__":
    sys.exit(main())
