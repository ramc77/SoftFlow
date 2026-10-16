"""Plot Phase-5 time series from a previous run.

Reads ``output/05_tumor_growth/history.npz`` (or any path the user
supplies via ``--input``) and produces three figures:

  - division_history.png   (capsule count + cumulative divisions)
  - cluster_size.png       (largest cluster vs time)
  - flow_rate.png          (Q/Q_0 vs time, with cluster span overlay)

Skipped silently if matplotlib isn't installed.

**Coarse-grained mechano-chemical proxy. Not a validated cancer
model.** See README.md.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np


HERE = pathlib.Path(__file__).resolve()
PROJECT = HERE.parents[2]


def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", default=None,
                    help="path to history.npz (default: "
                         "output/05_tumor_growth/history.npz)")
    p.add_argument("--out-dir", default=None,
                    help="output directory for figures (default: same as input)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv or sys.argv[1:])
    in_path = pathlib.Path(args.input or
        PROJECT / "output" / "05_tumor_growth" / "history.npz")
    if not in_path.is_file():
        print(f"ERROR: {in_path} not found. Run examples/05_tumor_growth/run.py first.")
        return 2
    out_dir = pathlib.Path(args.out_dir or in_path.parent)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(in_path)
    h = data["history"]

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping plots")
        return 0

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(h["time"], h["n_capsules"], label="capsules", color="C0")
    ax.plot(h["time"], h["n_divisions_total"],
             label="cumulative divisions", color="C1")
    ax.set_xlabel("lattice time"); ax.set_ylabel("count")
    ax.set_title("Phase-5: capsule count and cumulative divisions")
    ax.legend(); fig.tight_layout()
    fig.savefig(out_dir / "division_history.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(h["time"], h["largest_cluster"], color="C2")
    ax.set_xlabel("lattice time")
    ax.set_ylabel("largest cluster size (#capsules)")
    ax.set_title("Phase-5: largest connected cluster vs time")
    fig.tight_layout()
    fig.savefig(out_dir / "cluster_size.png", dpi=150)
    plt.close(fig)

    if "flow_rate" in data and "cluster_span" in data:
        fig, ax1 = plt.subplots(figsize=(7, 4))
        ax1.plot(h["time"], data["flow_rate"], color="C3",
                  label="Q(t) / Q_0")
        ax1.set_xlabel("lattice time")
        ax1.set_ylabel("flow-rate ratio", color="C3")
        ax1.axhline(0.5, ls=":", color="C3", alpha=0.4,
                     label="embolization threshold")
        ax2 = ax1.twinx()
        ax2.plot(h["time"], data["cluster_span"], color="C4",
                  label="cluster y-span")
        ax2.set_ylabel("spanning-cluster y-extent", color="C4")
        ax1.set_title("Phase-5: flow rate and cluster span")
        ax1.legend(loc="upper left")
        ax2.legend(loc="upper right")
        fig.tight_layout()
        fig.savefig(out_dir / "flow_rate.png", dpi=150)
        plt.close(fig)

    print(f"Wrote figures to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
