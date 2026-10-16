# SoftFlow

**A 2D Lattice-Boltzmann + Immersed-Boundary simulator for soft
particles in microfluidic channels.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10–3.12](https://img.shields.io/badge/python-3.10%E2%80%933.12-blue.svg)](https://www.python.org)
[![macOS · Linux](https://img.shields.io/badge/platform-macOS%20%7C%20Linux-lightgrey.svg)](INSTALL.md)

SoftFlow lets you set up a deformable-particle suspension simulation
in ~20 lines of Python and run it on a laptop. The C++ engine handles
the LBM fluid, the IBM coupling, the membrane mechanics, and an
optional advection-diffusion-reaction solver; the Python wrapper
exposes everything through a declarative LAMMPS-style API.

```python
from pysoftflow import SoftFlowSimulation

sim = SoftFlowSimulation()
sim.domain(nx=400, ny=80)
sim.boundary(x="periodic", y="wall")
sim.fluid(tau=0.8, collision="regularized")
sim.body_force(fx=3e-5)
sim.particle_type("rbc", model="skalak", G_s=0.06)
sim.region("seed", x=(20, 380), y=(10, 70))
sim.generate("rbc", count=30, region="seed", radius=3.0, seed=42)
sim.output(format="vtk", directory="output", interval=500)
sim.thermo(interval=500)
sim.run(50000)
```

Run the script, open the resulting `output/*.pvd` files in ParaView,
and watch 30 red-blood-cell-like capsules flowing through a periodic
Poiseuille channel.

---

## Why SoftFlow?

| Feature | What you get |
|---|---|
| **D2Q9 LBM** | BGK / MRT / **regularized** collision; built-in stability monitoring |
| **IBM coupling** | Multi-direct forcing (Luo 2007), Peskin 4-pt delta |
| **Membrane mechanics** | Hookean, Neo-Hookean, **Skalak** (RBCs), **WLC** (cytoskeleton); Helfrich bending |
| **Adhesion** | Reversible **Bell** and **catch-slip** bonds (Thomas-Vogel-Sokurenko 2008) |
| **Chemistry** | Advection-diffusion + Fick leaching + Langmuir adsorption |
| **Lubrication** | Sub-grid squeeze-film correction for dense suspensions |
| **Two-phase fluid** | Shan-Chen multiphase with Carnahan-Starling EOS |
| **Free surface** | Wet-dry LBM for dam-break / gravity-driven flows |
| **Diagnostics** | Segregation, mixing entropy, RDF, **Hoshen-Kopelman** clusters, lane order |
| **Reproducibility** | Per-run `run_manifest.json` with git SHA, compiler flags, full config, seed |
| **Output** | ParaView (`.pvd` / `.vti` / `.vtp`) + CSV time-series for ML |
| **Performance** | OpenMP-parallel C++ core; ~10k lattice cells × 30 capsules at ~40 steps/s on a laptop |

---

## Install

Quick path (macOS / Linux):

```bash
git clone https://github.com/ramc77/SoftFlow.git
cd SoftFlow
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python examples/01_poiseuille_lbm/run.py --smoke
```

The build invokes CMake under the hood (`scikit-build-core`) and
compiles the C++ engine for your platform. First build takes 3–6
minutes; later edits rebuild incrementally.

**Full install instructions for macOS (Intel + Apple Silicon) and
Ubuntu are in [`INSTALL.md`](INSTALL.md), including troubleshooting.**

---

## Documentation map

| Doc | What's in it |
|---|---|
| **[`INSTALL.md`](INSTALL.md)** | Prerequisites + platform-specific install + troubleshooting |
| **[`docs/api/python_api.md`](docs/api/python_api.md)** | Full reference for every `sim.X()` method |
| [`examples/README.md`](examples/README.md) | Five canonical runnable demos |
| [`docs/theory/`](docs/theory/) | LBM, IBM, ADR, adhesion, lubrication, membrane derivations |
| [`docs/drug_delivery.md`](docs/drug_delivery.md) | Drug-delivery module |
| [`docs/tumor_growth.md`](docs/tumor_growth.md) | Tumour-growth proxy module |
| [`docs/analysis/`](docs/analysis/) | Segregation / jamming / pattern diagnostics |

---

## Examples

Five runnable scripts in [`examples/`](examples/), each with a
`--smoke` flag for a 5–30 s sanity run and a full production length:

| # | Topic | Run | Headline output |
|---|---|---|---|
| 1 | Poiseuille channel, hardened LBM | `python examples/01_poiseuille_lbm/run.py` | Parabolic profile (< 1 % L2 vs analytical) |
| 2 | Bidisperse segregation in a 400×80 channel | `python examples/02_bidisperse_segregation/run.py` | Margination, lane order, RDF |
| 3 | Segregation + jamming diagnostics | `python examples/03_segregation_diagnostics/run.py` | Hoshen-Kopelman cluster sizes, Z-matrix |
| 4 | Drug delivery (5 release modes) | `python examples/04_drug_delivery/run.py --mode first_order` | Delivery efficiency η, dose maps |
| 5 | Tumour-growth proxy in flow | `python examples/05_tumor_growth/run.py` | Division events, embolization detection |

Open the resulting `output/<name>/{fluid,particles}.pvd` files in
ParaView and hit the play button.

---

---

## Verification & validation

A non-negotiable test suite (see [`tests/verification/`](tests/verification/)):

| Case | Expected result | Tolerance |
|---|---|---|
| Poiseuille flow (no particles) | Analytical parabolic profile | < 1 % L2 |
| Taylor–Green vortex decay | Exponential decay rate | < 2 % |
| Single capsule in shear | Tank-treading frequency vs Ca | < 5 % vs published |
| Rigid disk in Poiseuille | Segré–Silberberg radial focusing | < 10 % radial position |
| Pure diffusion of scalar | `σ² = 2Dt` | < 2 % |
| Advection–diffusion | `D_eff = D + Pe²·D/210` (Taylor dispersion) | < 5 % |

Run with `ctest --test-dir build --output-on-failure` or
`pytest tests/ -q`.

---

## Reproducibility

Every `sim.run()` writes a `config/run_manifest.json` into the
output directory containing:

- Git SHA + branch + dirty flag
- Compiler ID, version, full resolved CXX flags
- OpenMP on/off and thread count
- All declared parameters (units, geometry, fluid, particles, …)
- All RNG seeds

This is verbatim what reviewers ask for. Re-running with the same
manifest reproduces the result bit-for-bit.

---

## Limitations

- **2D only.** A 3D extension is on the roadmap but not in this
  release.
- **No GPU.** The data layout is GPU-friendly (SoA), but the CUDA /
  HIP kernels are not implemented yet.
- **Newtonian fluid by default.** A power-law option exists but
  non-Newtonian rheology beyond that is out of scope.
- **Single-species chemistry** (or a small number); full reaction
  networks are not supported.
- **Patient-specific geometries** require a polygon-mask import
  step (the engine accepts polygon domains, but no DICOM/STL
  importer ships in this release).

The tumour-growth and drug-delivery modules are **coarse-grained
mechano-chemical proxies, not validated oncology / pharmacokinetic
models** — that language is required wherever they're reported.

---

## Citing

If you use SoftFlow in published work, please cite:

```bibtex
@software{softflow_2026,
  author       = {Chand, Ram},
  title        = {{SoftFlow: A 2D LBM-IBM simulator for soft particles
                   in microfluidic channels}},
  year         = 2026,
  publisher    = {Zenodo},
  version      = {0.3.0},
  url          = {https://github.com/ramc77/SoftFlow},
}
```

A short methods paper is in preparation.

---

## Acknowledgements

Developed at the **Department of Natural Sciences, The Begum Nusrat
Bhutto Women University (BNBWU), Sukkur, Sindh, Pakistan**, in
support of graduate training in computational soft matter and
microfluidics.

The implementation follows established practice from:

- T. Krüger *et al.*, **The Lattice Boltzmann Method** (Springer, 2017)
- C. S. Peskin, **The Immersed Boundary Method** (Acta Numerica 2002)
- R. Skalak *et al.*, **Strain energy function of red blood cell
  membranes** (Biophys. J. 1973)
- W. E. Thomas, V. Vogel, E. Sokurenko, **Biophysics of catch bonds**
  (Annu. Rev. Biophys. 2008)
- D. W. Holm *et al.*, **DLD sorting** (Lab Chip 2011)

## License

MIT — see [`LICENSE`](LICENSE). You are free to use, modify, and
redistribute SoftFlow in research, teaching, and (subject to
appropriate caveats on the biomedical modules) commercial work.

---

## Contact

**Dr. Ram Chand** — Department of Natural Sciences, BNBWU, Sukkur,
Sindh, Pakistan
GitHub issues are the preferred channel for bug reports, feature
requests, and install problems.
