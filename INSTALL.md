# Installation guide

SoftFlow is a C++ engine with a Python wrapper. It builds from source
on **macOS (Intel + Apple Silicon)** and **Linux (Ubuntu 20.04+)**
with the same `pip install -e .` command. Windows is not officially
supported (WSL2 works fine — follow the Linux instructions).

---

## 1. Prerequisites by platform

| Platform | C++ toolchain | Build system | OpenMP | Python |
|---|---|---|---|---|
| macOS — Apple Silicon (M1/M2/M3/M4) | Xcode Command Line Tools (Apple Clang) | CMake ≥ 3.18 | `libomp` via Homebrew | 3.10–3.12 |
| macOS — Intel | Xcode Command Line Tools (Apple Clang) | CMake ≥ 3.18 | `libomp` via Homebrew | 3.10–3.12 |
| Ubuntu / Debian | `build-essential` (GCC ≥ 11) | CMake ≥ 3.18 | comes with GCC | 3.10–3.12 |

Python ≥ 3.13 is not yet supported because some downstream
dependencies still lack wheels.

---

## 2. macOS — Apple Silicon **and** Intel

The same commands work on both. Homebrew installs the correct
architecture automatically (`/opt/homebrew` on Apple Silicon,
`/usr/local` on Intel).

### 2.1 Install system prerequisites

```bash
# Xcode Command Line Tools (provides Apple Clang)
xcode-select --install                          # accept the dialog

# Homebrew (skip if you already have it)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# CMake + OpenMP runtime
brew install cmake libomp git

# (optional but recommended) pyenv to avoid touching the system Python
brew install pyenv
pyenv install 3.11.9
pyenv global 3.11.9
```

Verify the toolchain:

```bash
clang++ --version          # should print Apple clang 14.x or newer
cmake --version            # ≥ 3.18
python3 --version          # 3.10, 3.11, or 3.12
```

### 2.2 Clone and install

```bash
git clone https://github.com/ramc77/SoftFlow.git
cd SoftFlow

# Create an isolated Python environment (strongly recommended)
python3 -m venv .venv
source .venv/bin/activate

# Install the Python wrapper + native extension in editable mode.
# This invokes CMake under the hood (scikit-build-core).
pip install --upgrade pip
pip install -e ".[dev]"
```

The first `pip install` takes 3–6 minutes (CMake configures, then
compiles ~30 C++ translation units). Subsequent edits to `.cpp`
files trigger an incremental rebuild via `pip install -e .` again
or directly with `cmake --build build`.

### 2.3 Verify

```bash
python -c "import pysoftflow; print('SoftFlow', pysoftflow.__version__, 'OK')"

# Run the Poiseuille smoke test (10 s)
python examples/01_poiseuille_lbm/run.py --smoke
```

You should see a thermo table, then `Run complete. … steps/s` and
VTK files in `examples/01_poiseuille_lbm/vtk_*/`.

---

## 3. Ubuntu / Debian Linux

### 3.1 Install system prerequisites

```bash
sudo apt update
sudo apt install -y build-essential cmake git python3-dev python3-venv

# GCC 11+ is required for C++20.  On Ubuntu 20.04 / 22.04 you may need:
sudo apt install -y gcc-11 g++-11
export CC=gcc-11 CXX=g++-11      # add to ~/.bashrc to persist
```

Verify:

```bash
g++ --version                    # ≥ 11
cmake --version                  # ≥ 3.18
python3 --version                # 3.10, 3.11, or 3.12
```

### 3.2 Clone and install

```bash
git clone https://github.com/ramc77/SoftFlow.git
cd SoftFlow

python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -e ".[dev]"
```

### 3.3 Verify

```bash
python -c "import pysoftflow; print('SoftFlow', pysoftflow.__version__, 'OK')"
python examples/01_poiseuille_lbm/run.py --smoke
```

---

## 4. Optional: build only the C++ engine

If you want the C++ library without the Python bindings (for
profiling, debugging, or integrating into another C++ project):

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
                    -DBUILD_PYTHON=OFF \
                    -DBUILD_TESTS=ON \
                    -DUSE_OPENMP=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
```

---

## 5. Updating

```bash
cd SoftFlow
git pull
pip install -e ".[dev]"    # rebuilds incrementally
```

---

## 6. Troubleshooting

### macOS — `libomp not found`

```
CMake Error: Could not find OpenMP_CXX
```

Install (or reinstall) libomp and reconfigure:

```bash
brew install libomp
brew --prefix libomp       # should print a path
rm -rf build _skbuild
pip install -e ".[dev]"
```

### macOS — wrong Homebrew prefix on a dual-Mac household

If you have *both* Apple Silicon and Intel Homebrew on the same
machine (unusual but possible), tell CMake which one to use:

```bash
LIBOMP_PREFIX=$(brew --prefix libomp) \
CMAKE_PREFIX_PATH=$LIBOMP_PREFIX pip install -e ".[dev]"
```

### Ubuntu — `error: 'std::ranges' has not been declared`

Your GCC is too old. Upgrade:

```bash
sudo apt install -y gcc-11 g++-11
export CC=gcc-11 CXX=g++-11
rm -rf build _skbuild
pip install -e ".[dev]"
```

### `ImportError: dynamic module does not define module export function`

The compiled extension is from a different Python version than the
one you're running. Wipe the build and rebuild against the active
interpreter:

```bash
rm -rf build _skbuild **/__pycache__
pip install -e ".[dev]"
```

### ParaView can't open `*.pvd`

The `.pvd` files reference per-step `.vti` / `.vtp` files by
relative path. Open the `.pvd` (not the per-step files) from the
directory where it lives. If you moved the directory, also move
the `fluid/` and `particles/` subfolders along with the `.pvd`.

### Long simulations stall on macOS sleep

macOS sleeps the CPU when the lid closes. Use:

```bash
caffeinate -i python examples/05_tumor_growth/run.py
```

or run inside `tmux` / `screen` over ssh.

---

## 7. Uninstall

```bash
pip uninstall pysoftflow
rm -rf build _skbuild .venv
```

To remove everything: `rm -rf` the cloned directory.

---

## 8. Reporting an install failure

Please file an issue at
`https://github.com/ramc77/SoftFlow/issues` with:

```bash
uname -a
sw_vers                # macOS only
cmake --version
python3 --version
$CXX --version
pip install -e ".[dev]" 2>&1 | tail -80     # paste the failing log
```

The build prints a `build_info.h` fingerprint (compiler, flags,
git SHA) into every output's `run_manifest.json`, which makes
remote diagnosis tractable.
