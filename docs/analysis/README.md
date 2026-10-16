# Phase-3 analysis diagnostics

Twelve post-processing diagnostics for SoftFlow simulations, organised
into four families per the project specification: **mixing**, **patterns**,
**jamming**, and **HDF5 export**. All implementations are pure Python
(numpy + scipy) and consume `SimulationSnapshot` objects extracted
from a running `Simulation` or replayed from saved trajectories.

The C++ engine is untouched. Diagnostics can be re-run on the same
trajectory without re-simulating, which decouples Phase-3 work from
the Phase-1 V&V suite.

Source layout:

```
python/pysoftflow/analysis/
├── snapshot.py       — SimulationSnapshot (data layer)
├── mixing.py         — Lacey, Danckwerts, contact-asymmetry
├── rdf.py            — species-resolved g_AB(r)
├── patterns.py       — lane order, Hoshen-Kopelman, persistence
├── jamming.py        — packing field, contact number, per-type Z_ij,
│                       force percolation, MSD, non-affine D²_min
└── hdf5_export.py    — write/read all diagnostics to HDF5
```

The showcase example is
[`examples/02_bidisperse_segregation/analyse.py`](../../examples/02_bidisperse_segregation/analyse.py).

---

## SimulationSnapshot (the data layer)

Frozen per-particle view at one timestep:

| Field | Shape | Notes |
|---|---|---|
| `step`, `time` | scalars | discrete + physical time |
| `positions` | (N, 2) | centroids in lattice units |
| `velocities` | (N, 2) | mean of node velocities (NaN if uncoupled) |
| `radii` | (N,) | `effectiveRadius()` = `sqrt(area / π)` of the discretised polygon |
| `types` | (N,) | integer type labels |
| `box` | (Lx, Ly) | lattice box |
| `periodic_x` | bool | streamwise wrap |
| `bonds` | (M, 2) | adhesion bond pairs (capsule indices); wall bonds dropped |

Two extractors: `from_simulation(sim)` (live) and `from_arrays(...)` (synthetic).
Periodic-x minimum-image is applied throughout.

---

## 1. Mixing / segregation indices (`mixing.py`)

### Lacey index

Lacey (1954). For an axis bin partition ``n_bins`` along ``axis ∈ {x, y}``,
let `p_k` be the type-A fraction in bin k and `p̄` the global type-A
fraction. With sample variance σ², binomial baseline σ²_R = p̄(1−p̄)/N̄,
and segregated maximum σ²_M = p̄(1−p̄):

```
M_L = (σ²_M − σ²) / (σ²_M − σ²_R)
```

`M_L → 1` = random / well-mixed; `M_L → 0` = fully segregated.

### Danckwerts intensity of segregation

Danckwerts (1952). Same family, normalised by σ²_M alone:

```
I_S = σ² / (p̄ (1 − p̄))     ∈ [0, 1]
```

`I_S → 0` mixed, `I_S → 1` segregated.

### Contact asymmetry

```
A = (n_AA − n_BB) / (n_AA + n_BB)     ∈ [-1, +1]
```

with pairs in contact when `‖c_i − c_j‖ < r_i + r_j + cutoff`.

**Reference.** P. M. C. Lacey, *J. Appl. Chem.* **4**, 257 (1954).
P. V. Danckwerts, *Appl. Sci. Res. A* **3**, 279 (1952).

---

## 2. Species-resolved RDF (`rdf.py`)

Standard 2-D radial distribution between types A and B:

```
g_AB(r) = ⟨n_B(r; r+dr)⟩_A / (N_B / (Lx · Ly) · 2π r dr)
```

Periodic-x is handled by including ±Lx image copies of the B
population in the KDTree query; the y-direction is bounded.

For same-species (`type_a == type_b`), unordered pairs are counted
once and the conventional double-count factor is restored at the end.

**Reference.** Allen & Tildesley, *Computer Simulation of Liquids*,
2nd ed. (2017), §2.6.

---

## 3. Pattern detection (`patterns.py`)

### Lane order parameter Φ

```
Φ_lane(axis) = ⟨cos(2 θ)⟩,    θ = ∠(velocity, axis)
```

Φ = +1: motion fully aligned with axis (lane-formed);
Φ = −1: fully transverse;  Φ ≈ 0: isotropic.
Particles with `|v| ≤ speed_threshold` are excluded.

### Hoshen–Kopelman cluster labelling

Connected-component labelling of the contact / bond graph using
`scipy.sparse.csgraph.connected_components` (algorithmically
equivalent to HK for this problem). Labels are sorted by cluster
size descending so cluster 0 is always the largest.

### Cluster persistence

Lag-1 Jaccard similarity over same-cluster pairs:

```
P = |pairs_in_same_cluster_a ∩ pairs_in_same_cluster_b|
    / |pairs_in_same_cluster_a ∪ pairs_in_same_cluster_b|
```

`P → 1` structure preserved; `P → 0` clusters re-formed.

**References.** Hoshen & Kopelman, *Phys. Rev. B* **14**, 3438 (1976).
Vissers et al., *Phys. Rev. Lett.* **106**, 228303 (2011) — laning in
driven binary mixtures.

---

## 4. Jamming diagnostics (`jamming.py`)

### Local packing-fraction field φ(x, y)

Coarse-grained on an `n_x × n_y` grid; each disk's area is assigned
to its centroid's cell. For uniform fills ``mean(φ) ≈ N · π · ⟨r²⟩ / (Lx · Ly)``.

### Bulk contact number

```
Z(k) = #{ℓ : d_kℓ < r_k + r_ℓ + cutoff, ℓ ≠ k}
Z̄    = ⟨Z(k)⟩
```

Hex lattice + cutoff slightly above spacing → Z̄ → 6 for interior
particles.

### Per-type contact statistics matrix Z_ij

```
Z_matrix[i, i] = 2 · N_ii / n_i        (each AA pair contributes to two A's)
Z_matrix[i, j] = N_ij / n_i            (asymmetric for n_i ≠ n_j)
```

with the symmetric pair-count matrix `n_pairs[i, j] = N_ij`. The
identity `n_i · Z_matrix[i, j] == n_j · Z_matrix[j, i]` holds by
construction. This is the diagnostic that lets you say things like

  "the small-particle population averages 4.2 contacts with other
   small particles and 1.8 with large particles."

### Force-network percolation

Hoshen–Kopelman on the contact / bond graph; check whether any
single cluster contains a particle in the bottom band
``y < band_fraction · Ly`` *and* a particle in the top band
``y > (1 − band_fraction) · Ly``. Returns the spanning cluster's
size if any, plus the largest cluster's size for context.

### Mean-squared displacement

```
MSD(t) = ⟨|r(t) − r(0)|²⟩
```

across a sequence of snapshots with a fixed particle set. Plateau
detection via late-half log-log slope: when the slope falls below
``plateau_slope_threshold``, we report the plateau value. Periodic-x
is unwrapped via minimum-image at the per-pair level.

### Non-affine D²_min (Falk & Langer 1998)

For each particle k with neighbour set N_k(t_0):

```
F_k     = argmin_F  Σ_{ℓ ∈ N_k} ‖(r_ℓ(t_1) − r_k(t_1))
                                  − F (r_ℓ(t_0) − r_k(t_0))‖²
D²_min(k) = (1 / |N_k|) · ‖residual at the optimum F‖²
```

Pure affine deformations give D²_min ≡ 0 (the local F absorbs them
exactly); plastic / non-affine motion produces nonzero values.

**References.** O'Hern et al., *Phys. Rev. E* **68**, 011306 (2003) —
jamming and Z. Stauffer & Aharony, *Introduction to Percolation
Theory* (1994). Falk & Langer, *Phys. Rev. E* **57**, 7192 (1998) —
D²_min.

---

## 5. HDF5 export (`hdf5_export.py`)

`write_diagnostics_h5(path, diagnostics, snapshots=…, manifest=…)`
writes one HDF5 file with three top-level groups:

```
/manifest      attrs flattened from run_manifest.json
/snapshots     one subgroup per saved snapshot
/diagnostics   one subgroup per diagnostic
```

Dataclass fields with array-shaped values become datasets; scalars
become attrs. Nested dataclasses recurse, so a per-snapshot dict of
diagnostics lands as nested groups rather than as `repr` strings.

`load_diagnostics_h5(path)` returns a nested Python dict for
inspection (not a strict round-trip back to dataclasses — use the
result for exploratory plotting and for cross-run comparisons).

`load_run_manifest(output_dir)` searches both the modern layout
(`output_dir/config/run_manifest.json`) and the legacy flat layout
(`output_dir/run_manifest.json`).

---

## Tests

73 Python tests cover every diagnostic with synthetic inputs of
known answer:

| File | Tests |
|---|---|
| `test_analysis_snapshot.py` | 6 |
| `test_analysis_mixing.py` | 11 |
| `test_analysis_rdf.py` | 5 |
| `test_analysis_patterns.py` | 13 |
| `test_analysis_jamming.py` | 15 |
| `test_analysis_hdf5_export.py` | 6 |
| `test_example_analyse.py` | 1 |

Plus the inherited Phase-1 / Phase-2 tests (snapshot extractor against
a live Simulation, end-to-end facade round-trip, etc.). Total Python
suite: 73 tests in ~14 seconds.

## Limitations / follow-ups

1. **MSD periodic unwrap** is per-pair minimum-image. For trajectories
   that wrap multiple times, save absolute positions to CSV and
   build snapshots from those — the C++ writer already records
   absolute coordinates.
2. **D²_min uses a fixed neighbour cutoff** at t₀. For very large
   non-affine motions, the neighbour set itself changes; a Pearson-
   style adaptive choice is a follow-up.
3. **Plot/notebook layer** is intentionally absent — diagnostics emit
   numpy arrays and HDF5; users plot with matplotlib in their own
   notebooks. A canonical set of paper-quality plots is Phase-4
   territory.
4. **Auto-collect during a run** (vs post-process from snapshots) is
   not wired in yet. The current pattern (sample snapshots per N
   steps, run diagnostics afterwards) is easier to debug and to
   re-run; auto-collect can land later if a user asks.
