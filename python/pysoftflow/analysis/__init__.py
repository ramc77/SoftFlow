"""Phase-3 analysis package — segregation, mixing, jamming, patterns.

Designed as a post-processing layer that operates on numpy arrays
extracted from a running ``Simulation`` (or from saved CSV / HDF5
snapshots), so the C++ engine stays untouched and diagnostics can be
re-run on the same trajectory without re-simulating.

Submodules:

    snapshot   — SimulationSnapshot dataclass + extractors
    mixing     — Lacey, Danckwerts, contact-asymmetry
    rdf        — species-resolved radial distribution
    patterns   — lane order, Hoshen-Kopelman, cluster persistence
    jamming    — packing field, contact number, per-type contact matrix,
                 force percolation, MSD, non-affine D²_min
    hdf5_export — write all diagnostics to a single HDF5 file with
                  the run's manifest copied in for traceability
    trajectory — load CSV / HDF5 trajectory dumps as numpy stacks
"""

from __future__ import annotations

from .snapshot import SimulationSnapshot

__all__ = ["SimulationSnapshot"]
