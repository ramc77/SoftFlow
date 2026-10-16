"""
PySoftFlow: Python interface for the SoftFlow LBM-IBM capsule simulation.

Modules are imported separately so you only load what you need:

    # Full simulation setup (loads C++ core)
    from pysoftflow import SoftFlowSimulation

    # Pre-processing only (geometry helpers, no C++ needed)
    from pysoftflow import preprocess

    # Post-processing only (analysis functions, no C++ needed)
    from pysoftflow import postprocess

    # Visualization (needs matplotlib, deferred import)
    from pysoftflow import visualization

    # Preset configurations
    from pysoftflow import presets
"""

__version__ = "0.3.0"

# ── Lightweight submodules (no C++ dependency) ──────────────
# These can be imported without the simulation engine:
#   from pysoftflow.preprocess import staggered_pillars
#   from pysoftflow.postprocess import compute_segregation_index

from . import preprocess
from . import postprocess

# ── C++ core bindings (loaded on demand) ────────────────────
# Heavy imports are deferred to avoid slow startup when only
# doing pre/post-processing.

def _load_core():
    """Lazily import C++ bindings and return the module."""
    import softflow_core
    return softflow_core


# Core types — available via pysoftflow.Vec2d etc.
# These use __getattr__ for lazy loading.
_CORE_NAMES = {
    "Vec2d", "CellType", "BoundaryType",
    "FluidParams", "MembraneParams", "RepulsionParams",
    "ShanChenParams", "MLParams", "SimulationParams",
    "LatticeField", "Capsule", "CapsuleSystem",
    "CircleObstacle", "RectObstacle", "ChannelBuilder",
    "MembraneModel", "CollisionModel",
    "LubricationParams", "AdhesionParams", "ScalarParams",
    "PolygonObstacle", "SegregationResults",
}

# High-level wrapper names
_WRAPPER_NAMES = {
    "SoftFlowSimulation", "Simulation",
    "_CoreSimulation",
}


def __getattr__(name):
    """Lazy import: C++ types and high-level wrappers are loaded on first access."""
    if name in _CORE_NAMES:
        core = _load_core()
        obj = getattr(core, name)
        globals()[name] = obj  # cache for next access
        return obj

    if name == "SoftFlowSimulation":
        from .setup import SoftFlowSimulation
        globals()["SoftFlowSimulation"] = SoftFlowSimulation
        return SoftFlowSimulation

    if name == "Simulation":
        from .simulation import Simulation
        globals()["Simulation"] = Simulation
        return Simulation

    if name == "_CoreSimulation":
        core = _load_core()
        obj = core.Simulation
        globals()["_CoreSimulation"] = obj
        return obj

    if name == "visualization":
        from . import visualization
        globals()["visualization"] = visualization
        return visualization

    if name == "presets":
        from . import presets
        globals()["presets"] = presets
        return presets

    raise AttributeError(f"module 'pysoftflow' has no attribute {name!r}")


__all__ = [
    # Submodules (lightweight, always available)
    "preprocess",
    "postprocess",
    # Submodules (loaded on access)
    "visualization",
    "presets",
    # High-level API (loaded on access)
    "SoftFlowSimulation",
    "Simulation",
    # Core types (loaded on access)
    "Vec2d", "CellType", "BoundaryType",
    "FluidParams", "MembraneParams", "RepulsionParams",
    "ShanChenParams", "MLParams", "SimulationParams",
    "LatticeField", "Capsule", "CapsuleSystem",
    "CircleObstacle", "RectObstacle", "ChannelBuilder",
    "MembraneModel", "CollisionModel",
    "LubricationParams", "AdhesionParams", "ScalarParams",
    "PolygonObstacle", "SegregationResults",
    "_CoreSimulation",
]
