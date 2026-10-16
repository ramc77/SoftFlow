"""Wall / region absorbers for drug-delivery simulations.

A ``WallAbsorber`` models a patch of channel wall (or any rectangular
sub-region) that takes scalar mass out of the fluid every timestep
according to either first-order or Michaelis-Menten kinetics. The
patch is specified as integer lattice cell ranges; the absorber owns
its own cumulative-uptake counter so the metrics layer can compute
delivery efficiency directly.

Kinetics::

  first_order:        J(C) = k · C
  michaelis_menten:   J(C) = k · C / (K_M + C)

In both cases ``J`` has units of [scalar concentration / time], so
the per-cell uptake per timestep is ``ΔC = −J(C) · dt`` (clamped to
not over-deplete the cell). Mass conservation is exact in the sense
that whatever is removed from ``C`` is added to
``cumulative_absorbed``.

Boundary-condition vs. real tissue
-----------------------------------
This is a *coarse-grained sink*. It does not represent any specific
tissue type, transporter, or pharmacokinetic compartment. Use it as
a phenomenological "target-tissue uptake" for methodological studies;
clinical claims require a separately validated tissue model coupled
to the SoftFlow output.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


__all__ = ["WallAbsorber", "AbsorberStep"]


@dataclass
class AbsorberStep:
    """One-step uptake record returned by ``WallAbsorber.step``."""
    absorbed:    float    # mass taken this step
    cumulative:  float    # cumulative mass since registration


@dataclass
class WallAbsorber:
    """Rectangular absorber patch.

    Attributes
    ----------
    i_range : tuple[int, int]
        ``(i_lo, i_hi)`` lattice columns (inclusive lower, exclusive
        upper, like Python slices).
    j_range : tuple[int, int]
        ``(j_lo, j_hi)`` lattice rows.
    species : int
        Which scalar species this absorber consumes. Default 0.
    mode : str
        ``"first_order"`` or ``"michaelis_menten"``.
    k : float
        First-order rate constant ``k`` or M-M ``k_cat``.
    K_M : float
        Michaelis constant. Ignored for first-order mode but kept on
        the dataclass so the parameter shape is uniform.
    label : str
        Human-readable name (``"target"``, ``"off_target"``, …) for
        the metrics layer.
    cumulative_absorbed : float
        Running total of mass absorbed since registration.
    """

    i_range: tuple[int, int]
    j_range: tuple[int, int]
    species: int = 0
    mode:    str = "first_order"
    k:       float = 0.01
    K_M:     float = 1.0
    label:   str = "absorber"
    cumulative_absorbed: float = 0.0

    # Track the per-step uptake history so the metrics layer can
    # build residence-time / dose vs. time curves without a second pass.
    _step_history: list[float] = field(default_factory=list, init=False,
                                        repr=False, compare=False)

    def __post_init__(self):
        i_lo, i_hi = self.i_range
        j_lo, j_hi = self.j_range
        if not (i_hi > i_lo and j_hi > j_lo):
            raise ValueError(
                "WallAbsorber: i_range and j_range must be non-empty "
                f"(got i={self.i_range}, j={self.j_range})")
        if self.mode not in ("first_order", "michaelis_menten"):
            raise ValueError(
                f"WallAbsorber: mode must be 'first_order' or "
                f"'michaelis_menten' (got {self.mode!r})")
        if not (self.k >= 0.0):
            raise ValueError("WallAbsorber: k must be ≥ 0")
        if self.mode == "michaelis_menten" and not (self.K_M > 0.0):
            raise ValueError("WallAbsorber: K_M must be > 0 for MM mode")
        if self.species < 0:
            raise ValueError("WallAbsorber: species must be ≥ 0")

    @property
    def n_cells(self) -> int:
        """Number of lattice cells covered by this absorber."""
        return ((self.i_range[1] - self.i_range[0])
                * (self.j_range[1] - self.j_range[0]))

    def step(self, scalar_field: np.ndarray, dt: float) -> AbsorberStep:
        """Apply the sink to the scalar field in-place; return uptake.

        Parameters
        ----------
        scalar_field : np.ndarray, shape (ny, nx)
            The C++ ``AdvectionDiffusion.concentration(species)``
            view (writable). Modified in place: each cell in the
            absorber patch loses ``min(C, J(C) · dt)`` mass.
        dt : float
            Simulation timestep (lattice units).

        Returns
        -------
        AbsorberStep
        """
        i_lo, i_hi = self.i_range
        j_lo, j_hi = self.j_range
        # The C++ AdvectionDiffusion buffer is row-major (ny, nx); rows
        # are y-coordinates, cols are x. We slice both axes.
        patch = scalar_field[j_lo:j_hi, i_lo:i_hi]

        if self.mode == "first_order":
            J = self.k * patch
        else:                                         # michaelis_menten
            J = self.k * patch / (self.K_M + patch)

        # ΔC clamped to current cell mass (no over-depletion).
        deltaC = np.minimum(patch, J * dt)
        absorbed_mass = float(np.sum(deltaC))
        patch -= deltaC

        self.cumulative_absorbed += absorbed_mass
        self._step_history.append(absorbed_mass)
        return AbsorberStep(absorbed=absorbed_mass,
                             cumulative=self.cumulative_absorbed)

    @property
    def history(self) -> np.ndarray:
        """Per-step uptake history as a 1-D numpy array."""
        return np.asarray(self._step_history, dtype=np.float64)
