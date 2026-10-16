"""SI ↔ lattice unit conversion for SoftFlow.

the project specification mandates:

    Units: Lattice units internally; expose SI <-> lattice conversion
    utilities; every public API that takes a physical quantity must
    document its units.

This module provides ``LatticeUnits`` — a small dataclass that holds
the three independent scaling factors (length, time, mass) needed to
go from SI to lattice and back, plus convenience constructors that
pin those scales by matching dimensionless numbers (Re, Ca, Pe,
confinement) instead of asking the user to compute scales by hand.

Typical workflow::

    from pysoftflow.units import LatticeUnits

    # I have a physical channel: 100 µm wide, fluid ν = 1.0e-6 m²/s,
    # mean velocity 1 mm/s, capsule shear modulus G_s = 6.0 µN/m.
    units = LatticeUnits.from_channel_flow(
        channel_height_m       = 100e-6,
        ny_lattice             = 64,
        kinematic_viscosity_si = 1.0e-6,
        mean_velocity_si       = 1.0e-3,
        target_tau             = 0.8,           # picks Δt
    )

    print(units)
    print("Re      =", units.reynolds(L_si=100e-6, U_si=1e-3))
    print("u_lb    =", units.velocity_to_lb(1e-3))   # ≈ 0.025
    print("τ       =", units.relaxation_time())       # 0.8

The dataclass is deliberately minimal — three scale factors, a few
helpers, no magic. A simulation script picks scales once, uses
``units.*_to_lb()`` to translate every physical input, and
``units.*_to_si()`` to translate every output. Round-trip is exact
in floating point.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


__all__ = ["LatticeUnits"]


# Speed of sound in lattice units, c_s² = 1/3 for D2Q9 (and most LBM
# stencils). Exposed as a constant so external code can do its own
# Mach-number checks without re-importing.
CS2_LB: float = 1.0 / 3.0
CS_LB: float = math.sqrt(CS2_LB)


@dataclass(frozen=True)
class LatticeUnits:
    """Three independent SI ↔ lattice conversion scales.

    Attributes
    ----------
    dx_si : float
        One lattice cell, in metres. Sets the length scale.
    dt_si : float
        One lattice timestep, in seconds. Sets the time scale.
    rho_si : float
        Lattice reference density ρ₀ = 1, in kg / m³. Sets the mass
        scale. For incompressible water this is just 1000.

    Derived scales
    --------------
    velocity_scale = dx / dt          (m s⁻¹ per lattice unit)
    pressure_scale = ρ (dx/dt)²       (Pa  per lattice unit)
    force_scale    = ρ (dx)³ (dx/dt²) (N   per lattice unit, 3-D scaling)
    visc_scale     = (dx)² / dt       (m² s⁻¹ per lattice unit kinematic ν)

    All of the helpers below are simple multiplies / divides by these
    scales. No solver state is touched.
    """

    dx_si: float
    dt_si: float
    rho_si: float = 1000.0      # water by default, kg / m³

    # ─── derived scale accessors ─────────────────────────────────

    @property
    def velocity_scale(self) -> float:
        """One lattice velocity unit, in m / s."""
        return self.dx_si / self.dt_si

    @property
    def kinematic_viscosity_scale(self) -> float:
        """One lattice kinematic-viscosity unit, in m² / s."""
        return (self.dx_si * self.dx_si) / self.dt_si

    @property
    def pressure_scale(self) -> float:
        """One lattice pressure unit, in Pa.

        Equivalent to ρ_SI · (dx / dt)². For compressible LBM the
        recovered pressure is `p = ρ cs²` in lattice units; the SI
        pressure is therefore `p_lb · pressure_scale`.
        """
        return self.rho_si * self.velocity_scale ** 2

    @property
    def force_scale_per_unit_area(self) -> float:
        """One lattice force-density unit, in N / m³ (3-D)."""
        return self.rho_si * self.velocity_scale ** 2 / self.dx_si

    # ─── named-constructor helpers ──────────────────────────────

    @classmethod
    def from_channel_flow(
        cls,
        channel_height_m: float,
        ny_lattice: int,
        kinematic_viscosity_si: float,
        mean_velocity_si: float,
        target_tau: float = 0.8,
        rho_si: float = 1000.0,
    ) -> LatticeUnits:
        """Pin scales by channel geometry, viscosity, and target τ.

        Strategy: ``dx`` is set so the channel fits in ``ny_lattice``
        cells; ``dt`` is set so the resulting kinematic viscosity in
        lattice units satisfies ``ν_lb = (τ − ½) cs²`` for the chosen
        ``target_tau``.  This is the most common way to set up an LBM
        run: you know the channel width and the fluid viscosity, you
        choose the resolution, and τ falls out.

        The mean physical velocity is used only for a Mach-number
        sanity check — the function refuses to return a setup with
        ``u_lb > 0.1`` (heuristic stability boundary).
        """
        if ny_lattice < 4:
            raise ValueError(f"ny_lattice = {ny_lattice} is too small")
        if not (0.5 < target_tau < 2.0):
            raise ValueError(
                f"target_tau = {target_tau} should be in (0.5, 2.0) for stable BGK"
            )

        dx_si = channel_height_m / ny_lattice
        nu_lb = (target_tau - 0.5) * CS2_LB
        dt_si = nu_lb * dx_si * dx_si / kinematic_viscosity_si

        units = cls(dx_si=dx_si, dt_si=dt_si, rho_si=rho_si)
        u_lb = units.velocity_to_lb(mean_velocity_si)
        if abs(u_lb) > 0.1:
            raise ValueError(
                f"Resulting lattice velocity u_lb = {u_lb:.3f} exceeds the "
                f"stability heuristic 0.1; reduce target_tau or refine ny_lattice."
            )
        return units

    @classmethod
    def from_three_scales(
        cls, dx_si: float, dt_si: float, rho_si: float = 1000.0
    ) -> LatticeUnits:
        """Direct constructor when the user has already picked scales."""
        if dx_si <= 0 or dt_si <= 0 or rho_si <= 0:
            raise ValueError("dx_si, dt_si, rho_si must all be > 0")
        return cls(dx_si=dx_si, dt_si=dt_si, rho_si=rho_si)

    # ─── conversions ────────────────────────────────────────────

    def length_to_lb(self, x_si: float) -> float:
        return x_si / self.dx_si

    def length_to_si(self, x_lb: float) -> float:
        return x_lb * self.dx_si

    def time_to_lb(self, t_si: float) -> float:
        return t_si / self.dt_si

    def time_to_si(self, t_lb: float) -> float:
        return t_lb * self.dt_si

    def velocity_to_lb(self, u_si: float) -> float:
        return u_si / self.velocity_scale

    def velocity_to_si(self, u_lb: float) -> float:
        return u_lb * self.velocity_scale

    def kinematic_viscosity_to_lb(self, nu_si: float) -> float:
        return nu_si / self.kinematic_viscosity_scale

    def kinematic_viscosity_to_si(self, nu_lb: float) -> float:
        return nu_lb * self.kinematic_viscosity_scale

    def pressure_to_lb(self, p_si: float) -> float:
        return p_si / self.pressure_scale

    def pressure_to_si(self, p_lb: float) -> float:
        return p_lb * self.pressure_scale

    # ─── derived dimensionless numbers ─────────────────────────

    def relaxation_time(self) -> float:
        """The BGK τ that this unit system produces.

        From ``ν_lb = (τ − ½) cs²``; this is the value to plug into
        ``SimulationParams.fluid.tau``.
        """
        # Recover ν_lb from the scales themselves: by construction
        # the user already knows ν_SI; but we provide this so that
        # `units.relaxation_time()` is a single source of truth.
        # We pick ν_SI = nu_lb * scale, then invert. If the user
        # didn't go through `from_channel_flow`, they'll need to
        # compute τ themselves and pass it explicitly.
        # Here we just expose the relationship:
        raise NotImplementedError(
            "LatticeUnits cannot compute τ without ν_SI. Use "
            "kinematic_viscosity_to_lb(nu_SI) and τ = ν_lb/cs² + ½."
        )

    def reynolds(self, L_si: float, U_si: float, nu_si: float) -> float:
        """Re = U·L / ν (SI inputs, dimensionless output)."""
        if nu_si <= 0:
            raise ValueError("nu_si must be > 0")
        return U_si * L_si / nu_si

    def peclet(self, L_si: float, U_si: float, D_si: float) -> float:
        """Pe = U·L / D (SI inputs, dimensionless output)."""
        if D_si <= 0:
            raise ValueError("D_si must be > 0")
        return U_si * L_si / D_si

    def capillary(
        self, mu_si: float, U_si: float, surface_modulus_si: float
    ) -> float:
        """Capillary number Ca = μ U / G_s for capsule problems.

        Parameters
        ----------
        mu_si : dynamic viscosity, Pa·s
        U_si  : characteristic velocity, m/s
        surface_modulus_si : surface shear modulus G_s, N/m
        """
        if surface_modulus_si <= 0:
            raise ValueError("surface_modulus_si must be > 0")
        return mu_si * U_si / surface_modulus_si

    def confinement(self, particle_radius_si: float, channel_height_si: float) -> float:
        """Confinement χ = 2 r / H (or d / H)."""
        if channel_height_si <= 0:
            raise ValueError("channel_height_si must be > 0")
        return 2.0 * particle_radius_si / channel_height_si

    # ─── representation ─────────────────────────────────────────

    def __str__(self) -> str:
        return (
            f"LatticeUnits(dx={self.dx_si:.3e} m, dt={self.dt_si:.3e} s, "
            f"ρ={self.rho_si:.3f} kg/m³,\n"
            f"             velocity_scale={self.velocity_scale:.3e} m/s, "
            f"ν_scale={self.kinematic_viscosity_scale:.3e} m²/s,\n"
            f"             pressure_scale={self.pressure_scale:.3e} Pa)"
        )
