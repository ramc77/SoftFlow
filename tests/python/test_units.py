"""Round-trip and dimensional-number tests for pysoftflow.units."""

import math
import sys
import pathlib

# Add the python/ dir to sys.path so we can run pytest without installing.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "python"))

from pysoftflow.units import LatticeUnits, CS2_LB     # noqa: E402


def test_round_trip_length():
    u = LatticeUnits(dx_si=1e-6, dt_si=1e-9)
    for x in [0.0, 1e-9, 1e-6, 5e-3]:
        assert math.isclose(u.length_to_si(u.length_to_lb(x)), x, rel_tol=1e-12)


def test_round_trip_velocity():
    u = LatticeUnits(dx_si=1e-6, dt_si=1e-9)
    for v in [0.0, 1e-3, 0.1, 50.0]:
        assert math.isclose(u.velocity_to_si(u.velocity_to_lb(v)), v, rel_tol=1e-12)


def test_round_trip_viscosity():
    u = LatticeUnits(dx_si=2e-6, dt_si=5e-10)
    for nu in [1e-7, 1e-6, 1e-5]:
        assert math.isclose(
            u.kinematic_viscosity_to_si(u.kinematic_viscosity_to_lb(nu)),
            nu,
            rel_tol=1e-12,
        )


def test_from_channel_flow_pins_tau():
    """`from_channel_flow` must give a setup whose ν_lb = (τ − ½) cs²."""
    target_tau = 0.8
    u = LatticeUnits.from_channel_flow(
        channel_height_m=100e-6,
        ny_lattice=64,
        kinematic_viscosity_si=1.0e-6,
        mean_velocity_si=1.0e-3,
        target_tau=target_tau,
    )
    nu_lb = u.kinematic_viscosity_to_lb(1.0e-6)
    expected_nu_lb = (target_tau - 0.5) * CS2_LB
    assert math.isclose(nu_lb, expected_nu_lb, rel_tol=1e-12)


def test_reynolds_dimensionless():
    u = LatticeUnits(dx_si=1e-6, dt_si=1e-9)
    Re = u.reynolds(L_si=1e-3, U_si=1e-3, nu_si=1e-6)
    assert math.isclose(Re, 1.0)


def test_capillary_dimensionless():
    u = LatticeUnits(dx_si=1e-6, dt_si=1e-9)
    Ca = u.capillary(mu_si=1e-3, U_si=1e-3, surface_modulus_si=6e-6)
    assert math.isclose(Ca, 1e-3 * 1e-3 / 6e-6)


def test_confinement():
    u = LatticeUnits(dx_si=1e-6, dt_si=1e-9)
    chi = u.confinement(particle_radius_si=2e-6, channel_height_si=10e-6)
    assert math.isclose(chi, 0.4)


def test_mach_guard():
    """from_channel_flow should refuse setups with u_lb above ~0.1."""
    import pytest

    with pytest.raises(ValueError, match="u_lb"):
        LatticeUnits.from_channel_flow(
            channel_height_m=100e-6,
            ny_lattice=64,
            kinematic_viscosity_si=1.0e-6,
            mean_velocity_si=10.0,            # absurdly fast → u_lb > 0.1
            target_tau=0.8,
        )
