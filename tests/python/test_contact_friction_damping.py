"""Regression tests for the DEM-style dissipative/frictional contact (Path A).

The inter-capsule contact force is

    F = n * F_n  +  F_t,

with the conservative power-law repulsion ``F_rep = epsilon (sigma/r)^power``
along the contact normal ``n`` (pointing from the source node toward the
target node), an optional normal viscoelastic dashpot, and optional tangential
Coulomb friction:

    F_n = max(0, F_rep - gamma_n * v_n),     v_n = (v_a - v_b) . n
    F_t = -mu * F_n * (v_t / |v_t|),         v_t = (v_a - v_b) - v_n * n

These tests pin the qualitative physics that defines the model:

  * gamma_n = mu = 0 reproduces the legacy pure repulsion (opt-in safety);
  * a dashpot resists approach (|F_n| up) and is clamped on separation
    (cohesionless: no spurious attraction) -> coefficient of restitution < 1;
  * Coulomb friction opposes sliding and is capped at mu*|F_n|.

References: Cundall & Strack, Geotechnique 29, 47 (1979);
Brilliantov et al., Phys. Rev. E 53, 5382 (1996).
"""

from __future__ import annotations

import pathlib
import sys

import pytest

# Make the compiled engine importable from the in-tree build (like run.py).
_REPO = pathlib.Path(__file__).resolve().parents[2]
for _d in ("build", "build_phase2", "build_phase1"):
    _cand = _REPO / _d / "python"
    if _cand.is_dir():
        sys.path.insert(0, str(_cand))

sc = pytest.importorskip("softflow_core", reason="C++ engine not built")


def _net_force_on_c0(gamma_n, mu, vrel, sep=5.0):
    """Net contact force on capsule 0 from capsule 1.

    Capsule 0 is at x=10, capsule 1 at x=10+sep (both radius 2). At sep=5 the
    surfaces are 1 LU apart (non-overlapping) and every node of c0 lies left of
    every node of c1, so the contact normal on c0 points cleanly in -x.
    """
    rp = sc.RepulsionParams()
    rp.epsilon, rp.sigma, rp.r_cut, rp.power = 0.05, 1.0, 3.5, 4
    rp.damping_normal = gamma_n
    rp.friction_coeff = mu
    rep = sc.RepulsionForce(rp)

    system = sc.CapsuleSystem()
    mp = sc.MembraneParams()
    system.addCapsule(sc.Vec2d(10.0, 10.0), 2.0, 12, mp, 0)
    system.addCapsule(sc.Vec2d(10.0 + sep, 10.0), 2.0, 12, mp, 0)
    c0, c1 = system[0], system[1]
    for k in range(c0.numNodes()):
        c0.setNodeVelocity(k, sc.Vec2d(vrel[0], vrel[1]))
    for k in range(c1.numNodes()):
        c1.setNodeVelocity(k, sc.Vec2d(0.0, 0.0))
    c0.clearForces()
    c1.clearForces()
    rep.computeInterCapsule(c0, c1)
    fx = sum(c0.nodeForce(k).x for k in range(c0.numNodes()))
    fy = sum(c0.nodeForce(k).y for k in range(c0.numNodes()))
    return fx, fy


def test_zero_params_reproduce_legacy_repulsion():
    """gamma_n = mu = 0 must reproduce pure repulsion, regardless of velocity."""
    fx_static, fy_static = _net_force_on_c0(0.0, 0.0, (0.0, 0.0))
    fx_moving, fy_moving = _net_force_on_c0(0.0, 0.0, (0.3, 0.2))
    assert fx_static == pytest.approx(fx_moving)
    assert fy_static == pytest.approx(fy_moving)
    # And the legacy force is purely repulsive (pushes c0 in -x, away from c1).
    assert fx_static < 0.0
    assert fy_static == pytest.approx(0.0, abs=1e-9)


def test_damping_with_zero_velocity_is_inert():
    """With v_rel = 0 the dashpot/friction add nothing (reduces to repulsion)."""
    fx_rep, fy_rep = _net_force_on_c0(0.0, 0.0, (0.0, 0.0))
    fx_dem, fy_dem = _net_force_on_c0(0.5, 0.3, (0.0, 0.0))
    assert fx_dem == pytest.approx(fx_rep)
    assert fy_dem == pytest.approx(fy_rep, abs=1e-12)


def test_normal_damping_resists_approach_and_is_cohesionless():
    """Dashpot strengthens the push on approach and is clamped on separation."""
    fx_rep, _ = _net_force_on_c0(0.0, 0.0, (0.0, 0.0))
    fx_app, _ = _net_force_on_c0(0.5, 0.0, (+0.05, 0.0))   # c0 -> c1 (approach)
    fx_sep, _ = _net_force_on_c0(0.5, 0.0, (-0.05, 0.0))   # c0 <- c1 (separate)
    # Approaching: dissipative dashpot adds to the repulsion (|F| grows).
    assert abs(fx_app) > abs(fx_rep)
    # Separating: force is reduced and clamped >= 0 (never attractive: still -x).
    assert abs(fx_sep) < abs(fx_rep)
    assert fx_sep <= 0.0


def test_friction_opposes_sliding_and_is_capped():
    """Tangential Coulomb friction opposes the slide, magnitude <= mu*|F_n|."""
    mu = 0.3
    fx, fy = _net_force_on_c0(0.0, mu, (0.0, +0.05))   # c0 slides +y past c1
    # Friction acts along -y (opposing the +y slide).
    assert fy < 0.0
    # Coulomb cap: |F_t| <= mu * |F_normal|. Here |F_normal| ~ |fx| (normal is x).
    assert abs(fy) <= mu * abs(fx) + 1e-9
