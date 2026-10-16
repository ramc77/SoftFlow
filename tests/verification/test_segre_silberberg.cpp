// V&V: inertial lateral migration of a near-rigid particle (Segré-Silberberg).
//
// A neutrally buoyant, near-rigid particle in plane Poiseuille flow at
// finite Reynolds number migrates across streamlines to an off-centre
// equilibrium between the centreline and the wall (Segré & Silberberg 1962;
// for 2D channels see Krüger et al. 2017 §11 and Sui et al. 2008).
//
// The particle is a stiff elastic capsule (Hookean ring with discrete-
// elastica bending), which exerts its membrane forces on the fluid and is
// therefore hydrodynamically active.  (The is_rigid mode only projects the
// interpolated node velocities onto a rigid motion and spreads no force, so
// it is a passive tracer and does not migrate.)
//
// Protocol: two runs released on opposite sides of the expected
// equilibrium, one near the centreline and one near the wall.  Both must
// migrate by a clear margin and settle at the same off-centre position:
//   * |y_eq - centre| / (H/2) in (0.2, 0.9) for both runs,
//   * the two equilibria agree to 0.05 (H/2),
//   * each particle moves at least 0.15 (H/2) from its release point,
//   * the equilibrium is genuinely off the wall: particle surface at
//     least 1 lu from the wall surface (a particle held at the node clamp
//     next to the wall must not pass).

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "engine/simulation.h"
#include "lbm/lattice_field.h"
#include "membrane/capsule_system.h"
#include "core/parameters.h"

#include <cmath>
#include <string>
#include "vv_dump.h"

using namespace softflow;

namespace {

struct Migration { Real y0, y_eq; int steps; };

Migration release(Real y0, FILE* out) {
    SimulationParams p;
    p.nx = 100;
    p.ny = 64;
    p.dt = 1.0;
    p.fluid.tau           = 0.8;           // nu = 0.1
    p.fluid.rho0          = 1.0;
    p.fluid.boundary_type = BoundaryType::PERIODIC;
    p.fluid.periodic_y    = false;
    // u_max = g H^2 / (8 nu) ~ 0.048: channel Re = u_max H / nu ~ 30,
    // Mach ~ 0.08.  (At Re ~ 130 the particle overshoots to the wall.)
    p.fluid.body_force_x  = 1.0e-5;
    p.fluid.collision_model = CollisionModel::REGULARIZED;
    p.scalar.enabled = false;
    p.enable_stability_checks = false;
    p.enable_profiling        = false;
    p.metrics_interval        = 0;
    p.vtk_dump_every  = 0;
    p.csv_dump_every  = 0;
    p.probe_dump_every= 0;
    p.stats_dump_every= 0;
    p.output_dir = "/tmp/softflow_vv_segre";

    Simulation sim(p);
    sim.initialize();

    MembraneParams mp;                       // stiff, hydrodynamically active
    mp.model       = MembraneModel::HOOKEAN;
    mp.k_stretch   = 0.5;
    mp.use_helfrich_bending = true;
    mp.k_bend      = 0.06;
    mp.k_area      = 2.0;
    mp.k_perimeter = 0.2;
    mp.gamma_visc  = 0.0;
    mp.eta_membrane= 0.0;

    const Real R  = 4.0;
    const int  Nn = 25;                      // node spacing ~1 lu
    sim.capsules().addCapsule(Vec2d{p.nx / 2.0, y0}, R, Nn, mp, 0);

    const int max_steps = 600000, every = 1000;
    const Real centre = p.ny / 2.0;
    Real y = y0, y_prev = y0;
    int still = 0, s = 0;
    for (s = 0; s < max_steps; ++s) {
        sim.step();
        if ((s + 1) % every == 0) {
            y = sim.capsules()[0].centroid().y;
            if (out) std::fprintf(out, "%.2f,%d,%.6f\n", y0, s + 1, y);
            still = (std::abs(y - y_prev) < 0.002) ? still + 1 : 0;
            y_prev = y;
            if (still >= 40 && std::abs(y - y0) > 1.0) break;
        }
    }
    (void)centre;
    return {y0, y, s + 1};
}

} // namespace

TEST_CASE("Segre-Silberberg: near-rigid particle migrates to the same "
          "off-centre equilibrium from both sides") {
    const Real centre = 32.0, half_H = 31.0;
    FILE* out = vv::open("segre_silberberg", "y0,step,y");
    Migration inner = release(centre - 0.12 * half_H, out);   // near centreline
    Migration outer = release(centre - 0.85 * half_H, out);   // near wall
    vv::close(out);

    const Real oi = std::abs(inner.y_eq - centre) / half_H;
    const Real oo = std::abs(outer.y_eq - centre) / half_H;
    INFO("inner: y0 = " << inner.y0 << " -> y_eq = " << inner.y_eq
         << " (offset " << oi << ", " << inner.steps << " steps)");
    INFO("outer: y0 = " << outer.y0 << " -> y_eq = " << outer.y_eq
         << " (offset " << oo << ", " << outer.steps << " steps)");
    CHECK(oi > 0.2);
    CHECK(oi < 0.9);
    CHECK(oo > 0.2);
    CHECK(oo < 0.9);
    CHECK(std::abs(oi - oo) < 0.05);
    CHECK(std::abs(inner.y_eq - inner.y0) / half_H > 0.15);
    CHECK(std::abs(outer.y_eq - outer.y0) / half_H > 0.15);
    const Real R = 4.0;
    CHECK(half_H - std::abs(inner.y_eq - centre) - R > 1.0);
    CHECK(half_H - std::abs(outer.y_eq - centre) - R > 1.0);
}
