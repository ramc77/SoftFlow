// V&V: body-force-driven Poiseuille flow for every collision operator.
//
// test_poiseuille.cpp exercises BGK only.  This test runs the same
// channel with BGK, MRT and regularised BGK and checks both the profile
// (L2 < 1%) and the volumetric flow rate against
//
//   u(y) = (G / 2 nu) y (H - y),     Q = G H^3 / (12 nu),
//
// at two relaxation times.  The flow-rate check is sensitive to an error
// in the fraction of the body force applied per time step: regularising
// only the second-order non-equilibrium moment under Guo forcing applies
// (3 - 1/tau)/2 of the force, i.e. a 12.5% deficit at tau = 0.8, which
// this test would catch.
//
// Tolerance is 2%: with halfway bounce-back the effective wall position of
// BGK depends on tau (error grows as (tau - 1/2)^2), giving ~1.1% L2 at
// tau = 1.2 on H = 32.  That is a known discretisation property, not a
// defect, and is an order of magnitude below the force-scaling error the
// test is designed to detect.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "engine/simulation.h"
#include "lbm/lattice_field.h"

#include <algorithm>
#include <cmath>
#include <string>
#ifdef _OPENMP
#include <omp.h>
#endif

using namespace softflow;

namespace {

struct Result { Real l2; Real q_ratio; };

Result runChannel(CollisionModel model, Real tau) {
    SimulationParams p;
    p.nx = 8;
    p.ny = 34;                       // H = 32 fluid rows
    p.dt = 1.0;
    p.fluid.tau  = tau;
    p.fluid.rho0 = 1.0;
    p.fluid.boundary_type   = BoundaryType::PERIODIC;
    p.fluid.body_force_x    = 1.0e-6;
    p.fluid.collision_model = model;
    p.enable_stability_checks = false;
    p.enable_profiling        = false;
    p.output_interval         = 0;
    p.vtk_dump_every          = 0;
    p.csv_dump_every          = 0;
    p.probe_dump_every        = 0;
    p.stats_dump_every        = 0;
    p.metrics_interval        = 0;
    p.output_dir = "/tmp/softflow_vv_poiseuille_ops";

    Simulation sim(p);
    sim.initialize();

    const Real nu = p.kinematicViscosity();
    const int  H  = p.ny - 2;
    // ~5 viscous relaxation times H^2/nu.
    const int n_steps = static_cast<int>(5.0 * H * H / nu);
    for (int s = 0; s < n_steps; ++s) sim.step();

    const auto& field = sim.lbmSolver().field();
    const Real G = p.fluid.body_force_x;
    Real num = 0.0, den = 0.0, q = 0.0;
    for (int y = 1; y <= H; ++y) {
        Real ux = field.getUx(p.nx / 2, y);
        Real yp = static_cast<Real>(y - 1) + 0.5;
        Real ua = (G / (2.0 * nu)) * yp * (static_cast<Real>(H) - yp);
        num += (ux - ua) * (ux - ua);
        den += ua * ua;
        q   += ux;
    }
    const Real q_an = G * H * H * H / (12.0 * nu);
    return {std::sqrt(num / den), q / q_an};
}

} // namespace

TEST_CASE("Poiseuille: every collision operator recovers profile and flow rate") {
#ifdef _OPENMP
    omp_set_num_threads(1);   // 8 x 34 grid: thread overhead dominates
#endif
    const struct { CollisionModel m; std::string name; } ops[] = {
        {CollisionModel::BGK,         "BGK"},
        {CollisionModel::MRT,         "MRT"},
        {CollisionModel::REGULARIZED, "regularized"},
    };
    for (Real tau : {0.8, 1.2}) {
        for (const auto& op : ops) {
            Result r = runChannel(op.m, tau);
            INFO(op.name << " tau=" << tau << "  L2=" << r.l2
                 << "  Q/Q_an=" << r.q_ratio);
            MESSAGE(op.name << " tau=" << tau << "  L2=" << r.l2
                    << "  Q/Q_an=" << r.q_ratio);
            CHECK(r.l2 < 0.02);
            CHECK(std::abs(r.q_ratio - 1.0) < 0.02);
        }
    }
}

// A periodic channel is invariant under translation in x, so the steady
// profile must be identical in every column and independent of the
// domain length.  Halfway bounce-back that does not wrap the fluid
// neighbour across the periodic seam leaves the diagonal populations at
// the two seam corners unreflected; that injects mass and momentum there
// and breaks both properties (it gave an 8% velocity error in the
// near-wall node at the seam, decaying over ~10 columns).
TEST_CASE("Poiseuille: periodic channel is translation invariant") {
#ifdef _OPENMP
    omp_set_num_threads(1);
#endif
    auto profileSpread = [](int nx, Real& l2_mid) {
        SimulationParams p;
        p.nx = nx;
        p.ny = 34;
        p.dt = 1.0;
        p.fluid.tau = 0.8;
        p.fluid.rho0 = 1.0;
        p.fluid.boundary_type = BoundaryType::PERIODIC;
        p.fluid.body_force_x = 1.0e-6;
        p.fluid.collision_model = CollisionModel::BGK;
        p.enable_stability_checks = false;
        p.enable_profiling = false;
        p.output_interval = 0;
        p.vtk_dump_every = p.csv_dump_every = p.probe_dump_every = 0;
        p.stats_dump_every = p.metrics_interval = 0;
        p.output_dir = "/tmp/softflow_vv_poiseuille_seam";
        Simulation sim(p);
        sim.initialize();
        for (int s = 0; s < 20000; ++s) sim.step();
        const auto& f = sim.lbmSolver().field();
        Real spread = 0.0, umax = 0.0;
        for (int y = 1; y <= p.ny - 2; ++y) {
            Real lo = 1e30, hi = -1e30;
            for (int x = 0; x < nx; ++x) {
                lo = std::min(lo, f.getUx(x, y));
                hi = std::max(hi, f.getUx(x, y));
            }
            spread = std::max(spread, hi - lo);
            umax = std::max(umax, hi);
        }
        const Real nu = p.kinematicViscosity();
        const int H = p.ny - 2;
        Real num = 0.0, den = 0.0;
        for (int y = 1; y <= H; ++y) {
            Real yp = y - 0.5;
            Real ua = (p.fluid.body_force_x / (2.0 * nu)) * yp * (H - yp);
            Real ux = f.getUx(nx / 2, y);
            num += (ux - ua) * (ux - ua);
            den += ua * ua;
        }
        l2_mid = std::sqrt(num / den);
        return spread / umax;
    };
    Real l2_4 = 0.0, l2_32 = 0.0;
    const Real s4 = profileSpread(4, l2_4);
    const Real s32 = profileSpread(32, l2_32);
    MESSAGE("relative column spread nx=4: " << s4 << ", nx=32: " << s32
            << ";  L2 nx=4: " << l2_4 << ", nx=32: " << l2_32);
    CHECK(s4 < 1e-10);
    CHECK(s32 < 1e-10);
    CHECK(std::abs(l2_4 - l2_32) < 1e-8);
}
