// V&V: Chemistry mass conservation (Fick leaching + Langmuir adsorption).
//
// Invariant for a closed periodic system with no inlet, no walls, no
// advection (u = 0):
//
//   M_total(t) = sum_c M_p(c, t)
//              + sum_{cells} C(t)
//              + sum_c sum_k Gamma(c, k, t) * ds(c, k)
//
//                   ≡ M_total(0)        within roundoff.
//
// Physical interpretation:
//   • M_p(c, t)             — capsule c's chemical reservoir (drained by leaching)
//   • sum C(t)              — free concentration in the bulk fluid (D2Q9 ADR)
//   • sum Gamma · ds        — surface-bound mass via Langmuir adsorption
//
// This is the protocol the manuscript Figure 10 / Table 1 ought to cite
// for its 'global mass conservation' claim.
//
// We construct a single circular capsule with Nn nodes chosen so the
// average edge length ds_k ≈ 1 lu (so that the per-node and per-edge
// flux conventions agree exactly, see notes in advection_diffusion.cpp).
// Initial M_p > 0, C_eq above background, k_a = k_d = 0 (Langmuir off
// for the leaching-only subtest). Run a few hundred ADR steps and
// audit the budget.
//
// Tolerance: 1e-10 relative — looser than 1e-12 because the budget
// involves O(thousands) of additive flux updates per node-step that
// each accumulate rounding error.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "engine/simulation.h"
#include "lbm/advection_diffusion.h"
#include "lbm/lattice_field.h"
#include "membrane/capsule_system.h"
#include "core/parameters.h"

#include <cmath>
#include <vector>

using namespace softflow;

namespace {

// Sum free concentration over the lattice (dx = 1).
Real sumLatticeC(const AdvectionDiffusion& ad, int species = 0) {
    const Real* C = ad.concentrationData(species);
    Real s = 0.0;
    int N = ad.getNx() * ad.getNy();
    for (int n = 0; n < N; ++n) s += C[n];
    return s;
}

// Sum bound surface mass Sum_c Sum_k Gamma_k * ds_k.
Real sumSurfaceMass(const AdvectionDiffusion& ad,
                    const CapsuleSystem& caps,
                    int periodic_nx) {
    Real Lx = static_cast<Real>(periodic_nx);
    Real s = 0.0;
    for (int c = 0; c < caps.numCapsules(); ++c) {
        const Capsule& cap = caps[c];
        int Nn = cap.numNodes();
        for (int k = 0; k < Nn; ++k) {
            int kprev = (k - 1 + Nn) % Nn;
            int knext = (k + 1) % Nn;
            auto dsfn = [&](Vec2d a, Vec2d b) {
                Vec2d d = a - b;
                if (Lx > 0.0) {
                    if (d.x >  0.5 * Lx) d.x -= Lx;
                    if (d.x < -0.5 * Lx) d.x += Lx;
                }
                return d.norm();
            };
            Real ds_prev = dsfn(cap.nodePosition(k),     cap.nodePosition(kprev));
            Real ds_next = dsfn(cap.nodePosition(knext), cap.nodePosition(k));
            Real ds = 0.5 * (ds_prev + ds_next);
            s += ad.getNodeGamma(c, k) * ds;
        }
    }
    return s;
}

// Sum capsule reservoir mass Sum_c M_p(c), skipping infinite (-1) entries.
Real sumReservoir(const AdvectionDiffusion& ad, int ncaps) {
    Real s = 0.0;
    for (int c = 0; c < ncaps; ++c) {
        Real mp = ad.getCapsuleMp(c);
        if (mp >= 0.0) s += mp;
    }
    return s;
}

// Build a periodic-x / periodic-y simulation with no walls, no body force,
// one capsule of radius R at the centre with Nn nodes, chemistry params
// per leach_rate / Ceq / Mp_init / kads / kdes / Gmax.
SimulationParams baseParams(int nx, int ny, Real R, int Nn,
                            Real k_leach, Real Ceq, Real Mp_init,
                            Real k_a, Real k_d, Real Gmax) {
    SimulationParams p;
    p.nx = nx;
    p.ny = ny;
    p.dt = 1.0;
    p.fluid.tau = 0.6;
    p.fluid.rho0 = 1.0;
    p.fluid.body_force_x = 0.0;
    p.fluid.boundary_type = BoundaryType::PERIODIC;
    p.fluid.periodic_y = true;
    p.fluid.collision_model = CollisionModel::BGK;

    p.scalar.enabled = true;
    p.scalar.n_species = 1;
    p.scalar.diffusivity         = {0.01};
    p.scalar.inlet_concentration = {0.0};
    p.scalar.periodic_y = true;
    p.scalar.k_leach     = {k_leach};
    p.scalar.C_eq        = {Ceq};
    p.scalar.M_p_initial = {Mp_init};
    p.scalar.k_adsorb    = {k_a};
    p.scalar.k_desorb    = {k_d};
    p.scalar.Gamma_max   = {Gmax};

    p.enable_stability_checks = false;
    p.enable_profiling        = false;
    p.metrics_interval        = 0;
    p.vtk_dump_every  = 0;
    p.csv_dump_every  = 0;
    p.probe_dump_every= 0;
    p.stats_dump_every= 0;
    p.output_dir = "/tmp/softflow_vv_chembudget";

    (void)R; (void)Nn;
    return p;
}

} // anonymous namespace

TEST_CASE("Chemistry: leaching-only mass conservation (M_p + sum C constant)") {
    // Nn = 32 on R = 32/(2π) → average edge length ds ≈ 1 lu exactly.
    const int   nx = 128, ny = 96;
    const Real  R  = 32.0 / (2.0 * M_PI);
    const int   Nn = 32;
    const Real  Mp_init = 5.0;
    const Real  k_leach = 0.01;
    const Real  Ceq     = 0.05;   // background = 0 → uniform leaching at start

    SimulationParams p = baseParams(nx, ny, R, Nn,
                                    k_leach, Ceq, Mp_init,
                                    /*ka=*/0.0, /*kd=*/0.0, /*Gmax=*/1.0);

    Simulation sim(p);
    sim.initialize();

    // Add one capsule at the centre. (Membrane stiffness doesn't matter —
    // we will not call computeMembraneForces; ADR runs independently.)
    MembraneParams mp;
    auto& caps = sim.capsules();
    caps.addCapsule(Vec2d{nx / 2.0, ny / 2.0}, R, Nn, mp, /*type=*/0);

    auto* ad = sim.advectionDiffusion();
    REQUIRE(ad != nullptr);

    // Drive the chemistry by hand to isolate the test: leaching only,
    // no advection (u = 0 because no body force, no IBM forces spread).
    const std::vector<Real> rel  = {0.0};   // constant-rate fallback unused
    const std::vector<Real> absn = {0.0};

    Real M0 = Mp_init;
    Real total_0 = M0 + sumLatticeC(*ad) + sumSurfaceMass(*ad, caps, nx);
    INFO("M_total(0) = " << total_0);
    REQUIRE(std::abs(total_0 - M0) < 1e-12);

    const int n_steps = 200;
    Real worst_rel = 0.0;
    for (int step = 0; step < n_steps; ++step) {
        // Run the ADR step (diffusion only — u = 0).
        ad->step(sim.lbmSolver().field());
        // Apply chemistry source: this is the path under test.
        ad->applyChemistry(caps, rel, absn, p.scalar);

        Real Mp_now  = sumReservoir(*ad, caps.numCapsules());
        Real sumC    = sumLatticeC(*ad);
        Real sumGds  = sumSurfaceMass(*ad, caps, nx);
        Real total_t = Mp_now + sumC + sumGds;
        Real rel_err = std::abs(total_t - total_0) / total_0;
        if (rel_err > worst_rel) worst_rel = rel_err;
    }

    Real Mp_final = sumReservoir(*ad, caps.numCapsules());
    Real C_final  = sumLatticeC(*ad);
    INFO("M_p(final)      = " << Mp_final);
    INFO("Sum C(final)    = " << C_final);
    INFO("worst rel. err. = " << worst_rel);
    CHECK(worst_rel < 1e-10);
    // Sanity: leaching actually happened.
    CHECK(Mp_final < Mp_init - 1e-6);
    CHECK(C_final  > 1e-6);
}

TEST_CASE("Chemistry: leaching + Langmuir mass conservation") {
    // Same geometry, now with adsorption turned on. Released mass spreads
    // to the lattice and then is partly adsorbed back onto the membrane
    // (Gamma > 0). The total budget M_p + sum C + sum Gamma·ds must still
    // be conserved.
    const int   nx = 128, ny = 96;
    const Real  R  = 32.0 / (2.0 * M_PI);
    const int   Nn = 32;
    const Real  Mp_init = 5.0;
    const Real  k_leach = 0.01;
    const Real  Ceq     = 0.05;
    const Real  k_a     = 0.02;
    const Real  k_d     = 0.005;
    const Real  Gmax    = 0.1;

    SimulationParams p = baseParams(nx, ny, R, Nn,
                                    k_leach, Ceq, Mp_init,
                                    k_a, k_d, Gmax);

    Simulation sim(p);
    sim.initialize();
    MembraneParams mp;
    sim.capsules().addCapsule(Vec2d{nx / 2.0, ny / 2.0}, R, Nn, mp, /*type=*/0);

    auto* ad = sim.advectionDiffusion();
    REQUIRE(ad != nullptr);

    const std::vector<Real> rel  = {0.0};
    const std::vector<Real> absn = {0.0};

    Real M0 = Mp_init;
    Real total_0 = M0
                 + sumLatticeC(*ad)
                 + sumSurfaceMass(*ad, sim.capsules(), nx);

    const int n_steps = 300;
    Real worst_rel = 0.0;
    for (int step = 0; step < n_steps; ++step) {
        ad->step(sim.lbmSolver().field());
        ad->applyChemistry(sim.capsules(), rel, absn, p.scalar);

        Real Mp_now  = sumReservoir(*ad, sim.capsules().numCapsules());
        Real sumC    = sumLatticeC(*ad);
        Real sumGds  = sumSurfaceMass(*ad, sim.capsules(), nx);
        Real total_t = Mp_now + sumC + sumGds;
        Real rel_err = std::abs(total_t - total_0) / total_0;
        if (rel_err > worst_rel) worst_rel = rel_err;
    }

    // Sanity: bound coverage built up.
    Real sumGds_final = sumSurfaceMass(*ad, sim.capsules(), nx);
    INFO("Sum Gamma·ds final = " << sumGds_final);
    INFO("worst rel. err.    = " << worst_rel);
    CHECK(worst_rel < 1e-10);
    CHECK(sumGds_final > 1e-6);
}
