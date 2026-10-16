// V&V: tank-treading frequency of a Skalak capsule in Couette flow at two
// lattice resolutions ().
//
// Both runs keep the capillary number Ca = mu*gamma*R/G_s and the
// confinement R/H fixed (U_w fixed, H and R doubled, G_s fixed).  The
// particle Reynolds number gamma*R^2/nu is NOT held fixed: it doubles with
// the resolution (tau and U_w are unchanged), so the difference between the
// two results is not a pure discretisation error and the test does not
// assert convergence.  (Convergence with all dimensionless groups fixed is
// checked by the Jeffery-orbit test, which scales the shear rate as 1/a^2.)
//
// What is asserted: the measured frequency lies within 10 % of the
// small-capillary-number, unbounded-flow limit gamma/(4 pi) at both
// resolutions.  A deficit of a few per cent is expected from the finite
// Ca and the confinement 2R/H = 0.16.
//
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "engine/simulation.h"
#include "membrane/capsule_system.h"
#include "core/parameters.h"

#include <cmath>
#include <vector>
#include "vv_dump.h"

using namespace softflow;

namespace {

struct ShearResult {
    Real f_meas;
    Real f_expect;
    Real rel_err;
    int  Nn;
    int  ny;
};

// Run a tank-treading shear sim at a given lattice refinement factor k.
// All physical scales are fixed: target Ca, R/H, Re identical at all k.
ShearResult runShear(int k, int max_steps) {
    SimulationParams p;
    p.nx = 80  * k;
    p.ny = 64  * k;                 // H = ny - 2
    p.dt = 1.0;
    // Holding ν = (τ-½)/3 = 0.1 and U_w = 0.02 fixes γ̇ scaling.
    // Refinement keeps Re = U_w * H * 2 / ν invariant since H ∝ k but
    // U_w scales as 1/k (LBM convention to keep advective CFL constant)
    // — except we don't scale U_w because for tank-treading the
    // relevant test is grid-density at fixed γ̇. Use the simpler
    // convention: fix τ, U_w; γ̇ ∝ 1/k. Then the absolute period in
    // *lattice timesteps* scales as k.
    p.fluid.tau           = 0.8;
    p.fluid.rho0          = 1.0;
    p.fluid.boundary_type = BoundaryType::PERIODIC;
    p.fluid.periodic_y    = false;
    p.fluid.body_force_x  = 0.0;
    p.fluid.collision_model = CollisionModel::REGULARIZED;
    const Real U_w = 0.02;
    p.fluid.top_wall_velocity    = +U_w;
    p.fluid.bottom_wall_velocity = -U_w;
    p.scalar.enabled = false;
    p.enable_stability_checks = false;
    p.enable_profiling        = false;
    p.metrics_interval        = 0;
    p.vtk_dump_every  = 0;
    p.csv_dump_every  = 0;
    p.probe_dump_every= 0;
    p.stats_dump_every= 0;
    p.output_dir = "/tmp/softflow_vv_shear_conv";

    Simulation sim(p);
    sim.initialize();

    MembraneParams mp;
    mp.model       = MembraneModel::SKALAK;
    mp.G_s         = 0.002;
    mp.C_skalak    = 10.0;
    mp.k_bend      = 0.001;
    mp.k_area      = 0.02;
    mp.k_perimeter = 0.005;
    mp.gamma_visc  = 0.0;
    mp.eta_membrane= 0.0;

    const Real R  = 5.0 * k;
    const int  Nn = Capsule::computeOptimalNodes(R);
    sim.capsules().addCapsule(Vec2d{p.nx / 2.0, p.ny / 2.0}, R, Nn, mp, 0);

    const Real H         = static_cast<Real>(p.ny - 2);
    const Real gamma_dot = 2.0 * U_w / H;
    const Real f_expect  = gamma_dot / (4.0 * M_PI);

    // Spinup scales with k (refinement makes the channel relaxation
    // slower in lattice steps).
    const int spinup = 8000 * k;
    for (int s = 0; s < spinup; ++s) sim.step();

    std::vector<Real> ry_hist;
    const int sample_every = 50;
    for (int s = 0; s < max_steps; ++s) {
        sim.step();
        if (s % sample_every == 0) {
            Vec2d c0 = sim.capsules()[0].centroid();
            Vec2d n0 = sim.capsules()[0].nodePosition(0);
            ry_hist.push_back(n0.y - c0.y);
            if (ry_hist.size() > 200) {
                int crossings = 0;
                size_t i0 = ry_hist.size() / 2;
                for (size_t i = i0 + 1; i < ry_hist.size(); ++i) {
                    if ((ry_hist[i-1] < 0.0 && ry_hist[i] >= 0.0) ||
                        (ry_hist[i-1] > 0.0 && ry_hist[i] <= 0.0))
                        ++crossings;
                }
                if (crossings >= 6) {
                    Real n_periods    = crossings / 2.0;
                    Real window_steps = (ry_hist.size() - i0) * sample_every;
                    Real T_meas       = window_steps / n_periods;
                    Real f_meas       = 1.0 / T_meas;
                    return {f_meas, f_expect, std::abs(f_meas - f_expect) / f_expect, Nn, p.ny};
                }
            }
        }
    }
    // Insufficient data — return what we have so the diagnostic prints.
    return {0.0, f_expect, 1.0, Nn, p.ny};
}

} // anonymous namespace

TEST_CASE("Tank-treading: frequency within 10% of the small-Ca limit at k = 1, 2") {
    // k = 1: nominal. k = 2: each axis doubled, R doubled, Nn doubled.
    // (k = 4 would be ideal but expensive — 32× per-step work, 4×
    // longer wall-clock to capture each period.)
    ShearResult r1 = runShear(/*k=*/1, /*max_steps=*/200000);
    ShearResult r2 = runShear(/*k=*/2, /*max_steps=*/600000);

    INFO("k = 1:  ny = " << r1.ny << "  Nn = " << r1.Nn
         << "  f = " << r1.f_meas << "  rel_err = " << r1.rel_err);
    INFO("k = 2:  ny = " << r2.ny << "  Nn = " << r2.Nn
         << "  f = " << r2.f_meas << "  rel_err = " << r2.rel_err);

    // Both resolutions should reach the tank-treading regime cleanly.
    REQUIRE(r1.f_meas > 0.0);
    REQUIRE(r2.f_meas > 0.0);
    if (FILE* out = vv::open("tank_treading_convergence",
                             "k,ny,Nn,f_measured,f_keller_skalak,rel_err")) {
        for (const auto& r : {r1, r2})
            std::fprintf(out, "%d,%d,%d,%.6e,%.6e,%.6e\n", r.ny / 64, r.ny, r.Nn,
                         r.f_meas, r.f_expect, r.rel_err);
        vv::close(out);
    }

    // Both resolutions must reproduce the small-Ca limit to within 10 %.
    CHECK(r1.rel_err < 0.10);
    CHECK(r2.rel_err < 0.10);
}
