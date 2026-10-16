// V&V: Taylor dispersion (the project specification, row 6;).
//
// In a 2D plane Poiseuille flow with mean velocity u_mean driving advection
// of a passive scalar with molecular diffusivity D, a localized pulse
// disperses with an effective long-time diffusivity
//
//   D_eff = D + Pe^2 D / 210,        Pe = u_mean * H / D,
//
// (Aris 1956 / Taylor 1953). We verify the second moment of the cross-
// section-averaged concentration C_bar(x, t) grows linearly with slope
// 2 D_eff in the long-time regime t >> tau_mix = H^2 / (4 D).
//
// Tolerance per the project specification: < 5%.
//
// Why this test was previously parked (tests/verification/TODO.md):
// the original implementation computed the variance in the lab frame via
// sum_x C(x) * (x - x_bar)^2. Once the Gaussian pulse wrapped around the
// periodic x boundary, the variance estimate became unbounded (mass on
// both sides of the periodic seam contributes 'far' to the centroid).
// The Aris-Taylor result is unchanged; only the diagnostic was broken.
//
// Fix (this rewrite): compute the centroid and variance using *circular
// statistics* on the periodic x ring.
//
//   theta(x) = 2π x / Lx,
//   <e^{iθ}> = sum_x C_bar(x) e^{iθ(x)} / sum_x C_bar(x),
//   x_bar    = (Lx / 2π) * atan2(Im, Re)              (wrapped to [0, Lx)),
//   <(Δx)^2> = sum_x C_bar(x) (minImage(x - x_bar))^2 / sum_x C_bar(x).
//
// This is the standard treatment for periodic-domain statistics (see
// e.g. Mardia & Jupp, Directional Statistics, 2000). It is exact for any
// distribution narrower than Lx/2.
//
// We also de-trend the test against u_mean by working entirely in the
// circular frame — the pulse's mean position drifts with u_mean but the
// variance is unaffected (the centroid moves; the spread about the
// centroid is what we measure).

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "engine/simulation.h"
#include "lbm/advection_diffusion.h"
#include "lbm/lattice_field.h"

#include <cmath>
#include <vector>
#include "vv_dump.h"

using namespace softflow;

namespace {

constexpr Real D2Q9_W[9] = {
    4.0/9.0,
    1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0,
    1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0
};

// Circular mean and circular variance of the cross-section-averaged
// concentration C_bar(x) on a periodic ring of length Lx.
struct CircMoments {
    Real mass;
    Real x_bar;
    Real var;
};

CircMoments circularMoments(const std::vector<Real>& Cbar, Real Lx) {
    const int nx = static_cast<int>(Cbar.size());
    Real mass = 0.0;
    Real Re = 0.0, Im = 0.0;
    for (int x = 0; x < nx; ++x) {
        mass += Cbar[x];
        Real theta = 2.0 * M_PI * x / Lx;
        Re += Cbar[x] * std::cos(theta);
        Im += Cbar[x] * std::sin(theta);
    }
    Re /= mass;
    Im /= mass;
    Real theta_bar = std::atan2(Im, Re);   // in (-π, π]
    Real x_bar = theta_bar * Lx / (2.0 * M_PI);
    if (x_bar < 0.0) x_bar += Lx;          // wrap to [0, Lx)

    // Variance via minimum-image distance.
    Real var = 0.0;
    for (int x = 0; x < nx; ++x) {
        Real dx = static_cast<Real>(x) - x_bar;
        if (dx >  0.5 * Lx) dx -= Lx;
        if (dx < -0.5 * Lx) dx += Lx;
        var += Cbar[x] * dx * dx;
    }
    var /= mass;
    return {mass, x_bar, var};
}

} // anonymous namespace

TEST_CASE("Taylor dispersion: D_eff = D + Pe^2 D / 210 (circular-frame variance, < 5%)") {
    SimulationParams p;
    p.nx = 1024;             // long domain so the pulse stays narrow vs Lx
    p.ny = 24;               // H = ny - 2 = 22
    p.dt = 1.0;
    p.fluid.tau           = 0.6;              // ν = (τ-½)/3 = 0.0333
    p.fluid.rho0          = 1.0;
    p.fluid.boundary_type = BoundaryType::PERIODIC;
    p.fluid.periodic_y    = false;            // walls at y = 0, y = ny-1
    p.fluid.body_force_x  = 1.0e-5;           // u_mean ~ 1.2e-2, Pe ~ 13: enhancement ~0.8
    p.fluid.collision_model = CollisionModel::BGK;

    p.scalar.enabled              = true;
    p.scalar.n_species            = 1;
    p.scalar.diffusivity          = {0.02};
    p.scalar.inlet_concentration  = {0.0};
    p.scalar.periodic_y           = false;

    p.enable_stability_checks = false;
    p.enable_profiling        = false;
    p.metrics_interval        = 0;
    p.vtk_dump_every = p.csv_dump_every = p.probe_dump_every = p.stats_dump_every = 0;
    p.output_dir = "/tmp/softflow_vv_taylordisp";

    Simulation sim(p);
    sim.initialize();

    // Spin up the Poiseuille profile.
    const int spinup = 30000;
    for (int s = 0; s < spinup; ++s) sim.step();

    auto& field = sim.lbmSolver().field();
    Real u_mean = 0.0;
    int  count  = 0;
    for (int y = 1; y < p.ny - 1; ++y) {
        for (int x = 0; x < p.nx; ++x) {
            u_mean += field.getUx(x, y);
            ++count;
        }
    }
    u_mean /= count;
    REQUIRE(u_mean > 1e-6);

    const Real D    = p.scalar.diffusivity[0];
    const Real H    = static_cast<Real>(p.ny - 2);
    const Real Pe   = u_mean * H / D;
    const Real D_eff_expected = D + Pe * Pe * D / 210.0;

    INFO("u_mean = " << u_mean << ", Pe = " << Pe);
    INFO("D_eff (Aris-Taylor) = " << D_eff_expected);

    // Seed a Gaussian pulse centred at nx/4 across the channel.
    auto* adr = sim.advectionDiffusion();
    REQUIRE(adr != nullptr);
    Real* C = adr->concentrationData(0);
    const Real x0     = p.nx / 4.0;
    const Real sigma0 = 6.0;
    for (int y = 1; y < p.ny - 1; ++y) {
        for (int x = 0; x < p.nx; ++x) {
            Real dx = x - x0;
            C[y * p.nx + x] = std::exp(-dx * dx / (2.0 * sigma0 * sigma0));
        }
    }
    // Re-equilibrate the distributions.
    {
        Real* g = adr->gData(0);
        for (int n = 0; n < p.nx * p.ny; ++n) {
            for (int q = 0; q < 9; ++q) g[n * 9 + q] = D2Q9_W[q] * C[n];
        }
    }

    auto sectionMoments = [&]() -> CircMoments {
        std::vector<Real> Cbar(p.nx, 0.0);
        for (int x = 0; x < p.nx; ++x) {
            for (int y = 1; y < p.ny - 1; ++y) Cbar[x] += C[y * p.nx + x];
            Cbar[x] /= (p.ny - 2);
        }
        return circularMoments(Cbar, static_cast<Real>(p.nx));
    };

    // Reach the Aris-Taylor regime: variance growth is dominated by D_eff
    // only after t >> tau_mix = H^2 / (4D). Sample variance at two long
    // times and fit the slope.
    const int t_mix = static_cast<int>(H * H / (4.0 * D));
    const int t_start  = 6 * t_mix;
    const int t_window = 6 * t_mix;

    for (int s = 0; s < t_start; ++s) sim.step();
    CircMoments m1 = sectionMoments();
    for (int s = 0; s < t_window; ++s) sim.step();
    CircMoments m2 = sectionMoments();

    const Real D_eff_meas = (m2.var - m1.var) / (2.0 * t_window);
    const Real rel_err    = std::abs(D_eff_meas - D_eff_expected) / D_eff_expected;

    INFO("var(t_start)  = " << m1.var);
    INFO("var(t_finish) = " << m2.var);
    INFO("D_eff measured = " << D_eff_meas);
    INFO("relative error = " << rel_err);

    // Sanity: bulk-diffusion alone would give var growth = 2 D dt — we
    // should see the dispersion enhancement at this Pe.
    Real D_eff_naive = (m2.var - m1.var) / (2.0 * t_window);
    INFO("D_eff_naive (= measured slope) / D = " << D_eff_naive / D);

    if (FILE* out = vv::open("taylor_dispersion",
                             "Pe,D,D_eff_measured,D_eff_analytic,u_mean,H")) {
        std::fprintf(out, "%.6e,%.6e,%.6e,%.6e,%.6e,%.1f\n", Pe, D, D_eff_meas,
                     D_eff_expected, u_mean, H);
        vv::close(out);
    }
    CHECK(rel_err < 0.05);
}
