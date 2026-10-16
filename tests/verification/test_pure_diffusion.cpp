// V&V: pure diffusion of a Gaussian (the project specification, row 5)
//
// In a quiescent fluid (u = 0), an initial Gaussian concentration
// pulse C(x, 0) = C₀ exp(−x²/(2 σ₀²)) spreads as
//
//   C(x, t) = C₀ σ₀ / √(σ₀² + 2Dt)  ·  exp( −x² / (2 (σ₀² + 2Dt)) ).
//
// We check the recovered second-moment growth: σ²(t) = σ₀² + 2Dt
// over a 5000-step run on a periodic strip with no fluid forcing.
//
// Tolerance per the project specification: < 2% relative error.
//
// This test is also the principal regression for the Phase-1 ADR
// rewrite in src/lbm/advection_diffusion.cpp (race-free collide/stream
// + second-order equilibrium).

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "engine/simulation.h"
#include "lbm/advection_diffusion.h"

#include <cmath>
#include "vv_dump.h"

using namespace softflow;

TEST_CASE("Pure diffusion: Gaussian variance grows as σ² = σ₀² + 2Dt") {
    SimulationParams p;
    p.nx = 256;        // long enough that the Gaussian decays to zero before wrapping
    p.ny = 8;
    p.dt = 1.0;
    p.fluid.tau           = 0.6;          // ν small; immaterial — u=0 anyway
    p.fluid.boundary_type = BoundaryType::PERIODIC;
    p.fluid.periodic_y    = true;
    p.fluid.collision_model = CollisionModel::BGK;

    p.scalar.enabled              = true;
    p.scalar.n_species            = 1;
    p.scalar.diffusivity          = {0.05};       // D
    p.scalar.inlet_concentration  = {0.0};
    p.scalar.periodic_y           = true;

    p.enable_stability_checks = false;
    p.enable_profiling        = false;
    p.metrics_interval        = 0;
    p.vtk_dump_every = p.csv_dump_every = p.probe_dump_every = p.stats_dump_every = 0;
    p.output_dir = "/tmp/softflow_vv_purediff";

    Simulation sim(p);
    sim.initialize();

    // Seed a Gaussian centred at x = nx/2.
    auto* adr = sim.advectionDiffusion();
    REQUIRE(adr != nullptr);
    Real* C = adr->concentrationData(0);
    const int nx = p.nx, ny = p.ny;
    const Real x0     = nx / 2.0;
    const Real sigma0 = 8.0;
    const Real C0     = 1.0;
    for (int y = 0; y < ny; ++y) {
        for (int x = 0; x < nx; ++x) {
            Real dx = x - x0;
            C[y * nx + x] = C0 * std::exp(-dx * dx / (2.0 * sigma0 * sigma0));
        }
    }
    // Re-equilibrate the distribution functions to match the new C.
    {
        Real* g = adr->gData(0);
        for (int n = 0; n < nx * ny; ++n) {
            for (int q = 0; q < 9; ++q) {
                // D2Q9 weights inlined to avoid pulling in lattice.h here.
                static constexpr Real w[9] = {
                    4.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0,
                    1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0};
                g[n * 9 + q] = w[q] * C[n];
            }
        }
    }

    const int n_steps = 5000;
    for (int s = 0; s < n_steps; ++s) sim.step();

    // Compute mean and variance from the streamed concentration field
    // along y = 0 (any row is the same by symmetry — y is periodic).
    Real mass = 0.0, mean = 0.0, var = 0.0;
    for (int x = 0; x < nx; ++x) {
        Real c = C[0 * nx + x];
        mass += c;
        mean += c * x;
    }
    REQUIRE(mass > 1e-10);
    mean /= mass;
    for (int x = 0; x < nx; ++x) {
        Real c = C[0 * nx + x];
        Real dx = x - mean;
        var += c * dx * dx;
    }
    var /= mass;

    const Real D       = p.scalar.diffusivity[0];
    const Real sigma2_expected = sigma0 * sigma0 + 2.0 * D * n_steps;
    const Real rel_err = std::abs(var - sigma2_expected) / sigma2_expected;

    INFO("σ²(0)        = " << sigma0 * sigma0);
    INFO("σ²(t) measured  = " << var);
    INFO("σ²(t) expected  = " << sigma2_expected);
    INFO("relative error  = " << rel_err);
    if (FILE* out = vv::open("pure_diffusion", "steps,D,sigma2_0,sigma2_measured,sigma2_analytic,rel_err")) {
        std::fprintf(out, "%d,%.6e,%.6e,%.8e,%.8e,%.6e\n", n_steps, D, sigma0 * sigma0,
                     var, sigma2_expected, rel_err);
        vv::close(out);
    }
    CHECK(rel_err < 0.02);
}
