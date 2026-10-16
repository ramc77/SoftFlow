// V&V: Taylor–Green vortex decay (the project specification, row 2)
//
// A doubly-periodic 2D box with the initial condition
//
//   u_x(x,y,0) = -U₀ cos(k_x x) sin(k_y y),
//   u_y(x,y,0) =  U₀ sin(k_x x) cos(k_y y),
//   ρ(x,y,0)   = ρ₀ - (ρ₀ U₀² / (4 cs²)) [cos(2 k_x x) + cos(2 k_y y)],
//
// decays exactly under the Navier–Stokes equation as
//
//   u(t) = u(0) · exp(-ν (k_x² + k_y²) t),
//
// regardless of the spatial pattern. We verify the L2 norm of u after
// τ_decay = 1 / (ν (k_x² + k_y²)) matches exp(-1) within 2%.
//
// Tolerance per the project specification: < 2% on the recovered decay rate.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "engine/simulation.h"
#include "lbm/lattice_field.h"
#include "lbm/lattice.h"

#include <cmath>
#include "vv_dump.h"

using namespace softflow;

namespace {

// Initialize the lattice to the Taylor–Green analytic field at t=0.
// We seed the equilibrium distribution from (ρ, u) directly via
// LatticeField::setEquilibrium-style writes per cell.
void seedTaylorGreen(LatticeField& f, int nx, int ny,
                     Real U0, Real kx, Real ky, Real rho0) {
    const Real cs2 = D2Q9::cs2;
    for (int y = 0; y < ny; ++y) {
        for (int x = 0; x < nx; ++x) {
            Real ux = -U0 * std::cos(kx * x) * std::sin(ky * y);
            Real uy =  U0 * std::sin(kx * x) * std::cos(ky * y);
            Real rho = rho0
                - (rho0 * U0 * U0 / (4.0 * cs2))
                  * (std::cos(2.0 * kx * x) + std::cos(2.0 * ky * y));
            f.setRho(x, y, rho);
            f.setUx(x, y, ux);
            f.setUy(x, y, uy);
            f.initializeEquilibriumAt(x, y);
        }
    }
}

}  // namespace

TEST_CASE("Taylor–Green: exponential decay rate, < 2% error") {
    SimulationParams p;
    p.nx = 64;
    p.ny = 64;
    p.dt = 1.0;
    p.fluid.tau           = 0.8;          // ν = 0.1
    p.fluid.rho0          = 1.0;
    p.fluid.boundary_type = BoundaryType::PERIODIC;
    p.fluid.periodic_y    = true;
    p.fluid.collision_model = CollisionModel::BGK;
    p.enable_stability_checks = false;
    p.enable_profiling        = false;
    p.vtk_dump_every  = 0;
    p.csv_dump_every  = 0;
    p.probe_dump_every = 0;
    p.stats_dump_every = 0;
    p.metrics_interval = 0;
    p.output_dir = "/tmp/softflow_vv_taylorgreen";

    Simulation sim(p);
    sim.initialize();

    const Real U0  = 0.04;
    const Real kx  = 2.0 * M_PI / p.nx;
    const Real ky  = 2.0 * M_PI / p.ny;
    const Real nu  = p.kinematicViscosity();
    const Real k2  = kx * kx + ky * ky;
    const Real tau_decay = 1.0 / (nu * k2);

    seedTaylorGreen(sim.lbmSolver().field(), p.nx, p.ny, U0, kx, ky, p.fluid.rho0);

    auto kineticEnergy = [&](const LatticeField& f) {
        Real e = 0.0;
        for (int y = 0; y < p.ny; ++y) {
            for (int x = 0; x < p.nx; ++x) {
                Real ux = f.getUx(x, y), uy = f.getUy(x, y);
                e += ux * ux + uy * uy;
            }
        }
        return 0.5 * e;
    };
    const Real E0 = kineticEnergy(sim.lbmSolver().field());

    const int n_steps = static_cast<int>(tau_decay);   // run exactly one e-fold
    for (int s = 0; s < n_steps; ++s) sim.step();

    const Real Et = kineticEnergy(sim.lbmSolver().field());

    // E ~ |u|² so E(t)/E(0) = exp(-2 ν k² t). At t = τ_decay, ratio = exp(-2).
    const Real expected_ratio = std::exp(-2.0);
    const Real measured_ratio = Et / E0;
    const Real rel_err = std::abs(measured_ratio - expected_ratio) / expected_ratio;

    INFO("E0 = " << E0 << ", Et = " << Et);
    INFO("measured E(t)/E(0) = " << measured_ratio
         << ", expected exp(-2) = " << expected_ratio);
    INFO("relative error = " << rel_err);
    if (FILE* out = vv::open("taylor_green", "steps,E_ratio_measured,E_ratio_analytic,rel_err")) {
        std::fprintf(out, "%d,%.8e,%.8e,%.6e\n", n_steps, measured_ratio, expected_ratio, rel_err);
        vv::close(out);
    }
    CHECK(rel_err < 0.02);
}
