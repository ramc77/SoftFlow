// V&V: Poiseuille flow (the project specification, row 1)
//
// A 2D channel of height H = ny - 2 (excluding the two solid walls)
// driven by a constant body force F_x = G in the streamwise direction
// and periodic in x. The Navier–Stokes solution is a parabolic profile
//
//   u(y) = (G / (2 ν)) y (H − y),       0 ≤ y ≤ H,
//
// with peak speed
//
//   u_max = G H² / (8 ν).
//
// The lattice height counts the wall sites at y=0 and y=ny−1 as SOLID,
// so the fluid height H = ny − 2 and the y-coordinate of the centre
// fluid plane (in the channel-frame y' ≡ y − 1) is between 0 and H.
//
// Tolerance per the project specification: < 1% L2 error vs analytical.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "engine/simulation.h"
#include "lbm/lattice_field.h"

#include <cmath>

using namespace softflow;

TEST_CASE("Poiseuille: parabolic profile, L2 error < 1%") {
    // Channel: 16 wide x 34 tall (H = 32 fluid lattice cells).
    SimulationParams p;
    p.nx = 16;
    p.ny = 34;
    p.dt = 1.0;
    p.fluid.tau   = 0.8;          // ν = (τ−½)/3 = 0.1
    p.fluid.rho0  = 1.0;
    p.fluid.boundary_type = BoundaryType::PERIODIC;
    p.fluid.body_force_x  = 1.0e-6;     // G in lattice units
    p.fluid.collision_model = CollisionModel::BGK;
    p.enable_stability_checks = false;
    p.enable_profiling        = false;
    p.output_interval         = 0;
    p.vtk_dump_every          = 0;
    p.csv_dump_every          = 0;
    p.probe_dump_every        = 0;
    p.stats_dump_every        = 0;
    p.metrics_interval        = 0;
    p.output_dir = "/tmp/softflow_vv_poiseuille";

    Simulation sim(p);
    sim.initialize();

    // Poiseuille relaxation time: τ_relax ≈ H² / ν. With H = 32, ν = 0.1
    // that is ~10 000. Run 4 τ_relax to be safely converged.
    const int n_steps = 50000;
    for (int s = 0; s < n_steps; ++s) sim.step();

    // Sample the streamwise velocity profile at mid-channel (x = nx/2).
    const auto& field = sim.lbmSolver().field();
    const Real nu = p.kinematicViscosity();
    const Real G  = p.fluid.body_force_x;
    const int  H  = p.ny - 2;
    const Real u_max_analytic = G * H * H / (8.0 * nu);

    Real l2_num = 0.0, l2_den = 0.0;
    for (int y = 1; y <= H; ++y) {                 // y∈[1, H], fluid only
        Real ux = field.getUx(p.nx / 2, y);
        Real yp = static_cast<Real>(y - 1) + 0.5;  // wall-relative, midpoint of cell
        Real u_an = (G / (2.0 * nu)) * yp * (static_cast<Real>(H) - yp);
        l2_num += (ux - u_an) * (ux - u_an);
        l2_den += u_an * u_an;
    }
    Real l2_err = std::sqrt(l2_num / l2_den);

    INFO("u_max analytic = " << u_max_analytic);
    INFO("u_max measured = " << field.getUx(p.nx / 2, p.ny / 2));
    INFO("L2 error       = " << l2_err);
    CHECK(l2_err < 0.01);
}
