// A capsule added to a periodic simulation must use minimum-image
// distances in its membrane forces whenever it is added: before or after
// initialize(), or during a run by a dynamic inserter.  Previously the
// periodic length was applied only to capsules present at initialize(),
// so a capsule added afterwards (as the verification tests do) lost its
// wrapping and was disturbed whenever it straddled the periodic seam.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "engine/simulation.h"

using namespace softflow;

namespace {
SimulationParams periodicParams() {
    SimulationParams p;
    p.nx = 60;
    p.ny = 40;
    p.fluid.tau = 0.8;
    p.fluid.boundary_type = BoundaryType::PERIODIC;
    p.enable_stability_checks = false;
    p.enable_profiling = false;
    p.output_interval = 0;
    p.vtk_dump_every = p.csv_dump_every = p.probe_dump_every = 0;
    p.stats_dump_every = p.metrics_interval = 0;
    p.output_dir = "/tmp/softflow_unit_periodic_capsule";
    return p;
}
} // namespace

TEST_CASE("capsules added before and after initialize() are periodic") {
    SimulationParams p = periodicParams();
    MembraneParams mp;
    Simulation sim(p);
    sim.capsules().addCapsule({20.0, 20.0}, 5.0, 24, mp, 0);   // before
    sim.initialize();
    sim.capsules().addCapsule({40.0, 20.0}, 5.0, 24, mp, 0);   // after
    for (int c = 0; c < sim.capsules().numCapsules(); ++c)
        CHECK(sim.capsules()[c].getPeriodicX() == p.nx);
}

TEST_CASE("a capsule straddling the seam is force-free at rest") {
    SimulationParams p = periodicParams();
    MembraneParams mp;
    mp.k_stretch = 0.5;
    mp.use_helfrich_bending = true;
    mp.gamma_visc = 0.0;
    mp.eta_membrane = 0.0;
    Simulation sim(p);
    sim.initialize();
    sim.capsules().addCapsule({1.0, 20.0}, 5.0, 24, mp, 0);    // straddles x = 0
    Capsule& cap = sim.capsules()[0];
    // Wrap the nodes individually, as the periodic update does.
    for (auto& n : cap.positions()) {
        if (n.x < 0.0) n.x += p.nx;
        if (n.x >= p.nx) n.x -= p.nx;
    }
    cap.clearForces();
    cap.computeMembraneForces();
    Real fmax = 0.0;
    for (const auto& f : cap.forces()) fmax = std::max(fmax, f.norm());
    CHECK(fmax < 1e-10);
}
