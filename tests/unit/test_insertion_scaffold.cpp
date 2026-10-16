// Step 1 round-trip test for the insertion scaffolding.
//
// Exercises the full IInserter / InsertionContext / RectRegion /
// Monodisperse / overlap-helper / Simulation::insertCapsules pipeline
// using a minimal inline inserter that places one capsule at the
// region centroid. The structured layouts (square/hex/RSA/Poisson-disk)
// land in subsequent steps; this test only asserts that the API plumbs
// through correctly.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "core/insertion/inserter.h"
#include "core/insertion/region.h"
#include "core/insertion/size_distribution.h"
#include "engine/simulation.h"

using namespace softflow;
using namespace softflow::insertion;

namespace {

// Minimal inserter: deposit one Placement at the region centroid,
// or none at all if the centroid is invalid (overlaps something).
class CentroidInserter final : public IInserter {
public:
    CentroidInserter(const RectRegion& region, Real radius)
        : region_(region), radius_(radius) {}

    std::vector<Placement> generate(const InsertionContext& ctx,
                                    std::mt19937_64& /*rng*/) override {
        const auto [lo, hi] = region_.bbox();
        const Vec2d c{ 0.5 * (lo.x + hi.x), 0.5 * (lo.y + hi.y) };
        if (!region_.contains(c)) return {};
        if (!isPlacementValid(ctx, c, radius_)) return {};
        return { Placement{ c, radius_ } };
    }

private:
    RectRegion region_;
    Real       radius_;
};

}  // namespace

TEST_CASE("RectRegion: basic geometry") {
    RectRegion r(0.0, 200.0, 10.0, 90.0);
    CHECK(r.contains(Vec2d{100, 50}));
    CHECK_FALSE(r.contains(Vec2d{-1, 50}));
    CHECK_FALSE(r.contains(Vec2d{100, 5}));
    CHECK(r.area() == doctest::Approx(200.0 * 80.0));

    const auto [lo, hi] = r.bbox();
    CHECK(lo.x == 0.0);   CHECK(lo.y == 10.0);
    CHECK(hi.x == 200.0); CHECK(hi.y == 90.0);
}

TEST_CASE("Monodisperse: always returns the same radius") {
    Monodisperse d(3.0);
    std::mt19937_64 rng(42);
    for (int i = 0; i < 32; ++i) CHECK(d.sample(rng) == doctest::Approx(3.0));
    CHECK(d.minRadius() == 3.0);
    CHECK(d.maxRadius() == 3.0);
}

TEST_CASE("isPlacementValid: walls, obstacles, existing capsules") {
    InsertionContext ctx;
    ctx.nx = 100; ctx.ny = 100;
    ctx.wall_y_bottom = 0.5; ctx.wall_y_top = 99.5;
    ctx.min_gap       = 1.0;
    ctx.periodic_nx   = 0;

    // Free space → ok.
    CHECK(isPlacementValid(ctx, Vec2d{50, 50}, 5.0));

    // Touches the bottom wall → rejected.
    CHECK_FALSE(isPlacementValid(ctx, Vec2d{50, 5.0}, 5.0));

    // Existing capsule blocks the slot.
    ctx.existing_centers = { Vec2d{50, 50} };
    ctx.existing_radii   = { 5.0 };
    CHECK_FALSE(isPlacementValid(ctx, Vec2d{52, 50}, 5.0));    // overlap
    CHECK(isPlacementValid(ctx, Vec2d{70, 50}, 5.0));          // far enough
}

TEST_CASE("Simulation::insertCapsules: end-to-end round trip") {
    // A periodic 200×80 channel, no obstacles. Confirm that
    // CentroidInserter plus Simulation::insertCapsules deposits
    // exactly one capsule at (100, 40) and that the capsule count
    // increments by 1.
    SimulationParams p;
    p.nx = 200; p.ny = 80;
    p.fluid.boundary_type = BoundaryType::PERIODIC;
    p.enable_stability_checks = false;
    p.enable_profiling        = false;
    p.rng_seed                = 0xCAFEBABEull;
    p.output_dir              = "/tmp/softflow_phase2_step1";
    p.vtk_format              = "ascii";

    Simulation sim(p);

    REQUIRE(sim.capsules().numCapsules() == 0);

    MembraneParams mb;
    mb.model = MembraneModel::HOOKEAN;

    RectRegion       region(0, 200, 10, 70);
    CentroidInserter inserter(region, 5.0);

    int placed = sim.insertCapsules(inserter, mb, /*type=*/0,
                                    /*min_gap=*/1.0,
                                    /*num_nodes=*/16,
                                    /*seed_tag=*/"unit_test");
    CHECK(placed == 1);
    CHECK(sim.capsules().numCapsules() == 1);

    Vec2d c = sim.capsules()[0].centroid();
    CHECK(c.x == doctest::Approx(100.0));
    CHECK(c.y == doctest::Approx(40.0));

    // Re-running the same inserter should respect the existing
    // capsule and refuse to place a second at the same point.
    int second = sim.insertCapsules(inserter, mb, 0, 1.0, 16, "unit_test");
    CHECK(second == 0);
    CHECK(sim.capsules().numCapsules() == 1);
}
