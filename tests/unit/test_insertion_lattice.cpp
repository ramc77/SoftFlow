// Tests for SquareLatticeInserter and HexagonalLatticeInserter.
//
// Per the Phase-2 design note (design-review plan + the project specification),
// every concrete inserter ships with: count, no-overlap (pairwise +
// vs walls), reproducibility under a fixed seed, and graceful
// saturation. The structured layouts also have an analytical count,
// which we assert here.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "core/insertion/inserter.h"
#include "core/insertion/lattice_inserter.h"
#include "core/insertion/region.h"
#include "core/insertion/size_distribution.h"

#include <cmath>
#include <memory>
#include <random>

using namespace softflow;
using namespace softflow::insertion;

namespace {

InsertionContext makeChannelContext(int nx = 200, int ny = 80,
                                    Real min_gap = 1.0,
                                    int periodic_nx = 0) {
    InsertionContext ctx;
    ctx.nx = nx; ctx.ny = ny;
    ctx.wall_y_bottom = 0.5;
    ctx.wall_y_top    = static_cast<Real>(ny) - 1.5;
    ctx.min_gap       = min_gap;
    ctx.periodic_nx   = periodic_nx;
    return ctx;
}

// Pairwise no-overlap check that respects the periodic_nx convention
// in the context. Returns the number of violating pairs.
int countOverlaps(const InsertionContext& ctx,
                  const std::vector<Placement>& ps,
                  Real min_gap)
{
    int violations = 0;
    for (std::size_t i = 0; i < ps.size(); ++i) {
        for (std::size_t j = i + 1; j < ps.size(); ++j) {
            const Real d = distance(ctx, ps[i].center, ps[j].center);
            if (d < ps[i].radius + ps[j].radius + min_gap) ++violations;
        }
    }
    return violations;
}

}  // namespace

TEST_CASE("SquareLatticeInserter: count, no-overlap, seed reproducibility") {
    auto ctx     = makeChannelContext();
    auto region  = std::make_shared<RectRegion>(0.0, 200.0, 10.0, 70.0);
    auto sizes   = std::make_shared<Monodisperse>(2.0);

    // Spacing 8 in x and 10 in y inside a 200×60 region. The first
    // lattice node sits at (lo + sx/2, lo + sy/2), so the analytic
    // count is floor(200/8) × floor(60/10) = 25 × 6 = 150.
    SquareLatticeInserter inserter(region, /*sx=*/8.0, /*sy=*/10.0,
                                   sizes, /*jitter=*/0.0);

    std::mt19937_64 rng(42);
    auto a = inserter.generate(ctx, rng);

    CHECK(a.size() == 150);
    CHECK(countOverlaps(ctx, a, ctx.min_gap) == 0);

    // Reproducibility under fixed seed.
    std::mt19937_64 rng2(42);
    auto b = inserter.generate(ctx, rng2);
    REQUIRE(a.size() == b.size());
    for (std::size_t i = 0; i < a.size(); ++i) {
        CHECK(a[i].center.x == doctest::Approx(b[i].center.x));
        CHECK(a[i].center.y == doctest::Approx(b[i].center.y));
        CHECK(a[i].radius   == doctest::Approx(b[i].radius));
    }
}

TEST_CASE("SquareLatticeInserter: jitter perturbs but stays in region") {
    auto ctx     = makeChannelContext();
    auto region  = std::make_shared<RectRegion>(0.0, 200.0, 10.0, 70.0);
    auto sizes   = std::make_shared<Monodisperse>(2.0);

    SquareLatticeInserter inserter(region, 8.0, 10.0, sizes, /*jitter=*/0.1);

    std::mt19937_64 rng(7);
    auto out = inserter.generate(ctx, rng);

    // With jitter 0.1 and spacing 8, every centre should be within
    // 0.1*8 = 0.8 of its nominal lattice position. We don't check
    // exact positions, only that they are still in the region.
    for (const auto& p : out) {
        CHECK(p.center.x >= 0.0); CHECK(p.center.x <= 200.0);
        CHECK(p.center.y >= 10.0); CHECK(p.center.y <= 70.0);
    }
    CHECK(countOverlaps(ctx, out, ctx.min_gap) == 0);
}

TEST_CASE("HexagonalLatticeInserter: count matches hex-pack analytic, no overlap") {
    auto ctx     = makeChannelContext();
    auto region  = std::make_shared<RectRegion>(0.0, 200.0, 10.0, 70.0);
    auto sizes   = std::make_shared<Monodisperse>(2.0);

    // Spacing 8: row spacing dy = sqrt(3)/2 · 8 ≈ 6.928. The first
    // row centre is at lo.y + 0.5 · dy = 13.464; subsequent rows step
    // by 6.928. The last row whose centre fits in [10, 70] is at
    // 13.464 + 8·6.928 ≈ 68.89, giving 9 rows total (indices 0..8).
    // Each row has floor(200 / 8) = 25 nodes (even rows start at x=4,
    // odd rows at x=8 — both fit 25 within [0, 200]). Analytic count:
    // 9 × 25 = 225. Allow ±2 for floating-point edge effects on the
    // last lattice node.
    HexagonalLatticeInserter inserter(region, /*spacing=*/8.0,
                                      sizes, /*jitter=*/0.0);

    std::mt19937_64 rng(42);
    auto out = inserter.generate(ctx, rng);

    CHECK(out.size() >= 223u);
    CHECK(out.size() <= 227u);
    CHECK(countOverlaps(ctx, out, ctx.min_gap) == 0);

    // Reproducibility.
    std::mt19937_64 rng2(42);
    auto out2 = inserter.generate(ctx, rng2);
    REQUIRE(out.size() == out2.size());
    for (std::size_t i = 0; i < out.size(); ++i) {
        CHECK(out[i].center.x == doctest::Approx(out2[i].center.x));
        CHECK(out[i].center.y == doctest::Approx(out2[i].center.y));
    }
}

TEST_CASE("Lattice inserters: respect existing capsules and obstacles") {
    auto ctx = makeChannelContext();

    // Pre-occupy the centre of the channel so the inserter must skip
    // any lattice site within 5+2+min_gap = 8 lattice units of (100, 40).
    ctx.existing_centers.push_back(Vec2d{100, 40});
    ctx.existing_radii.push_back(5.0);

    auto region  = std::make_shared<RectRegion>(0.0, 200.0, 10.0, 70.0);
    auto sizes   = std::make_shared<Monodisperse>(2.0);
    SquareLatticeInserter inserter(region, 8.0, 10.0, sizes, /*jitter=*/0.0);

    std::mt19937_64 rng(0);
    auto out = inserter.generate(ctx, rng);

    // The blocked zone is roughly π·8² ≈ 200 lattice units of forbidden
    // area. With a 8×10 grid, that's ~2-3 lattice nodes excluded. Loose
    // bound: at least 145 placed, at most 149 (out of 150 nominal).
    CHECK(out.size() >= 145u);
    CHECK(out.size() <  150u);
    // None of them collides with the existing capsule.
    for (const auto& p : out) {
        const Real d = distance(ctx, p.center, Vec2d{100, 40});
        CHECK(d >= p.radius + 5.0 + ctx.min_gap);
    }
}

TEST_CASE("Lattice inserters: graceful saturation in tight regions") {
    auto ctx     = makeChannelContext(/*nx=*/40, /*ny=*/40);
    auto region  = std::make_shared<RectRegion>(0.0, 40.0, 1.0, 39.0);
    auto sizes   = std::make_shared<Monodisperse>(20.0);   // disks too big

    HexagonalLatticeInserter inserter(region, 8.0, sizes, 0.0);

    std::mt19937_64 rng(0);
    auto out = inserter.generate(ctx, rng);

    // Every lattice node would overlap walls or itself; result must
    // be empty, not throw, not infinite-spin.
    CHECK(out.empty());
}
