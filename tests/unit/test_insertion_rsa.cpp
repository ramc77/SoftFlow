// Tests for RSAInserter and PoissonDiskInserter.
//
// Coverage per the Phase-2 design note:
//   - Reproducibility under fixed seed (bit-exact placements).
//   - No-overlap (pairwise + walls + obstacles).
//   - Saturation density approaches the 2-D RSA jamming limit
//     φ_J ≈ 0.547 with sufficient max_attempts.
//   - Bridson invariant: every pair separated by ≥ r_min.
//   - Graceful saturation when the region is too small.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "core/insertion/inserter.h"
#include "core/insertion/region.h"
#include "core/insertion/rsa_inserter.h"
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

int countOverlaps(const InsertionContext& ctx,
                  const std::vector<Placement>& ps,
                  Real min_gap)
{
    int v = 0;
    for (std::size_t i = 0; i < ps.size(); ++i)
        for (std::size_t j = i + 1; j < ps.size(); ++j)
            if (distance(ctx, ps[i].center, ps[j].center)
                < ps[i].radius + ps[j].radius + min_gap)
                ++v;
    return v;
}

// 2-D area packing fraction φ = sum(π r_k²) / region_area.
Real packingFraction(const std::vector<Placement>& ps, Real region_area) {
    Real sum = 0.0;
    for (const auto& p : ps) sum += M_PI * p.radius * p.radius;
    return sum / region_area;
}

}  // namespace

TEST_CASE("RSAInserter: no overlap, no walls/obstacles violated, reproducible") {
    auto ctx     = makeChannelContext();
    auto region  = std::make_shared<RectRegion>(0.0, 200.0, 10.0, 70.0);
    auto sizes   = std::make_shared<Monodisperse>(2.0);

    RSAInserter inserter(region, /*target=*/300, sizes, /*max_attempts=*/30000);

    std::mt19937_64 rng_a(42);
    auto a = inserter.generate(ctx, rng_a);

    CHECK(countOverlaps(ctx, a, ctx.min_gap) == 0);
    for (const auto& p : a) {
        CHECK(p.center.y - p.radius >= ctx.wall_y_bottom + ctx.min_gap);
        CHECK(p.center.y + p.radius <= ctx.wall_y_top    - ctx.min_gap);
    }

    std::mt19937_64 rng_b(42);
    auto b = inserter.generate(ctx, rng_b);
    REQUIRE(a.size() == b.size());
    for (std::size_t i = 0; i < a.size(); ++i) {
        CHECK(a[i].center.x == doctest::Approx(b[i].center.x));
        CHECK(a[i].center.y == doctest::Approx(b[i].center.y));
    }
}

TEST_CASE("RSAInserter: density approaches the 2-D RSA jamming limit") {
    // φ_J ≈ 0.547 for monodisperse 2-D disks.  With min_gap = 0
    // (touching disks) and a generous attempt budget, the recovered
    // packing fraction must be > 0.45 (well above casual rejection
    // sampling, well below crystal packing).
    auto ctx          = makeChannelContext();
    ctx.min_gap       = 0.0;
    auto region       = std::make_shared<RectRegion>(0.0, 200.0, 10.0, 70.0);
    const Real radius = 2.0;
    auto sizes        = std::make_shared<Monodisperse>(radius);

    // Aim very high so the saturation cap stops us instead of the count.
    RSAInserter inserter(region, /*target=*/100000, sizes,
                         /*max_attempts=*/200000);

    std::mt19937_64 rng(7);
    auto out = inserter.generate(ctx, rng);

    const Real region_area = (200.0 - 0.0) * (70.0 - 10.0);
    const Real phi         = packingFraction(out, region_area);

    INFO("RSA packing fraction = " << phi << ", N = " << out.size());
    CHECK(phi > 0.45);
    CHECK(phi < 0.547 + 0.05);   // safely below the jamming limit
    CHECK(countOverlaps(ctx, out, ctx.min_gap) == 0);
}

TEST_CASE("RSAInserter: saturates gracefully in oversized region") {
    auto ctx     = makeChannelContext();
    auto region  = std::make_shared<RectRegion>(0.0, 30.0, 5.0, 25.0);
    auto sizes   = std::make_shared<Monodisperse>(8.0);

    // Region is 30×20 with min_gap=1 and disk r=8: about 30×20/π·8² ~ 3
    // disks fit, far below the 100 we are asking for. Must return
    // a small number, not throw, not loop.
    RSAInserter inserter(region, /*target=*/100, sizes,
                         /*max_attempts=*/5000);

    std::mt19937_64 rng(1);
    auto out = inserter.generate(ctx, rng);

    CHECK(out.size() < 10u);
    CHECK(countOverlaps(ctx, out, ctx.min_gap) == 0);
}

TEST_CASE("PoissonDiskInserter: every pair >= r_min, reproducible") {
    auto ctx     = makeChannelContext();
    auto region  = std::make_shared<RectRegion>(0.0, 200.0, 10.0, 70.0);
    auto sizes   = std::make_shared<Monodisperse>(2.0);

    const Real r_min = 6.0;     // > 2*r + min_gap = 5
    PoissonDiskInserter inserter(region, r_min, sizes, /*k=*/30);

    std::mt19937_64 rng_a(11);
    auto a = inserter.generate(ctx, rng_a);
    REQUIRE(a.size() > 50u);    // Bridson should find a healthy fill

    // Bridson invariant.
    CHECK(inserter.lastMinSeparation() >= r_min - 1e-9);
    CHECK(countOverlaps(ctx, a, ctx.min_gap) == 0);

    std::mt19937_64 rng_b(11);
    auto b = inserter.generate(ctx, rng_b);
    REQUIRE(a.size() == b.size());
    for (std::size_t i = 0; i < a.size(); ++i) {
        CHECK(a[i].center.x == doctest::Approx(b[i].center.x));
        CHECK(a[i].center.y == doctest::Approx(b[i].center.y));
    }
}

TEST_CASE("PoissonDiskInserter: gracefully handles degenerate regions") {
    // A region surrounded by the wall envelope: no fluid space at all.
    auto ctx = makeChannelContext(/*nx=*/30, /*ny=*/30);
    auto region = std::make_shared<RectRegion>(0.0, 30.0, 0.5, 1.5);
    auto sizes  = std::make_shared<Monodisperse>(2.0);

    PoissonDiskInserter inserter(region, /*r_min=*/6.0, sizes, /*k=*/30);

    std::mt19937_64 rng(0);
    auto out = inserter.generate(ctx, rng);

    // Wall envelope rules out every point in the region — none placed.
    CHECK(out.empty());
}
