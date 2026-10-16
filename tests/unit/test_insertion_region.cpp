// Tests for IRegion implementations: RectRegion (covered in
// scaffolding), CircleRegion, PolygonRegion. Each region must
// correctly answer contains/bbox/area, and inserters that consume
// the region must respect its membership when sampling.

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
                                    Real min_gap = 1.0)
{
    InsertionContext ctx;
    ctx.nx = nx; ctx.ny = ny;
    ctx.wall_y_bottom = 0.5;
    ctx.wall_y_top    = static_cast<Real>(ny) - 1.5;
    ctx.min_gap       = min_gap;
    return ctx;
}

}  // namespace

TEST_CASE("CircleRegion: containment and area") {
    CircleRegion d(Vec2d{50, 40}, 10.0);

    CHECK(d.contains(Vec2d{50, 40}));
    CHECK(d.contains(Vec2d{60, 40}));         // boundary
    CHECK(d.contains(Vec2d{55, 45}));
    CHECK_FALSE(d.contains(Vec2d{61, 40}));
    CHECK_FALSE(d.contains(Vec2d{50, 51}));
    CHECK(d.area() == doctest::Approx(M_PI * 100.0));

    const auto [lo, hi] = d.bbox();
    CHECK(lo.x == doctest::Approx(40.0));
    CHECK(lo.y == doctest::Approx(30.0));
    CHECK(hi.x == doctest::Approx(60.0));
    CHECK(hi.y == doctest::Approx(50.0));

    CHECK_THROWS_AS(CircleRegion(Vec2d{0,0}, -1.0), std::invalid_argument);
}

TEST_CASE("CircleRegion: every RSA placement actually lies in the circle") {
    auto ctx     = makeChannelContext();
    auto region  = std::make_shared<CircleRegion>(Vec2d{100, 40}, 25.0);
    auto sizes   = std::make_shared<Monodisperse>(2.0);

    RSAInserter inserter(region, /*target=*/40, sizes, /*max_attempts=*/4000);
    std::mt19937_64 rng(0);
    auto out = inserter.generate(ctx, rng);

    REQUIRE(out.size() > 20u);
    for (const auto& p : out) {
        const Real dx = p.center.x - 100.0;
        const Real dy = p.center.y - 40.0;
        CHECK(std::sqrt(dx * dx + dy * dy) <= 25.0 + 1e-9);
    }
}

TEST_CASE("PolygonRegion: triangle membership and shoelace area") {
    // Equilateral triangle with vertices roughly at (0,0), (10,0), (5, 8.66).
    PolygonRegion tri({
        Vec2d{0.0, 0.0},
        Vec2d{10.0, 0.0},
        Vec2d{5.0,  10.0 * std::sqrt(3.0) / 2.0},
    });

    CHECK(tri.contains(Vec2d{5.0, 1.0}));    // well inside
    CHECK(tri.contains(Vec2d{5.0, 4.0}));
    CHECK_FALSE(tri.contains(Vec2d{-1.0, 1.0}));
    CHECK_FALSE(tri.contains(Vec2d{5.0, 9.0}));   // above apex
    CHECK_FALSE(tri.contains(Vec2d{0.0, 5.0}));   // outside left edge

    const Real expected_area = 0.5 * 10.0 * (10.0 * std::sqrt(3.0) / 2.0);
    CHECK(tri.area() == doctest::Approx(expected_area));

    const auto [lo, hi] = tri.bbox();
    CHECK(lo.x == doctest::Approx(0.0));   CHECK(lo.y == doctest::Approx(0.0));
    CHECK(hi.x == doctest::Approx(10.0));  CHECK(hi.y == doctest::Approx(8.66025).epsilon(1e-3));
}

TEST_CASE("PolygonRegion: orientation-independent area") {
    // Same square traversed CW and CCW must report identical area.
    std::vector<Vec2d> ccw = {{0,0}, {10,0}, {10,10}, {0,10}};
    std::vector<Vec2d> cw  = {{0,0}, {0,10}, {10,10}, {10,0}};
    PolygonRegion p_ccw(ccw), p_cw(cw);
    CHECK(p_ccw.area() == doctest::Approx(100.0));
    CHECK(p_cw.area()  == doctest::Approx(100.0));
}

TEST_CASE("PolygonRegion: rejects degenerate inputs") {
    CHECK_THROWS_AS(PolygonRegion(std::vector<Vec2d>{{0,0}, {1,0}}),
                    std::invalid_argument);
    CHECK_THROWS_AS(PolygonRegion(std::vector<Vec2d>{{0,0}, {1,0}, {2,0}}),
                    std::invalid_argument);  // collinear → zero area
}

TEST_CASE("PolygonRegion: every RSA placement lies in the polygon") {
    auto ctx = makeChannelContext();

    // L-shape concave polygon to stress the crossing-number test.
    auto region = std::make_shared<PolygonRegion>(std::vector<Vec2d>{
        Vec2d{20, 10}, Vec2d{80, 10}, Vec2d{80, 30},
        Vec2d{40, 30}, Vec2d{40, 70}, Vec2d{20, 70},
    });
    auto sizes  = std::make_shared<Monodisperse>(1.5);
    RSAInserter inserter(region, /*target=*/100, sizes, /*max_attempts=*/20000);

    std::mt19937_64 rng(7);
    auto out = inserter.generate(ctx, rng);

    REQUIRE(out.size() > 30u);
    for (const auto& p : out) {
        CHECK(region->contains(p.center));
    }
}
