// Tests for ImageMaskRegion.
//
// Coverage:
//   - In-memory construction: contains/bbox/area sanity, y-flip, threshold.
//   - Round-trip through P2 (ASCII) and P5 (binary) PGM files.
//   - Inserter integration: every RSA placement in an image-mask
//     region passes the mask's contains() test.
//   - Rejection of malformed input.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "core/insertion/image_mask_region.h"
#include "core/insertion/inserter.h"
#include "core/insertion/rsa_inserter.h"
#include "core/insertion/size_distribution.h"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <memory>
#include <random>
#include <vector>

namespace fs = std::filesystem;
using namespace softflow;
using namespace softflow::insertion;

namespace {

// Build a 4×3 mask whose pattern is (rows top-to-bottom in the image,
// matching PGM row-major):
//
//   row 0:   0  0  0  0     ← image top
//   row 1:   0 255 255  0
//   row 2:   0  0  0  0     ← image bottom
//
// In lattice coords (with the y-flip), the inside cells are at
// (i,j_lat) = (1, 1) and (2, 1), which under origin=(0,0), scale=1
// cover x ∈ [1, 3], y ∈ [1, 2].
std::vector<std::uint8_t> threeByFourCenterStripe() {
    return {
          0,   0,   0,   0,
          0, 255, 255,   0,
          0,   0,   0,   0,
    };
}

}  // namespace

TEST_CASE("ImageMaskRegion: in-memory contains/bbox/area, y-flip") {
    ImageMaskRegion m(threeByFourCenterStripe(),
                      /*width=*/4, /*height=*/3,
                      /*origin=*/Vec2d{0.0, 0.0},
                      /*scale=*/1.0,
                      /*threshold=*/127);

    // bbox spans [origin, origin + (w·s, h·s)).
    const auto [lo, hi] = m.bbox();
    CHECK(lo.x == 0.0); CHECK(lo.y == 0.0);
    CHECK(hi.x == 4.0); CHECK(hi.y == 3.0);

    // Area: 2 inside pixels × scale² = 2.
    CHECK(m.area() == doctest::Approx(2.0));

    // The two "on" pixels are at image (i=1,j=1) and (i=2,j=1).
    // After y-flip, these are at j_lattice = (height-1-j) = 1.
    // → cells (i=1, j_lat=1) and (i=2, j_lat=1). With origin (0,0)
    //   and scale 1, those are x∈[1,2], x∈[2,3], y∈[1,2].
    CHECK(m.contains(Vec2d{1.5, 1.5}));
    CHECK(m.contains(Vec2d{2.5, 1.5}));
    CHECK_FALSE(m.contains(Vec2d{0.5, 1.5}));     // i=0 column → off
    CHECK_FALSE(m.contains(Vec2d{3.5, 1.5}));     // i=3 column → off
    CHECK_FALSE(m.contains(Vec2d{1.5, 0.5}));     // bottom row → off
    CHECK_FALSE(m.contains(Vec2d{1.5, 2.5}));     // top row    → off
    // Out of bbox.
    CHECK_FALSE(m.contains(Vec2d{-1.0, 1.5}));
    CHECK_FALSE(m.contains(Vec2d{1.5, 5.0}));
}

TEST_CASE("ImageMaskRegion::fromPGM: P2 ASCII round-trip") {
    fs::path p = fs::temp_directory_path() / "softflow_imask_p2.pgm";
    {
        std::ofstream f(p);
        // 4×3 ASCII PGM matching threeByFourCenterStripe().
        f << "P2\n# softflow test\n4 3\n255\n";
        f << "  0   0   0   0\n";
        f << "  0 255 255   0\n";
        f << "  0   0   0   0\n";
    }

    ImageMaskRegion m = ImageMaskRegion::fromPGM(p.string(),
        /*origin=*/Vec2d{10.0, 5.0}, /*scale=*/2.0);

    CHECK(m.width()  == 4);
    CHECK(m.height() == 3);

    // Two inside pixels × scale² = 2 × 4 = 8.
    CHECK(m.area() == doctest::Approx(8.0));

    // Inside pixel (i=1, j_lat=1) → x ∈ [12, 14], y ∈ [7, 9].
    CHECK(m.contains(Vec2d{13.0, 8.0}));
    CHECK_FALSE(m.contains(Vec2d{11.0, 6.0}));
    CHECK_FALSE(m.contains(Vec2d{50.0, 50.0}));

    fs::remove(p);
}

TEST_CASE("ImageMaskRegion::fromPGM: P5 binary round-trip") {
    fs::path p = fs::temp_directory_path() / "softflow_imask_p5.pgm";
    {
        std::ofstream f(p, std::ios::binary);
        f << "P5\n4 3\n255\n";
        const auto px = threeByFourCenterStripe();
        f.write(reinterpret_cast<const char*>(px.data()),
                static_cast<std::streamsize>(px.size()));
    }

    ImageMaskRegion m = ImageMaskRegion::fromPGM(p.string(),
        /*origin=*/Vec2d{0.0, 0.0}, /*scale=*/1.0);

    CHECK(m.area() == doctest::Approx(2.0));
    CHECK(m.contains(Vec2d{1.5, 1.5}));
    CHECK_FALSE(m.contains(Vec2d{0.5, 1.5}));

    fs::remove(p);
}

TEST_CASE("ImageMaskRegion::fromPGM: rejects malformed input") {
    fs::path p = fs::temp_directory_path() / "softflow_imask_bad.pgm";

    SUBCASE("non-PGM magic") {
        { std::ofstream f(p); f << "P3\n2 2\n255\n0 0 0 0\n"; }
        CHECK_THROWS_AS(ImageMaskRegion::fromPGM(p.string(),
                            Vec2d{0,0}, 1.0),
                        std::runtime_error);
    }
    SUBCASE("16-bit graymap unsupported") {
        { std::ofstream f(p); f << "P2\n2 2\n65535\n0 0 0 0\n"; }
        CHECK_THROWS_AS(ImageMaskRegion::fromPGM(p.string(),
                            Vec2d{0,0}, 1.0),
                        std::runtime_error);
    }
    SUBCASE("missing payload") {
        { std::ofstream f(p); f << "P2\n2 2\n255\n0 0\n"; }
        CHECK_THROWS_AS(ImageMaskRegion::fromPGM(p.string(),
                            Vec2d{0,0}, 1.0),
                        std::runtime_error);
    }
    SUBCASE("non-existent file") {
        CHECK_THROWS_AS(ImageMaskRegion::fromPGM("/tmp/no_such_file.pgm",
                            Vec2d{0,0}, 1.0),
                        std::runtime_error);
    }
    fs::remove(p);
}

TEST_CASE("ImageMaskRegion: every RSA placement satisfies the mask") {
    // 8x8 mask with a centred 4x4 "on" block.
    std::vector<std::uint8_t> px(64, 0);
    for (int j = 2; j < 6; ++j) for (int i = 2; i < 6; ++i) px[j * 8 + i] = 255;
    auto region = std::make_shared<ImageMaskRegion>(
        std::move(px), 8, 8, Vec2d{10.0, 10.0}, /*scale=*/4.0);

    InsertionContext ctx;
    ctx.nx = 100; ctx.ny = 100;
    ctx.wall_y_bottom = 0.5;
    ctx.wall_y_top    = 99.5;
    ctx.min_gap       = 0.5;
    ctx.periodic_nx   = 0;

    auto sizes = std::make_shared<Monodisperse>(2.0);
    RSAInserter inserter(region, /*target=*/30, sizes, /*max_attempts=*/4000);
    std::mt19937_64 rng(11);
    auto out = inserter.generate(ctx, rng);

    REQUIRE(out.size() > 8u);
    for (const auto& p : out) {
        CHECK(region->contains(p.center));
    }
}
