// Tests for ISizeDistribution implementations.
//
// Each distribution is tested for:
//   - reproducibility under fixed seed,
//   - bounds (every draw within [minRadius, maxRadius]),
//   - statistical agreement with the requested PDF (large-N average
//     for moments; for Bidisperse the fraction is asserted within
//     a 1/√N tolerance).

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "core/insertion/size_distribution.h"

#include <cmath>
#include <random>
#include <vector>

using namespace softflow;
using namespace softflow::insertion;

TEST_CASE("Bidisperse: requested fraction recovered within 1/sqrt(N)") {
    Bidisperse d(/*r_small=*/2.0, /*r_large=*/4.0, /*fraction_small=*/0.3);
    std::mt19937_64 rng(42);

    const int N = 50000;
    int n_small = 0;
    for (int i = 0; i < N; ++i) {
        Real r = d.sample(rng);
        REQUIRE((r == 2.0 || r == 4.0));
        if (r == 2.0) ++n_small;
    }
    const Real frac = static_cast<Real>(n_small) / N;
    // 4σ bound for binomial(N, 0.3): σ = sqrt(0.3·0.7/N) ≈ 0.00205
    CHECK(std::abs(frac - 0.3) < 0.01);

    CHECK(d.minRadius() == 2.0);
    CHECK(d.maxRadius() == 4.0);
}

TEST_CASE("Bidisperse: reproducibility under fixed seed") {
    Bidisperse d(2.0, 4.0, 0.5);
    std::mt19937_64 a(7), b(7);
    for (int i = 0; i < 1000; ++i) {
        CHECK(d.sample(a) == d.sample(b));
    }
}

TEST_CASE("Lognormal: hard truncation respected, reproducibility") {
    // Median exp(0) = 1. σ = 0.4 → most draws in roughly [0.4, 2.5].
    // Truncate to [0.5, 2.0] → some rejection is expected.
    Lognormal d(/*mu_log=*/0.0, /*sigma_log=*/0.4,
                /*r_min=*/0.5, /*r_max=*/2.0);

    std::mt19937_64 a(123), b(123);
    const int N = 5000;
    Real sum = 0.0;
    for (int i = 0; i < N; ++i) {
        Real ra = d.sample(a);
        Real rb = d.sample(b);
        CHECK(ra >= 0.5);
        CHECK(ra <= 2.0);
        CHECK(ra == rb);   // bit-exact reproducibility
        sum += ra;
    }
    const Real mean = sum / N;
    // Truncated median is still ≈ 1; mean should be close to it.
    CHECK(mean > 0.85);
    CHECK(mean < 1.20);

    CHECK(d.minRadius() == 0.5);
    CHECK(d.maxRadius() == 2.0);
}

TEST_CASE("UserDiscrete: requested weights recovered, reproducibility") {
    // Three discrete sizes with weights 1:2:3 (so probabilities
    // 1/6, 2/6, 3/6 = 0.1667, 0.3333, 0.5).
    std::vector<Real> radii   = {1.0, 2.0, 3.0};
    std::vector<Real> weights = {1.0, 2.0, 3.0};
    UserDiscrete d(radii, weights);

    std::mt19937_64 rng(31);
    const int N = 60000;
    int counts[3] = {0, 0, 0};
    for (int i = 0; i < N; ++i) {
        Real r = d.sample(rng);
        if (r == 1.0)      ++counts[0];
        else if (r == 2.0) ++counts[1];
        else if (r == 3.0) ++counts[2];
        else FAIL("unexpected radius");
    }
    const Real f0 = static_cast<Real>(counts[0]) / N;
    const Real f1 = static_cast<Real>(counts[1]) / N;
    const Real f2 = static_cast<Real>(counts[2]) / N;

    // 4σ bound for each is ~0.008 — comfortable.
    CHECK(std::abs(f0 - 1.0/6.0) < 0.01);
    CHECK(std::abs(f1 - 2.0/6.0) < 0.01);
    CHECK(std::abs(f2 - 3.0/6.0) < 0.01);

    CHECK(d.minRadius() == 1.0);
    CHECK(d.maxRadius() == 3.0);

    // Reproducibility.
    std::mt19937_64 ra(99), rb(99);
    for (int i = 0; i < 1000; ++i) CHECK(d.sample(ra) == d.sample(rb));
}

TEST_CASE("UserDiscrete: rejects degenerate inputs") {
    CHECK_THROWS_AS(UserDiscrete({}, {}),                  std::invalid_argument);
    CHECK_THROWS_AS(UserDiscrete({1.0, 2.0}, {1.0}),       std::invalid_argument);
    CHECK_THROWS_AS(UserDiscrete({1.0}, {-1.0}),           std::invalid_argument);
    CHECK_THROWS_AS(UserDiscrete({1.0}, {0.0}),            std::invalid_argument);
    CHECK_THROWS_AS(UserDiscrete({-1.0}, {1.0}),           std::invalid_argument);
}

TEST_CASE("Bidisperse: rejects degenerate inputs") {
    CHECK_THROWS_AS(Bidisperse(-1.0, 4.0, 0.5),    std::invalid_argument);
    CHECK_THROWS_AS(Bidisperse(2.0, -1.0, 0.5),    std::invalid_argument);
    CHECK_THROWS_AS(Bidisperse(2.0, 4.0, -0.1),    std::invalid_argument);
    CHECK_THROWS_AS(Bidisperse(2.0, 4.0,  1.1),    std::invalid_argument);
}
