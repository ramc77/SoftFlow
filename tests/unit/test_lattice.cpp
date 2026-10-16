// Sanity test for the D2Q9 lattice constants.
// Verifies the algebraic identities every D2Q9 implementation must satisfy.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "lbm/lattice.h"

#include <cmath>

using namespace softflow;

TEST_CASE("D2Q9 weights sum to one") {
    Real s = 0.0;
    for (int q = 0; q < D2Q9::Q; ++q) s += D2Q9::w[q];
    CHECK(std::abs(s - 1.0) < 1e-15);
}

TEST_CASE("D2Q9 first moment of weights vanishes") {
    Real sx = 0.0, sy = 0.0;
    for (int q = 0; q < D2Q9::Q; ++q) {
        sx += D2Q9::w[q] * D2Q9::cx[q];
        sy += D2Q9::w[q] * D2Q9::cy[q];
    }
    CHECK(std::abs(sx) < 1e-15);
    CHECK(std::abs(sy) < 1e-15);
}

TEST_CASE("D2Q9 second moment of weights equals cs^2 I") {
    Real sxx = 0.0, syy = 0.0, sxy = 0.0;
    for (int q = 0; q < D2Q9::Q; ++q) {
        sxx += D2Q9::w[q] * D2Q9::cx[q] * D2Q9::cx[q];
        syy += D2Q9::w[q] * D2Q9::cy[q] * D2Q9::cy[q];
        sxy += D2Q9::w[q] * D2Q9::cx[q] * D2Q9::cy[q];
    }
    CHECK(std::abs(sxx - D2Q9::cs2) < 1e-15);
    CHECK(std::abs(syy - D2Q9::cs2) < 1e-15);
    CHECK(std::abs(sxy)             < 1e-15);
}

TEST_CASE("D2Q9 opposite directions are involutive") {
    for (int q = 0; q < D2Q9::Q; ++q) {
        CHECK(D2Q9::opp[D2Q9::opp[q]] == q);
        CHECK(D2Q9::cx[D2Q9::opp[q]] == -D2Q9::cx[q]);
        CHECK(D2Q9::cy[D2Q9::opp[q]] == -D2Q9::cy[q]);
    }
}

TEST_CASE("D2Q9 equilibrium recovers density and momentum at zero velocity") {
    const Real rho = 1.234;
    Real m0 = 0.0, mx = 0.0, my = 0.0;
    for (int q = 0; q < D2Q9::Q; ++q) {
        Real f = D2Q9::feq(q, rho, 0.0, 0.0);
        m0 += f;
        mx += f * D2Q9::cx[q];
        my += f * D2Q9::cy[q];
    }
    CHECK(std::abs(m0 - rho)  < 1e-13);
    CHECK(std::abs(mx)        < 1e-13);
    CHECK(std::abs(my)        < 1e-13);
}

TEST_CASE("D2Q9 equilibrium recovers density and momentum at small finite u") {
    const Real rho = 1.0;
    const Real ux = 0.05, uy = -0.03;
    Real m0 = 0.0, mx = 0.0, my = 0.0;
    for (int q = 0; q < D2Q9::Q; ++q) {
        Real f = D2Q9::feq(q, rho, ux, uy);
        m0 += f;
        mx += f * D2Q9::cx[q];
        my += f * D2Q9::cy[q];
    }
    CHECK(std::abs(m0 - rho)        < 1e-13);
    CHECK(std::abs(mx - rho * ux)   < 1e-13);
    CHECK(std::abs(my - rho * uy)   < 1e-13);
}
