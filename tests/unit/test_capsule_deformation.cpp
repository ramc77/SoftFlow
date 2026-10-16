// Taylor deformation parameter D = (L - B)/(L + B) of the equivalent
// ellipse, and the inclination of its major axis.  D must not depend on
// the orientation of the capsule (a bounding-box measure reads 0 for an
// ellipse at 45 degrees) and must survive wrapping across a periodic x
// boundary.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "membrane/capsule.h"

#include <cmath>

using namespace softflow;

namespace {

Capsule makeEllipse(Real a, Real b, Real angle, Vec2d c, int n = 256) {
    MembraneParams mp;
    mp.shape = CapsuleShape::ELLIPSE;
    mp.aspect_ratio = b / a;
    Capsule cap(0, c, a, n, mp);
    // Rotate the rest shape about its centre.
    const Real cs = std::cos(angle), sn = std::sin(angle);
    for (auto& p : cap.positions()) {
        Real dx = p.x - c.x, dy = p.y - c.y;
        p = Vec2d{c.x + cs * dx - sn * dy, c.y + sn * dx + cs * dy};
    }
    return cap;
}

} // namespace

TEST_CASE("circle has D = 0") {
    Capsule cap = makeEllipse(6.0, 6.0, 0.0, {20.0, 20.0});
    CHECK(cap.deformationIndex() < 1e-6);
}

TEST_CASE("ellipse D = (a-b)/(a+b) at any orientation") {
    const Real a = 8.0, b = 4.0, D_exact = (a - b) / (a + b);
    for (Real deg : {0.0, 20.0, 45.0, 70.0, 90.0, 135.0}) {
        const Real ang = deg * PI / 180.0;
        Capsule cap = makeEllipse(a, b, ang, {30.0, 25.0});
        INFO("angle " << deg);
        CHECK(std::abs(cap.deformationIndex() - D_exact) < 2e-3);
        // Inclination in (-90, 90] degrees, modulo 180.
        Real inc = cap.inclinationAngle() * 180.0 / PI;
        Real diff = std::fmod(inc - deg + 270.0, 180.0) - 90.0;
        CHECK(std::abs(diff) < 0.2);
    }
}

TEST_CASE("D is unchanged across a periodic x boundary") {
    const Real a = 8.0, b = 5.0;
    Capsule inside = makeEllipse(a, b, 0.6, {50.0, 25.0});
    Capsule wrapped = makeEllipse(a, b, 0.6, {99.0, 25.0});
    wrapped.setPeriodicX(100);
    for (auto& p : wrapped.positions())
        if (p.x >= 100.0) p.x -= 100.0;
    CHECK(std::abs(wrapped.deformationIndex() - inside.deformationIndex()) < 1e-9);
}
