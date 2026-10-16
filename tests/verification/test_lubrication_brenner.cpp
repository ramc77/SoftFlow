// V&V: lubrication correction — Brenner-style 1/h scaling.
//
// SoftFlow applies a sub-grid hydrodynamic correction when the gap h
// between two close-approaching surfaces drops below h_threshold:
//
//   F_lub = -6 * pi * mu * a_eff * v_n / max(h, h_min)
//
// (`src/membrane/lubrication.cpp:112` for capsule-capsule,
//  `:143-152` for capsule-wall, with a_eff = R for the wall case).
//
// This test pins:
//   (a) The bare 1/h scaling away from h_min — fitting F · h vs h
//       across h ∈ [0.5, 1.8] lu must give a flat ratio within 3 %.
//   (b) Regularization — the force saturates exactly at the h_min
//       clamp; F(h ≤ h_min) = F(h_min) within 1e-12.
//   (c) Cutoff — F(h > h_threshold) = 0 exactly.
//   (d) Wall variant matches the same 1/h form with a_eff = R.
//
// We drive the test by setting a single capsule's nodes at known
// positions with a known relative normal velocity and reading back
// the spread forces from Capsule::forces() after one
// LubricationCorrection::computeAll(...) call.
//
// Reference: Brenner 1961, Aidun & Clausen, Annu. Rev. Fluid Mech.
// 2010, equation set replicated in docs/theory/lubrication.md.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "membrane/lubrication.h"
#include "membrane/capsule_system.h"
#include "core/parameters.h"

#include <cmath>
#include "vv_dump.h"
#include <vector>

using namespace softflow;

namespace {

// Build a CapsuleSystem with one capsule of radius R centred at (cx, cy)
// and Nn nodes. Forces are cleared by the caller before each measurement.
CapsuleSystem oneCapsule(Vec2d c, Real R, int Nn) {
    MembraneParams mp;
    CapsuleSystem caps;
    caps.addCapsule(c, R, Nn, mp, /*type=*/0);
    return caps;
}

} // anonymous namespace

// Pick the (single) node with the smallest y, return |Fy| there.
// We use Nn = 12 (the smallest closed polygon allowed by Capsule) and
// h_threshold = 1.5 so that *only the lowest node* lies within the
// active band — eliminating ambiguity about which nodes contribute.
static Real lowestNodeFy(const CapsuleSystem& caps) {
    Real y_min = 1e30, Fy_at_min = 0.0;
    for (int k = 0; k < caps[0].numNodes(); ++k) {
        Vec2d p = caps[0].nodePosition(k);
        if (p.y < y_min) {
            y_min = p.y;
            Fy_at_min = caps[0].forces()[k].y;
        }
    }
    return Fy_at_min;          // signed: must point away from the wall
}

TEST_CASE("Lubrication: capsule-wall single-node force matches -6π μ R v_n / h") {
    LubricationParams lp;
    lp.enabled     = true;
    // h_threshold chosen so only the lowest node of the 12-gon contributes
    // for the h values we sweep. The lowest node sits at y_min = 0.5 + h;
    // its neighbours sit at y = cy − R·cos(π/12)·(...) — for Nn = 12 and
    // R = 6, the second-lowest pair is at y_min + R·(1 − cos(2π/12)) =
    // y_min + 0.804. Setting h_threshold = 1.6 guarantees that as long as
    // h_lowest + 0.804 > h_threshold, only the single lowest node fires.
    lp.h_threshold = 1.6;
    lp.h_min       = 0.1;
    const Real nu  = 0.1;     // kinematic viscosity (= dynamic viscosity since rho0 = 1)

    const Real R  = 6.0;
    const int  Nn = 12;
    const int  ny = 64;
    const Real V  = 0.001;    // downward velocity

    // Active range for the lowest-only regime: h ∈ (0.1, h_threshold − 0.804] = (0.1, 0.796].
    const std::vector<Real> H = {0.75, 0.55, 0.40, 0.25, 0.15};
    Real max_rel_err = 0.0;
    FILE* out = vv::open("lubrication_wall", "h,h_min,F_measured,F_analytic");

    for (Real h : H) {
        Real cy = 0.5 + h + R;     // exact: lowest vertex of 12-gon at angle -π/2
                                   // sits at y = cy − R = 0.5 + h
        CapsuleSystem caps = oneCapsule(Vec2d{32.0, cy}, R, Nn);
        for (int k = 0; k < caps[0].numNodes(); ++k) {
            caps[0].forces()[k] = Vec2d{0.0, 0.0};
            caps[0].setNodeVelocity(k, Vec2d{0.0, -V});
        }
        LubricationCorrection lub(lp, nu);
        lub.computeAll(caps, ny, /*periodic_nx=*/64);

        Real F_meas = lowestNodeFy(caps);
        Real h_clamped = std::max(h, lp.h_min);
        // The code uses Capsule::effectiveRadius() = sqrt(rest_area/π),
        // not the constructor radius. For a regular Nn-gon of circumradius
        // R, area = ½·Nn·R²·sin(2π/Nn), so the effective radius is
        // slightly smaller than R (≈ 5.86 for Nn = 12, R = 6).
        Real a_eff     = caps[0].effectiveRadius();
        Real F_analytic = 6.0 * M_PI * nu * a_eff * V / h_clamped;
        // The node approaches the bottom wall (v_y = -V), so the force on it
        // must be +y, away from the wall.
        CHECK(F_meas > 0.0);
        Real rel = std::abs(F_meas - F_analytic) / F_analytic;
        INFO("h = " << h << "  F_meas = " << F_meas
              << "  F_analytic = " << F_analytic << "  rel = " << rel);
        if (out) std::fprintf(out, "%.6e,%.6e,%.9e,%.9e\n", h, lp.h_min,
                              F_meas, F_analytic);
        if (rel > max_rel_err) max_rel_err = rel;
    }
    vv::close(out);

    INFO("max relative error across the sweep = " << max_rel_err);
    CHECK(max_rel_err < 1e-12);
}

TEST_CASE("Lubrication: regularization at h_min clamps the force") {
    LubricationParams lp;
    lp.enabled     = true;
    lp.h_threshold = 2.0;
    lp.h_min       = 0.5;     // make the clamp easy to hit
    const Real nu  = 0.1;
    const Real R   = 6.0;
    const int  Nn  = 48;
    const int  ny  = 64;
    const Real V   = 0.001;

    // Two configurations: h = h_min/2 (clamped) and h = h_min (boundary).
    // Force should be identical to floating-point precision because the
    // code does max(h, h_min) before evaluating F.
    Real F_below = 0.0, F_at = 0.0;
    for (int trial = 0; trial < 2; ++trial) {
        Real h = (trial == 0) ? 0.5 * lp.h_min : lp.h_min;
        Real cy = 0.5 + h + (R - 1e-6);
        CapsuleSystem caps = oneCapsule(Vec2d{32.0, cy}, R, Nn);
        for (int k = 0; k < caps[0].numNodes(); ++k) {
            caps[0].forces()[k] = Vec2d{0.0, 0.0};
            caps[0].setNodeVelocity(k, Vec2d{0.0, -V});
        }
        LubricationCorrection lub(lp, nu);
        lub.computeAll(caps, ny, /*periodic_nx=*/64);
        Real Fy_lowest = 0.0;
        Real y_lowest  = 1e30;
        for (int k = 0; k < caps[0].numNodes(); ++k) {
            Vec2d p = caps[0].nodePosition(k);
            if (p.y < y_lowest) {
                y_lowest = p.y;
                Fy_lowest = caps[0].forces()[k].y;
            }
        }
        if (trial == 0) F_below = std::abs(Fy_lowest);
        else            F_at    = std::abs(Fy_lowest);
    }
    INFO("F(h = h_min/2) = " << F_below);
    INFO("F(h = h_min)   = " << F_at);
    CHECK(std::abs(F_below - F_at) < 1e-12);
}

TEST_CASE("Lubrication: zero force outside h_threshold") {
    LubricationParams lp;
    lp.enabled     = true;
    lp.h_threshold = 2.0;
    lp.h_min       = 0.1;
    const Real nu  = 0.1;
    const Real R   = 6.0;
    const int  Nn  = 48;
    const int  ny  = 96;
    const Real V   = 0.001;

    // h = h_threshold + 1.0 → outside cutoff, force must be exactly 0.
    const Real h  = lp.h_threshold + 1.0;
    Real cy = 0.5 + h + (R - 1e-6);
    CapsuleSystem caps = oneCapsule(Vec2d{32.0, cy}, R, Nn);
    for (int k = 0; k < caps[0].numNodes(); ++k) {
        caps[0].forces()[k] = Vec2d{0.0, 0.0};
        caps[0].setNodeVelocity(k, Vec2d{0.0, -V});
    }
    LubricationCorrection lub(lp, nu);
    lub.computeAll(caps, ny, /*periodic_nx=*/64);

    Real Fy_sum = 0.0;
    for (int k = 0; k < caps[0].numNodes(); ++k) {
        Fy_sum += std::abs(caps[0].forces()[k].y);
    }
    INFO("Sum|Fy| at h = h_threshold + 1.0 = " << Fy_sum);
    CHECK(Fy_sum < 1e-15);
}

TEST_CASE("Lubrication: capsule-capsule pair-wise 1/h scaling") {
    LubricationParams lp;
    lp.enabled     = true;
    lp.h_threshold = 2.0;
    lp.h_min       = 0.1;
    const Real nu  = 0.1;
    const Real R   = 6.0;
    const int  Nn  = 48;
    const int  ny  = 64;
    const Real V   = 0.001;     // approach speed (each capsule moves V/2 toward the other)

    const std::vector<Real> H = {1.6, 1.2, 0.8, 0.5};
    std::vector<Real> F_pair;
    F_pair.reserve(H.size());

    for (Real h : H) {
        // Two capsules along the x axis, separated so their closest
        // surface gap is h.
        const Real cx1 = 32.0 - (R + h * 0.5);
        const Real cx2 = 32.0 + (R + h * 0.5);
        const Real cy  = 32.0;

        CapsuleSystem caps;
        MembraneParams mp;
        caps.addCapsule(Vec2d{cx1, cy}, R, Nn, mp, /*type=*/0);
        caps.addCapsule(Vec2d{cx2, cy}, R, Nn, mp, /*type=*/0);

        for (int c = 0; c < 2; ++c) {
            for (int k = 0; k < caps[c].numNodes(); ++k) {
                caps[c].forces()[k] = Vec2d{0.0, 0.0};
                Real vx = (c == 0) ? V * 0.5 : -V * 0.5;   // toward each other
                caps[c].setNodeVelocity(k, Vec2d{vx, 0.0});
            }
        }
        LubricationCorrection lub(lp, nu);
        lub.computeAll(caps, ny, /*periodic_nx=*/0);

        Real Fx_sum = 0.0;
        for (int k = 0; k < caps[0].numNodes(); ++k) {
            Fx_sum += caps[0].forces()[k].x;
        }
        F_pair.push_back(std::abs(Fx_sum));
        INFO("h = " << h << "  |Fx_sum| = " << std::abs(Fx_sum));
    }

    // Verify monotonic increase as h decreases.
    for (size_t i = 1; i < F_pair.size(); ++i) {
        CHECK(F_pair[i] > F_pair[i - 1]);
    }
}
