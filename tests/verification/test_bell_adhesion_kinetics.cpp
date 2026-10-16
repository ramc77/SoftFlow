// V&V: Bell-model adhesion kinetics .
//
// SoftFlow's reversible adhesion model implements two complementary
// pieces:
//
//   • Stochastic formation: a candidate (i, j) node pair within d_bond
//     forms a bond per step with probability p_on = k_on * dt
//     (`src/membrane/adhesion.cpp:69`, `:105`).
//
//   • Stochastic dissociation, Bell 1978:
//        k_off(F) = k_off^0  *  exp(F / F_crit)
//        p_off     = k_off(F) * dt
//     (`adhesion.cpp:186-188`).
//
// In the absence of formation (k_on = 0 or empty type matrix), the
// expected fraction of bonds surviving t time-units is
//
//     N(t) / N(0)  =  exp( -k_off(F) t ).
//
// For the catch-slip variant (params.use_catch_slip = true),
//
//     k_off(F) = k_catch * exp(-F/F_catch) + k_slip * exp(F/F_slip).
//
// We seed N_pairs independent bonded capsule pairs with a known
// pre-stretched bond (so that computeBondForces sets each bond's
// current_force to a known F), advance the AdhesionModel for n_steps,
// and compare the measured survival-probability decay rate to the
// analytical k_off(F).
//
// Tolerance: 8 % relative — accounts for the inherent N^{-1/2} sampling
// noise at N_pairs = 800 and the 30+ steps used to estimate the rate.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "membrane/adhesion.h"
#include "membrane/capsule_system.h"
#include "core/parameters.h"

#include <cmath>
#include "vv_dump.h"
#include <vector>

using namespace softflow;

namespace {

// Build N_pairs of capsules along x. Each pair is at vertical offset y0;
// capsule centres separated by 2R + gap so node-0 (rightmost of i, leftmost
// of j) sits at distance ≈ gap, smaller than d_bond. We then push exactly
// one bond per pair into AdhesionModel's bond list.
struct PairsSetup {
    CapsuleSystem caps;
    Real          rest_length;   // bond rest length (lattice units)
};

PairsSetup buildPairs(int N_pairs, Real R, Real gap) {
    PairsSetup s;
    MembraneParams mp;
    const int Nn = Capsule::computeOptimalNodes(R);
    const Real dx = 4.0 * R + 4.0;     // horizontal stride per pair (well-separated)
    const Real y0 = 50.0;
    for (int p = 0; p < N_pairs; ++p) {
        Real cxA = 10.0 + p * dx;
        Real cxB = cxA + 2.0 * R + gap;
        s.caps.addCapsule(Vec2d{cxA, y0}, R, Nn, mp, /*type=*/0);
        s.caps.addCapsule(Vec2d{cxB, y0}, R, Nn, mp, /*type=*/0);
    }
    s.rest_length = gap;     // matches the gap between facing nodes
    return s;
}

// For each pair (capsule_index 2p and 2p+1), find the (ni, nj) with the
// smallest centre-to-centre distance and push a Bond into the model.
void seedBonds(AdhesionModel& adh, const CapsuleSystem& caps, Real rest_length) {
    auto& bonds = adh.getBondsMutable();
    bonds.clear();
    int ncaps = caps.numCapsules();
    for (int p = 0; 2 * p + 1 < ncaps; ++p) {
        const Capsule& ci = caps[2 * p];
        const Capsule& cj = caps[2 * p + 1];
        Real best = 1e30;
        int  bi = 0, bj = 0;
        for (int ni = 0; ni < ci.numNodes(); ++ni) {
            for (int nj = 0; nj < cj.numNodes(); ++nj) {
                Real d = (cj.nodePosition(nj) - ci.nodePosition(ni)).norm();
                if (d < best) { best = d; bi = ni; bj = nj; }
            }
        }
        bonds.push_back({2 * p, bi, 2 * p + 1, bj, rest_length, 0.0});
    }
}

// Repeatedly call AdhesionModel::update for `n_steps`. Capsules are held
// frozen by zeroing velocities each step. Returns the number of surviving
// bonds at each step (history.size() = n_steps + 1).
std::vector<int> runDecay(AdhesionModel& adh, CapsuleSystem& caps, int n_steps,
                          Real dt, int ny, int periodic_nx) {
    std::vector<int> hist;
    hist.reserve(n_steps + 1);
    hist.push_back(adh.getNumBonds());
    for (int s = 0; s < n_steps; ++s) {
        // Freeze the capsules so geometry — and hence the bond force — is
        // invariant from step to step.
        for (int c = 0; c < caps.numCapsules(); ++c) {
            for (int k = 0; k < caps[c].numNodes(); ++k) {
                caps[c].setNodeVelocity(k, Vec2d{0.0, 0.0});
                caps[c].forces()[k] = Vec2d{0.0, 0.0};
            }
        }
        adh.update(caps, dt, ny, periodic_nx);
        hist.push_back(adh.getNumBonds());
    }
    return hist;
}

// Fit the decay history to N(t) = N0 * exp(-k * t) by least squares on log N.
Real fitDecayRate(const std::vector<int>& hist) {
    Real sx = 0.0, sy = 0.0, sxx = 0.0, sxy = 0.0;
    int  m  = 0;
    for (size_t t = 0; t < hist.size(); ++t) {
        if (hist[t] < 5) break;             // statistics fall apart below 5
        Real x = static_cast<Real>(t);
        Real y = std::log(static_cast<Real>(hist[t]));
        sx  += x; sy  += y; sxx += x*x; sxy += x*y;
        ++m;
    }
    Real denom = m * sxx - sx * sx;
    return -(m * sxy - sx * sy) / denom;     // -slope = k
}

} // anonymous namespace

TEST_CASE("Bell adhesion: bond survival decays as exp(-k_off(F) * t)") {
    // Reusable parameters
    AdhesionParams ap;
    ap.enabled              = true;
    ap.adhesion_matrix      = {{ true }};         // type 0 ↔ type 0 allowed
    ap.k_on                 = 0.0;                // no new formation
    ap.k_off                = 0.02;               // k_off^0
    ap.k_bond               = 1.0;                // bond spring stiffness
    ap.d_bond               = 2.0;
    ap.F_crit               = 0.05;
    ap.max_bonds_per_node   = 4;
    ap.use_catch_slip       = false;
    ap.wall_adhesion        = false;

    const Real R   = 4.0;
    const int  ny  = 100;
    const Real dt  = 1.0;
    const int  N_pairs = 1500;
    const int  n_steps = 60;

    // Three forces spanning the Bell exponential — at F = F_crit, lifetime
    // is shortened by exactly e.
    const std::vector<Real> F_test = {0.0, 0.5 * ap.F_crit, ap.F_crit};
    Real worst_rel = 0.0;
    Real worst_F   = 0.0;
    FILE* out = vv::open("bell_survival", "F,F_crit,step,N,N0,k_off_analytic,k_off_measured");

    for (Real F : F_test) {
        // Choose stretch so that k_bond*(d - rest_length) = F.
        Real stretch = F / ap.k_bond;       // bond extension to give force F
        Real gap     = 0.4;                 // facing-node gap (< d_bond)
        Real rest_length = gap - stretch;   // so current = rest + stretch
        if (rest_length < 0.0) rest_length = 0.05;     // floor
        // current_force = k_bond * (gap - rest_length)
        Real F_applied = ap.k_bond * (gap - rest_length);
        INFO("F_target = " << F << ", F_applied = " << F_applied);

        PairsSetup s   = buildPairs(N_pairs, R, gap);
        AdhesionModel adh(ap, /*seed=*/ 12345 + static_cast<unsigned>(F * 1000));
        seedBonds(adh, s.caps, rest_length);

        std::vector<int> hist = runDecay(adh, s.caps, n_steps, dt, ny, 0);

        Real k_meas = fitDecayRate(hist);
        Real k_anal = ap.k_off * std::exp(F_applied / ap.F_crit);
        Real rel    = std::abs(k_meas - k_anal) / k_anal;

        INFO("k_off^0          = " << ap.k_off);
        INFO("F_applied        = " << F_applied);
        INFO("k_off(F) analyt. = " << k_anal);
        INFO("k_off(F) measured= " << k_meas);
        INFO("relative error   = " << rel);

        // Sanity: N(0) = N_pairs, N(t_end) < N(0).
        CHECK(hist.front() == N_pairs);
        CHECK(hist.back()  <  hist.front());

        if (out)
            for (size_t t = 0; t < hist.size(); ++t)
                std::fprintf(out, "%.6e,%.6e,%zu,%d,%d,%.6e,%.6e\n", F_applied,
                             ap.F_crit, t, hist[t], hist.front(), k_anal, k_meas);
        if (rel > worst_rel) { worst_rel = rel; worst_F = F_applied; }
    }
    vv::close(out);
    INFO("worst relative error = " << worst_rel << " at F = " << worst_F);
    // Tolerance of 10 % covers the inherent N^{-1/2} sampling noise at
    // N_pairs = 1500 and the 60 steps used to estimate the rate; with
    // k_off = 0.02 exp(F/F_crit), expected survivors at the steepest
    // case (F = F_crit) are N(t_end) ~ 1500 * exp(-60*0.054) ~ 60,
    // giving Poisson-noise-limited rate estimate ~ 1/sqrt(60) ~ 13 %.
    CHECK(worst_rel < 0.10);
}

TEST_CASE("Bell adhesion: catch-slip lifetime is non-monotonic in F") {
    // Catch-slip lifetime tau(F) = 1 / k_off(F) where
    //   k_off(F) = k_catch * exp(-F/F_catch) + k_slip * exp(F/F_slip).
    // tau' = 0 at F* = (F_catch F_slip)/(F_catch + F_slip)
    //                  * ln[(k_catch F_slip)/(k_slip F_catch)],
    // and we expect tau to be larger at the optimum than at F = 0 and at
    // F = 2 F_slip. The test only checks this qualitative shape — pinning
    // the optimum location would amount to re-deriving the model.
    AdhesionParams ap;
    ap.enabled              = true;
    ap.adhesion_matrix      = {{ true }};
    ap.k_on                 = 0.0;
    ap.k_bond               = 1.0;            // higher stiffness so F up to ~0.25 fits in
                                              // gap ∈ [0, d_bond] without floor-clipping
    ap.d_bond               = 2.0;
    ap.max_bonds_per_node   = 4;
    ap.use_catch_slip       = true;
    ap.k_off_catch          = 0.5;
    ap.k_off_slip           = 0.005;
    ap.F_catch              = 0.02;
    ap.F_slip               = 0.05;
    ap.wall_adhesion        = false;

    const Real R = 4.0;
    const int  ny = 100;
    const Real dt = 1.0;
    const int  N_pairs = 1500;
    const int  n_steps = 60;

    // The catch-slip lifetime maximum (k_off minimum) is at
    //   F* = ln[(k_catch F_slip)/(k_slip F_catch)] / (1/F_catch + 1/F_slip)
    //      = ln(250) / 70 ≈ 0.0789.
    // Probe three forces: 0, just below the optimum (in the catch
    // regime), and well above the optimum (in the slip regime).
    auto measure_rate = [&](Real F_target, unsigned seed) {
        Real gap          = 0.5;
        Real rest_length  = gap - F_target / ap.k_bond;
        REQUIRE(rest_length > 0.01);     // catch the floor-clipping mistake
        Real F_app = ap.k_bond * (gap - rest_length);

        PairsSetup s = buildPairs(N_pairs, R, gap);
        AdhesionModel adh(ap, /*seed=*/seed);
        seedBonds(adh, s.caps, rest_length);
        std::vector<int> hist = runDecay(adh, s.caps, n_steps, dt, ny, 0);
        return std::pair<Real, Real>{F_app, fitDecayRate(hist)};
    };

    auto [F_lo,  k_lo ] = measure_rate(0.0,   71001);
    auto [F_mid, k_mid] = measure_rate(0.08,  71002);   // near optimum
    auto [F_hi,  k_hi ] = measure_rate(0.20,  71003);   // deep slip regime

    INFO("low  F = " << F_lo  << "  k_off = " << k_lo);
    INFO("mid  F = " << F_mid << "  k_off = " << k_mid);
    INFO("high F = " << F_hi  << "  k_off = " << k_hi);

    // Catch-slip signature: k_off is non-monotonic — drops from F=0 toward
    // the optimum, then increases at large F.
    CHECK(k_mid < k_lo);
    CHECK(k_hi  > k_mid);
}
