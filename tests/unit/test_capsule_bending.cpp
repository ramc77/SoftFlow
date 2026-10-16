// Discrete-elastica bending, E_b = (k_b/2) sum_i (theta_i - theta0_i)^2 / l0_i.
//
//  * The rest shape is force-free for any rest shape (biconcave included),
//    so a red-cell-like capsule is not pre-stressed.
//  * The force is the exact negative gradient of E_b (finite-difference
//    check) and carries no net force or torque.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "membrane/capsule.h"

#include <cmath>
#include <random>

using namespace softflow;

namespace {

MembraneParams bendingOnly(CapsuleShape shape) {
    MembraneParams mp;
    mp.model = MembraneModel::HOOKEAN;
    mp.k_stretch = 0.0;
    mp.k_bend = 0.01;
    mp.use_helfrich_bending = true;
    mp.k_area = 0.0;
    mp.k_perimeter = 0.0;
    mp.gamma_visc = 0.0;
    mp.eta_membrane = 0.0;
    mp.shape = shape;
    mp.indent_depth = 0.7;
    return mp;
}

Real maxForce(const Capsule& c) {
    Real m = 0.0;
    for (const auto& f : c.forces()) m = std::max(m, f.norm());
    return m;
}

void perturb(Capsule& c, Real amp, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<Real> u(-amp, amp);
    for (auto& p : c.positions()) p = Vec2d{p.x + u(rng), p.y + u(rng)};
}

} // namespace

TEST_CASE("rest shape is force-free (circle and biconcave)") {
    for (auto shape : {CapsuleShape::CIRCLE, CapsuleShape::BICONCAVE}) {
        Capsule c(0, {30.0, 30.0}, 8.0, 60, bendingOnly(shape));
        c.clearForces();
        c.computeMembraneForces();
        CHECK(maxForce(c) < 1e-12);
        CHECK(c.bendingEnergy() < 1e-20);
    }
}

TEST_CASE("bending force is -grad E_b, with zero net force and torque") {
    Capsule c(0, {30.0, 30.0}, 8.0, 60, bendingOnly(CapsuleShape::BICONCAVE));
    perturb(c, 0.3, 7);
    c.clearForces();
    c.computeMembraneForces();
    const auto F = c.forces();

    Vec2d net{0.0, 0.0};
    Real torque = 0.0, fmax = 0.0;
    const Vec2d ctr = c.centroid();
    for (int k = 0; k < c.numNodes(); ++k) {
        net += F[k];
        torque += (c.positions()[k] - ctr).cross(F[k]);
        fmax = std::max(fmax, F[k].norm());
    }
    CHECK(fmax > 1e-6);
    CHECK(net.norm() < 1e-12 * c.numNodes());
    CHECK(std::abs(torque) < 1e-11 * c.numNodes());

    const Real h = 1e-6;
    Real err = 0.0;
    for (int k = 0; k < c.numNodes(); k += 7) {
        for (int d = 0; d < 2; ++d) {
            Vec2d& p = c.positions()[k];
            Real& q = (d == 0) ? p.x : p.y;
            const Real q0 = q;
            q = q0 + h; Real Ep = c.bendingEnergy();
            q = q0 - h; Real Em = c.bendingEnergy();
            q = q0;
            Real fd = -(Ep - Em) / (2.0 * h);
            Real an = (d == 0) ? F[k].x : F[k].y;
            err = std::max(err, std::abs(fd - an));
        }
    }
    INFO("max |F - (-dE/dx)| = " << err << ", max |F| = " << fmax);
    CHECK(err < 1e-6 * fmax);
}
