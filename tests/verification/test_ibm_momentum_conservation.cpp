// V&V: IBM force-spreading momentum conservation.
//
// Peskin's 4-point regularized delta function is a partition of unity:
//
//   Sum_{x in lattice} phi(x - X_k) = 1   exactly,
//
// for every Lagrangian node X_k whose 4x4 stencil fits inside the lattice.
// Therefore the spread step
//
//   F(x) = Sum_k f_k * delta(x - X_k) * ds_k
//
// integrates exactly:
//
//   Sum_{x in lattice} F(x) * dx^2  =  Sum_k f_k * ds_k    (dx = 1 in lattice units).
//
// The accompanying manuscript reports Sum-residual < 1e-12 in Table 1;
// this test pins that claim. We construct a CapsuleSystem with a single
// circular capsule far from the walls, set every nodal force to a known
// uniform value, call ImmersedBoundary::spreadForces, and compare the
// summed lattice force to the analytical Sum_k f_k * ds_k = f * perimeter.
//
// Tolerance: 1e-12 relative, matching the manuscript claim. This is purely
// floating-point cancellation noise, not a discretization error.
//
// A node-spacing scan (Nn in {32, 64, 128, 256}) confirms the residual
// stays at machine precision and does not grow with N, demonstrating
// that the kernel sums to unity uniformly along the membrane.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "coupling/immersed_boundary.h"
#include "lbm/lattice_field.h"
#include "membrane/capsule_system.h"
#include "core/parameters.h"

#include <cmath>
#include "vv_dump.h"

using namespace softflow;

namespace {

// Build a single circular capsule of radius R at (cx, cy) with Nn nodes,
// far from the walls so the full Peskin 4-pt stencil fits the lattice.
CapsuleSystem makeRing(Vec2d center, Real R, int Nn) {
    MembraneParams mp;            // defaults: Hookean with k_stretch=0.1 — irrelevant
    CapsuleSystem caps;
    caps.addCapsule(center, R, Nn, mp, /*type=*/0);
    return caps;
}

// Sum F over all lattice cells (dx = 1 in lattice units, so this is the
// area integral).
Vec2d sumLatticeForce(const LatticeField& field) {
    int nx = field.getNx();
    int ny = field.getNy();
    Real sFx = 0.0, sFy = 0.0;
    for (int y = 0; y < ny; ++y) {
        for (int x = 0; x < nx; ++x) {
            sFx += field.Fx(x, y);
            sFy += field.Fy(x, y);
        }
    }
    return {sFx, sFy};
}

// Analytical total = Sum_k f * ds_k = f * Sum_k ds_k. The Capsule constructor
// stores rest segment lengths; for a uniformly sampled ring of N nodes on a
// circle of radius R, Sum_k ds_k = N * (2 sin(pi/N) * R) which approaches
// 2*pi*R. We compute it from the actual node positions for an exact match.
Vec2d expectedTotal(const CapsuleSystem& caps, Vec2d f_per_node, int periodic_nx) {
    const Capsule& c = caps[0];
    int Nn = c.numNodes();
    Real total_ds = 0.0;
    Real Lx = static_cast<Real>(periodic_nx);
    for (int k = 0; k < Nn; ++k) {
        int kprev = (k - 1 + Nn) % Nn;
        int knext = (k + 1) % Nn;
        auto ds_periodic = [&](Vec2d a, Vec2d b) {
            Vec2d d = a - b;
            if (Lx > 0.0) {
                if (d.x >  0.5 * Lx) d.x -= Lx;
                if (d.x < -0.5 * Lx) d.x += Lx;
            }
            return d.norm();
        };
        Real ds_prev = ds_periodic(c.nodePosition(k),     c.nodePosition(kprev));
        Real ds_next = ds_periodic(c.nodePosition(knext), c.nodePosition(k));
        total_ds += 0.5 * (ds_prev + ds_next);
    }
    return {f_per_node.x * total_ds, f_per_node.y * total_ds};
}

} // anonymous namespace

TEST_CASE("IBM spread: total lattice force matches Sum_k f_k*ds_k to 1e-12") {
    const int nx = 128, ny = 64;
    const Vec2d center{nx / 2.0, ny / 2.0};
    const Real  R = 12.0;
    const int   Nn = 64;
    const Vec2d f_per_node{0.001, -0.0007};   // arbitrary nonzero force

    LatticeField field(nx, ny);
    field.clearForces();

    CapsuleSystem caps = makeRing(center, R, Nn);
    // Set every node's force to the same value.
    for (int k = 0; k < caps[0].numNodes(); ++k) {
        caps[0].forces()[k] = f_per_node;
    }

    ImmersedBoundary ibm;
    ibm.spreadForces(caps, field);

    Vec2d got      = sumLatticeForce(field);
    Vec2d expected = expectedTotal(caps, f_per_node, nx);

    Real abs_x = std::abs(got.x - expected.x);
    Real abs_y = std::abs(got.y - expected.y);
    Real norm  = std::abs(expected.x) + std::abs(expected.y);
    Real rel   = (abs_x + abs_y) / norm;

    INFO("Sum F_x measured  = " << got.x);
    INFO("Sum F_x expected  = " << expected.x);
    INFO("Sum F_y measured  = " << got.y);
    INFO("Sum F_y expected  = " << expected.y);
    INFO("relative residual = " << rel);
    CHECK(rel < 1e-12);
}

TEST_CASE("IBM spread: residual stays at machine precision across node counts") {
    const int nx = 256, ny = 128;
    const Vec2d center{nx / 2.0, ny / 2.0};
    const Real  R = 16.0;
    const Vec2d f_per_node{0.002, 0.001};

    Real worst_rel = 0.0;
    int  worst_N   = 0;
    FILE* out = vv::open("ibm_partition", "Nn,rel_residual");

    for (int Nn : {32, 64, 128, 256}) {
        LatticeField field(nx, ny);
        field.clearForces();

        CapsuleSystem caps = makeRing(center, R, Nn);
        for (int k = 0; k < caps[0].numNodes(); ++k) {
            caps[0].forces()[k] = f_per_node;
        }

        ImmersedBoundary ibm;
        ibm.spreadForces(caps, field);

        Vec2d got      = sumLatticeForce(field);
        Vec2d expected = expectedTotal(caps, f_per_node, nx);

        Real rel = (std::abs(got.x - expected.x) + std::abs(got.y - expected.y))
                 / (std::abs(expected.x) + std::abs(expected.y));

        INFO("Nn = " << Nn << "  rel residual = " << rel);
        CHECK(rel < 1e-12);

        if (out) std::fprintf(out, "%d,%.6e\n", Nn, rel);
        if (rel > worst_rel) { worst_rel = rel; worst_N = Nn; }
    }
    vv::close(out);

    INFO("worst-case residual = " << worst_rel << "  at Nn = " << worst_N);
    CHECK(worst_rel < 1e-12);
}

TEST_CASE("IBM spread: capsule near periodic seam still conserves momentum") {
    // Place the capsule centre at x = 2, so the 4-pt stencil straddles the
    // periodic x = 0 seam. Momentum conservation must survive the wrap.
    const int nx = 64, ny = 64;
    const Vec2d center{2.0, ny / 2.0};
    const Real  R = 5.0;
    const int   Nn = 48;
    const Vec2d f_per_node{0.0015, 0.0005};

    LatticeField field(nx, ny);
    field.clearForces();

    CapsuleSystem caps = makeRing(center, R, Nn);
    caps[0].setPeriodicX(nx);
    for (int k = 0; k < caps[0].numNodes(); ++k) {
        caps[0].forces()[k] = f_per_node;
    }

    ImmersedBoundary ibm;
    ibm.spreadForces(caps, field);

    Vec2d got      = sumLatticeForce(field);
    Vec2d expected = expectedTotal(caps, f_per_node, nx);

    Real rel = (std::abs(got.x - expected.x) + std::abs(got.y - expected.y))
             / (std::abs(expected.x) + std::abs(expected.y));

    INFO("Sum F measured  = (" << got.x << ", " << got.y << ")");
    INFO("Sum F expected  = (" << expected.x << ", " << expected.y << ")");
    INFO("relative residual = " << rel);
    CHECK(rel < 1e-12);
}
