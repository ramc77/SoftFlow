#include "immersed_boundary.h"
#include "../lbm/lattice_field.h"
#include "../membrane/capsule_system.h"
#include "../membrane/capsule.h"
#include <cmath>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace softflow {

// Minimum-image arc length between adjacent nodes.
// When capsule straddles a periodic boundary, raw difference can be ~nx
// instead of ~1 lattice unit. This fixes the ds computation.
static inline Real minImageDs(const Vec2d& a, const Vec2d& b, Real Lx) {
    Vec2d d = a - b;
    if (Lx > 0.0) {
        if (d.x >  0.5 * Lx) d.x -= Lx;
        if (d.x < -0.5 * Lx) d.x += Lx;
    }
    return d.norm();
}

void ImmersedBoundary::spreadForces(const CapsuleSystem& capsules, LatticeField& field) {
    // NOTE: lattice forces must be cleared by caller (Simulation::step) before
    // calling this method.

    int ncaps = capsules.numCapsules();
    if (ncaps == 0) return;

    int nx = field.getNx();
    int ny = field.getNy();
    Real Lx = static_cast<Real>(nx); // periodic domain width

#ifdef _OPENMP
    int N = nx * ny;
    int num_threads = omp_get_max_threads();

    // Allocate thread-local force buffers to avoid atomic writes
    std::vector<std::vector<Real>> local_Fx(num_threads, std::vector<Real>(N, 0.0));
    std::vector<std::vector<Real>> local_Fy(num_threads, std::vector<Real>(N, 0.0));

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        Real* my_Fx = local_Fx[tid].data();
        Real* my_Fy = local_Fy[tid].data();

        #pragma omp for schedule(dynamic)
        for (int c = 0; c < ncaps; ++c) {
            const Capsule& cap = capsules[c];
            int Nn = cap.numNodes();
            for (int k = 0; k < Nn; ++k) {
                Vec2d pos = cap.nodePosition(k);
                Vec2d force = cap.nodeForce(k);

                int kprev = (k - 1 + Nn) % Nn;
                int knext = (k + 1) % Nn;
                // Use minimum-image for ds to handle periodic boundary crossing
                Real ds_prev = minImageDs(cap.nodePosition(k), cap.nodePosition(kprev), Lx);
                Real ds_next = minImageDs(cap.nodePosition(knext), cap.nodePosition(k), Lx);
                Real ds = 0.5 * (ds_prev + ds_next);

                spreadNodeForceLocal(pos, force, ds, nx, ny, my_Fx, my_Fy);
            }
        }

        // Parallel reduction: merge thread-local buffers into field
        #pragma omp for schedule(static)
        for (int n = 0; n < N; ++n) {
            Real sum_fx = 0.0, sum_fy = 0.0;
            for (int t = 0; t < num_threads; ++t) {
                sum_fx += local_Fx[t][n];
                sum_fy += local_Fy[t][n];
            }
            field.FxData()[n] += sum_fx;
            field.FyData()[n] += sum_fy;
        }
    }
#else
    // Serial fallback
    for (int c = 0; c < ncaps; ++c) {
        const Capsule& cap = capsules[c];
        int Nn = cap.numNodes();
        for (int k = 0; k < Nn; ++k) {
            Vec2d pos = cap.nodePosition(k);
            Vec2d force = cap.nodeForce(k);

            int kprev = (k - 1 + Nn) % Nn;
            int knext = (k + 1) % Nn;
            Real ds_prev = minImageDs(cap.nodePosition(k), cap.nodePosition(kprev), Lx);
            Real ds_next = minImageDs(cap.nodePosition(knext), cap.nodePosition(k), Lx);
            Real ds = 0.5 * (ds_prev + ds_next);

            spreadNodeForce(pos, force, ds, field);
        }
    }
#endif
}

void ImmersedBoundary::spreadNodeForce(const Vec2d& node_pos, const Vec2d& force,
                                        Real ds, LatticeField& field) {
    // Guard against NaN/Inf
    if (std::isnan(node_pos.x) || std::isnan(node_pos.y) ||
        std::isinf(node_pos.x) || std::isinf(node_pos.y)) return;
    if (std::isnan(force.x) || std::isnan(force.y)) return;

    int nx = field.getNx();
    int ny = field.getNy();

    int ix0 = static_cast<int>(std::floor(node_pos.x));
    int iy0 = static_cast<int>(std::floor(node_pos.y));

    if (iy0 < -PeskinDelta::SUPPORT || iy0 >= ny + PeskinDelta::SUPPORT) return;

    for (int iy = iy0 - PeskinDelta::SUPPORT + 1; iy <= iy0 + PeskinDelta::SUPPORT; ++iy) {
        for (int ix = ix0 - PeskinDelta::SUPPORT + 1; ix <= ix0 + PeskinDelta::SUPPORT; ++ix) {
            int jx = ix;
            if (jx < 0) jx += nx;
            if (jx >= nx) jx -= nx;
            int jy = iy;
            if (jy < 0 || jy >= ny) continue;

            Real rx = node_pos.x - static_cast<Real>(ix);
            Real ry = node_pos.y - static_cast<Real>(iy);
            Real w = PeskinDelta::delta2d(rx, ry);

            if (w > 1e-15) {
                field.addExternalForce(jx, jy, force.x * w * ds, force.y * w * ds);
            }
        }
    }
}

void ImmersedBoundary::spreadNodeForceLocal(const Vec2d& node_pos, const Vec2d& force,
                                             Real ds, int nx, int ny,
                                             Real* local_Fx, Real* local_Fy) {
    // Guard against NaN/Inf positions (from blow-up)
    if (std::isnan(node_pos.x) || std::isnan(node_pos.y) ||
        std::isinf(node_pos.x) || std::isinf(node_pos.y)) return;
    if (std::isnan(force.x) || std::isnan(force.y)) return;

    int ix0 = static_cast<int>(std::floor(node_pos.x));
    int iy0 = static_cast<int>(std::floor(node_pos.y));

    // Sanity check: if node is far outside domain, skip
    if (iy0 < -PeskinDelta::SUPPORT || iy0 >= ny + PeskinDelta::SUPPORT) return;

    int N = nx * ny;
    for (int iy = iy0 - PeskinDelta::SUPPORT + 1; iy <= iy0 + PeskinDelta::SUPPORT; ++iy) {
        for (int ix = ix0 - PeskinDelta::SUPPORT + 1; ix <= ix0 + PeskinDelta::SUPPORT; ++ix) {
            int jx = ix;
            if (jx < 0) jx += nx;
            if (jx >= nx) jx -= nx;
            int jy = iy;
            if (jy < 0 || jy >= ny) continue;

            Real rx = node_pos.x - static_cast<Real>(ix);
            Real ry = node_pos.y - static_cast<Real>(iy);
            Real w = PeskinDelta::delta2d(rx, ry);

            if (w > 1e-15) {
                int n = jy * nx + jx;
                if (n >= 0 && n < N) {
                    local_Fx[n] += force.x * w * ds;
                    local_Fy[n] += force.y * w * ds;
                }
            }
        }
    }
}

void ImmersedBoundary::interpolateVelocity(const LatticeField& field,
                                             CapsuleSystem& capsules) {
    int ncaps = capsules.numCapsules();
#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic)
#endif
    for (int c = 0; c < ncaps; ++c) {
        Capsule& cap = capsules[c];
        int N = cap.numNodes();
        for (int k = 0; k < N; ++k) {
            Vec2d pos = cap.nodePosition(k);
            Vec2d u_interp = interpolateAtPoint(pos, field);
            cap.setNodeVelocity(k, u_interp);
        }
    }
}

Vec2d ImmersedBoundary::interpolateAtPoint(const Vec2d& pos,
                                            const LatticeField& field) {
    // Guard against NaN/Inf
    if (std::isnan(pos.x) || std::isnan(pos.y) ||
        std::isinf(pos.x) || std::isinf(pos.y))
        return {0.0, 0.0};

    int nx = field.getNx();
    int ny = field.getNy();

    int ix0 = static_cast<int>(std::floor(pos.x));
    int iy0 = static_cast<int>(std::floor(pos.y));

    if (iy0 < -PeskinDelta::SUPPORT || iy0 >= ny + PeskinDelta::SUPPORT)
        return {0.0, 0.0};

    Vec2d u_interp{0.0, 0.0};

    for (int iy = iy0 - PeskinDelta::SUPPORT + 1; iy <= iy0 + PeskinDelta::SUPPORT; ++iy) {
        for (int ix = ix0 - PeskinDelta::SUPPORT + 1; ix <= ix0 + PeskinDelta::SUPPORT; ++ix) {
            int jx = ix;
            if (jx < 0) jx += nx;
            if (jx >= nx) jx -= nx;
            int jy = iy;
            if (jy < 0 || jy >= ny) continue;

            Real rx = pos.x - static_cast<Real>(ix);
            Real ry = pos.y - static_cast<Real>(iy);
            Real w = PeskinDelta::delta2d(rx, ry);

            if (w > 1e-15) {
                u_interp.x += field.getUx(jx, jy) * w;
                u_interp.y += field.getUy(jx, jy) * w;
            }
        }
    }

    return u_interp;
}

void ImmersedBoundary::multiDirectForcing(CapsuleSystem& capsules,
                                           LatticeField& field,
                                           int iterations) {
    // Standard spread (iteration 0)
    spreadForces(capsules, field);

    if (iterations <= 1) return;

    // Iterative correction (Luo et al., PRE 2007)
    // For each subsequent iteration:
    //   1. Recompute macroscopic velocity from distributions + forces
    //   2. Re-interpolate velocity at Lagrangian points
    //   3. Compute correction: deltaF = 2*rho*(U_desired - U_interp)/dt
    //   4. Spread correction forces to lattice
    for (int iter = 1; iter < iterations; ++iter) {
        // Recompute macroscopic velocity including current forces
        field.computeMacroscopic();

        // Interpolate velocity at membrane nodes
        const int ncaps = capsules.numCapsules();
        for (int c = 0; c < ncaps; ++c) {
            auto& cap = capsules[c];
            for (int k = 0; k < cap.numNodes(); ++k) {
                Vec2d pos = cap.nodePosition(k);
                Vec2d u_interp = interpolateAtPoint(pos, field);
                Vec2d u_desired = cap.nodeVelocity(k); // from previous step
                Vec2d delta_u = u_desired - u_interp;

                // Correction force: deltaF = 2 * rho * delta_u / dt
                // In lattice units, dt = 1, rho ≈ 1
                Real rho_local = 1.0; // approximate
                int ix = static_cast<int>(pos.x + 0.5);
                int iy = static_cast<int>(pos.y + 0.5);
                // Periodic wrapping for rho lookup
                if (ix < 0) ix += field.getNx();
                if (ix >= field.getNx()) ix -= field.getNx();
                if (iy >= 0 && iy < field.getNy()) {
                    rho_local = field.getRho(ix, iy);
                }

                Vec2d delta_F = delta_u * (2.0 * rho_local);

                // Compute arc length for this node (with periodic minimum-image)
                int prev_k = (k - 1 + cap.numNodes()) % cap.numNodes();
                int next_k = (k + 1) % cap.numNodes();
                Real Lx = static_cast<Real>(field.getNx());
                Real ds_prev = minImageDs(cap.nodePosition(k), cap.nodePosition(prev_k), Lx);
                Real ds_next = minImageDs(cap.nodePosition(next_k), cap.nodePosition(k), Lx);
                Real ds = 0.5 * (ds_prev + ds_next);

                // Spread correction force
                spreadNodeForce(pos, delta_F, ds, field);
            }
        }
    }
}

} // namespace softflow
