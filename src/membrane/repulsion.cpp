#include "repulsion.h"
#include <cmath>

namespace softflow {

Vec2d RepulsionForce::pairForce(const Vec2d& pos_a, const Vec2d& vel_a,
                                const Vec2d& pos_b, const Vec2d& vel_b) const {
    Vec2d diff = pos_a - pos_b;     // points from b to a
    diff.x = minImageDx(diff.x);    // periodic image in x
    Real r = diff.norm();
    if (r < 1e-15 || r >= params_.r_cut) return Vec2d{0.0, 0.0};

    Vec2d n = diff / r;             // unit normal pointing from b toward a
    // Conservative power-law repulsion (the contact "spring").
    Real F_rep = params_.epsilon * std::pow(params_.sigma / r, params_.power);
    Real F_n = F_rep;
    Vec2d F_t{0.0, 0.0};

    if (params_.damping_normal > 0.0 || params_.friction_coeff > 0.0) {
        Vec2d v_rel = vel_a - vel_b;
        // Normal relative velocity (>0 separating, <0 approaching).
        Real v_n = v_rel.x * n.x + v_rel.y * n.y;
        // Normal viscoelastic damping; cohesionless => clamp total normal >= 0
        // (a dashpot would otherwise pull the pair together while separating).
        F_n = F_rep - params_.damping_normal * v_n;
        if (F_n < 0.0) F_n = 0.0;
        // Tangential Coulomb friction opposing the sliding direction, capped
        // at mu * |F_n| by construction.
        if (params_.friction_coeff > 0.0) {
            Vec2d v_t = v_rel - n * v_n;   // tangential relative velocity
            Real v_t_mag = v_t.norm();
            if (v_t_mag > 1e-12) {
                F_t = (v_t / v_t_mag) * (-params_.friction_coeff * F_n);
            }
        }
    }
    return n * F_n + F_t;
}

void RepulsionForce::computeInterCapsule(Capsule& ci, Capsule& cj) {
    auto& pos_i = ci.positions();
    auto& pos_j = cj.positions();
    auto& vel_i = ci.velocities();
    auto& vel_j = cj.velocities();
    auto& f_i = ci.forces();
    auto& f_j = cj.forces();

    for (int a = 0; a < ci.numNodes(); ++a) {
        for (int b = 0; b < cj.numNodes(); ++b) {
            // pairForce(a,b) == -pairForce(b,a), so Newton's 3rd law holds.
            Vec2d F = pairForce(pos_i[a], vel_i[a], pos_j[b], vel_j[b]);
            f_i[a] += F;
            f_j[b] -= F;
        }
    }
}

void RepulsionForce::computeWallRepulsion(Capsule& c, Real y_bottom, Real y_top) {
    auto& pos = c.positions();
    auto& vel = c.velocities();
    auto& f = c.forces();
    const Vec2d kStatic{0.0, 0.0};  // walls are stationary

    // Reuse pairForce with a projected surface point directly below/above the
    // node: the normal comes out as +y (bottom) or -y (top), and damping +
    // friction against the (static) wall follow automatically.
    for (int i = 0; i < c.numNodes(); ++i) {
        f[i] += pairForce(pos[i], vel[i], Vec2d{pos[i].x, y_bottom}, kStatic);
        f[i] += pairForce(pos[i], vel[i], Vec2d{pos[i].x, y_top}, kStatic);
    }
}

/// One-sided repulsion: force on target from source only.
/// Thread-safe when each thread owns a unique target capsule.
void RepulsionForce::computeOneSidedRepulsion(Capsule& target, const Capsule& source) {
    auto& pos_t = target.positions();
    auto& vel_t = target.velocities();
    const auto& pos_s = source.positions();
    const auto& vel_s = source.velocities();
    auto& f_t = target.forces();

    for (int a = 0; a < target.numNodes(); ++a) {
        for (int b = 0; b < source.numNodes(); ++b) {
            f_t[a] += pairForce(pos_t[a], vel_t[a], pos_s[b], vel_s[b]);
        }
    }
}

void RepulsionForce::computeAll(CapsuleSystem& system, Real y_bottom, Real y_top) {
    int n = system.numCapsules();
    if (n == 0) return;

    // Use cell-list acceleration when there are enough capsules to benefit.
    // The cell list reduces O(N_nodes^2) pair checks to O(N_nodes * avg_neighbors).
    if (system.totalNodes() > 100) {
        computeAllCellList(system, y_bottom, y_top);
        return;
    }

#ifdef _OPENMP
    // Parallel: one-sided force — each thread owns its target capsule.
    // Doubles FLOPs vs Newton-3rd-law but eliminates write conflicts.
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i == j) continue;
            computeOneSidedRepulsion(system[i], system[j]);
        }
        computeWallRepulsion(system[i], y_bottom, y_top);
    }
#else
    // Serial: use Newton's 3rd law (compute each pair once)
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            computeInterCapsule(system[i], system[j]);
        }
    }
    for (int i = 0; i < n; ++i) {
        computeWallRepulsion(system[i], y_bottom, y_top);
    }
#endif
}

void RepulsionForce::computeAllCellList(CapsuleSystem& system,
                                         Real y_bottom, Real y_top) {
    const int ncaps = system.numCapsules();
    const Real r_cut = params_.r_cut;

    // Build the cell list from current node positions.
    // Cell size = r_cut so any interacting pair shares the same or adjacent cells.
    cell_list_.build(system, r_cut, Lx_, y_bottom, y_top);

    // One-sided, cell-list accelerated repulsion.
    // Outer loop: iterate over each capsule (and each of its nodes).
    // For each node, look up its cell and the 3x3 neighbor stencil,
    // then only check nodes from OTHER capsules found in those cells.
    //
    // Thread safety: each thread owns capsule i and only writes to
    // system[i].forces(), so no write conflicts occur.

#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic)
#endif
    for (int i = 0; i < ncaps; ++i) {
        auto& cap_i = system[i];
        auto& pos_i = cap_i.positions();
        auto& vel_i = cap_i.velocities();
        auto& f_i = cap_i.forces();
        const int nnodes_i = cap_i.numNodes();

        for (int a = 0; a < nnodes_i; ++a) {
            int ci = cell_list_.cellIndex(pos_i[a]);
            int cx = ci % cell_list_.ncx();
            int cy = ci / cell_list_.ncx();

            int neighbors[9];
            int nneigh = cell_list_.neighborCells(cx, cy, neighbors);

            for (int nc = 0; nc < nneigh; ++nc) {
                const auto& cell_nodes = cell_list_.cell(
                    neighbors[nc] % cell_list_.ncx(),
                    neighbors[nc] / cell_list_.ncx());

                for (const auto& ref : cell_nodes) {
                    if (ref.capsule == i) continue; // skip self-capsule

                    const Vec2d& pos_b = system[ref.capsule].nodePosition(ref.node);
                    const Vec2d vel_b = system[ref.capsule].nodeVelocity(ref.node);
                    // one-sided: only write to target node a
                    f_i[a] += pairForce(pos_i[a], vel_i[a], pos_b, vel_b);
                }
            }
        }

        // Wall repulsion (always needed, independent of cell list)
        computeWallRepulsion(cap_i, y_bottom, y_top);
    }
}

void RepulsionForce::computeObstacleRepulsion(CapsuleSystem& system, const Obstacle& obs) {
    const Real r_cut = params_.r_cut;
    const Real epsilon = params_.epsilon;
    const Real sigma = params_.sigma;
    const int power = params_.power;
    const Real gamma_n = params_.damping_normal;
    const Real mu = params_.friction_coeff;
    const int ncaps = system.numCapsules();

#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic)
#endif
    for (int c = 0; c < ncaps; ++c) {
        auto& pos = system[c].positions();
        auto& vel = system[c].velocities();
        auto& f = system[c].forces();

        for (int i = 0; i < system[c].numNodes(); ++i) {
            Real sd = obs.signedDistance(pos[i].x, pos[i].y);

            // sd > 0: outside obstacle (normal repulsion + optional friction)
            // sd < 0: INSIDE obstacle (pure emergency push-out)
            if (sd >= r_cut) continue;  // too far away

            Real r = std::abs(sd);
            if (r < 1e-15) r = 1e-15;  // avoid division by zero

            Vec2d normal = obs.normalAt(pos[i].x, pos[i].y);
            Real F_rep = epsilon * std::pow(sigma / r, power);

            if (sd < 0.0) {
                // Inside the obstacle: strong emergency push-out, no damping.
                f[i] += normal * (F_rep * 10.0);
                continue;
            }

            // Outside contact: optional normal damping + Coulomb friction
            // against the (static) obstacle surface.
            Real F_n = F_rep;
            Vec2d F_t{0.0, 0.0};
            if (gamma_n > 0.0 || mu > 0.0) {
                const Vec2d& v = vel[i];
                Real v_n = v.x * normal.x + v.y * normal.y;
                F_n = F_rep - gamma_n * v_n;
                if (F_n < 0.0) F_n = 0.0;
                if (mu > 0.0) {
                    Vec2d v_t = v - normal * v_n;
                    Real v_t_mag = v_t.norm();
                    if (v_t_mag > 1e-12) {
                        F_t = (v_t / v_t_mag) * (-mu * F_n);
                    }
                }
            }
            f[i] += normal * F_n + F_t;
        }
    }
}

} // namespace softflow
