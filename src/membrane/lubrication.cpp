#include "lubrication.h"
#include "capsule_system.h"
#include "cell_list.h"
#include <cmath>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace softflow {

LubricationCorrection::LubricationCorrection(const LubricationParams& params,
                                               Real kinematic_viscosity)
    : h_threshold_(params.h_threshold),
      h_min_(params.h_min),
      mu_(kinematic_viscosity)  // rho=1 in lattice units, so mu = nu
{}

void LubricationCorrection::computeAll(CapsuleSystem& capsules, int ny,
                                        int periodic_nx) {
    computeCapsulePairLubrication(capsules, periodic_nx);
    computeWallLubrication(capsules, ny);
}

void LubricationCorrection::computeCapsulePairLubrication(
    CapsuleSystem& capsules, int periodic_nx) {
    const int ncaps = capsules.numCapsules();
    if (ncaps < 2) return;

    const Real Lx = (periodic_nx > 0) ? static_cast<Real>(periodic_nx) : 0.0;

    // ── Use CellList for O(N) neighbor finding ──────────────
    // Build cell list with h_threshold as cutoff
    Real y_min = 0.0, y_max = 1000.0; // conservative bounds
    // Find actual y range from capsule positions
    for (int c = 0; c < ncaps; ++c) {
        const auto& pos = capsules[c].positions();
        for (const auto& p : pos) {
            if (p.y < y_min) y_min = p.y;
            if (p.y > y_max) y_max = p.y;
        }
    }
    y_min -= h_threshold_;
    y_max += h_threshold_;

    CellList cell_list;
    cell_list.build(capsules, h_threshold_, Lx, y_min, y_max);

    // For each capsule, find closest interacting node from other capsules
    // using the cell list for O(N) neighbor search.
    // One-sided force accumulation: each thread owns its target capsule ci.
#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic)
#endif
    for (int ci = 0; ci < ncaps; ++ci) {
        auto& capI = capsules[ci];
        Real rI = capI.effectiveRadius();

        for (int ni = 0; ni < capI.numNodes(); ++ni) {
            Vec2d pi = capI.nodePosition(ni);

            // Look up cell for this node
            int ci_cell = cell_list.cellIndex(pi);
            int cx = ci_cell % cell_list.ncx();
            int cy = ci_cell / cell_list.ncx();

            // Query neighbor cells
            int nbr_cells[9];
            int n_nbr = cell_list.neighborCells(cx, cy, nbr_cells);

            // Find closest node from a different capsule
            Real min_gap = h_threshold_;
            int best_cj = -1, best_nj = -1;
            Vec2d best_normal{0, 0};

            for (int nc = 0; nc < n_nbr; ++nc) {
                const auto& nodes = cell_list.cell(nbr_cells[nc] % cell_list.ncx(),
                                                    nbr_cells[nc] / cell_list.ncx());
                for (const auto& ref : nodes) {
                    if (ref.capsule == ci) continue; // skip self-capsule

                    Vec2d pj = capsules[ref.capsule].nodePosition(ref.node);
                    Vec2d d = pj - pi;
                    if (Lx > 0.0) {
                        if (d.x >  0.5 * Lx) d.x -= Lx;
                        if (d.x < -0.5 * Lx) d.x += Lx;
                    }
                    Real dist = d.norm();
                    if (dist < min_gap && dist > 1e-15) {
                        min_gap = dist;
                        best_cj = ref.capsule;
                        best_nj = ref.node;
                        best_normal = d / dist;
                    }
                }
            }

            if (best_cj < 0 || min_gap >= h_threshold_) continue;

            // Regularize gap
            Real h = std::max(min_gap, h_min_);
            Real rJ = capsules[best_cj].effectiveRadius();
            Real a_eff = rI * rJ / (rI + rJ);

            // Relative velocity in normal direction
            Vec2d vI = capI.nodeVelocity(ni);
            Vec2d vJ = capsules[best_cj].nodeVelocity(best_nj);
            Vec2d v_rel = vJ - vI;
            Real v_n = v_rel.dot(best_normal);

            // 2D lubrication force
            Real F_mag = -6.0 * PI * mu_ * a_eff * v_n / h;
            Vec2d F_lub = best_normal * F_mag;

            // One-sided: only write to capI (owned by this thread)
            capI.forces()[ni] -= F_lub;
        }
    }
}

void LubricationCorrection::computeWallLubrication(CapsuleSystem& capsules,
                                                     int ny) {
    const int ncaps = capsules.numCapsules();
    const Real y_bottom = 0.5;  // wall position (half-way bounce-back)
    const Real y_top = static_cast<Real>(ny) - 0.5;

#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic)
#endif
    for (int ci = 0; ci < ncaps; ++ci) {
        auto& cap = capsules[ci];
        Real a = cap.effectiveRadius();

        for (int ni = 0; ni < cap.numNodes(); ++ni) {
            Vec2d pos = cap.nodePosition(ni);
            Vec2d vel = cap.nodeVelocity(ni);

            // The wall is at rest, so the lubrication force opposes the
            // node's wall-normal velocity at either wall:
            //   F_y = -6 pi mu a v_y / h
            // (away from the wall on approach, towards it on separation).
            // An earlier version had the opposite sign, which accelerated
            // nodes into the wall.
            Real h_bot = pos.y - y_bottom;
            if (h_bot < h_threshold_ && h_bot > 0) {
                Real h = std::max(h_bot, h_min_);
                cap.forces()[ni].y += -6.0 * PI * mu_ * a * vel.y / h;
            }

            Real h_top = y_top - pos.y;
            if (h_top < h_threshold_ && h_top > 0) {
                Real h = std::max(h_top, h_min_);
                cap.forces()[ni].y += -6.0 * PI * mu_ * a * vel.y / h;
            }
        }
    }
}

} // namespace softflow
