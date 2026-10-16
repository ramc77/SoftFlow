#include "adhesion.h"
#include "capsule_system.h"
#include <cmath>
#include <algorithm>
#include <iostream>

namespace softflow {

AdhesionModel::AdhesionModel(const AdhesionParams& params, unsigned seed)
    : params_(params), rng_(seed)
{
    // Phase-1 behaviour change (the project specification / the design review):
    // an empty adhesion_matrix used to mean "every type bonds with
    // every other type" — a silent foot-gun that produced unintended
    // adhesion when users enabled the model without configuring
    // type pairs. The new contract is "empty matrix means no
    // adhesion". The change is announced loudly the first time the
    // model is constructed so old scripts surface the regression
    // immediately, with a clear migration message.
    if (params_.enabled && params_.adhesion_matrix.empty()) {
        std::cerr
            << "*** AdhesionModel: adhesion is enabled but adhesion_matrix "
            << "is empty. No bonds will form. Set "
            << "AdhesionParams::adhesion_matrix[i][j] = true for each pair "
            << "(i, j) of capsule types that should be allowed to bond."
            << std::endl;
    }
}

bool AdhesionModel::canBond(int type_i, int type_j) const {
    // Empty matrix => no bonds (Phase-1 change). See the constructor
    // for the migration note. Out-of-range indices also yield no bond.
    if (params_.adhesion_matrix.empty()) return false;
    const int n = static_cast<int>(params_.adhesion_matrix.size());
    if (type_i < 0 || type_j < 0 || type_i >= n || type_j >= n) return false;
    return params_.adhesion_matrix[type_i][type_j];
}

void AdhesionModel::update(CapsuleSystem& capsules, Real dt, int ny,
                            int periodic_nx) {
    int ncaps = capsules.numCapsules();

    // Initialize bond count tracking
    bonds_per_node_.resize(ncaps);
    for (int c = 0; c < ncaps; ++c) {
        bonds_per_node_[c].assign(capsules[c].numNodes(), 0);
    }
    // Count existing bonds
    for (const auto& b : bonds_) {
        if (b.capsule_i >= 0 && b.capsule_i < ncaps)
            bonds_per_node_[b.capsule_i][b.node_i]++;
        if (b.capsule_j >= 0 && b.capsule_j < ncaps)
            bonds_per_node_[b.capsule_j][b.node_j]++;
    }

    tryBreakBonds(dt);
    tryFormBonds(capsules, dt, periodic_nx);
    if (params_.wall_adhesion) {
        tryFormWallBonds(capsules, dt, ny);
    }
    computeBondForces(capsules, periodic_nx);
    detectClusters(ncaps);
}

void AdhesionModel::tryFormBonds(CapsuleSystem& capsules, Real dt,
                                  int periodic_nx) {
    int ncaps = capsules.numCapsules();
    std::uniform_real_distribution<Real> dist(0.0, 1.0);
    Real p_on = params_.k_on * dt;

    for (int ci = 0; ci < ncaps; ++ci) {
        auto& capI = capsules[ci];
        for (int cj = ci + 1; cj < ncaps; ++cj) {
            auto& capJ = capsules[cj];

            if (!canBond(capI.getType(), capJ.getType())) continue;

            // Quick centroid distance check
            Vec2d dc = capJ.centroid() - capI.centroid();
            if (periodic_nx > 0) {
                Real Lx = static_cast<Real>(periodic_nx);
                if (dc.x >  0.5 * Lx) dc.x -= Lx;
                if (dc.x < -0.5 * Lx) dc.x += Lx;
            }
            Real rI = capI.effectiveRadius();
            Real rJ = capJ.effectiveRadius();
            if (dc.norm() > rI + rJ + params_.d_bond) continue;

            for (int ni = 0; ni < capI.numNodes(); ++ni) {
                if (bonds_per_node_[ci][ni] >= params_.max_bonds_per_node) continue;

                Vec2d pi = capI.nodePosition(ni);
                for (int nj = 0; nj < capJ.numNodes(); ++nj) {
                    if (bonds_per_node_[cj][nj] >= params_.max_bonds_per_node) continue;

                    Vec2d pj = capJ.nodePosition(nj);
                    Vec2d d = pj - pi;
                    if (periodic_nx > 0) {
                        Real Lx = static_cast<Real>(periodic_nx);
                        if (d.x >  0.5 * Lx) d.x -= Lx;
                        if (d.x < -0.5 * Lx) d.x += Lx;
                    }
                    Real gap = d.norm();

                    if (gap < params_.d_bond && dist(rng_) < p_on) {
                        // Check if bond already exists
                        bool exists = false;
                        for (const auto& b : bonds_) {
                            if (b.capsule_i == ci && b.node_i == ni &&
                                b.capsule_j == cj && b.node_j == nj) {
                                exists = true;
                                break;
                            }
                        }
                        if (!exists) {
                            bonds_.push_back({ci, ni, cj, nj, gap, 0.0});
                            bonds_per_node_[ci][ni]++;
                            bonds_per_node_[cj][nj]++;
                        }
                    }
                }
            }
        }
    }
}

void AdhesionModel::tryFormWallBonds(CapsuleSystem& capsules, Real dt, int ny) {
    int ncaps = capsules.numCapsules();
    std::uniform_real_distribution<Real> dist(0.0, 1.0);
    Real p_on = params_.wall_k_on * dt;
    Real y_bot = 0.5;
    Real y_top = static_cast<Real>(ny) - 0.5;

    for (int ci = 0; ci < ncaps; ++ci) {
        auto& cap = capsules[ci];
        for (int ni = 0; ni < cap.numNodes(); ++ni) {
            if (bonds_per_node_[ci][ni] >= params_.max_bonds_per_node) continue;

            Vec2d pos = cap.nodePosition(ni);
            Real h_bot = pos.y - y_bot;
            Real h_top = y_top - pos.y;

            if (h_bot < params_.d_bond && h_bot > 0 && dist(rng_) < p_on) {
                bonds_.push_back({ci, ni, -1, 0, h_bot, 0.0}); // -1 = wall
                bonds_per_node_[ci][ni]++;
            }
            if (h_top < params_.d_bond && h_top > 0 && dist(rng_) < p_on) {
                bonds_.push_back({ci, ni, -2, 0, h_top, 0.0}); // -2 = top wall
                bonds_per_node_[ci][ni]++;
            }
        }
    }
}

void AdhesionModel::tryBreakBonds(Real dt) {
    std::uniform_real_distribution<Real> dist(0.0, 1.0);

    bonds_.erase(
        std::remove_if(bonds_.begin(), bonds_.end(),
            [&](const Bond& b) {
                Real F = b.current_force;

                Real p_off;
                if (params_.use_catch_slip) {
                    // Catch-slip bond model (Thomas et al. 2008, Pereverzev et al. 2005)
                    //
                    // k_off(F) = k_catch * exp(-F/F_catch) + k_slip * exp(F/F_slip)
                    //
                    // At low force: catch pathway dominates → k_off DECREASES
                    //   (bond is strengthened by moderate force)
                    // At high force: slip pathway dominates → k_off INCREASES
                    //   (bond breaks under large force)
                    // This creates a biphasic lifetime curve with maximum
                    // bond lifetime at an intermediate "optimal" force.
                    //
                    // Biological examples: P-selectin/PSGL-1, FimH/mannose
                    Real k_catch = params_.k_off_catch;
                    Real k_slip  = params_.k_off_slip;
                    Real F_c = params_.F_catch;
                    Real F_s = params_.F_slip;

                    Real rate = k_catch * std::exp(-F / F_c) +
                                k_slip  * std::exp( F / F_s);
                    p_off = rate * dt;
                } else {
                    // Standard Bell model: k_off(F) = k_off * exp(F/F_crit)
                    Real k_off = (b.capsule_j >= 0) ? params_.k_off : params_.wall_k_off;
                    p_off = k_off * std::exp(F / params_.F_crit) * dt;
                }

                return dist(rng_) < p_off;
            }),
        bonds_.end()
    );
}

void AdhesionModel::computeBondForces(CapsuleSystem& capsules,
                                       int periodic_nx) {
    for (auto& bond : bonds_) {
        auto& capI = capsules[bond.capsule_i];
        Vec2d pi = capI.nodePosition(bond.node_i);

        Vec2d pj;
        if (bond.capsule_j >= 0) {
            // Cell-cell bond
            pj = capsules[bond.capsule_j].nodePosition(bond.node_j);
        } else if (bond.capsule_j == -1) {
            // Bottom wall bond
            pj = Vec2d{pi.x, 0.5};
        } else {
            // Top wall bond (capsule_j == -2)
            // ny not available here, use rest_length to approximate
            pj = Vec2d{pi.x, pi.y + bond.rest_length};
        }

        Vec2d d = pj - pi;
        if (periodic_nx > 0) {
            Real Lx = static_cast<Real>(periodic_nx);
            if (d.x >  0.5 * Lx) d.x -= Lx;
            if (d.x < -0.5 * Lx) d.x += Lx;
        }
        Real dist = d.norm();
        if (dist < 1e-15) continue;

        Real k = (bond.capsule_j >= 0) ? params_.k_bond : params_.wall_k_bond;
        Real F_mag = k * (dist - bond.rest_length);
        bond.current_force = std::abs(F_mag);

        Vec2d F = (d / dist) * F_mag;
        capI.forces()[bond.node_i] += F;
        if (bond.capsule_j >= 0) {
            capsules[bond.capsule_j].forces()[bond.node_j] -= F;
        }
    }
}

int AdhesionModel::getBondsForCapsule(int capsule_id) const {
    int count = 0;
    for (const auto& b : bonds_) {
        if (b.capsule_i == capsule_id || b.capsule_j == capsule_id) count++;
    }
    return count;
}

// Union-find for cluster detection
int AdhesionModel::find(int x) {
    while (parent_[x] != x) {
        parent_[x] = parent_[parent_[x]]; // path compression
        x = parent_[x];
    }
    return x;
}

void AdhesionModel::unite(int x, int y) {
    int rx = find(x), ry = find(y);
    if (rx != ry) parent_[rx] = ry;
}

void AdhesionModel::detectClusters(int ncaps) {
    parent_.resize(ncaps);
    for (int i = 0; i < ncaps; ++i) parent_[i] = i;

    // Unite capsules connected by bonds
    for (const auto& b : bonds_) {
        if (b.capsule_j >= 0 && b.capsule_j < ncaps) {
            unite(b.capsule_i, b.capsule_j);
        }
    }

    // Assign cluster IDs
    cluster_ids_.resize(ncaps);
    std::vector<int> root_to_cluster(ncaps, -1);
    num_clusters_ = 0;
    for (int i = 0; i < ncaps; ++i) {
        int root = find(i);
        if (root_to_cluster[root] < 0) {
            root_to_cluster[root] = num_clusters_++;
        }
        cluster_ids_[i] = root_to_cluster[root];
    }

    // Compute cluster sizes
    cluster_sizes_.assign(num_clusters_, 0);
    for (int i = 0; i < ncaps; ++i) {
        cluster_sizes_[cluster_ids_[i]]++;
    }
}

} // namespace softflow
