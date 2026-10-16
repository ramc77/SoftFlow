#pragma once
#include "../core/types.h"
#include "../core/parameters.h"
#include <vector>
#include <random>

namespace softflow {

class CapsuleSystem;

/// A single adhesion bond between two surface nodes.
struct Bond {
    int capsule_i;   // first capsule index
    int node_i;      // node on first capsule
    int capsule_j;   // second capsule index (-1 for wall bond)
    int node_j;      // node on second capsule (or wall receptor index)
    Real rest_length; // equilibrium bond length
    Real current_force; // current tensile force magnitude
};

/// Cell adhesion model based on Bell (1978) receptor-ligand kinetics.
/// Supports cell-cell and cell-wall adhesion with stochastic bond
/// formation/breaking and force-dependent dissociation.
class AdhesionModel {
public:
    explicit AdhesionModel(const AdhesionParams& params, unsigned seed = 42);

    /// Update bonds: form new, break existing, compute forces.
    void update(CapsuleSystem& capsules, Real dt, int ny, int periodic_nx);

    /// Access bond list for output
    const std::vector<Bond>& getBonds() const { return bonds_; }
    std::vector<Bond>& getBondsMutable() { return bonds_; }
    int getNumBonds() const { return static_cast<int>(bonds_.size()); }

    /// Get cluster assignments (computed during update via union-find)
    const std::vector<int>& getClusterIds() const { return cluster_ids_; }
    const std::vector<int>& getClusterSizes() const { return cluster_sizes_; }
    int getNumClusters() const { return num_clusters_; }

    /// Count bonds for a specific capsule
    int getBondsForCapsule(int capsule_id) const;

private:
    AdhesionParams params_;
    std::vector<Bond> bonds_;
    std::mt19937 rng_;

    // Cluster detection (union-find)
    std::vector<int> cluster_ids_;
    std::vector<int> cluster_sizes_;
    int num_clusters_ = 0;

    // Count bonds per node to enforce max_bonds_per_node
    std::vector<std::vector<int>> bonds_per_node_; // [capsule][node] = count

    /// Check if types i,j can form bonds
    bool canBond(int type_i, int type_j) const;

    /// Try forming new bonds between close nodes
    void tryFormBonds(CapsuleSystem& capsules, Real dt, int periodic_nx);

    /// Try forming bonds with walls
    void tryFormWallBonds(CapsuleSystem& capsules, Real dt, int ny);

    /// Break bonds based on force-dependent dissociation
    void tryBreakBonds(Real dt);

    /// Compute spring forces from active bonds
    void computeBondForces(CapsuleSystem& capsules, int periodic_nx);

    /// Detect clusters using union-find on bonded capsules
    void detectClusters(int ncaps);

    // Union-find helpers
    std::vector<int> parent_;
    int find(int x);
    void unite(int x, int y);
};

} // namespace softflow
