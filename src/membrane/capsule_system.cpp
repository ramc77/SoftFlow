#include "capsule_system.h"

namespace softflow {

void CapsuleSystem::addCapsule(Vec2d center, Real radius, int num_nodes,
                                const MembraneParams& params, int type) {
    capsules_.emplace_back(next_id_++, center, radius, num_nodes, params, type);
    if (periodic_nx_ > 0) capsules_.back().setPeriodicX(periodic_nx_);
}

void CapsuleSystem::computeAllMembraneForces() {
    int n = static_cast<int>(capsules_.size());
#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic)
#endif
    for (int i = 0; i < n; ++i) {
        capsules_[i].computeMembraneForces();
    }
}

void CapsuleSystem::clearAllForces() {
    int n = static_cast<int>(capsules_.size());
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < n; ++i) {
        capsules_[i].clearForces();
    }
}

void CapsuleSystem::moveAllNodes(Real dt) {
    int n = static_cast<int>(capsules_.size());
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < n; ++i) {
        capsules_[i].moveNodes(dt);
    }
}

int CapsuleSystem::totalNodes() const {
    int total = 0;
    for (const auto& c : capsules_) {
        total += c.numNodes();
    }
    return total;
}

} // namespace softflow
