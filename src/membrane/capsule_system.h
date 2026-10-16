#pragma once
#include "capsule.h"
#include <vector>

namespace softflow {

class CapsuleSystem {
public:
    void addCapsule(Vec2d center, Real radius, int num_nodes,
                    const MembraneParams& params, int type = 0);

    /// Periodic length in x (0 = not periodic).  Applied to every capsule now in
    /// the system and to every capsule added later, so the order of
    /// initialize() and addCapsule() cannot change the physics.
    void setPeriodicX(int nx) {
        periodic_nx_ = nx;
        for (auto& c : capsules_) c.setPeriodicX(nx);
    }
    int periodicX() const { return periodic_nx_; }

    void computeAllMembraneForces();
    void clearAllForces();
    void moveAllNodes(Real dt);

    int numCapsules() const { return static_cast<int>(capsules_.size()); }
    int totalNodes() const;

    Capsule& operator[](int i) { return capsules_[i]; }
    const Capsule& operator[](int i) const { return capsules_[i]; }

    std::vector<Capsule>& capsules() { return capsules_; }
    const std::vector<Capsule>& capsules() const { return capsules_; }

private:
    std::vector<Capsule> capsules_;
    int next_id_ = 0;
    int periodic_nx_ = 0;
};

} // namespace softflow
