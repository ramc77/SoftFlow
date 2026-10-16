#pragma once
#include "lattice_field.h"
#include <vector>

namespace softflow {

class BounceBack {
public:
    // Apply halfway bounce-back for all SOLID cells.
    // Must be called after streaming (on the post-stream distributions).

    // Original method: iterates all nodes checking type (fallback)
    void apply(LatticeField& field);

    // Optimized method: uses precomputed list of solid node indices
    // Avoids per-step type-checking over all nodes.
    void apply(LatticeField& field, const std::vector<int>& solid_nodes);

    // Periodic directions: the fluid neighbour of a wall node across a
    // periodic boundary must be found by wrapping, otherwise the diagonal
    // populations at the seam corners are never reflected.
    void setPeriodic(bool periodic_x, bool periodic_y) {
        periodic_x_ = periodic_x;
        periodic_y_ = periodic_y;
    }

private:
    bool periodic_x_ = false;
    bool periodic_y_ = false;
};

} // namespace softflow
