#pragma once
#include "../core/types.h"
#include <string>

namespace softflow {

class Simulation;

/// Save and load complete simulation state for checkpoint/restart.
/// Binary format: distributions, capsule positions/velocities, bonds, scalar fields.
class Checkpoint {
public:
    /// Save full simulation state to binary file
    static bool save(const Simulation& sim, const std::string& filename);

    /// Load simulation state from checkpoint file
    static bool load(Simulation& sim, const std::string& filename);
};

} // namespace softflow
