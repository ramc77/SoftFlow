#pragma once
#include "../core/types.h"
#include "../core/parameters.h"

namespace softflow {

class CapsuleSystem;

/// Lubrication correction forces for near-contact interactions.
/// When the gap between two surfaces is smaller than the LBM can resolve
/// (~1.5 lattice units), analytical corrections are applied.
/// Reference: Aidun & Clausen, Annu. Rev. Fluid Mech. 2010
class LubricationCorrection {
public:
    explicit LubricationCorrection(const LubricationParams& params, Real kinematic_viscosity);

    /// Compute lubrication corrections between all close capsule pairs
    /// and between capsules and walls.
    void computeAll(CapsuleSystem& capsules, int ny, int periodic_nx);

private:
    Real h_threshold_;  // max gap for correction
    Real h_min_;        // regularization cutoff
    Real mu_;           // dynamic viscosity = rho * nu

    /// Capsule-capsule lubrication between two close nodes
    void computeCapsulePairLubrication(CapsuleSystem& capsules, int periodic_nx);

    /// Capsule-wall lubrication
    void computeWallLubrication(CapsuleSystem& capsules, int ny);
};

} // namespace softflow
