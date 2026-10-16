#pragma once
#include "../core/types.h"
#include <vector>

namespace softflow {

class CapsuleSystem;
class LatticeField;

/// Spatially varying viscosity field for viscosity contrast simulations.
/// Uses ray-casting to classify lattice nodes as inside or outside capsules,
/// then sets local relaxation time tau accordingly.
///
/// Inside capsule: tau_in = 3 * nu_in + 0.5 = 3 * lambda * nu_out + 0.5
/// Outside capsule: tau_out = tau (default)
class ViscosityField {
public:
    ViscosityField(int nx, int ny, Real tau_out);

    /// Update the inside/outside classification and tau_local field.
    /// Should be called every ~10 timesteps (expensive).
    void update(const CapsuleSystem& capsules);

    /// Get local tau at a lattice node
    Real getTauLocal(int x, int y) const {
        return tau_local_[y * nx_ + x];
    }

    /// Get pointer to tau_local array (for collision operator)
    const Real* tauLocalData() const { return tau_local_.data(); }
    Real* tauLocalData() { return tau_local_.data(); }

private:
    int nx_, ny_;
    Real tau_out_;
    std::vector<Real> tau_local_;

    /// Point-in-polygon test for a single capsule (ray-casting)
    bool isInsideCapsule(Real x, Real y,
                         const std::vector<Vec2d>& nodes, int periodic_nx) const;
};

} // namespace softflow
