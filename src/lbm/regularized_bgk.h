#pragma once
#include "lattice_field.h"
#include "../core/parameters.h"

namespace softflow {

/// Regularized BGK collision operator (Latt & Chopard, Phys. Rev. E 2006).
/// Projects the non-equilibrium part of the distribution onto the second-order
/// polynomial (stress tensor) form, then relaxes. This eliminates spurious
/// higher-order non-equilibrium modes, improving stability at low viscosity.
class RegularizedBGK {
public:
    explicit RegularizedBGK(Real tau);

    /// Perform regularized collision on the entire lattice field.
    /// Includes Guo forcing term for external forces.
    void collide(LatticeField& field);

private:
    Real tau_;
    Real omega_;  // 1/tau
};

} // namespace softflow
