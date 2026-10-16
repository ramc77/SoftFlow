#pragma once
#include "lattice_field.h"
#include "../core/parameters.h"

namespace softflow {

class BGKCollision {
public:
    explicit BGKCollision(Real tau);

    // Set relaxation time
    void setTau(Real tau) { tau_ = tau; omega_ = 1.0 / tau; }
    Real getTau() const { return tau_; }

    // Perform BGK collision with Guo forcing on entire field
    void collide(LatticeField& field);

private:
    Real tau_;    // relaxation time
    Real omega_;  // 1/tau
};

} // namespace softflow
