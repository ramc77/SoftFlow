#pragma once
#include "lattice_field.h"
#include "../core/parameters.h"
#include <vector>

namespace softflow {

class ZouHeBoundary {
public:
    ZouHeBoundary(Real inlet_velocity, Real outlet_density);

    void setInletVelocity(Real ux) { inlet_ux_ = ux; }
    void setOutletDensity(Real rho) { outlet_rho_ = rho; }

    // Apply Zou-He conditions: velocity inlet (left), pressure outlet (right)
    void apply(LatticeField& field);

    // Optimized: use precomputed inlet/outlet node lists
    void apply(LatticeField& field,
               const std::vector<int>& inlet_nodes,
               const std::vector<int>& outlet_nodes);

    void applyInletLeft(LatticeField& field);
    void applyOutletRight(LatticeField& field);

    Real getInletVelocity() const { return inlet_ux_; }
    Real getOutletDensity() const { return outlet_rho_; }

private:
    Real inlet_ux_;
    Real outlet_rho_;
};

} // namespace softflow
