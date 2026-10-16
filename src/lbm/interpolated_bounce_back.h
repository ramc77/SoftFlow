#pragma once
#include "lattice_field.h"
#include "../geometry/obstacle.h"
#include <vector>
#include <memory>

namespace softflow {

/// Interpolated bounce-back (Bouzidi et al., Phys. Fluids 2001).
/// Uses linear interpolation based on the fractional distance q
/// from the fluid node to the wall. Much more accurate than
/// staircase bounce-back for curved boundaries.
class InterpolatedBounceBack {
public:
    /// Pre-compute fractional distances for all fluid-solid links
    void initialize(const LatticeField& field,
                    const std::vector<std::shared_ptr<Obstacle>>& obstacles);

    /// Apply IBB after streaming
    void apply(LatticeField& field);

    /// Enable periodic-x wrap so the q<½ upstream-link fallback wraps
    /// rather than collapsing to halfway BB when an obstacle sits on
    /// the streamwise periodic seam. Pass 0 to disable. By default the
    /// IBB assumes a non-periodic domain.
    void setPeriodicX(int nx) { periodic_nx_ = nx; }

private:
    struct IBBLink {
        int x_fluid, y_fluid;  // fluid node position
        int q;                  // direction pointing into solid
        Real q_frac;            // fractional distance to wall [0, 1]
    };

    std::vector<IBBLink> links_;
    int periodic_nx_ = 0;       // 0 = non-periodic; >0 = wrap modulus
};

} // namespace softflow
