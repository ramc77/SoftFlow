#include "viscosity_field.h"
#include "../membrane/capsule_system.h"
#include <cmath>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace softflow {

ViscosityField::ViscosityField(int nx, int ny, Real tau_out)
    : nx_(nx), ny_(ny), tau_out_(tau_out)
{
    tau_local_.resize(nx * ny, tau_out);
}

bool ViscosityField::isInsideCapsule(Real x, Real y,
                                      const std::vector<Vec2d>& nodes,
                                      int periodic_nx) const {
    // Ray-casting algorithm for point-in-polygon
    int n = static_cast<int>(nodes.size());
    bool inside = false;

    for (int i = 0, j = n - 1; i < n; j = i++) {
        Vec2d vi = nodes[i];
        Vec2d vj = nodes[j];

        // Handle periodic wrapping
        if (periodic_nx > 0) {
            Real Lx = static_cast<Real>(periodic_nx);
            // Unwrap relative to point
            Real dx_i = vi.x - x;
            if (dx_i >  0.5 * Lx) vi.x -= Lx;
            if (dx_i < -0.5 * Lx) vi.x += Lx;
            Real dx_j = vj.x - x;
            if (dx_j >  0.5 * Lx) vj.x -= Lx;
            if (dx_j < -0.5 * Lx) vj.x += Lx;
        }

        if (((vi.y > y) != (vj.y > y)) &&
            (x < (vj.x - vi.x) * (y - vi.y) / (vj.y - vi.y) + vi.x)) {
            inside = !inside;
        }
    }
    return inside;
}

void ViscosityField::update(const CapsuleSystem& capsules) {
    int ncaps = capsules.numCapsules();

    // Reset to default tau
    std::fill(tau_local_.begin(), tau_local_.end(), tau_out_);

    if (ncaps == 0) return;

    Real nu_out = (tau_out_ - 0.5) / 3.0;

#ifdef _OPENMP
    #pragma omp parallel for collapse(2) schedule(dynamic)
#endif
    for (int y = 1; y < ny_ - 1; ++y) {
        for (int x = 0; x < nx_; ++x) {
            Real px = static_cast<Real>(x);
            Real py = static_cast<Real>(y);

            for (int c = 0; c < ncaps; ++c) {
                const auto& cap = capsules[c];

                // Quick bounding check
                Vec2d cen = cap.centroid();
                Real r = cap.effectiveRadius();
                Real dx = px - cen.x;
                Real dy = py - cen.y;
                if (dx * dx + dy * dy > (r + 2.0) * (r + 2.0)) continue;

                if (isInsideCapsule(px, py, cap.positions(),
                                     cap.getPeriodicX())) {
                    Real lambda = cap.getViscosityRatio();
                    Real nu_in = lambda * nu_out;
                    Real tau_in = 3.0 * nu_in + 0.5;
                    tau_local_[y * nx_ + x] = tau_in;
                    break; // node can only be inside one capsule
                }
            }
        }
    }
}

} // namespace softflow
