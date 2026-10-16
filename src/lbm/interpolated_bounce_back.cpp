#include "interpolated_bounce_back.h"
#include "lattice.h"
#include <cmath>

namespace softflow {

void InterpolatedBounceBack::initialize(
    const LatticeField& field,
    const std::vector<std::shared_ptr<Obstacle>>& obstacles) {

    links_.clear();
    const int nx = field.getNx();
    const int ny = field.getNy();

    for (int y = 0; y < ny; ++y) {
        for (int x = 0; x < nx; ++x) {
            if (field.cellType(x, y) == CellType::SOLID) continue;

            for (int q = 1; q < D2Q9::Q; ++q) {
                int xn = x + D2Q9::cx[q];
                int yn = y + D2Q9::cy[q];

                if (xn < 0 || xn >= nx || yn < 0 || yn >= ny) continue;
                if (field.cellType(xn, yn) != CellType::SOLID) continue;

                // This fluid node has a link pointing into a solid node.
                // Find the fractional distance to the wall.
                Real x0 = static_cast<Real>(x);
                Real y0 = static_cast<Real>(y);
                Real dx = static_cast<Real>(D2Q9::cx[q]);
                Real dy = static_cast<Real>(D2Q9::cy[q]);

                // Find closest obstacle to determine exact wall position
                Real min_q = 1.0; // default: wall at neighbor node
                for (const auto& obs : obstacles) {
                    // Binary search for wall intersection along the link
                    Real lo = 0.0, hi = 1.0;
                    for (int iter = 0; iter < 20; ++iter) {
                        Real mid = 0.5 * (lo + hi);
                        Real xm = x0 + mid * dx;
                        Real ym = y0 + mid * dy;
                        if (obs->contains(xm, ym)) {
                            hi = mid;
                        } else {
                            lo = mid;
                        }
                    }
                    if (hi < min_q) min_q = hi;
                }

                links_.push_back({x, y, q, min_q});
            }
        }
    }
}

void InterpolatedBounceBack::apply(LatticeField& field) {
    // Bouzidi linear interpolation:
    // q_frac < 0.5: f(x, opp[q]) = 2*q*f*(x, q) + (1-2*q)*f*(x-e, q)
    // q_frac >= 0.5: f(x, opp[q]) = (1/(2*q))*f*(x, q) + (1-1/(2*q))*f(x, opp[q])

    for (const auto& link : links_) {
        int x = link.x_fluid;
        int y = link.y_fluid;
        int q = link.q;
        Real qf = link.q_frac;
        int opp = D2Q9::opp[q];

        if (qf < 0.5) {
            // Need the upstream node (x − e_q). When the obstacle sits
            // on the periodic-x seam (e.g. a circular obstacle whose
            // bounding box crosses x=0 or x=nx−1) the upstream link can
            // fall outside the lattice; wrap it through periodic_nx_
            // before falling back to halfway BB. The y axis stays
            // bounded because the channel walls are at y∈{0, ny−1}.
            int xu = x - D2Q9::cx[q];
            int yu = y - D2Q9::cy[q];
            if (periodic_nx_ > 0) {
                xu = ((xu % periodic_nx_) + periodic_nx_) % periodic_nx_;
            }
            if (xu >= 0 && xu < field.getNx() &&
                yu >= 0 && yu < field.getNy()) {
                field.f(x, y, opp) = 2.0 * qf * field.f(x, y, q)
                    + (1.0 - 2.0 * qf) * field.f(xu, yu, q);
            } else {
                // Truly out of bounds (y direction at non-periodic
                // domain edge): fall back to halfway BB.
                field.f(x, y, opp) = field.f(x, y, q);
            }
        } else {
            Real inv2q = 1.0 / (2.0 * qf);
            field.f(x, y, opp) = inv2q * field.f(x, y, q)
                + (1.0 - inv2q) * field.f(x, y, opp);
        }
    }
}

} // namespace softflow
