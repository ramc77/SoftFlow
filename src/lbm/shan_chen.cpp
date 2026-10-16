#include "shan_chen.h"
#include "lattice.h"
#include "lattice_field.h"
#include <cmath>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace softflow {

ShanChen::ShanChen(const ShanChenParams& params, BoundaryType bc)
    : G_(params.G), rho0_(params.rho0_sc), bc_(bc),
      eos_type_(params.eos_type),
      cs_a_(params.cs_a), cs_b_(params.cs_b),
      cs_T_(params.cs_T), cs_R_(params.cs_R)
{}

// Carnahan-Starling equation of state:
//   p = rho*R*T * (1 + eta + eta^2 - eta^3) / (1 - eta)^3 - a*rho^2
// where eta = b*rho/4 is the packing fraction.
// Reference: Yuan & Schaefer, Phys. Fluids 18, 042101 (2006)
Real ShanChen::p_carnahan_starling(Real rho) const {
    Real eta = cs_b_ * rho / 4.0;
    eta = std::min(eta, 0.95);  // clamp to avoid singularity
    Real eta2 = eta * eta;
    Real eta3 = eta2 * eta;
    Real one_minus_eta = 1.0 - eta;
    Real ome3 = one_minus_eta * one_minus_eta * one_minus_eta;

    Real p_repulsive = rho * cs_R_ * cs_T_ * (1.0 + eta + eta2 - eta3) / ome3;
    Real p_attractive = -cs_a_ * rho * rho;
    return p_repulsive + p_attractive;
}

Real ShanChen::psi(Real rho) const {
    if (rho < 1e-12) return 0.0;

    if (eos_type_ == 1) {
        // Yuan-Schaefer pseudopotential from Carnahan-Starling EOS:
        //   psi = sqrt(2*(p_EOS - rho*cs2) / (G*cs2))
        // For G < 0, argument is positive when p_EOS < rho*cs2 (two-phase region).
        Real p_eos = p_carnahan_starling(rho);
        Real cs2 = D2Q9::cs2;  // 1/3
        Real arg = 2.0 * (p_eos - rho * cs2) / (G_ * cs2);
        if (arg < 0.0) arg = 0.0;  // clamp outside two-phase region
        return std::sqrt(arg);
    }

    // EOS 0: Original Shan-Chen exponential
    return rho0_ * (1.0 - std::exp(-rho / rho0_));
}

void ShanChen::computeForce(LatticeField& field) {
    int nx = field.getNx();
    int ny = field.getNy();
    bool periodic_x = (bc_ == BoundaryType::PERIODIC);

    #ifdef _OPENMP
    #pragma omp parallel for collapse(2) schedule(static)
    #endif
    for (int y = 1; y < ny - 1; ++y) {
        for (int x = 0; x < nx; ++x) {
            if (field.getCellType(x, y) != CellType::FLUID) continue;

            Real psi_here = psi(field.getRho(x, y));
            Real Fx = 0.0, Fy = 0.0;

            // F_SC(x) = -G * psi(x) * sum_q w_q * psi(x + e_q) * e_q
            for (int q = 1; q < D2Q9::Q; ++q) {
                int xn = x + D2Q9::cx[q];
                int yn = y + D2Q9::cy[q];

                // X-direction wrapping depends on boundary type
                if (periodic_x) {
                    if (xn < 0) xn += nx;
                    if (xn >= nx) xn -= nx;
                } else {
                    // Non-periodic (CLOSED, INLET_OUTLET): skip out-of-bounds
                    if (xn < 0 || xn >= nx) continue;
                }
                // Clamp in y
                if (yn < 0 || yn >= ny) continue;

                if (field.getCellType(xn, yn) == CellType::SOLID) {
                    // Solid nodes: use local density for neutral wetting (90° contact)
                    Fx += D2Q9::w[q] * psi(field.getRho(x, y)) * D2Q9::cx[q];
                    Fy += D2Q9::w[q] * psi(field.getRho(x, y)) * D2Q9::cy[q];
                } else {
                    Real psi_n = psi(field.getRho(xn, yn));
                    Fx += D2Q9::w[q] * psi_n * D2Q9::cx[q];
                    Fy += D2Q9::w[q] * psi_n * D2Q9::cy[q];
                }
            }

            Fx *= -G_ * psi_here;
            Fy *= -G_ * psi_here;

            field.addExternalForce(x, y, Fx, Fy);
        }
    }
}

} // namespace softflow
