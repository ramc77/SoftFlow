#include "lattice_field.h"
#include "lattice.h"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace softflow {

LatticeField::LatticeField(int nx, int ny)
    : nx_(nx), ny_(ny), N_(nx * ny),
      f_(static_cast<size_t>(nx) * ny * D2Q9::Q, 0.0),
      f_tmp_(static_cast<size_t>(nx) * ny * D2Q9::Q, 0.0),
      rho_(static_cast<size_t>(nx) * ny, 0.0),
      ux_(static_cast<size_t>(nx) * ny, 0.0),
      uy_(static_cast<size_t>(nx) * ny, 0.0),
      Fx_(static_cast<size_t>(nx) * ny, 0.0),
      Fy_(static_cast<size_t>(nx) * ny, 0.0),
      flags_(static_cast<size_t>(nx) * ny, CellType::FLUID)
{
}

void LatticeField::clearForces() {
    Fx_.fill(0.0);
    Fy_.fill(0.0);
}

void LatticeField::computeMacroscopic() {
    const int N = N_;
    // Gather pointers to each q-plane for direct access
    const Real* fp[D2Q9::Q];
    for (int q = 0; q < D2Q9::Q; ++q) {
        fp[q] = f_.data() + q * N;
    }

    Real* __restrict__ rho_ptr = rho_.data();
    Real* __restrict__ ux_ptr  = ux_.data();
    Real* __restrict__ uy_ptr  = uy_.data();
    const Real* __restrict__ fx_ptr = Fx_.data();
    const Real* __restrict__ fy_ptr = Fy_.data();
    const CellType* __restrict__ fl = flags_.data();

#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int n = 0; n < N; ++n) {
        if (fl[n] == CellType::SOLID || fl[n] == CellType::EMPTY) {
            rho_ptr[n] = 0.0;
            ux_ptr[n]  = 0.0;
            uy_ptr[n]  = 0.0;
            continue;
        }

        // Gather f values from each q-plane (SoA layout: fp[q][n])
        Real r = 0.0, vx = 0.0, vy = 0.0;
        for (int q = 0; q < D2Q9::Q; ++q) {
            Real fi = fp[q][n];
            r  += fi;
            vx += fi * static_cast<Real>(D2Q9::cx[q]);
            vy += fi * static_cast<Real>(D2Q9::cy[q]);
        }

        rho_ptr[n] = r;
        // Include Guo forcing correction: u = (1/rho) * (sum f_i e_i + F*dt/2)
        if (r > 1e-15) {
            ux_ptr[n] = (vx + 0.5 * fx_ptr[n]) / r;
            uy_ptr[n] = (vy + 0.5 * fy_ptr[n]) / r;
        } else {
            ux_ptr[n] = 0.0;
            uy_ptr[n] = 0.0;
        }
    }
}

void LatticeField::setEquilibrium(Real rho0, Real ux0, Real uy0) {
    const int N = N_;

#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int n = 0; n < N; ++n) {
        rho_[n] = rho0;
        ux_[n]  = ux0;
        uy_[n]  = uy0;

        for (int q = 0; q < D2Q9::Q; ++q) {
            f_[q * N + n] = D2Q9::feq(q, rho0, ux0, uy0);
        }
    }
}

void LatticeField::initializeEquilibriumAt(int x, int y) {
    const int n = idx(x, y);
    const int N = N_;
    Real r  = rho_[n];
    Real vx = ux_[n];
    Real vy = uy_[n];
    for (int q = 0; q < D2Q9::Q; ++q) {
        f_[q * N + n] = D2Q9::feq(q, r, vx, vy);
    }
}

void LatticeField::initializeAllEquilibrium() {
    const int N = N_;
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int n = 0; n < N; ++n) {
        for (int q = 0; q < D2Q9::Q; ++q) {
            f_[q * N + n] = D2Q9::feq(q, rho_[n], ux_[n], uy_[n]);
        }
    }
}

void LatticeField::swapBuffers() {
    f_.swap(f_tmp_);
}

} // namespace softflow
