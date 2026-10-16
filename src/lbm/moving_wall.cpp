#include "moving_wall.h"
#include "lattice.h"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace softflow {

MovingWall::MovingWall(Real top_wall_velocity, Real bottom_wall_velocity)
    : top_vel_(top_wall_velocity), bot_vel_(bottom_wall_velocity)
{}

void MovingWall::apply(LatticeField& field) {
    const int nx = field.getNx();
    const int ny = field.getNy();
    const Real cs2 = 1.0 / 3.0;

    // Bottom wall (y=0): moving with velocity (bot_vel_, 0)
    if (std::abs(bot_vel_) > 1e-15) {
#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (int x = 0; x < nx; ++x) {
            if (field.cellType(x, 0) != CellType::SOLID) continue;

            for (int q = 1; q < D2Q9::Q; ++q) {
                int xn = x - D2Q9::cx[q];
                int yn = 0 - D2Q9::cy[q];
                if (periodic_x_) xn = (xn + nx) % nx;
                if (xn < 0 || xn >= nx || yn < 0 || yn >= ny) continue;
                if (field.cellType(xn, yn) == CellType::SOLID) continue;

                Real rho = field.rho(xn, yn);
                Real eu_wall = D2Q9::cx[q] * bot_vel_; // u_wall = (bot_vel_, 0)
                field.f(xn, yn, D2Q9::opp[q]) = field.f(x, 0, q)
                    - 2.0 * D2Q9::w[q] * rho * eu_wall / cs2;
            }
        }
    }

    // Top wall (y=ny-1): moving with velocity (top_vel_, 0)
    if (std::abs(top_vel_) > 1e-15) {
        int y_top = ny - 1;
#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (int x = 0; x < nx; ++x) {
            if (field.cellType(x, y_top) != CellType::SOLID) continue;

            for (int q = 1; q < D2Q9::Q; ++q) {
                int xn = x - D2Q9::cx[q];
                int yn = y_top - D2Q9::cy[q];
                if (periodic_x_) xn = (xn + nx) % nx;
                if (xn < 0 || xn >= nx || yn < 0 || yn >= ny) continue;
                if (field.cellType(xn, yn) == CellType::SOLID) continue;

                Real rho = field.rho(xn, yn);
                Real eu_wall = D2Q9::cx[q] * top_vel_;
                field.f(xn, yn, D2Q9::opp[q]) = field.f(x, y_top, q)
                    - 2.0 * D2Q9::w[q] * rho * eu_wall / cs2;
            }
        }
    }
}

} // namespace softflow
