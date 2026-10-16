#include "bounce_back.h"
#include "lattice.h"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace softflow {

// Optimized bounce-back using precomputed solid node list.
// Only iterates over known SOLID cells — no type-check branching.
void BounceBack::apply(LatticeField& field, const std::vector<int>& solid_nodes) {
    const int nx = field.getNx();
    const int ny = field.getNy();
    const int ns = static_cast<int>(solid_nodes.size());

#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int si = 0; si < ns; ++si) {
        int n = solid_nodes[si];
        int x = n % nx;
        int y = n / nx;

        for (int q = 1; q < D2Q9::Q; ++q) {  // skip rest direction
            int xn = x - D2Q9::cx[q];
            int yn = y - D2Q9::cy[q];
            if (periodic_x_) xn = (xn + nx) % nx;
            if (periodic_y_) yn = (yn + ny) % ny;

            if (xn < 0 || xn >= nx || yn < 0 || yn >= ny) continue;
            {
                CellType ct_n = field.cellType(xn, yn);
                if (ct_n == CellType::SOLID || ct_n == CellType::EMPTY) continue;
            }

            // Reflect: population heading into solid/empty → bounce back
            field.f(xn, yn, D2Q9::opp[q]) = field.f(x, y, q);
        }
    }
}

// Fallback: iterate all nodes (for backward compatibility)
void BounceBack::apply(LatticeField& field) {
    const int nx = field.getNx();
    const int ny = field.getNy();

#ifdef _OPENMP
    #pragma omp parallel for collapse(2) schedule(static)
#endif
    for (int y = 0; y < ny; ++y) {
        for (int x = 0; x < nx; ++x) {
            if (field.cellType(x, y) != CellType::SOLID) continue;

            for (int q = 1; q < D2Q9::Q; ++q) {
                int xn = x - D2Q9::cx[q];
                int yn = y - D2Q9::cy[q];
                if (periodic_x_) xn = (xn + nx) % nx;
                if (periodic_y_) yn = (yn + ny) % ny;
                if (xn < 0 || xn >= nx || yn < 0 || yn >= ny) continue;
                if (field.cellType(xn, yn) == CellType::SOLID) continue;
                field.f(xn, yn, D2Q9::opp[q]) = field.f(x, y, q);
            }
        }
    }
}

} // namespace softflow
