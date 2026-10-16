#include "zou_he_boundary.h"
#include "lattice.h"

namespace softflow {

ZouHeBoundary::ZouHeBoundary(Real inlet_velocity, Real outlet_density)
    : inlet_ux_(inlet_velocity), outlet_rho_(outlet_density)
{
}

void ZouHeBoundary::apply(LatticeField& field) {
    applyInletLeft(field);
    applyOutletRight(field);
}

// Optimized: use precomputed node lists
void ZouHeBoundary::apply(LatticeField& field,
                           const std::vector<int>& inlet_nodes,
                           const std::vector<int>& outlet_nodes) {
    const int nx = field.getNx();
    const int ny = field.getNy();

    // Apply inlet on precomputed nodes
    for (int idx : inlet_nodes) {
        int x = idx % nx;
        int y = idx / nx;
        if (x != 0) continue; // Zou-He inlet only on left wall

        // Parabolic velocity profile at inlet
        Real y_center = static_cast<Real>(ny) * 0.5;
        Real H = static_cast<Real>(ny - 2);
        Real y_norm = (static_cast<Real>(y) - y_center) / (H * 0.5);
        Real ux = inlet_ux_ * (1.0 - y_norm * y_norm);
        Real uy = 0.0;

        Real f0 = field.f(x, y, 0);
        Real f2 = field.f(x, y, 2);
        Real f3 = field.f(x, y, 3);
        Real f4 = field.f(x, y, 4);
        Real f6 = field.f(x, y, 6);
        Real f7 = field.f(x, y, 7);

        Real rho = (f0 + f2 + f4 + 2.0 * (f3 + f6 + f7)) / (1.0 - ux);

        field.f(x, y, 1) = f3 + (2.0 / 3.0) * rho * ux;
        field.f(x, y, 5) = f7 + 0.5 * (f4 - f2) + (1.0 / 6.0) * rho * ux + 0.5 * rho * uy;
        field.f(x, y, 8) = f6 - 0.5 * (f4 - f2) + (1.0 / 6.0) * rho * ux - 0.5 * rho * uy;

        field.rho(x, y) = rho;
        field.ux(x, y) = ux;
        field.uy(x, y) = uy;
    }

    // Apply outlet on precomputed nodes
    for (int idx : outlet_nodes) {
        int x = idx % nx;
        int y = idx / nx;
        if (x != nx - 1) continue; // Zou-He outlet only on right wall

        Real rho = outlet_rho_;

        Real f0 = field.f(x, y, 0);
        Real f1 = field.f(x, y, 1);
        Real f2 = field.f(x, y, 2);
        Real f4 = field.f(x, y, 4);
        Real f5 = field.f(x, y, 5);
        Real f8 = field.f(x, y, 8);

        Real ux = -1.0 + (f0 + f2 + f4 + 2.0 * (f1 + f5 + f8)) / rho;
        Real uy = 0.0;

        field.f(x, y, 3) = f1 - (2.0 / 3.0) * rho * ux;
        field.f(x, y, 7) = f5 - 0.5 * (f4 - f2) - (1.0 / 6.0) * rho * ux - 0.5 * rho * uy;
        field.f(x, y, 6) = f8 + 0.5 * (f4 - f2) - (1.0 / 6.0) * rho * ux + 0.5 * rho * uy;

        field.rho(x, y) = rho;
        field.ux(x, y) = ux;
        field.uy(x, y) = uy;
    }
}

// Zou-He velocity inlet (x = 0) — proper formulation
void ZouHeBoundary::applyInletLeft(LatticeField& field) {
    const int ny = field.getNy();
    const int x = 0;

    for (int y = 1; y < ny - 1; ++y) {
        if (field.cellType(x, y) == CellType::SOLID) continue;

        Real y_center = static_cast<Real>(ny) * 0.5;
        Real H = static_cast<Real>(ny - 2);
        Real y_norm = (static_cast<Real>(y) - y_center) / (H * 0.5);
        Real ux = inlet_ux_ * (1.0 - y_norm * y_norm);
        Real uy = 0.0;

        Real f0 = field.f(x, y, 0);
        Real f2 = field.f(x, y, 2);
        Real f3 = field.f(x, y, 3);
        Real f4 = field.f(x, y, 4);
        Real f6 = field.f(x, y, 6);
        Real f7 = field.f(x, y, 7);

        Real rho = (f0 + f2 + f4 + 2.0 * (f3 + f6 + f7)) / (1.0 - ux);

        field.f(x, y, 1) = f3 + (2.0 / 3.0) * rho * ux;
        field.f(x, y, 5) = f7 + 0.5 * (f4 - f2) + (1.0 / 6.0) * rho * ux + 0.5 * rho * uy;
        field.f(x, y, 8) = f6 - 0.5 * (f4 - f2) + (1.0 / 6.0) * rho * ux - 0.5 * rho * uy;

        field.rho(x, y) = rho;
        field.ux(x, y) = ux;
        field.uy(x, y) = uy;
    }
}

// Zou-He pressure outlet (x = nx-1) — proper formulation
void ZouHeBoundary::applyOutletRight(LatticeField& field) {
    const int nx = field.getNx();
    const int ny = field.getNy();
    const int x = nx - 1;

    for (int y = 1; y < ny - 1; ++y) {
        if (field.cellType(x, y) == CellType::SOLID) continue;

        Real rho = outlet_rho_;

        Real f0 = field.f(x, y, 0);
        Real f1 = field.f(x, y, 1);
        Real f2 = field.f(x, y, 2);
        Real f4 = field.f(x, y, 4);
        Real f5 = field.f(x, y, 5);
        Real f8 = field.f(x, y, 8);

        Real ux = -1.0 + (f0 + f2 + f4 + 2.0 * (f1 + f5 + f8)) / rho;
        Real uy = 0.0;

        field.f(x, y, 3) = f1 - (2.0 / 3.0) * rho * ux;
        field.f(x, y, 7) = f5 - 0.5 * (f4 - f2) - (1.0 / 6.0) * rho * ux - 0.5 * rho * uy;
        field.f(x, y, 6) = f8 + 0.5 * (f4 - f2) - (1.0 / 6.0) * rho * ux + 0.5 * rho * uy;

        field.rho(x, y) = rho;
        field.ux(x, y) = ux;
        field.uy(x, y) = uy;
    }
}

} // namespace softflow
