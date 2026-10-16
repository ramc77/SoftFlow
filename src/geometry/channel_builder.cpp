#include "channel_builder.h"
#include "../lbm/lattice_field.h"

namespace softflow {

void ChannelBuilder::applyToField(LatticeField& field) const {
    int nx = field.getNx();
    int ny = field.getNy();

    for (int y = 0; y < ny; ++y) {
        for (int x = 0; x < nx; ++x) {
            // Default: fluid
            CellType ct = CellType::FLUID;

            // Top and bottom walls
            if (has_bottom_wall_ && y == 0) ct = CellType::SOLID;
            if (has_top_wall_ && y == ny - 1) ct = CellType::SOLID;

            // Inlet (left) and outlet (right) — only for non-periodic BCs
            if (boundary_type_ == BoundaryType::INLET_OUTLET) {
                if (x == 0 && ct == CellType::FLUID) ct = CellType::INLET;
                if (x == nx - 1 && ct == CellType::FLUID) ct = CellType::OUTLET;
            }

            // Obstacles
            for (const auto& obs : obstacles_) {
                if (obs->contains(static_cast<Real>(x), static_cast<Real>(y))) {
                    ct = CellType::SOLID;
                    break;
                }
            }

            field.setCellType(x, y, ct);
        }
    }
}

} // namespace softflow
