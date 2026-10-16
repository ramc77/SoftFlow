#include "free_surface.h"
#include "lattice.h"
#include <algorithm>
#include <cstring>

namespace softflow {

FreeSurface::FreeSurface(LBMSolver& solver, Real rho_atm, Real threshold)
    : solver_(solver),
      nx_(solver.getNx()),
      ny_(solver.getNy()),
      rho_atm_(rho_atm),
      threshold_(threshold)
{
    originally_empty_.assign(static_cast<size_t>(nx_) * ny_, false);
}

void FreeSurface::markEmpty(int x0, int y0, int x1, int y1) {
    x0 = std::max(x0, 0);  x1 = std::min(x1, nx_ - 1);
    y0 = std::max(y0, 0);  y1 = std::min(y1, ny_ - 1);
    for (int y = y0; y <= y1; ++y)
        for (int x = x0; x <= x1; ++x)
            originally_empty_[idx(x, y)] = true;
}

void FreeSurface::apply(LatticeField& field) {
    // Set originally-empty cells to EMPTY type and fill with atmospheric equilibrium.
    // EMPTY cells are NOT treated as SOLID — they are pre-filled with feq(rho_atm, 0)
    // so that during streaming they provide an atmospheric pressure inlet to adjacent
    // FLUID cells.  No bounce-back is applied to EMPTY cells (lbm_solver excludes them
    // from solid_nodes_).
    for (int y = 0; y < ny_; ++y) {
        for (int x = 0; x < nx_; ++x) {
            if (!originally_empty_[idx(x, y)]) continue;
            if (field.getCellType(x, y) != CellType::FLUID) continue;

            field.setCellType(x, y, CellType::EMPTY);
            field.setRho(x, y, rho_atm_);   // needed by initializeEquilibriumAt
            field.setUx(x, y, 0.0);
            field.setUy(x, y, 0.0);
            field.initializeEquilibriumAt(x, y);
            field.setRho(x, y, 0.0);        // reset: EMPTY shows rho=0 in VTK
        }
    }
    // Rebuild: EMPTY cells are no longer in solid_nodes_
    solver_.rebuildBoundaryNodeLists();
}

void FreeSurface::step(LatticeField& field) {
    // ─────────────────────────────────────────────────────────────────────────
    // Called AFTER lbm_solver_->step().
    //
    // Physics:
    //  • EMPTY cells were pre-filled with feq(rho_atm, 0) before the LBM step.
    //  • During streaming, FLUID cells pulled feq(rho_atm, 0) from EMPTY
    //    neighbors → atmospheric pressure BC on the fluid side.
    //  • EMPTY cells pulled actual distributions from FLUID neighbors → mass
    //    and momentum have physically flowed from the FLUID side into EMPTY.
    //
    // Steps:
    //  1. Identify EMPTY cells whose FLUID neighbor has rho > rho_trigger.
    //  2. Convert them to FLUID using their ACTUAL current f values
    //     (mass-conserving: the FLUID side lost this mass via streaming).
    //  3. Re-fill remaining EMPTY cells with feq(rho_atm, 0) for the next step.
    //  4. Rebuild solid_nodes_ if any conversions occurred.
    // ─────────────────────────────────────────────────────────────────────────

    const Real rho_trigger = rho_atm_ * (1.0 + threshold_);

    // ── Step 1: EMPTY→FLUID (wet cells where fluid pressure exceeds rho_trigger) ──
    std::vector<int> to_wet;

    for (int y = 1; y < ny_ - 1; ++y) {
        for (int x = 0; x < nx_; ++x) {
            if (field.getCellType(x, y) != CellType::EMPTY) continue;

            for (int q = 1; q < D2Q9::Q; ++q) {
                int xn = x + D2Q9::cx[q];
                int yn = y + D2Q9::cy[q];
                if (xn < 0 || xn >= nx_ || yn < 0 || yn >= ny_) continue;

                CellType ct = field.getCellType(xn, yn);
                if (ct != CellType::FLUID && ct != CellType::INLET) continue;

                if (field.getRho(xn, yn) > rho_trigger) {
                    to_wet.push_back(idx(x, y));
                    break;
                }
            }
        }
    }

    // ── Step 2: FLUID→EMPTY (drain top free-surface cells when rho < rho_drain) ──
    // A FLUID cell converts to air when:
    //   (a) its density drops below rho_atm*(1 - 5*threshold) — genuine rarefaction, AND
    //   (b) it has an EMPTY neighbor in the UPWARD half-sphere (cy > 0: N, NE, NW).
    //
    // The upward-only direction check is the key physical constraint:
    //   - Horizontal free surface (top of water): EMPTY is ABOVE → drains ✓
    //     (this is what makes the left water level visibly drop)
    //   - Vertical dam face (x=NX/2): EMPTY is to the SIDE (cy=0) → does NOT drain ✓
    //     (those cells should flow rightward, not disappear)
    //
    // Why 5× threshold for drain (not 1×)?
    //   Each EMPTY→FLUID conversion sends a small acoustic pulse through the water.
    //   Pressure fluctuations from these pulses are O(1-3 × threshold), so draining
    //   at 1× threshold would cascade unphysically into the interior.
    //   At 5× threshold (~1% below atmospheric), only the genuine rarefaction wave
    //   from the dam break — where kinetic pressure lowers density by u²/cs² ≈ 1-3%
    //   — will trigger drainage.  This matches dam-break physics: the receding wave
    //   arrives at the top-left surface and the density there genuinely falls ~1%.
    const Real rho_drain = rho_atm_ * (1.0 - 5.0 * threshold_);
    std::vector<int> to_drain;

    for (int y = 1; y < ny_ - 1; ++y) {
        for (int x = 0; x < nx_; ++x) {
            if (field.getCellType(x, y) != CellType::FLUID) continue;
            if (field.getRho(x, y) >= rho_drain) continue;  // still pressurized

            // Only drain if an EMPTY cell exists ABOVE (cy[q] > 0)
            bool has_empty_above = false;
            for (int q = 1; q < D2Q9::Q && !has_empty_above; ++q) {
                if (D2Q9::cy[q] <= 0) continue;   // skip horizontal and downward
                int xn = x + D2Q9::cx[q];
                int yn = y + D2Q9::cy[q];
                if (xn < 0 || xn >= nx_ || yn < 0 || yn >= ny_) continue;
                if (field.getCellType(xn, yn) == CellType::EMPTY)
                    has_empty_above = true;
            }
            if (has_empty_above) to_drain.push_back(idx(x, y));
        }
    }

    // ── Apply EMPTY→FLUID conversions ─────────────────────────────────────────
    // Use ACTUAL streamed f values (mass already transferred from adjacent FLUID).
    for (int n : to_wet) {
        const int y_c = n / nx_;
        const int x_c = n % nx_;

        field.setCellType(x_c, y_c, CellType::FLUID);

        Real rho_c = 0.0, ux_c = 0.0, uy_c = 0.0;
        for (int q = 0; q < D2Q9::Q; ++q) {
            Real fq = field.f(x_c, y_c, q);
            rho_c += fq;
            ux_c  += static_cast<Real>(D2Q9::cx[q]) * fq;
            uy_c  += static_cast<Real>(D2Q9::cy[q]) * fq;
        }
        if (rho_c > 1e-10) { ux_c /= rho_c; uy_c /= rho_c; }
        else               { ux_c = 0.0;    uy_c = 0.0; }

        field.setRho(x_c, y_c, rho_c);
        field.setUx(x_c, y_c, ux_c);
        field.setUy(x_c, y_c, uy_c);
    }

    // ── Apply FLUID→EMPTY conversions ─────────────────────────────────────────
    // Initialize as atmospheric pressure so they provide the correct inlet BC.
    for (int n : to_drain) {
        const int y_c = n / nx_;
        const int x_c = n % nx_;

        field.setCellType(x_c, y_c, CellType::EMPTY);
        field.setRho(x_c, y_c, rho_atm_);
        field.setUx(x_c, y_c, 0.0);
        field.setUy(x_c, y_c, 0.0);
        field.initializeEquilibriumAt(x_c, y_c);
        field.setRho(x_c, y_c, 0.0);  // reset: EMPTY shows rho=0 in VTK
    }

    // Rebuild boundary node lists if any conversions occurred.
    if (!to_wet.empty() || !to_drain.empty()) {
        solver_.rebuildBoundaryNodeLists();
        total_wetted_ += static_cast<int>(to_wet.size());
    }

    // Step 3: Re-fill ALL remaining EMPTY cells with feq(rho_atm, 0).
    // This provides the atmospheric pressure inlet BC for the next LBM step:
    // FLUID cells adjacent to EMPTY will pull feq(rho_atm, 0) during streaming,
    // which acts as an open boundary at atmospheric pressure.
    //
    // IMPORTANT: initializeEquilibriumAt() reads rho_/ux_/uy_ to build feq,
    // so we must temporarily set rho_=rho_atm before calling it.
    // Afterwards reset rho_ to 0 so that VTK output shows EMPTY cells as
    // rho=0 (visually empty), not rho=rho_atm (which would look like water).
    for (int y = 0; y < ny_; ++y) {
        for (int x = 0; x < nx_; ++x) {
            if (field.getCellType(x, y) != CellType::EMPTY) continue;
            field.setRho(x, y, rho_atm_);   // needed by initializeEquilibriumAt
            field.setUx(x, y, 0.0);
            field.setUy(x, y, 0.0);
            field.initializeEquilibriumAt(x, y);
            field.setRho(x, y, 0.0);        // reset: EMPTY cells show rho=0 in VTK
        }
    }
}

bool FreeSurface::isOriginallyEmpty(int x, int y) const {
    return originally_empty_[idx(x, y)];
}

} // namespace softflow
