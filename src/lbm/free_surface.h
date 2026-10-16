#pragma once
#include "lattice_field.h"
#include "lbm_solver.h"
#include <vector>

namespace softflow {

/**
 * Mass-conserving free-surface LBM for gravity-driven dam-break flows.
 *
 * Algorithm (atmospheric-pressure inlet + pressure-threshold wetting):
 *
 *   Initialization (apply()):
 *     - Empty cells are marked EMPTY and pre-filled with feq(rho_atm, 0).
 *     - EMPTY cells are NOT added to solid_nodes_, so NO bounce-back.
 *
 *   Each timestep — LBM step first, then free_surface->step():
 *     - During streaming: FLUID cells pull feq(rho_atm, 0) from EMPTY neighbors
 *       → atmospheric open-boundary on the fluid side.
 *     - EMPTY cells receive actual distributions from FLUID neighbors.
 *     - free_surface->step():
 *       1. Find EMPTY cells whose FLUID neighbor has rho > rho_atm*(1+threshold).
 *       2. Convert to FLUID using their ACTUAL current f values (MASS-CONSERVING:
 *          mass streamed from FLUID side — left water level physically drops).
 *       3. Re-fill remaining EMPTY cells with feq(rho_atm, 0) for next step.
 *       4. Rebuild solid_nodes_ if conversions occurred.
 *
 * Physics:
 *   Gravity builds hydrostatic pressure rho_bottom ≈ rho0*(1+3*|gy|*H).
 *   Bottom cells convert first.  Momentum in the actual f values drives waves.
 *
 * Reference: inspired by Korner et al. (2005) Modelling Simul. Mater. Sci. Eng.
 *            13, 723-737. Simplified single-phase variant without fill fractions.
 *
 * Unit conventions: dx=1 mm/lu, g_lat=1e-4 lu/ts^2 = 9.81 m/s^2 (NOT 9.81 lu/ts^2!)
 */
class FreeSurface {
public:
    /**
     * @param solver     Reference to the LBM solver (needed to rebuild node lists)
     * @param rho_atm    Atmospheric density — EMPTY cells are pre-filled at this density
     * @param threshold  Fractional pressure excess that triggers wetting:
     *                   rho_fluid_neighbor > rho_atm * (1 + threshold)
     *                   Default 0.002 → 0.2% excess, wets bottom cells first under gravity
     */
    explicit FreeSurface(LBMSolver& solver,
                         Real rho_atm   = 1.0,
                         Real threshold = 0.002);

    /**
     * Mark a rectangular region as "empty" (initially solid, can be wetted).
     * Call before apply(). Coordinates are inclusive [x0,x1] × [y0,y1].
     */
    void markEmpty(int x0, int y0, int x1, int y1);

    /**
     * Apply the empty-cell marking to the lattice field.
     * Call AFTER simulation.initialize() and BEFORE any steps.
     * Sets EMPTY cell type and rebuilds boundary node lists.
     */
    void apply(LatticeField& field);

    /**
     * Mass-conserving wetting: convert EMPTY→FLUID using actual streamed f values,
     * then re-fill remaining EMPTY cells with feq(rho_atm, 0).
     * Call AFTER each lbm_solver_->step().
     */
    void step(LatticeField& field);

    // Query
    int  totalWetted()          const { return total_wetted_; }
    bool isOriginallyEmpty(int x, int y) const;

private:
    LBMSolver& solver_;
    int        nx_, ny_;
    Real       rho_atm_;
    Real       threshold_;
    int        total_wetted_ = 0;

    // originally_empty_[y*nx+x] = true for cells that were marked empty
    std::vector<bool> originally_empty_;

    int idx(int x, int y) const { return y * nx_ + x; }
};

} // namespace softflow
