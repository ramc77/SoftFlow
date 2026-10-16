#pragma once
#include "lattice_field.h"
#include "bgk_collision.h"
#include "mrt_collision.h"
#include "regularized_bgk.h"
#include "zou_he_boundary.h"
#include "bounce_back.h"
#include "moving_wall.h"
#include "interpolated_bounce_back.h"
#include "../core/parameters.h"
#include "../geometry/obstacle.h"
#include <memory>
#include <vector>

namespace softflow {

class LBMSolver {
public:
    explicit LBMSolver(const SimulationParams& params);

    // Initialize: set top/bottom walls as SOLID, set equilibrium everywhere
    void initialize();

    // Perform one LBM timestep: collide -> stream -> bounce-back -> BCs -> macroscopic
    void step();

    // Access lattice field (for coupling with IBM, output, etc.)
    LatticeField&       field()       { return field_; }
    const LatticeField& field() const { return field_; }

    // Access sub-components
    BGKCollision&     collision()   { return bgk_collision_; }
    ZouHeBoundary&    boundary()    { return boundary_; }
    BounceBack&       bounceBack()  { return bounce_back_; }

    int  getNx() const { return params_.nx; }
    int  getNy() const { return params_.ny; }
    int  getStep() const { return current_step_; }
    bool isPeriodicX() const { return periodic_x_; }
    bool isPeriodicY() const { return periodic_y_; }

    // Set obstacles for IBB initialization
    void setObstacles(const std::vector<std::shared_ptr<Obstacle>>& obstacles);

    // Rebuild boundary node lists (call after mid-simulation cell type changes,
    // e.g. after free-surface wet-dry conversion converts EMPTY → FLUID)
    void rebuildBoundaryNodeLists() { buildBoundaryNodeLists(); }

    // Set spatially varying tau (for viscosity contrast)
    void setTauLocal(const Real* tau_local);
    bool hasLocalTau() const { return use_local_tau_; }

    // Non-Newtonian: update local tau from strain rate field
    void updateNonNewtonianTau();

    // Set fluid in a region (for multi-fluid initialization)
    void setFluidRegion(int x0, int y0, int x1, int y1,
                        Real rho, Real ux, Real uy);

    // Precomputed neighbor indices for streaming (SoA layout)
    const int* neighborIndices() const { return neighbor_idx_.data(); }

private:
    // Pull-style streaming with precomputed neighbor lookup (SoA layout)
    void stream();

    // Precompute streaming neighbor lookup table (SoA: per q-plane)
    void precomputeNeighborIndices();

    // Precompute boundary node lists (TASK 8: avoid per-step type checking)
    void buildBoundaryNodeLists();

    SimulationParams params_;
    LatticeField     field_;
    BGKCollision     bgk_collision_;
    std::unique_ptr<MRTCollision> mrt_collision_;
    std::unique_ptr<RegularizedBGK> reg_collision_;
    ZouHeBoundary    boundary_;
    BounceBack       bounce_back_;
    std::unique_ptr<MovingWall> moving_wall_;
    std::unique_ptr<InterpolatedBounceBack> ibb_;
    int              current_step_;
    bool             periodic_x_ = false;
    bool             periodic_y_ = false;
    bool             use_mrt_ = false;
    bool             use_regularized_ = false;
    bool             use_local_tau_ = false;
    std::vector<Real> tau_local_;

    // Precomputed neighbor indices for streaming (SoA layout)
    // neighbor_idx_[q * N + n] = source spatial index for (q, n)
    std::vector<int>  neighbor_idx_;

    // Precomputed boundary node lists (TASK 8)
    std::vector<int> solid_nodes_;    // flat indices of SOLID cells
    std::vector<int> inlet_nodes_;    // flat indices of INLET cells
    std::vector<int> outlet_nodes_;   // flat indices of OUTLET cells
};

} // namespace softflow
