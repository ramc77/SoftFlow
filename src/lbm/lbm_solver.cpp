#include "lbm_solver.h"
#include "lattice.h"
#include <algorithm>
#include <cmath>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace softflow {

LBMSolver::LBMSolver(const SimulationParams& params)
    : params_(params),
      field_(params.nx, params.ny),
      bgk_collision_(params.fluid.tau),
      boundary_(params.fluid.inlet_velocity, params.fluid.outlet_density),
      bounce_back_(),
      current_step_(0),
      periodic_x_(params.fluid.boundary_type == BoundaryType::PERIODIC),
      periodic_y_(params.fluid.periodic_y),
      use_mrt_(params.fluid.use_mrt ||
               params.fluid.collision_model == CollisionModel::MRT),
      use_regularized_(params.fluid.collision_model == CollisionModel::REGULARIZED)
{
    if (use_mrt_) {
        mrt_collision_ = std::make_unique<MRTCollision>(
            params.fluid.tau,
            params.fluid.mrt_se,
            params.fluid.mrt_s_eps,
            params.fluid.mrt_sq);
    }

    if (use_regularized_) {
        reg_collision_ = std::make_unique<RegularizedBGK>(params.fluid.tau);
    }

    // Moving wall
    if (std::abs(params.fluid.top_wall_velocity) > 1e-15 ||
        std::abs(params.fluid.bottom_wall_velocity) > 1e-15) {
        moving_wall_ = std::make_unique<MovingWall>(
            params.fluid.top_wall_velocity,
            params.fluid.bottom_wall_velocity);
        moving_wall_->setPeriodicX(periodic_x_);
    }

    // Halfway bounce-back must reflect across periodic boundaries too.
    bounce_back_.setPeriodic(periodic_x_, params.fluid.periodic_y);

    // Interpolated bounce-back. When the streamwise direction is
    // periodic, the IBB upstream-link fallback (q<½ branch) needs to
    // wrap rather than collapse to halfway BB at the seam — see
    // src/lbm/interpolated_bounce_back.cpp.
    if (params.fluid.use_interpolated_bb) {
        ibb_ = std::make_unique<InterpolatedBounceBack>();
        if (periodic_x_) ibb_->setPeriodicX(params.nx);
    }

    // Precompute neighbor lookup (SoA layout)
    precomputeNeighborIndices();

    // Initialize local tau for non-Newtonian if needed
    if (params.fluid.non_newtonian_model != NonNewtonianModel::NONE) {
        int N = params.nx * params.ny;
        tau_local_.resize(N, params.fluid.tau);
        use_local_tau_ = true;
    }
}

// Precompute neighbor indices for SoA streaming.
// Layout: neighbor_idx_[q * N + n] = source spatial index for direction q at node n.
// The streaming pull copies: f_tmp[q][n] = f[q][ neighbor_idx_[q*N + n] ]
void LBMSolver::precomputeNeighborIndices() {
    const int nx = params_.nx;
    const int ny = params_.ny;
    const int N = nx * ny;

    neighbor_idx_.resize(N * D2Q9::Q);

#ifdef _OPENMP
    #pragma omp parallel for collapse(2) schedule(static)
#endif
    for (int y = 0; y < ny; ++y) {
        for (int x = 0; x < nx; ++x) {
            const int n = y * nx + x;
            for (int q = 0; q < D2Q9::Q; ++q) {
                int xs = x - D2Q9::cx[q];
                int ys = y - D2Q9::cy[q];

                if (periodic_x_) xs = (xs + nx) % nx;
                if (periodic_y_) ys = (ys + ny) % ny;

                bool oob_x = !periodic_x_ && (xs < 0 || xs >= nx);
                bool oob_y = !periodic_y_ && (ys < 0 || ys >= ny);

                if (oob_x || oob_y) {
                    // Self-reference for out-of-bounds
                    neighbor_idx_[q * N + n] = n;
                } else {
                    neighbor_idx_[q * N + n] = ys * nx + xs;
                }
            }
        }
    }
}

// Build precomputed lists of boundary node indices (TASK 8).
// Called once after initialize() sets up cell types.
// Eliminates per-step type checking in bounce-back.
void LBMSolver::buildBoundaryNodeLists() {
    const int nx = params_.nx;
    const int ny = params_.ny;

    solid_nodes_.clear();
    inlet_nodes_.clear();
    outlet_nodes_.clear();

    for (int y = 0; y < ny; ++y) {
        for (int x = 0; x < nx; ++x) {
            CellType ct = field_.cellType(x, y);
            int n = y * nx + x;
            if (ct == CellType::SOLID)  solid_nodes_.push_back(n);
            // EMPTY cells are NOT in solid_nodes_: they act as atmospheric-pressure
            // inlets via streaming (free_surface.cpp pre-fills them with feq(rho_atm,0)).
            if (ct == CellType::INLET)  inlet_nodes_.push_back(n);
            if (ct == CellType::OUTLET) outlet_nodes_.push_back(n);
        }
    }
}

void LBMSolver::initialize() {
    const int nx = params_.nx;
    const int ny = params_.ny;

    // Mark top and bottom rows as SOLID walls
    for (int x = 0; x < nx; ++x) {
        field_.cellType(x, 0)      = CellType::SOLID;
        field_.cellType(x, ny - 1) = CellType::SOLID;
    }

    // Mark left/right columns based on boundary type
    if (params_.fluid.boundary_type == BoundaryType::INLET_OUTLET) {
        for (int y = 1; y < ny - 1; ++y) {
            field_.cellType(0, y)      = CellType::INLET;
            field_.cellType(nx - 1, y) = CellType::OUTLET;
        }
    } else if (params_.fluid.boundary_type == BoundaryType::CLOSED) {
        for (int y = 0; y < ny; ++y) {
            field_.cellType(0, y)      = CellType::SOLID;
            field_.cellType(nx - 1, y) = CellType::SOLID;
        }
    }

    // Periodic y: remove top/bottom walls
    if (periodic_y_) {
        for (int x = 0; x < nx; ++x) {
            field_.cellType(x, 0)      = CellType::FLUID;
            field_.cellType(x, ny - 1) = CellType::FLUID;
        }
    }

    // Initialize all cells to equilibrium
    field_.setEquilibrium(params_.fluid.rho0, 0.0, 0.0);
    field_.clearForces();
    field_.computeMacroscopic();

    // Build precomputed boundary node lists
    buildBoundaryNodeLists();

    current_step_ = 0;
}

void LBMSolver::step() {
    // 1. Collision (BGK, MRT, or Regularized, with Guo forcing)
    if (use_regularized_ && reg_collision_) {
        reg_collision_->collide(field_);
    } else if (use_mrt_ && mrt_collision_) {
        mrt_collision_->collide(field_);
    } else {
        bgk_collision_.collide(field_);
    }

    // 2. Streaming (SoA pull scheme using precomputed neighbor indices)
    stream();

    // 3. Bounce-back for solid walls (uses precomputed node list)
    bounce_back_.apply(field_, solid_nodes_);

    // 3b. Moving wall (if configured)
    if (moving_wall_) {
        moving_wall_->apply(field_);
    }

    // 3c. Interpolated bounce-back for curved obstacles
    if (ibb_) {
        ibb_->apply(field_);
    }

    // 4. Zou-He boundary conditions (inlet / outlet)
    if (params_.fluid.boundary_type == BoundaryType::INLET_OUTLET) {
        boundary_.apply(field_, inlet_nodes_, outlet_nodes_);
    }

    // 5. Compute macroscopic quantities
    field_.computeMacroscopic();

    ++current_step_;
}

// SoA pull-style streaming using precomputed neighbor lookup table.
// For each direction q, copies an entire spatial plane at once.
// Layout: f_tmp[q*N + n] = f[q*N + neighbor_idx_[q*N + n]]
void LBMSolver::stream() {
    const int N = field_.size();

    Real*       dst = field_.fTmpData();
    const Real* src = field_.fData();
    const int*  nb  = neighbor_idx_.data();

    // Stream each q-plane independently (contiguous memory access)
    for (int q = 0; q < D2Q9::Q; ++q) {
        const int qN = q * N;
        const Real* src_q = src + qN;
        Real* dst_q = dst + qN;
        const int* nb_q = nb + qN;

#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (int n = 0; n < N; ++n) {
            dst_q[n] = src_q[nb_q[n]];
        }
    }

    field_.swapBuffers();
}

// Non-Newtonian viscosity: compute local strain rate and update tau
void LBMSolver::updateNonNewtonianTau() {
    if (params_.fluid.non_newtonian_model == NonNewtonianModel::NONE) return;

    const int nx = field_.getNx();
    const int ny = field_.getNy();
    const int N = nx * ny;
    const CellType* flags = field_.cellTypeData();

    const Real tau_min = params_.fluid.nn_tau_min;
    const Real tau_max = params_.fluid.nn_tau_max;

#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int n = 0; n < N; ++n) {
        if (flags[n] != CellType::FLUID) {
            tau_local_[n] = params_.fluid.tau;
            continue;
        }

        int x = n % nx;
        int y = n / nx;

        // Compute strain rate from velocity field
        int xp = std::min(x + 1, nx - 1), xm = std::max(x - 1, 0);
        int yp = std::min(y + 1, ny - 1), ym = std::max(y - 1, 0);

        Real dux_dx = (field_.getUx(xp, y) - field_.getUx(xm, y)) / static_cast<Real>(xp - xm);
        Real duy_dy = (field_.getUy(x, yp) - field_.getUy(x, ym)) / static_cast<Real>(yp - ym);
        Real dux_dy = (field_.getUx(x, yp) - field_.getUx(x, ym)) / static_cast<Real>(yp - ym);
        Real duy_dx = (field_.getUy(xp, y) - field_.getUy(xm, y)) / static_cast<Real>(xp - xm);

        Real s11 = dux_dx;
        Real s22 = duy_dy;
        Real s12 = 0.5 * (dux_dy + duy_dx);
        Real gamma_dot = std::sqrt(2.0 * (s11 * s11 + s22 * s22 + 2.0 * s12 * s12));

        Real nu_local;
        if (params_.fluid.non_newtonian_model == NonNewtonianModel::POWER_LAW) {
            Real K = params_.fluid.nn_K;
            Real nn = params_.fluid.nn_n;
            nu_local = K * std::pow(std::max(gamma_dot, 1e-10), nn - 1.0);
        } else {
            Real nu_0 = params_.fluid.nn_nu_0;
            Real nu_inf = params_.fluid.nn_nu_inf;
            Real lam = params_.fluid.nn_lambda;
            Real nn = params_.fluid.nn_n;
            Real lg = lam * gamma_dot;
            nu_local = nu_inf + (nu_0 - nu_inf) * std::pow(1.0 + lg * lg, (nn - 1.0) / 2.0);
        }

        Real tau = 3.0 * nu_local + 0.5;
        tau_local_[n] = std::clamp(tau, tau_min, tau_max);
    }
}

void LBMSolver::setObstacles(
    const std::vector<std::shared_ptr<Obstacle>>& obstacles) {
    if (ibb_) {
        ibb_->initialize(field_, obstacles);
    }
}

void LBMSolver::setTauLocal(const Real* tau_local) {
    int N = params_.nx * params_.ny;
    tau_local_.resize(N);
    std::copy(tau_local, tau_local + N, tau_local_.begin());
    use_local_tau_ = true;
}

void LBMSolver::setFluidRegion(int x0, int y0, int x1, int y1,
                                Real rho, Real ux, Real uy) {
    for (int y = std::max(0, y0); y < std::min(params_.ny, y1); ++y) {
        for (int x = std::max(0, x0); x < std::min(params_.nx, x1); ++x) {
            if (field_.cellType(x, y) == CellType::SOLID) continue;
            field_.setRho(x, y, rho);
            field_.setUx(x, y, ux);
            field_.setUy(x, y, uy);
            field_.initializeEquilibriumAt(x, y);
        }
    }
}

} // namespace softflow
