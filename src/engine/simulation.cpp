#include "simulation.h"
#include "../io/checkpoint.h"
#include <softflow/build_info.h>     // CMake-generated: git SHA, compiler, flags
#include <iostream>
#include <random>
#include <cmath>
#include <algorithm>
#include <sys/stat.h>
#include <fstream>
#include <sstream>
#include <chrono>
#include <ctime>
#include <iomanip>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

// Enum-to-string helpers used by the manifest writer.
// Recording string names (rather than integer enum values) keeps the
// manifest interpretable even if the enum order ever changes.
const char* boundaryTypeName(softflow::BoundaryType bt) {
    using BT = softflow::BoundaryType;
    switch (bt) {
        case BT::PERIODIC:     return "PERIODIC";
        case BT::INLET_OUTLET: return "INLET_OUTLET";
        case BT::CLOSED:       return "CLOSED";
    }
    return "UNKNOWN";
}

const char* collisionModelName(softflow::CollisionModel cm) {
    using CM = softflow::CollisionModel;
    switch (cm) {
        case CM::BGK:         return "BGK";
        case CM::MRT:         return "MRT";
        case CM::REGULARIZED: return "REGULARIZED";
    }
    return "UNKNOWN";
}

const char* membraneModelName(softflow::MembraneModel mm) {
    using MM = softflow::MembraneModel;
    switch (mm) {
        case MM::HOOKEAN:     return "HOOKEAN";
        case MM::NEO_HOOKEAN: return "NEO_HOOKEAN";
        case MM::SKALAK:      return "SKALAK";
        case MM::WLC:         return "WLC";
    }
    return "UNKNOWN";
}

const char* nonNewtonianModelName(softflow::NonNewtonianModel nm) {
    using NN = softflow::NonNewtonianModel;
    switch (nm) {
        case NN::NONE:      return "NONE";
        case NN::POWER_LAW: return "POWER_LAW";
        case NN::CARREAU:   return "CARREAU";
    }
    return "UNKNOWN";
}

const char* capsuleShapeName(softflow::CapsuleShape cs) {
    using CS = softflow::CapsuleShape;
    switch (cs) {
        case CS::CIRCLE:    return "CIRCLE";
        case CS::ELLIPSE:   return "ELLIPSE";
        case CS::BICONCAVE: return "BICONCAVE";
        case CS::FIBER:     return "FIBER";
    }
    return "UNKNOWN";
}

// Minimal JSON helpers — keeps the writer self-contained (no third-party
// JSON library) at the cost of writing each field by hand. This is fine
// because the manifest schema is small and stable; if it grows beyond a
// few hundred lines we should pull in nlohmann/json.
std::string jsonEscape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 2);
    for (char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:   out += c;
        }
    }
    return out;
}

std::string isoTimestamp() {
    auto now    = std::chrono::system_clock::now();
    auto t      = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
#if defined(_WIN32)
    gmtime_s(&tm, &t);
#else
    gmtime_r(&t, &tm);
#endif
    std::ostringstream os;
    os << std::put_time(&tm, "%Y-%m-%dT%H:%M:%SZ");
    return os.str();
}

template <typename T>
std::string vec2json(const std::vector<T>& v) {
    std::ostringstream os;
    os << "[";
    for (size_t i = 0; i < v.size(); ++i) {
        if (i) os << ", ";
        os << v[i];
    }
    os << "]";
    return os.str();
}

}  // anonymous namespace

namespace softflow {

Simulation::Simulation(const SimulationParams& params)
    : params_(params), repulsion_(params.repulsion) {
    lbm_solver_ = std::make_unique<LBMSolver>(params);

    // Enable periodic minimum-image for inter-capsule repulsion
    if (params.fluid.boundary_type == BoundaryType::PERIODIC) {
        repulsion_.setPeriodicX(static_cast<Real>(params.nx));
        capsules_.setPeriodicX(params.nx);
    }

    if (params.shan_chen.enabled) {
        shan_chen_ = std::make_unique<ShanChen>(params.shan_chen,
                                                 params.fluid.boundary_type);
    }

    // Lubrication corrections
    if (params.lubrication.enabled) {
        lubrication_ = std::make_unique<LubricationCorrection>(
            params.lubrication, params.kinematicViscosity());
    }

    // Cell adhesion
    if (params.adhesion.enabled) {
        adhesion_ = std::make_unique<AdhesionModel>(params.adhesion);
    }

    // Advection-diffusion (scalar transport)
    if (params.scalar.enabled) {
        advection_diffusion_ = std::make_unique<AdvectionDiffusion>(
            params.nx, params.ny, params.scalar);
    }

    // Viscosity contrast
    if (params.fluid.viscosity_contrast) {
        viscosity_field_ = std::make_unique<ViscosityField>(
            params.nx, params.ny, params.fluid.tau);
    }

    // Segregation metrics
    if (params.metrics_interval > 0) {
        seg_metrics_ = std::make_unique<SegregationMetrics>(params.ny);
    }

    if (params.ml.enabled) {
        ml_model_ = std::make_unique<SimpleNN>(params.ml);
        ml_data_ = std::make_unique<TrainingData>(params.ml.buffer_size);
    }

    // ── Create output directory structure ──
    mkdir(params.output_dir.c_str(), 0755);

    // I/O writers
    vtk_writer_ = std::make_unique<VTKWriter>(params.output_dir);
    if (params.fluid.boundary_type == BoundaryType::PERIODIC) {
        vtk_writer_->setPeriodicX(params.nx);
    }
    vtk_writer_->setFormat(params.vtk_format);
    if (params.vtk_format == "legacy") {
        vtk_writer_->setLegacyMode(true);
    }
    if (advection_diffusion_) {
        vtk_writer_->setAdvectionDiffusion(advection_diffusion_.get());
    }

    csv_writer_ = std::make_unique<CSVWriter>(params.output_dir);
    probe_writer_ = std::make_unique<FluidProbeWriter>(params.output_dir);
    stats_writer_ = std::make_unique<GlobalStatsWriter>(params.output_dir);

    // ── Initialize profiler timers ──
    if (params.enable_profiling) {
        timer_membrane_      = profiler_.addTimer("Membrane forces");
        timer_repulsion_     = profiler_.addTimer("Repulsion forces");
        timer_lubrication_   = profiler_.addTimer("Lubrication");
        timer_adhesion_      = profiler_.addTimer("Adhesion");
        timer_ibm_spread_    = profiler_.addTimer("IBM spread");
        timer_ibm_interp_    = profiler_.addTimer("IBM interpolate");
        timer_lbm_           = profiler_.addTimer("LBM step");
        timer_body_force_    = profiler_.addTimer("Body/external force");
        timer_shan_chen_     = profiler_.addTimer("Shan-Chen");
        timer_viscosity_     = profiler_.addTimer("Viscosity update");
        timer_scalar_        = profiler_.addTimer("Scalar transport");
        timer_output_        = profiler_.addTimer("Output I/O");
        timer_non_newtonian_ = profiler_.addTimer("Non-Newtonian tau");
    }
}

void Simulation::setChannelBuilder(const ChannelBuilder& builder) {
    channel_builder_ = std::make_unique<ChannelBuilder>(builder);
}

void Simulation::addCapsule(Vec2d center, Real radius, int num_nodes,
                             const MembraneParams& mparams, int type) {
    capsules_.addCapsule(center, radius, num_nodes, mparams, type);
}

void Simulation::addCapsuleRandom(int count, Real x0, Real y0, Real x1, Real y1,
                                   Real radius_min, Real radius_max, int num_nodes,
                                   const MembraneParams& mparams, int type,
                                   int seed, Real min_gap, int max_attempts_in) {
    std::mt19937 rng(static_cast<unsigned>(seed));
    std::uniform_real_distribution<Real> dist_x(x0, x1);
    std::uniform_real_distribution<Real> dist_y(y0, y1);
    std::uniform_real_distribution<Real> dist_r(radius_min, radius_max);

    int placed = 0;
    int total_max = (max_attempts_in > 0) ? max_attempts_in : count * 500;
    int attempts = 0;

    while (placed < count && attempts < total_max) {
        Real cx = dist_x(rng);
        Real cy = dist_y(rng);
        Real r = dist_r(rng);

        // Check overlap with existing capsules
        bool overlap = false;
        for (int c = 0; c < capsules_.numCapsules(); ++c) {
            Vec2d oc = capsules_[c].centroid();
            Real or_ = capsules_[c].effectiveRadius();
            Real dist = (Vec2d{cx, cy} - oc).norm();
            if (dist < r + or_ + min_gap) {
                overlap = true;
                break;
            }
        }

        // Check wall distance
        if (channel_builder_) {
            if (cy - r < channel_builder_->wallBottom() + min_gap ||
                cy + r > channel_builder_->wallTop() - min_gap) {
                overlap = true;
            }
            for (const auto& obs : channel_builder_->getObstacles()) {
                if (obs->signedDistance(cx, cy) < r + min_gap) {
                    overlap = true;
                    break;
                }
            }
        }

        if (!overlap) {
            int nn = (num_nodes > 0) ? num_nodes : Capsule::computeOptimalNodes(r);
            capsules_.addCapsule({cx, cy}, r, nn, mparams, type);
            placed++;
        }
        attempts++;
    }

    std::cout << "Placed " << placed << "/" << count << " capsules ("
              << attempts << " attempts)" << std::endl;
}

namespace {

// FNV-1a 64-bit. Used to derive a stable per-inserter seed from the
// canonical params_.rng_seed and a string tag. Bit-exact across
// platforms because it operates on raw byte streams with fixed
// constants. Imported here (rather than in a shared utils header)
// because Phase 1 has no other consumer; if a second use appears we
// promote it to src/core/.
uint64_t fnv1a64(const std::string& s, uint64_t seed) {
    uint64_t h = seed ^ 0xcbf29ce484222325ull;       // standard FNV offset
    for (unsigned char c : s) {
        h ^= c;
        h *= 0x100000001b3ull;                        // FNV prime
    }
    return h ? h : 0x9e3779b97f4a7c15ull;             // never return 0
}

}  // namespace

int Simulation::insertCapsules(insertion::IInserter& inserter,
                               const MembraneParams& mparams,
                               int type,
                               Real min_gap,
                               int num_nodes,
                               const std::string& seed_tag)
{
    // Build the context from the current channel + existing capsules.
    // The rest of the engine (capsule.cpp, repulsion.cpp) treats walls
    // as the lattice rows y=0 and y=ny-1; the *fluid* extent stops half
    // a lattice unit before the wall row centres, matching the y-clamp
    // in simulation.cpp:392 and the y-bot/y-top used in repulsion.
    insertion::InsertionContext ctx;
    ctx.nx           = params_.nx;
    ctx.ny           = params_.ny;
    ctx.wall_y_bottom = 0.5;
    ctx.wall_y_top    = static_cast<Real>(params_.ny) - 1.5;
    if (channel_builder_) {
        ctx.wall_y_bottom = channel_builder_->wallBottom() + 0.5;
        ctx.wall_y_top    = channel_builder_->wallTop()    - 0.5;
        ctx.obstacles     = channel_builder_->getObstacles();
    }
    ctx.periodic_nx = (params_.fluid.boundary_type == BoundaryType::PERIODIC)
                      ? params_.nx : 0;
    ctx.min_gap     = min_gap;

    const int n_existing = capsules_.numCapsules();
    ctx.existing_centers.reserve(n_existing);
    ctx.existing_radii.reserve(n_existing);
    for (int c = 0; c < n_existing; ++c) {
        ctx.existing_centers.push_back(capsules_[c].centroid());
        ctx.existing_radii.push_back(capsules_[c].effectiveRadius());
    }

    // Derive a stable seed from the canonical params_.rng_seed and the
    // user-supplied tag. Seed 0 in params_ falls back to a fixed
    // entropy source (the tag's hash alone) so runs are still
    // reproducible — but the user is on the hook for setting
    // params_.rng_seed if they want cross-tag independence.
    std::mt19937_64 rng(fnv1a64(seed_tag, params_.rng_seed));

    auto placements = inserter.generate(ctx, rng);

    int n_placed = 0;
    for (const auto& p : placements) {
        const int nn = (num_nodes > 0)
                       ? num_nodes
                       : Capsule::computeOptimalNodes(p.radius);
        capsules_.addCapsule(p.center, p.radius, nn, mparams, type);
        ++n_placed;
    }

    std::cout << "insertCapsules[" << seed_tag << "]: placed "
              << n_placed << " of " << placements.size()
              << " requested (existing capsules: " << n_existing << ")"
              << std::endl;

    return n_placed;
}

void Simulation::registerDynamicInserter(
    std::shared_ptr<insertion::IDynamicInserter> inserter,
    const MembraneParams& mparams,
    int type,
    Real min_gap,
    int num_nodes,
    const std::string& seed_tag)
{
    if (!inserter) {
        throw std::invalid_argument(
            "Simulation::registerDynamicInserter: inserter must not be null");
    }
    DynamicInserterEntry entry{
        std::move(inserter),
        mparams,
        type,
        min_gap,
        num_nodes,
        seed_tag,
        std::mt19937_64(fnv1a64(seed_tag, params_.rng_seed)),
    };
    dynamic_inserters_.push_back(std::move(entry));
}

// Drain every registered dynamic inserter, adding placements as
// capsules immediately. Called from step() after the y-clamp
// (post-step 10) and before ML/metrics so freshly added capsules see
// full physics on the next step. The InsertionContext is rebuilt per
// inserter so later inserters see capsules placed by earlier ones in
// the same drain.
void Simulation::runDynamicInserters() {
    if (dynamic_inserters_.empty()) return;

    insertion::InsertionContext base_ctx;
    base_ctx.nx           = params_.nx;
    base_ctx.ny           = params_.ny;
    base_ctx.wall_y_bottom = 0.5;
    base_ctx.wall_y_top    = static_cast<Real>(params_.ny) - 1.5;
    if (channel_builder_) {
        base_ctx.wall_y_bottom = channel_builder_->wallBottom() + 0.5;
        base_ctx.wall_y_top    = channel_builder_->wallTop()    - 0.5;
        base_ctx.obstacles     = channel_builder_->getObstacles();
    }
    base_ctx.periodic_nx = (params_.fluid.boundary_type == BoundaryType::PERIODIC)
                           ? params_.nx : 0;

    for (auto& entry : dynamic_inserters_) {
        // Refresh the existing-capsule field from the *current* state
        // (including capsules placed earlier in this same drain).
        insertion::InsertionContext ctx = base_ctx;
        ctx.min_gap = entry.min_gap;
        const int n_existing = capsules_.numCapsules();
        ctx.existing_centers.reserve(n_existing);
        ctx.existing_radii.reserve(n_existing);
        for (int c = 0; c < n_existing; ++c) {
            ctx.existing_centers.push_back(capsules_[c].centroid());
            ctx.existing_radii.push_back(capsules_[c].effectiveRadius());
        }

        auto placements = entry.inserter->step(ctx, params_.dt, entry.rng);
        for (const auto& p : placements) {
            const int nn = (entry.num_nodes > 0)
                           ? entry.num_nodes
                           : Capsule::computeOptimalNodes(p.radius);
            capsules_.addCapsule(p.center, p.radius, nn, entry.mparams, entry.type);
            // Critical: the new capsule must be visible to subsequent
            // inserters in this same drain. Re-extending ctx is cheap
            // for the small per-step counts we expect.
            ctx.existing_centers.push_back(p.center);
            ctx.existing_radii.push_back(p.radius);
        }
    }
}

void Simulation::initialize() {
    // Apply channel geometry to lattice
    if (channel_builder_) {
        channel_builder_->setBoundaryType(params_.fluid.boundary_type);
        channel_builder_->applyToField(lbm_solver_->field());
    }

    // Initialize LBM to equilibrium
    lbm_solver_->initialize();

    // Set obstacles for interpolated bounce-back (if enabled)
    if (channel_builder_ && params_.fluid.use_interpolated_bb) {
        lbm_solver_->setObstacles(channel_builder_->getObstacles());
    }

    // Initialize advection-diffusion
    if (advection_diffusion_) {
        advection_diffusion_->initialize(0.0);
    }

    // Apply free-surface empty regions (must come after lbm_solver_->initialize()
    // so that wall SOLID cells are already set and won't be overwritten by EMPTY)
    if (free_surface_) {
        free_surface_->apply(lbm_solver_->field());
    }

    // Set periodic wrapping on capsules for per-node minimum-image forces
    if (params_.fluid.boundary_type == BoundaryType::PERIODIC) {
        capsules_.setPeriodicX(params_.nx);   // also applies to capsules added later
    }

    step_count_ = 0;

    // Write simulation config JSON for reproducibility
    writeSimulationConfig();

    std::cout << "Simulation initialized: " << params_.nx << "x" << params_.ny
              << " lattice, " << capsules_.numCapsules() << " capsules ("
              << capsules_.totalNodes() << " membrane nodes)" << std::endl;
}

void Simulation::step() {
    const bool profiling = params_.enable_profiling;

    // 0. Clear fluid lattice forces from previous step
    lbm_solver_->field().clearForces();

    // 1. Clear all forces on membrane nodes
    capsules_.clearAllForces();

    // 2. Compute membrane internal forces (stretch, bend, area, perimeter, viscous)
    if (profiling) profiler_.start(timer_membrane_);
    capsules_.computeAllMembraneForces();
    if (profiling) profiler_.stop(timer_membrane_);

    // 3. Compute repulsion forces (inter-capsule + wall)
    if (profiling) profiler_.start(timer_repulsion_);
    Real y_bot = 0.5, y_top = static_cast<Real>(params_.ny) - 1.5;
    if (channel_builder_) {
        y_bot = channel_builder_->wallBottom() + 0.5;
        y_top = channel_builder_->wallTop() - 0.5;
    }
    repulsion_.computeAll(capsules_, y_bot, y_top);

    if (channel_builder_) {
        for (const auto& obs : channel_builder_->getObstacles()) {
            repulsion_.computeObstacleRepulsion(capsules_, *obs);
        }
    }
    if (profiling) profiler_.stop(timer_repulsion_);

    // 3b. Lubrication corrections
    if (lubrication_) {
        if (profiling) profiler_.start(timer_lubrication_);
        int periodic_nx = (params_.fluid.boundary_type == BoundaryType::PERIODIC)
                          ? params_.nx : 0;
        lubrication_->computeAll(capsules_, params_.ny, periodic_nx);
        if (profiling) profiler_.stop(timer_lubrication_);
    }

    // 3c. Adhesion
    if (adhesion_) {
        if (profiling) profiler_.start(timer_adhesion_);
        int periodic_nx = (params_.fluid.boundary_type == BoundaryType::PERIODIC)
                          ? params_.nx : 0;
        adhesion_->update(capsules_, params_.dt, params_.ny, periodic_nx);
        if (profiling) profiler_.stop(timer_adhesion_);
    }

    // 3d. Capsule buoyancy (density != rho0 → net gravity on capsule)
    if (params_.fluid.apply_gravity_to_capsules) {
        const Real gx = params_.fluid.gravity_x;
        const Real gy = params_.fluid.gravity_y;
        if (gx != 0.0 || gy != 0.0) {
            const Real rho0 = params_.fluid.rho0;
            for (int c = 0; c < capsules_.numCapsules(); ++c) {
                capsules_[c].applyBuoyancyForce(gx, gy, rho0);
            }
        }
    }

    // 4. IBM: Spread membrane forces to LBM lattice
    if (profiling) profiler_.start(timer_ibm_spread_);
    if (params_.ibm_iterations > 1) {
        ibm_.multiDirectForcing(capsules_, lbm_solver_->field(),
                                params_.ibm_iterations);
    } else {
        ibm_.spreadForces(capsules_, lbm_solver_->field());
    }
    if (profiling) profiler_.stop(timer_ibm_spread_);

    // 5. Apply external forces to fluid
    if (profiling) profiler_.start(timer_body_force_);
    {
        const Real bfx = params_.fluid.body_force_x;
        const Real bfy = params_.fluid.body_force_y;
        const Real gx  = params_.fluid.gravity_x;
        const Real gy  = params_.fluid.gravity_y;
        const bool has_body_force = (bfx != 0.0 || bfy != 0.0);
        const bool has_gravity    = params_.fluid.apply_gravity_to_fluid &&
                                    (gx != 0.0 || gy != 0.0);

        if (has_body_force || has_gravity) {
            auto& field = lbm_solver_->field();
            const int N = field.size();
#ifdef _OPENMP
            #pragma omp parallel for schedule(static)
#endif
            for (int n = 0; n < N; ++n) {
                if (field.cellTypeData()[n] == CellType::FLUID) {
                    if (has_body_force) {
                        field.FxData()[n] += bfx;
                        field.FyData()[n] += bfy;
                    }
                    if (has_gravity) {
                        Real rho = field.rhoData()[n];
                        field.FxData()[n] += rho * gx;
                        field.FyData()[n] += rho * gy;
                    }
                }
            }
        }
    }
    if (profiling) profiler_.stop(timer_body_force_);

    // 6. Optional: Shan-Chen surface tension forces
    if (shan_chen_) {
        if (profiling) profiler_.start(timer_shan_chen_);
        shan_chen_->computeForce(lbm_solver_->field());
        if (profiling) profiler_.stop(timer_shan_chen_);
    }

    // 6b. Force regularization for IBM-LBM stability.
    //
    // Caps |F_lattice| at params_.max_lattice_force (default 0.01 lattice
    // units, the historical hard-coded value). Set to a large number
    // (e.g. 1e9) to disable. The number of nodes that hit the cap is
    // accumulated in last_capped_nodes_ and reported by checkStability()
    // so users can see when the cap is silently rescaling physics.
    {
        const Real Fxy_max = params_.max_lattice_force;
        const Real Fxy_max2 = Fxy_max * Fxy_max;
        auto& field = lbm_solver_->field();
        const int N = field.size();
        int capped = 0;
#ifdef _OPENMP
        #pragma omp parallel for schedule(static) reduction(+:capped)
#endif
        for (int n = 0; n < N; ++n) {
            Real fx = field.FxData()[n];
            Real fy = field.FyData()[n];
            Real fmag2 = fx * fx + fy * fy;
            if (fmag2 > Fxy_max2) {
                Real scale = Fxy_max / std::sqrt(fmag2);
                field.FxData()[n] *= scale;
                field.FyData()[n] *= scale;
                ++capped;
            }
        }
        last_capped_nodes_ = capped;
    }

    // 6c. Update viscosity field (spatially varying tau)
    if (viscosity_field_ && step_count_ % params_.fluid.viscosity_update_interval == 0) {
        if (profiling) profiler_.start(timer_viscosity_);
        viscosity_field_->update(capsules_);
        lbm_solver_->setTauLocal(viscosity_field_->tauLocalData());
        if (profiling) profiler_.stop(timer_viscosity_);
    }

    // 6d. Non-Newtonian viscosity update
    if (params_.fluid.non_newtonian_model != NonNewtonianModel::NONE) {
        if (profiling) profiler_.start(timer_non_newtonian_);
        lbm_solver_->updateNonNewtonianTau();
        if (profiling) profiler_.stop(timer_non_newtonian_);
    }

    // 7. LBM step (collision → stream → BCs → macroscopic)
    if (profiling) profiler_.start(timer_lbm_);
    lbm_solver_->step();
    if (profiling) profiler_.stop(timer_lbm_);

    // 7b. Free-surface: convert EMPTY cells that are now under fluid pressure
    if (free_surface_) {
        free_surface_->step(lbm_solver_->field());
    }

    // 7c. Advection-diffusion step (scalar transport)
    if (advection_diffusion_) {
        if (profiling) profiler_.start(timer_scalar_);
        advection_diffusion_->step(lbm_solver_->field());

        // Use physics-based chemistry when any kinetic param is set;
        // otherwise fall back to constant-rate for backward compatibility.
        bool has_physics = !scalar_params_cache_.k_leach.empty()
                        || !scalar_params_cache_.k_adsorb.empty();
        if (has_physics) {
            advection_diffusion_->applyChemistry(
                capsules_, scalar_release_rates_, scalar_absorption_rates_,
                scalar_params_cache_);
        } else if (!scalar_release_rates_.empty() || !scalar_absorption_rates_.empty()) {
            advection_diffusion_->applySourceSink(
                capsules_, scalar_release_rates_, scalar_absorption_rates_);
        }
        if (profiling) profiler_.stop(timer_scalar_);
    }

    // 8. IBM: Interpolate fluid velocity to membrane nodes
    if (profiling) profiler_.start(timer_ibm_interp_);
    ibm_.interpolateVelocity(lbm_solver_->field(), capsules_);
    if (profiling) profiler_.stop(timer_ibm_interp_);

    // 9. Move membrane nodes with interpolated velocity
    capsules_.moveAllNodes(params_.dt);

    // 10. Clamp capsule nodes within the y-wall boundaries
    if (!params_.fluid.periodic_y) {
        Real y_min = 1.5;
        Real y_max = static_cast<Real>(params_.ny) - 1.5;
        for (int c = 0; c < capsules_.numCapsules(); ++c) {
            auto& nodes = capsules_[c].positions();
            for (auto& p : nodes) {
                if (p.y < y_min) p.y = y_min;
                if (p.y > y_max) p.y = y_max;
            }
        }
    }

    // 10a. Periodic-x recirculation of capsules, decoupled from the fluid BC.
    // A capsule whose centroid leaves [0, nx) is rigidly translated by +-nx so
    // it re-enters at the opposite end (conveyor recirculation). This lets a
    // finite suspension recirculate at sustained concentration while the fluid
    // keeps its own BC (e.g. velocity-driven inlet/outlet). Inter-capsule and
    // IBM forces remain non-periodic, consistent with the non-periodic fluid;
    // the seam crossing is a one-step teleport (valid for capsules small
    // relative to nx). When the fluid itself is PERIODIC this is redundant
    // (the standard periodic path already wraps), so it is only needed for the
    // periodic-capsules + non-periodic-fluid combination.
    if (params_.fluid.capsule_periodic_x &&
        params_.fluid.boundary_type != BoundaryType::PERIODIC) {
        const Real Lx = static_cast<Real>(params_.nx);
        for (int c = 0; c < capsules_.numCapsules(); ++c) {
            const Vec2d cen = capsules_[c].centroid();
            Real shift = 0.0;
            if (cen.x >= Lx)      shift = -Lx;
            else if (cen.x < 0.0) shift =  Lx;
            if (shift != 0.0) {
                auto& nodes = capsules_[c].positions();
                for (auto& p : nodes) p.x += shift;
            }
        }
    }

    // 10b. Drain registered dynamic inserters. Placed here so that
    // (a) freshly added capsules see full physics on the *next* step,
    // (b) they are visible to ML data collection and segregation
    //     metrics on this step, and
    // (c) capsule motion has settled before any insertion considers
    //     where the existing field is.
    runDynamicInserters();

    // 11. ML data collection and surrogate training
    if (ml_data_ && ml_model_) {
        collectMLData();
        if (step_count_ > 0 && step_count_ % params_.ml.retrain_interval == 0) {
            trainMLModel();
        }
    }

    // 12. Segregation metrics
    if (seg_metrics_ && params_.metrics_interval > 0 &&
        step_count_ % params_.metrics_interval == 0) {
        last_seg_results_ = seg_metrics_->compute(capsules_, adhesion_.get());
        std::string metrics_file = params_.output_dir + "/metrics_" +
                                   std::to_string(step_count_) + ".csv";
        seg_metrics_->writeCSV(metrics_file, last_seg_results_,
                               step_count_,
                               static_cast<Real>(step_count_) * params_.dt);
    }

    // 13. Stability checks
    if (params_.enable_stability_checks && params_.stability_check_interval > 0 &&
        step_count_ % params_.stability_check_interval == 0) {
        checkStability();
    }

    // 14. Output
    if (profiling) profiler_.start(timer_output_);
    {
        Real time = static_cast<Real>(step_count_) * params_.dt;
        int vtk_interval = (params_.vtk_dump_every > 0) ? params_.vtk_dump_every : params_.output_interval;
        int csv_interval = (params_.csv_dump_every > 0) ? params_.csv_dump_every : params_.output_interval;

        // VTK output
        if (vtk_interval > 0 && step_count_ % vtk_interval == 0) {
            vtk_writer_->writeFluidField(lbm_solver_->field(), step_count_);
            vtk_writer_->writeCapsules(capsules_, step_count_);
            vtk_writer_->recordTimestep(step_count_, time);
        }

        // CSV particle output
        if (csv_interval > 0 && step_count_ % csv_interval == 0) {
            csv_writer_->writeTimestep(capsules_, step_count_, time);

            // Extra CSV writers
            for (auto& ecw : extra_csv_writers_) {
                ecw->writeTimestep(capsules_, step_count_, time);
            }
        }

        // Fluid probe output
        if (params_.probe_dump_every > 0 && step_count_ % params_.probe_dump_every == 0) {
            probe_writer_->writeTimestep(lbm_solver_->field(), step_count_, time);
        }

        // Global statistics
        if (params_.stats_dump_every > 0 && step_count_ % params_.stats_dump_every == 0) {
            stats_writer_->writeTimestep(lbm_solver_->field(), capsules_,
                                          step_count_, time,
                                          advection_diffusion_.get());
        }
    }
    if (profiling) profiler_.stop(timer_output_);

    // 15. Checkpoint
    if (params_.checkpoint_interval > 0 &&
        step_count_ > 0 && step_count_ % params_.checkpoint_interval == 0) {
        std::string cp_file = params_.output_dir + "/checkpoint_" +
                              std::to_string(step_count_) + ".sfck";
        Checkpoint::save(*this, cp_file);
    }

    // 16. Custom callback
    if (step_callback_) {
        step_callback_(*this, step_count_);
    }

    step_count_++;
}

void Simulation::run(int num_steps) {
    std::cout << "Running " << num_steps << " timesteps..." << std::endl;
    for (int s = 0; s < num_steps; ++s) {
        step();
        if (s % 1000 == 0 && s > 0) {
            std::cout << "  Step " << step_count_ << "/" << num_steps << std::endl;
        }
    }

    finalize();

    std::cout << "Simulation complete. " << step_count_ << " total steps." << std::endl;
}

void Simulation::finalize() {
    // Write PVD collection files for ParaView time-series
    if (vtk_writer_) vtk_writer_->writePVDFiles();

    // Close CSV writers
    if (csv_writer_) csv_writer_->close();
    for (auto& ecw : extra_csv_writers_) ecw->close();
    if (probe_writer_) probe_writer_->close();
    if (stats_writer_) stats_writer_->close();

    // Print profiling report
    if (params_.enable_profiling) {
        profiler_.printReport(step_count_);
    }
}

// ── Stability monitoring ──────────────────────────────────────────
void Simulation::checkStability() {
    const auto& field = lbm_solver_->field();
    int nx = field.getNx();
    int ny = field.getNy();
    Real rho0 = params_.fluid.rho0;
    Real max_dev = params_.max_density_deviation;
    Real max_vel = params_.max_velocity;

    bool nan_detected = false;
    bool density_warning = false;
    bool velocity_warning = false;
    Real worst_rho = rho0, worst_vel = 0.0;

    for (int y = 0; y < ny && !nan_detected; ++y) {
        for (int x = 0; x < nx; ++x) {
            CellType ct = field.getCellType(x, y);
            if (ct == CellType::SOLID || ct == CellType::EMPTY) continue;
            Real rho = field.getRho(x, y);
            Real ux = field.getUx(x, y);
            Real uy = field.getUy(x, y);

            // NaN/Inf check
            if (std::isnan(rho) || std::isinf(rho) ||
                std::isnan(ux) || std::isinf(ux) ||
                std::isnan(uy) || std::isinf(uy)) {
                std::cerr << "*** STABILITY WARNING: NaN/Inf detected at step "
                          << step_count_ << " at (" << x << "," << y << ")" << std::endl;
                nan_detected = true;
                break;
            }

            // Density deviation check
            Real dev = std::abs(rho - rho0) / rho0;
            if (dev > max_dev) {
                if (!density_warning || std::abs(rho - rho0) > std::abs(worst_rho - rho0)) {
                    worst_rho = rho;
                }
                density_warning = true;
            }

            // Velocity check
            Real vel = std::sqrt(ux * ux + uy * uy);
            if (vel > max_vel) {
                if (vel > worst_vel) worst_vel = vel;
                velocity_warning = true;
            }
        }
    }

    if (density_warning) {
        std::cerr << "*** STABILITY WARNING: density deviation at step " << step_count_
                  << " — rho = " << worst_rho << " (rho0 = " << rho0
                  << ", deviation = " << std::abs(worst_rho - rho0) / rho0 * 100.0
                  << "%)" << std::endl;
    }

    if (velocity_warning) {
        std::cerr << "*** STABILITY WARNING: high velocity at step " << step_count_
                  << " — |u| = " << worst_vel << " (limit = " << max_vel
                  << ", Ma = " << worst_vel / std::sqrt(1.0 / 3.0) << ")" << std::endl;
    }

    // Surface the IBM-LBM force-cap activity. A non-zero count means
    // params_.max_lattice_force is silently rescaling forces — the user
    // should know whether that is acceptable for their physics.
    if (last_capped_nodes_ > 0) {
        std::cerr << "*** STABILITY NOTICE: max_lattice_force cap rescaled "
                  << last_capped_nodes_ << " fluid nodes at step " << step_count_
                  << " (cap = " << params_.max_lattice_force << ")" << std::endl;
    }
}

Real Simulation::getMaxSpeed() const {
    const auto& field = lbm_solver_->field();
    int nx = field.getNx();
    int ny = field.getNy();
    Real maxSp = 0.0;
    for (int y = 0; y < ny; ++y) {
        for (int x = 0; x < nx; ++x) {
            if (field.getCellType(x, y) == CellType::SOLID) continue;
            Real vx = field.getUx(x, y);
            Real vy = field.getUy(x, y);
            Real sp = vx * vx + vy * vy;
            if (sp > maxSp) maxSp = sp;
        }
    }
    return std::sqrt(maxSp);
}

void Simulation::setFluidRegion(int x0, int y0, int x1, int y1,
                                 Real rho, Real ux, Real uy) {
    lbm_solver_->setFluidRegion(x0, y0, x1, y1, rho, ux, uy);
}

void Simulation::enableFreeSurface(Real rho_atm, Real threshold) {
    free_surface_ = std::make_unique<FreeSurface>(*lbm_solver_, rho_atm, threshold);
}

void Simulation::setEmptyRegion(int x0, int y0, int x1, int y1) {
    if (free_surface_)
        free_surface_->markEmpty(x0, y0, x1, y1);
}

void Simulation::setScalarRegion(int x0, int y0, int x1, int y1,
                                  Real concentration, int species) {
    if (advection_diffusion_) {
        advection_diffusion_->setRegion(x0, y0, x1, y1, concentration, species);
    }
}

void Simulation::setScalarReleaseRate(int capsule_type, Real rate) {
    if (capsule_type >= static_cast<int>(scalar_release_rates_.size())) {
        scalar_release_rates_.resize(capsule_type + 1, 0.0);
    }
    scalar_release_rates_[capsule_type] = rate;
}

void Simulation::setScalarAbsorptionRate(int capsule_type, Real rate) {
    if (capsule_type >= static_cast<int>(scalar_absorption_rates_.size())) {
        scalar_absorption_rates_.resize(capsule_type + 1, 0.0);
    }
    scalar_absorption_rates_[capsule_type] = rate;
}

void Simulation::setLeachingParams(int capsule_type, Real k_leach, Real C_eq) {
    if (capsule_type >= static_cast<int>(scalar_params_cache_.k_leach.size())) {
        scalar_params_cache_.k_leach.resize(capsule_type + 1, 0.0);
        scalar_params_cache_.C_eq.resize(capsule_type + 1, 0.0);
    }
    scalar_params_cache_.k_leach[capsule_type] = k_leach;
    scalar_params_cache_.C_eq[capsule_type]    = C_eq;
}

void Simulation::setAdsorptionParams(int capsule_type, Real k_a, Real k_d, Real Gamma_max) {
    if (capsule_type >= static_cast<int>(scalar_params_cache_.k_adsorb.size())) {
        scalar_params_cache_.k_adsorb.resize(capsule_type + 1, 0.0);
        scalar_params_cache_.k_desorb.resize(capsule_type + 1, 0.0);
        scalar_params_cache_.Gamma_max.resize(capsule_type + 1, 1.0);
    }
    scalar_params_cache_.k_adsorb[capsule_type]  = k_a;
    scalar_params_cache_.k_desorb[capsule_type]  = k_d;
    scalar_params_cache_.Gamma_max[capsule_type] = Gamma_max;
}

void Simulation::setParticleMass(int capsule_type, Real Mp0) {
    if (capsule_type >= static_cast<int>(scalar_params_cache_.M_p_initial.size()))
        scalar_params_cache_.M_p_initial.resize(capsule_type + 1, 0.0);
    scalar_params_cache_.M_p_initial[capsule_type] = Mp0;
}

// ── Output configuration ──────────────────────────────────────────

void Simulation::setOutputConfig(const OutputConfig& config) {
    output_config_ = config;
    vtk_writer_->setFluidFields(config.fluid_fields);
    vtk_writer_->setParticleFields(config.particle_vtk_fields);
    vtk_writer_->setFormat(config.vtk_format);
    csv_writer_->setFields(config.csv_fields);
    csv_writer_->setFilter(config.csv_filter);

    // Create extra CSV writers
    extra_csv_writers_.clear();
    for (const auto& ec : config.extra_csv) {
        auto w = std::make_unique<CSVWriter>(params_.output_dir, ec.filename,
                                              "csv", ec.append);
        w->setFields(ec.fields);
        w->setFilter(ec.filter);
        extra_csv_writers_.push_back(std::move(w));
    }

    // Add probes
    for (const auto& p : config.probes) {
        probe_writer_->addProbe(p.i, p.j, p.label);
    }
}

void Simulation::setVTKOutput(int dump_every, const std::string& format,
                               const FluidOutputFields& fluid_fields,
                               const ParticleVTKFields& particle_fields) {
    params_.vtk_dump_every = dump_every;
    vtk_writer_->setFormat(format);
    vtk_writer_->setFluidFields(fluid_fields);
    vtk_writer_->setParticleFields(particle_fields);
}

void Simulation::setCSVOutput(int dump_every, const std::string& format,
                               const ParticleOutputFields& fields,
                               const ParticleFilter& filter, bool append) {
    params_.csv_dump_every = dump_every;
    csv_writer_ = std::make_unique<CSVWriter>(params_.output_dir, "particle_data.csv",
                                               format, append);
    csv_writer_->setFields(fields);
    csv_writer_->setFilter(filter);
}

void Simulation::addExtraCSVOutput(const std::string& filename, int dump_every,
                                    const ParticleOutputFields& fields,
                                    const ParticleFilter& filter) {
    (void)dump_every; // stored in ExtraCSVConfig if needed
    auto w = std::make_unique<CSVWriter>(params_.output_dir, filename, "csv", true);
    w->setFields(fields);
    w->setFilter(filter);
    extra_csv_writers_.push_back(std::move(w));
}

void Simulation::addFluidProbe(int i, int j, const std::string& label) {
    probe_writer_->addProbe(i, j, label);
}

void Simulation::writeOutput() {
    vtk_writer_->writeFluidField(lbm_solver_->field(), step_count_);
    vtk_writer_->writeCapsules(capsules_, step_count_);
    csv_writer_->writeTimestep(capsules_, step_count_,
                                static_cast<Real>(step_count_) * params_.dt);
}

// Write run_manifest.json — the canonical reproducibility record.
//
// The manifest captures everything needed to re-run, audit, or debug a
// simulation post-mortem: the source revision (git SHA + dirty flag),
// the compiler ID/version/flags, the OpenMP thread count, the wall-clock
// timestamp, the canonical RNG seed, and the *full* resolved
// SimulationParams tree (every nested struct, every field). Enums are
// serialized as string names so the manifest stays meaningful if enum
// values are reordered later. See the project specification / §9.
//
// The hand-rolled JSON keeps the engine free of a third-party JSON
// dependency. If the schema grows past ~300 lines we should switch to
// nlohmann/json or a similar header-only library.
void Simulation::writeSimulationConfig() const {
    // Place the manifest under output_dir/config/ for the modern VTK
    // layout (which uses sub-directories for fluid/, particles/) and
    // directly under output_dir/ for the legacy flat layout.
    std::string path;
    if (params_.vtk_format == "legacy") {
        path = params_.output_dir + "/run_manifest.json";
    } else {
        std::string config_dir = params_.output_dir + "/config";
        mkdir(config_dir.c_str(), 0755);
        path = config_dir + "/run_manifest.json";
    }

    std::ofstream f(path);
    if (!f) {
        std::cerr << "WARNING: failed to open " << path
                  << " for writing run_manifest.json" << std::endl;
        return;
    }

    // OpenMP thread count is captured at runtime, not build-time, because
    // it depends on OMP_NUM_THREADS / process-level affinity. We sample
    // it here once.
    int omp_threads = 1;
#ifdef _OPENMP
    #pragma omp parallel
    {
        #pragma omp single
        omp_threads = omp_get_num_threads();
    }
#endif

    const auto& fl  = params_.fluid;
    const auto& mb  = params_.membrane;
    const auto& rep = params_.repulsion;
    const auto& lub = params_.lubrication;
    const auto& adh = params_.adhesion;
    const auto& sc  = params_.scalar;
    const auto& shc = params_.shan_chen;
    const auto& mll = params_.ml;

    f << std::setprecision(17);  // round-trip-safe doubles
    f << "{\n";
    f << "  \"manifest_schema_version\": 1,\n";
    f << "  \"timestamp_utc\": \"" << isoTimestamp() << "\",\n";

    // ── Build provenance ──
    f << "  \"build\": {\n";
    f << "    \"softflow_version\": \"" << build_info::version       << "\",\n";
    f << "    \"git_sha\":          \"" << build_info::git_sha       << "\",\n";
    f << "    \"git_branch\":       \"" << build_info::git_branch    << "\",\n";
    f << "    \"git_dirty\":        " << (build_info::git_dirty ? "true" : "false") << ",\n";
    f << "    \"build_type\":       \"" << build_info::build_type    << "\",\n";
    f << "    \"compiler_id\":      \"" << build_info::compiler_id   << "\",\n";
    f << "    \"compiler_version\": \"" << build_info::compiler_ver  << "\",\n";
    f << "    \"cxx_standard\":     \"" << build_info::cxx_standard  << "\",\n";
    f << "    \"cxx_flags\":        \"" << jsonEscape(build_info::cxx_flags) << "\",\n";
    f << "    \"system_name\":      \"" << build_info::system_name   << "\",\n";
    f << "    \"system_processor\": \"" << build_info::system_proc   << "\",\n";
    f << "    \"openmp\":           " << (build_info::openmp ? "true" : "false") << ",\n";
    f << "    \"openmp_threads\":   " << omp_threads << "\n";
    f << "  },\n";

    // ── Reproducibility seed ──
    f << "  \"rng_seed\": " << params_.rng_seed << ",\n";

    // ── Domain & timing ──
    f << "  \"domain\": { \"nx\": " << params_.nx
      <<              ", \"ny\": " << params_.ny << " },\n";
    f << "  \"timing\": { \"dt\": " << params_.dt
      <<              ", \"num_steps\": " << params_.num_steps
      <<              ", \"output_interval\": " << params_.output_interval << " },\n";
    f << "  \"ibm_iterations\": " << params_.ibm_iterations << ",\n";
    f << "  \"max_lattice_force\": " << params_.max_lattice_force << ",\n";

    // ── FluidParams ──
    f << "  \"fluid\": {\n";
    f << "    \"tau\": " << fl.tau << ",\n";
    f << "    \"kinematic_viscosity\": " << params_.kinematicViscosity() << ",\n";
    f << "    \"rho0\": " << fl.rho0 << ",\n";
    f << "    \"inlet_velocity\": " << fl.inlet_velocity << ",\n";
    f << "    \"outlet_density\": " << fl.outlet_density << ",\n";
    f << "    \"boundary_type\": \"" << boundaryTypeName(fl.boundary_type) << "\",\n";
    f << "    \"body_force_x\": " << fl.body_force_x << ",\n";
    f << "    \"body_force_y\": " << fl.body_force_y << ",\n";
    f << "    \"gravity_x\": " << fl.gravity_x << ",\n";
    f << "    \"gravity_y\": " << fl.gravity_y << ",\n";
    f << "    \"apply_gravity_to_fluid\": " << (fl.apply_gravity_to_fluid ? "true" : "false") << ",\n";
    f << "    \"apply_gravity_to_capsules\": " << (fl.apply_gravity_to_capsules ? "true" : "false") << ",\n";
    f << "    \"collision_model\": \"" << collisionModelName(fl.collision_model) << "\",\n";
    f << "    \"mrt\": { \"se\": " << fl.mrt_se
      <<           ", \"s_eps\": " << fl.mrt_s_eps
      <<           ", \"sq\": " << fl.mrt_sq << " },\n";
    f << "    \"viscosity_contrast\": " << (fl.viscosity_contrast ? "true" : "false") << ",\n";
    f << "    \"viscosity_update_interval\": " << fl.viscosity_update_interval << ",\n";
    f << "    \"non_newtonian\": {\n";
    f << "      \"model\": \"" << nonNewtonianModelName(fl.non_newtonian_model) << "\",\n";
    f << "      \"K\": " << fl.nn_K << ", \"n\": " << fl.nn_n << ",\n";
    f << "      \"nu_0\": " << fl.nn_nu_0 << ", \"nu_inf\": " << fl.nn_nu_inf << ",\n";
    f << "      \"lambda\": " << fl.nn_lambda << ",\n";
    f << "      \"tau_min\": " << fl.nn_tau_min << ", \"tau_max\": " << fl.nn_tau_max << "\n";
    f << "    },\n";
    f << "    \"periodic_y\": " << (fl.periodic_y ? "true" : "false") << ",\n";
    f << "    \"top_wall_velocity\": " << fl.top_wall_velocity << ",\n";
    f << "    \"bottom_wall_velocity\": " << fl.bottom_wall_velocity << ",\n";
    f << "    \"use_interpolated_bb\": " << (fl.use_interpolated_bb ? "true" : "false") << ",\n";
    f << "    \"use_pressure_bc\": " << (fl.use_pressure_bc ? "true" : "false") << ",\n";
    f << "    \"inlet_pressure\": " << fl.inlet_pressure << ",\n";
    f << "    \"outlet_pressure\": " << fl.outlet_pressure << ",\n";
    f << "    \"use_open_boundary\": " << (fl.use_open_boundary ? "true" : "false") << "\n";
    f << "  },\n";

    // ── MembraneParams ──
    f << "  \"membrane\": {\n";
    f << "    \"model\": \"" << membraneModelName(mb.model) << "\",\n";
    f << "    \"k_stretch\": " << mb.k_stretch << ",\n";
    f << "    \"G_s\": " << mb.G_s << ", \"C_skalak\": " << mb.C_skalak << ",\n";
    f << "    \"wlc\": { \"L_max_ratio\": " << mb.wlc_L_max_ratio
      <<           ", \"kBT_p\": " << mb.wlc_kBT_p
      <<           ", \"k_pow\": " << mb.wlc_k_pow << " },\n";
    f << "    \"k_bend\": " << mb.k_bend << ",\n";
    f << "    \"use_helfrich_bending\": " << (mb.use_helfrich_bending ? "true" : "false") << ",\n";
    f << "    \"kappa_0\": " << mb.kappa_0 << ",\n";
    f << "    \"k_area\": " << mb.k_area << ", \"k_perimeter\": " << mb.k_perimeter << ",\n";
    f << "    \"gamma_visc\": " << mb.gamma_visc << ", \"eta_membrane\": " << mb.eta_membrane << ",\n";
    f << "    \"viscosity_ratio\": " << mb.viscosity_ratio << ",\n";
    f << "    \"density\": " << mb.density << ",\n";
    f << "    \"shape\": \"" << capsuleShapeName(mb.shape) << "\",\n";
    f << "    \"aspect_ratio\": " << mb.aspect_ratio << ",\n";
    f << "    \"indent_depth\": " << mb.indent_depth << ",\n";
    f << "    \"is_rigid\": " << (mb.is_rigid ? "true" : "false") << "\n";
    f << "  },\n";

    // ── RepulsionParams ──
    f << "  \"repulsion\": {\n";
    f << "    \"epsilon\": " << rep.epsilon << ", \"sigma\": " << rep.sigma << ",\n";
    f << "    \"r_cut\": " << rep.r_cut << ", \"power\": " << rep.power << "\n";
    f << "  },\n";

    // ── LubricationParams ──
    f << "  \"lubrication\": { \"enabled\": " << (lub.enabled ? "true" : "false")
      <<                  ", \"h_threshold\": " << lub.h_threshold
      <<                  ", \"h_min\": " << lub.h_min << " },\n";

    // ── AdhesionParams ──
    f << "  \"adhesion\": {\n";
    f << "    \"enabled\": " << (adh.enabled ? "true" : "false") << ",\n";
    f << "    \"k_on\": " << adh.k_on << ", \"k_off\": " << adh.k_off << ",\n";
    f << "    \"k_bond\": " << adh.k_bond << ", \"d_bond\": " << adh.d_bond << ",\n";
    f << "    \"F_crit\": " << adh.F_crit << ",\n";
    f << "    \"max_bonds_per_node\": " << adh.max_bonds_per_node << ",\n";
    f << "    \"use_catch_slip\": " << (adh.use_catch_slip ? "true" : "false") << ",\n";
    f << "    \"k_off_catch\": " << adh.k_off_catch << ", \"F_catch\": " << adh.F_catch << ",\n";
    f << "    \"k_off_slip\": " << adh.k_off_slip << ", \"F_slip\": " << adh.F_slip << ",\n";
    f << "    \"wall_adhesion\": " << (adh.wall_adhesion ? "true" : "false") << ",\n";
    f << "    \"wall_k_on\": " << adh.wall_k_on << ", \"wall_k_off\": " << adh.wall_k_off << ",\n";
    f << "    \"wall_k_bond\": " << adh.wall_k_bond << ",\n";
    f << "    \"wall_receptor_spacing\": " << adh.wall_receptor_spacing << ",\n";
    f << "    \"adhesion_matrix_set\": " << (adh.adhesion_matrix.empty() ? "false" : "true") << "\n";
    f << "  },\n";

    // ── ScalarParams ──
    f << "  \"scalar\": {\n";
    f << "    \"enabled\": " << (sc.enabled ? "true" : "false") << ",\n";
    f << "    \"n_species\": " << sc.n_species << ",\n";
    f << "    \"diffusivity\": "          << vec2json(sc.diffusivity)          << ",\n";
    f << "    \"inlet_concentration\": "  << vec2json(sc.inlet_concentration)  << ",\n";
    f << "    \"periodic_y\": " << (sc.periodic_y ? "true" : "false") << ",\n";
    f << "    \"k_leach\": "               << vec2json(sc.k_leach)              << ",\n";
    f << "    \"C_eq\": "                  << vec2json(sc.C_eq)                 << ",\n";
    f << "    \"M_p_initial\": "           << vec2json(sc.M_p_initial)          << ",\n";
    f << "    \"k_adsorb\": "              << vec2json(sc.k_adsorb)             << ",\n";
    f << "    \"k_desorb\": "              << vec2json(sc.k_desorb)             << ",\n";
    f << "    \"Gamma_max\": "             << vec2json(sc.Gamma_max)            << "\n";
    f << "  },\n";

    // ── ShanChenParams ──
    f << "  \"shan_chen\": {\n";
    f << "    \"enabled\": " << (shc.enabled ? "true" : "false") << ",\n";
    f << "    \"G\": " << shc.G << ", \"rho0_sc\": " << shc.rho0_sc << ",\n";
    f << "    \"eos_type\": " << shc.eos_type << ",\n";
    f << "    \"cs_a\": " << shc.cs_a << ", \"cs_b\": " << shc.cs_b
      << ", \"cs_T\": " << shc.cs_T << ", \"cs_R\": " << shc.cs_R << ",\n";
    f << "    \"wetting_angle_deg\": " << shc.wetting_angle << ",\n";
    f << "    \"n_components\": " << shc.n_components << ",\n";
    f << "    \"G_12\": " << shc.G_12 << ", \"tau_2\": " << shc.tau_2 << "\n";
    f << "  },\n";

    // ── MLParams ──
    f << "  \"ml\": {\n";
    f << "    \"enabled\": " << (mll.enabled ? "true" : "false") << ",\n";
    f << "    \"warmup_steps\": " << mll.warmup_steps << ",\n";
    f << "    \"retrain_interval\": " << mll.retrain_interval << ",\n";
    f << "    \"error_threshold\": " << mll.error_threshold << ",\n";
    f << "    \"hidden_size\": " << mll.hidden_size << ",\n";
    f << "    \"learning_rate\": " << mll.learning_rate << ",\n";
    f << "    \"training_epochs\": " << mll.training_epochs << ",\n";
    f << "    \"buffer_size\": " << mll.buffer_size << ",\n";
    f << "    \"adam_beta1\": " << mll.adam_beta1 << ", \"adam_beta2\": " << mll.adam_beta2 << ",\n";
    f << "    \"use_pinn\": " << (mll.use_pinn ? "true" : "false")
      << ", \"pinn_lambda\": " << mll.pinn_lambda << ",\n";
    f << "    \"use_gnn\": " << (mll.use_gnn ? "true" : "false")
      << ", \"gnn_cutoff\": " << mll.gnn_cutoff
      << ", \"gnn_layers\": " << mll.gnn_layers << ",\n";
    f << "    \"adaptive_switching\": " << (mll.adaptive_switching ? "true" : "false")
      << ", \"validation_interval\": " << mll.validation_interval << "\n";
    f << "  },\n";

    // ── Output / monitoring ──
    f << "  \"output\": {\n";
    f << "    \"output_dir\": \""    << jsonEscape(params_.output_dir)    << "\",\n";
    f << "    \"output_format\": \"" << jsonEscape(params_.output_format) << "\",\n";
    f << "    \"checkpoint_interval\": " << params_.checkpoint_interval << ",\n";
    f << "    \"metrics_interval\": "    << params_.metrics_interval    << ",\n";
    f << "    \"vtk_dump_every\": "      << params_.vtk_dump_every      << ",\n";
    f << "    \"csv_dump_every\": "      << params_.csv_dump_every      << ",\n";
    f << "    \"probe_dump_every\": "    << params_.probe_dump_every    << ",\n";
    f << "    \"stats_dump_every\": "    << params_.stats_dump_every    << ",\n";
    f << "    \"vtk_format\": \"" << jsonEscape(params_.vtk_format) << "\"\n";
    f << "  },\n";
    f << "  \"stability\": {\n";
    f << "    \"enable_stability_checks\": " << (params_.enable_stability_checks ? "true" : "false") << ",\n";
    f << "    \"max_density_deviation\": " << params_.max_density_deviation << ",\n";
    f << "    \"max_velocity\": " << params_.max_velocity << ",\n";
    f << "    \"stability_check_interval\": " << params_.stability_check_interval << "\n";
    f << "  },\n";
    f << "  \"enable_profiling\": " << (params_.enable_profiling ? "true" : "false") << ",\n";

    // ── Setup-time counts ──
    f << "  \"setup\": { \"n_capsules\": " << capsules_.numCapsules()
      <<              ", \"total_membrane_nodes\": " << capsules_.totalNodes() << " }\n";
    f << "}\n";
}

void Simulation::collectMLData() {
    const auto& field = lbm_solver_->field();
    int nx = field.getNx();
    int ny = field.getNy();
    Real nu = params_.kinematicViscosity();
    Real channel_width = static_cast<Real>(ny);

    for (int c = 0; c < capsules_.numCapsules(); ++c) {
        const Capsule& cap = capsules_[c];
        Vec2d cen = cap.centroid();
        Real r = cap.effectiveRadius();

        int ix = std::clamp(static_cast<int>(cen.x), 0, nx - 1);
        int iy = std::clamp(static_cast<int>(cen.y), 0, ny - 1);
        Real uf_x = field.getUx(ix, iy);
        Real uf_y = field.getUy(ix, iy);

        Vec2d cap_vel{0, 0};
        for (int k = 0; k < cap.numNodes(); ++k) {
            cap_vel += cap.nodeVelocity(k);
        }
        cap_vel = cap_vel / static_cast<Real>(cap.numNodes());

        Real rel_vx = cap_vel.x - uf_x;
        Real rel_vy = cap_vel.y - uf_y;
        Real rel_v = std::sqrt(rel_vx * rel_vx + rel_vy * rel_vy);

        MLFeatures feat;
        feat.rel_vel_x = rel_vx;
        feat.rel_vel_y = rel_vy;
        feat.Re_p = (nu > 1e-15) ? rel_v * 2.0 * r / nu : 0.0;
        feat.local_rho = field.getRho(ix, iy);

        int ixp = std::min(ix + 1, nx - 1), ixm = std::max(ix - 1, 0);
        int iyp = std::min(iy + 1, ny - 1), iym = std::max(iy - 1, 0);
        feat.grad_rho_x = 0.5 * (field.getRho(ixp, iy) - field.getRho(ixm, iy));
        feat.grad_rho_y = 0.5 * (field.getRho(ix, iyp) - field.getRho(ix, iym));

        Real y_bot = 0.5, y_top = static_cast<Real>(ny) - 1.5;
        feat.wall_distance = std::min(cen.y - y_bot, y_top - cen.y);
        feat.radius_normalized = r / channel_width;
        feat.local_solid_fraction = 0.0;

        Vec2d F_hydro{0, 0};
        for (int k = 0; k < cap.numNodes(); ++k) {
            F_hydro += cap.nodeForce(k);
        }

        ml_data_->addSample(feat, F_hydro);
    }
}

void Simulation::trainMLModel() {
    if (ml_data_->size() < 100) return;
    std::cout << "Training ML surrogate model..." << std::endl;
    ml_model_->train(ml_data_->getInputs(), ml_data_->getTargets());
}

bool Simulation::saveCheckpoint(const std::string& filename) const {
    return Checkpoint::save(*this, filename);
}

bool Simulation::loadCheckpoint(const std::string& filename) {
    return Checkpoint::load(*this, filename);
}

} // namespace softflow
