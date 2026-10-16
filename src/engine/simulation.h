#pragma once
#include "../core/types.h"
#include "../core/parameters.h"
#include "../lbm/lbm_solver.h"
#include "../lbm/free_surface.h"
#include "../lbm/mrt_collision.h"
#include "../lbm/shan_chen.h"
#include "../lbm/advection_diffusion.h"
#include "../membrane/capsule_system.h"
#include "../membrane/repulsion.h"
#include "../membrane/lubrication.h"
#include "../membrane/adhesion.h"
#include "../coupling/immersed_boundary.h"
#include "../coupling/viscosity_field.h"
#include "../geometry/channel_builder.h"
#include "../analysis/segregation_metrics.h"
#include "../ml/simple_nn.h"
#include "../ml/training_data.h"
#include "../io/vtk_writer.h"
#include "../io/csv_writer.h"
#include "../io/checkpoint.h"
#include "../io/output_config.h"
#include "../io/fluid_probe_writer.h"
#include "../io/global_stats_writer.h"
#include "../io/profiler.h"
#include "../core/insertion/inserter.h"
#include "../core/insertion/dynamic_inserter.h"
#include <memory>
#include <functional>

namespace softflow {

// Top-level simulation orchestrator
// Ties together LBM fluid, deformable capsules, IBM coupling, and ML
class Simulation {
public:
    explicit Simulation(const SimulationParams& params);

    // Setup methods (call before initialize)
    void setChannelBuilder(const ChannelBuilder& builder);

    void addCapsule(Vec2d center, Real radius, int num_nodes,
                    const MembraneParams& mparams, int type = 0);

    void addCapsuleRandom(int count, Real x0, Real y0, Real x1, Real y1,
                          Real radius_min, Real radius_max, int num_nodes,
                          const MembraneParams& mparams, int type = 0,
                          int seed = 12345, Real min_gap = 1.0,
                          int max_attempts = 0);

    /// Place capsules according to an `IInserter` strategy.
    ///
    /// Builds an `InsertionContext` from the current channel geometry,
    /// existing capsule field, and the supplied `min_gap`, drives the
    /// inserter once, and adds each returned `Placement` as a capsule
    /// of the given `MembraneParams` and `type`. The inserter's RNG is
    /// derived from `params_.rng_seed` (via FNV-1a hash of a tag) so
    /// runs are reproducible without per-call seeds.
    ///
    /// `num_nodes <= 0` requests `Capsule::computeOptimalNodes(r)` per
    /// placement (matches the existing `addCapsule` convention).
    ///
    /// Returns the number of capsules actually placed (may be less
    /// than the inserter requested if the region saturates — see
    /// `IInserter` contract). Safe to call before or after
    /// `initialize()`.
    int insertCapsules(insertion::IInserter& inserter,
                       const MembraneParams& mparams,
                       int type = 0,
                       Real min_gap = 1.0,
                       int num_nodes = 0,
                       const std::string& seed_tag = "default");

    /// Register a dynamic inserter to fire every timestep.
    ///
    /// The inserter's `step()` is called from `Simulation::step()`
    /// after capsule motion (post node-clamp) and before metrics, so
    /// freshly-placed capsules see full physics on the *next* step
    /// without disturbing the in-flight one. Each registration owns
    /// a deterministic mt19937_64 sub-stream seeded from
    /// `params.rng_seed` and `seed_tag`.
    ///
    /// Multiple registrations are supported — they run in
    /// registration order. Pass a unique `seed_tag` per registration
    /// so each gets an independent sub-stream.
    ///
    /// Ownership is shared because pybind11's unique_ptr argument
    /// handling does not compose with the IInserter / IRegion /
    /// ISizeDistribution shared_ptr hierarchy. In practice Simulation
    /// is the only long-lived owner; the user-side handle goes out of
    /// scope right after registration.
    void registerDynamicInserter(
        std::shared_ptr<insertion::IDynamicInserter> inserter,
        const MembraneParams& mparams,
        int type = 0,
        Real min_gap = 1.0,
        int num_nodes = 0,
        const std::string& seed_tag = "dynamic");

    /// Number of dynamic inserters currently registered.
    int numDynamicInserters() const {
        return static_cast<int>(dynamic_inserters_.size());
    }

    // Initialize simulation state
    void initialize();

    // Advance one timestep
    void step();

    // Run N timesteps (entire loop in C++)
    void run(int num_steps);

    // Diagnostics
    Real getMaxSpeed() const;

    // Accessors
    int currentStep() const { return step_count_; }
    void setCurrentStep(int step) { step_count_ = step; }

    // Checkpoint save/load
    bool saveCheckpoint(const std::string& filename) const;
    bool loadCheckpoint(const std::string& filename);
    const LBMSolver& lbmSolver() const { return *lbm_solver_; }
    LBMSolver& lbmSolver() { return *lbm_solver_; }
    const CapsuleSystem& capsules() const { return capsules_; }
    CapsuleSystem& capsules() { return capsules_; }
    const SimulationParams& params() const { return params_; }

    // Advection-diffusion access
    AdvectionDiffusion* advectionDiffusion() {
        return advection_diffusion_.get();
    }
    const AdvectionDiffusion* advectionDiffusion() const {
        return advection_diffusion_.get();
    }

    // Adhesion access
    AdhesionModel* adhesion() { return adhesion_.get(); }
    const AdhesionModel* adhesion() const { return adhesion_.get(); }

    // Segregation metrics access
    SegregationMetrics* segregationMetrics() { return seg_metrics_.get(); }
    const SegregationResults& lastSegregationResults() const { return last_seg_results_; }

    // Viscosity field access
    ViscosityField* viscosityField() { return viscosity_field_.get(); }

    // Set fluid in a region
    void setFluidRegion(int x0, int y0, int x1, int y1,
                        Real rho, Real ux, Real uy);

    // Free-surface wet-dry dam-break support
    // Call enableFreeSurface() before initialize(), then setEmptyRegion() to
    // mark the "air" side. After each LBM step the FreeSurface converts EMPTY
    // cells to FLUID when adjacent fluid pressure exceeds rho_atm*(1+threshold).
    void enableFreeSurface(Real rho_atm = 1.0, Real threshold = 0.002);
    void setEmptyRegion(int x0, int y0, int x1, int y1);
    FreeSurface* freeSurface() { return free_surface_.get(); }

    // Set scalar concentration in a region
    void setScalarRegion(int x0, int y0, int x1, int y1,
                         Real concentration, int species = 0);

    // Set scalar source/sink rates per capsule type (constant-rate, backward-compatible)
    void setScalarReleaseRate(int capsule_type, Real rate);
    void setScalarAbsorptionRate(int capsule_type, Real rate);

    // Physics-based leaching: J = k_leach * (C_eq - C_surface)
    void setLeachingParams(int capsule_type, Real k_leach, Real C_eq);
    // Langmuir adsorption/desorption: dΓ/dt = k_a*C*(1-Γ/Γ_max) - k_d*Γ
    void setAdsorptionParams(int capsule_type, Real k_a, Real k_d, Real Gamma_max);
    // Initial particle chemical reservoir mass (0 = infinite)
    void setParticleMass(int capsule_type, Real Mp0);

    // Callback for custom per-step operations
    using StepCallback = std::function<void(Simulation&, int)>;
    void setStepCallback(StepCallback cb) { step_callback_ = std::move(cb); }

    // ── Output configuration API ──
    void setOutputConfig(const OutputConfig& config);

    // Set VTK output options
    void setVTKOutput(int dump_every, const std::string& format,
                      const FluidOutputFields& fluid_fields,
                      const ParticleVTKFields& particle_fields);

    // Set CSV output options
    void setCSVOutput(int dump_every, const std::string& format,
                      const ParticleOutputFields& fields,
                      const ParticleFilter& filter,
                      bool append = true);

    // Add extra CSV output with different filter
    void addExtraCSVOutput(const std::string& filename, int dump_every,
                           const ParticleOutputFields& fields,
                           const ParticleFilter& filter);

    // Add fluid probe
    void addFluidProbe(int i, int j, const std::string& label);

    // Finalize: write PVD files, close writers, print profiling
    void finalize();

    // Profiler access
    const Profiler& profiler() const { return profiler_; }

private:
    SimulationParams params_;
    std::unique_ptr<LBMSolver> lbm_solver_;
    CapsuleSystem capsules_;
    ImmersedBoundary ibm_;
    RepulsionForce repulsion_;
    std::unique_ptr<ChannelBuilder> channel_builder_;

    // Optional components
    std::unique_ptr<FreeSurface> free_surface_;
    std::unique_ptr<ShanChen> shan_chen_;
    std::unique_ptr<LubricationCorrection> lubrication_;
    std::unique_ptr<AdhesionModel> adhesion_;
    std::unique_ptr<AdvectionDiffusion> advection_diffusion_;
    std::unique_ptr<ViscosityField> viscosity_field_;
    std::unique_ptr<SegregationMetrics> seg_metrics_;
    SegregationResults last_seg_results_;

    // ML
    std::unique_ptr<SimpleNN> ml_model_;
    std::unique_ptr<TrainingData> ml_data_;

    // Scalar source/sink rates per capsule type
    std::vector<Real> scalar_release_rates_;
    std::vector<Real> scalar_absorption_rates_;

    // Physics-based chemistry params cache (populated by setLeachingParams etc.)
    ScalarParams scalar_params_cache_;

    // ── I/O (comprehensive output system) ──
    OutputConfig output_config_;
    std::unique_ptr<VTKWriter> vtk_writer_;
    std::unique_ptr<CSVWriter> csv_writer_;
    std::vector<std::unique_ptr<CSVWriter>> extra_csv_writers_;
    std::unique_ptr<FluidProbeWriter> probe_writer_;
    std::unique_ptr<GlobalStatsWriter> stats_writer_;

    // Profiler
    Profiler profiler_;
    int timer_membrane_ = -1;
    int timer_repulsion_ = -1;
    int timer_lubrication_ = -1;
    int timer_adhesion_ = -1;
    int timer_ibm_spread_ = -1;
    int timer_ibm_interp_ = -1;
    int timer_lbm_ = -1;
    int timer_body_force_ = -1;
    int timer_shan_chen_ = -1;
    int timer_viscosity_ = -1;
    int timer_scalar_ = -1;
    int timer_output_ = -1;
    int timer_non_newtonian_ = -1;

    int step_count_ = 0;
    StepCallback step_callback_;

    // Registered dynamic inserters. Each entry owns its own RNG
    // sub-stream so registrations are independently reproducible
    // when the canonical params_.rng_seed is set.
    struct DynamicInserterEntry {
        std::shared_ptr<insertion::IDynamicInserter> inserter;
        MembraneParams                               mparams;
        int                                          type;
        Real                                         min_gap;
        int                                          num_nodes;        // 0 = auto
        std::string                                  seed_tag;
        std::mt19937_64                              rng;
    };
    std::vector<DynamicInserterEntry> dynamic_inserters_;
    void runDynamicInserters();

    // Per-step count of fluid nodes whose IBM/body force exceeded
    // params_.max_lattice_force and was rescaled. Updated in step()
    // and reported by checkStability() so silent cap events become
    // visible. Written non-atomically by a single thread (after the
    // OpenMP reduction inside step()), read by checkStability().
    int last_capped_nodes_ = 0;

    // Random source needed for include of <random> via mt19937_64.
    // Pulled in transitively by inserter.h, kept explicit here so
    // any future trim of insertion headers does not break compilation.

    void writeOutput();
    void checkStability();
    void collectMLData();
    void trainMLModel();
    void writeSimulationConfig() const;
};

} // namespace softflow
