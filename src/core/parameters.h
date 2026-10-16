#pragma once
#include "types.h"
#include <vector>
#include <string>

namespace softflow {

// ── Membrane constitutive model enum ─────────────────────────────────
enum class MembraneModel : int {
    HOOKEAN     = 0,   // Linear spring: F = k*(L/L0 - 1)
    NEO_HOOKEAN = 1,   // Neo-Hookean: F = G_s*(lambda - 1/lambda^3)
    SKALAK      = 2,   // Skalak (1973): strain-hardening RBC membrane
    WLC         = 3    // Worm-Like Chain (Fedosov et al., Biophys J 2010)
};

// ── Capsule rest shape enum ───────────────────────────────────────────
enum class CapsuleShape : int {
    CIRCLE    = 0,   // default: x=R*cos, y=R*sin
    ELLIPSE   = 1,   // x=R*cos, y=R*aspect_ratio*sin  (aspect_ratio = b/a)
    BICONCAVE = 2,   // r(θ)=R*(1-indent_depth*sin²θ)  (RBC cross-section)
    FIBER     = 3    // alias for ELLIPSE with high aspect_ratio
};

// ── Collision operator model enum ────────────────────────────────────
enum class CollisionModel : int {
    BGK         = 0,
    MRT         = 1,
    REGULARIZED = 2    // Latt & Chopard (2006) regularized BGK
};

// ── Boundary type extensions ─────────────────────────────────────────
// (BoundaryType already in types.h: PERIODIC, INLET_OUTLET, CLOSED)

// ── Non-Newtonian viscosity model enum ──────────────────────────
enum class NonNewtonianModel : int {
    NONE       = 0,   // Newtonian (constant viscosity)
    POWER_LAW  = 1,   // Power-law: nu = K * |gamma_dot|^(n-1)
    CARREAU    = 2    // Carreau: nu = nu_inf + (nu_0 - nu_inf) * [1 + (lambda*gamma_dot)^2]^((n-1)/2)
};

struct FluidParams {
    Real tau = 0.8;          // BGK relaxation time (> 0.5 for stability)
    Real rho0 = 1.0;         // reference density
    Real inlet_velocity = 0.02; // inlet velocity (lattice units)
    Real outlet_density = 1.0;  // outlet density (lattice units)
    BoundaryType boundary_type = BoundaryType::INLET_OUTLET;
    Real body_force_x = 0.0; // constant body force (pressure gradient driving)
    Real body_force_y = 0.0;
    Real gravity_x = 0.0;    // gravity acceleration (density-dependent: F = rho * g)
    Real gravity_y = 0.0;
    bool apply_gravity_to_fluid = true;     // apply F=rho*g to fluid lattice nodes
    bool apply_gravity_to_capsules = true;  // apply buoyancy F=(rho_cap-rho_fluid)*A*g

    // Collision model selection
    CollisionModel collision_model = CollisionModel::BGK;
    bool use_mrt = false;    // legacy flag (prefer collision_model)

    // MRT collision (Lallemand & Luo, Phys. Rev. E 61, 2000)
    Real mrt_se = 1.4;       // energy mode relaxation rate
    Real mrt_s_eps = 1.4;    // energy-squared mode relaxation rate
    Real mrt_sq = 1.4;       // energy-flux mode relaxation rate

    // Viscosity contrast (spatially varying tau)
    bool viscosity_contrast = false;
    int viscosity_update_interval = 10; // update inside/outside map every N steps

    // Non-Newtonian viscosity
    NonNewtonianModel non_newtonian_model = NonNewtonianModel::NONE;
    // Power-law: nu = K * |gamma_dot|^(n-1)
    Real nn_K = 0.1;             // consistency index
    Real nn_n = 0.5;             // flow behavior index (n<1 shear-thinning, n>1 shear-thickening)
    // Carreau: nu = nu_inf + (nu_0 - nu_inf) * [1 + (lambda*gamma_dot)^2]^((n-1)/2)
    Real nn_nu_0 = 0.1667;       // zero-shear-rate viscosity (tau=1.0 equivalent)
    Real nn_nu_inf = 0.0033;     // infinite-shear-rate viscosity (tau=0.51)
    Real nn_lambda = 3.313;      // relaxation time (Carreau parameter)
    Real nn_tau_min = 0.505;     // minimum tau (stability limit)
    Real nn_tau_max = 2.0;       // maximum tau

    // Extended boundary conditions
    bool periodic_y = false;  // periodic also in y-direction
    // Recirculate capsules in x independently of the fluid BC: a capsule whose
    // centroid leaves [0, nx) is translated by +-nx so it re-enters at the
    // opposite end. Lets a finite suspension recirculate (sustained
    // concentration) while the fluid uses, e.g., inlet/outlet driving. Forces
    // and IBM stay non-periodic to match the (non-periodic) fluid.
    bool capsule_periodic_x = false;
    Real top_wall_velocity = 0.0;    // moving top wall (Couette)
    Real bottom_wall_velocity = 0.0; // moving bottom wall
    bool use_interpolated_bb = false; // Bouzidi interpolated bounce-back
    bool use_pressure_bc = false;     // pressure (density) inlet/outlet
    Real inlet_pressure = 1.0;       // inlet density for pressure BC
    Real outlet_pressure = 1.0;      // outlet density for pressure BC
    bool use_open_boundary = false;   // convective outlet BC
};

struct MembraneParams {
    // Model selection
    MembraneModel model = MembraneModel::HOOKEAN;

    // Hookean spring constant (used when model == HOOKEAN)
    Real k_stretch = 0.1;    // stretching stiffness

    // Neo-Hookean / Skalak surface shear modulus
    Real G_s = 0.1;          // surface shear elastic modulus
    Real C_skalak = 1.0;     // Skalak area dilation ratio (C >> 1 for area conservation)

    // WLC parameters (Fedosov et al., Biophys J 2010)
    Real wlc_L_max_ratio = 2.2;  // L_max / L0 ratio (max extensibility, typically 2.0-3.0)
    Real wlc_kBT_p = 0.0;        // kBT/p (thermal energy / persistence length)
                                  // If 0, auto-computed from G_s to match shear modulus
    Real wlc_k_pow = 0.0;        // repulsive power-law coefficient (area conservation)
                                  // If 0, auto-computed to balance WLC at rest

    // Bending
    Real k_bend = 0.005;     // bending stiffness
    bool use_helfrich_bending = false; // true = Helfrich curvature-based bending
    Real kappa_0 = 0.0;      // spontaneous curvature (for non-circular rest shapes)

    // Conservation penalties
    Real k_area = 0.5;       // area conservation stiffness
    Real k_perimeter = 0.1;  // perimeter conservation stiffness
    Real gamma_visc = 0.01;  // translational viscous damping
    Real eta_membrane = 0.05; // Kelvin-Voigt membrane viscosity (strain-rate damping)
                              // Resists rapid deformation while allowing slow deformation

    // Viscosity contrast
    Real viscosity_ratio = 1.0; // lambda = mu_in / mu_out (RBCs: ~5)

    // Density (lattice units, 1.0 = neutrally buoyant with fluid rho0)
    // When density != rho0, a net buoyancy force is applied: F = (rho_cap - rho_fluid) * A * g
    // Default 1.0 = massless IBM (standard Peskin 2002 approach, no gravity on capsule)
    Real density = 1.0;

    // Shape (rest configuration)
    CapsuleShape shape        = CapsuleShape::CIRCLE;
    Real         aspect_ratio = 1.0;   // b/a for ELLIPSE/FIBER; a = radius
    Real         indent_depth = 0.4;   // BICONCAVE: r(θ)=R*(1-d*sin²θ), d∈[0,1]

    // Rigid body: no membrane deformation; IBM velocities projected to translation + rotation
    bool is_rigid = false;
};

struct RepulsionParams {
    Real epsilon = 0.05;     // repulsion strength (prevents capsule overlap)
    Real sigma = 1.0;        // repulsion range parameter
    Real r_cut = 3.5;        // cutoff distance (lattice units)
    int power = 4;           // repulsion power law exponent (4 = softer, longer range)

    // --- DEM-style dissipative/frictional contact (Path A) ---------------
    // Both default to 0 => pure conservative repulsion (legacy behaviour).
    // Normal viscoelastic damping coefficient gamma_n: adds a dashpot force
    // -gamma_n * v_n along the contact normal (cohesionless: the total normal
    // force is clamped >= 0, so there is no spurious attraction). Dissipates
    // energy on approach => coefficient of restitution < 1.
    //   Refs: Cundall & Strack, Geotechnique 29, 47 (1979);
    //         Brilliantov et al., Phys. Rev. E 53, 5382 (1996).
    Real damping_normal = 0.0;
    // Coulomb friction coefficient mu: adds a tangential force opposing the
    // tangential relative velocity, with magnitude mu * |F_normal| (kinetic
    // Coulomb friction, regularised by the sliding direction).
    //   Ref: Cundall & Strack (1979); Coulomb friction.
    Real friction_coeff = 0.0;
};

struct LubricationParams {
    bool enabled = false;    // must be explicitly enabled (O(N²) — slow with many capsules)
    Real h_threshold = 2.0;  // apply when gap < threshold (lattice units)
    Real h_min = 0.1;        // regularization cutoff (avoid singularity)
};

struct AdhesionParams {
    bool enabled = false;

    // Cell-cell adhesion (Bell model, Bell 1978)
    Real k_on = 0.001;       // bond formation rate
    Real k_off = 0.01;       // bond dissociation rate (base)
    Real k_bond = 0.05;      // bond spring constant
    Real d_bond = 2.0;       // max distance for bond formation
    Real F_crit = 0.01;      // critical force for slip bond
    int max_bonds_per_node = 3;

    // Catch-slip bond model (Thomas et al., 2008; Pereverzev et al., 2005)
    // Extends Bell model with catch behavior: force initially DECREASES
    // dissociation rate before slip regime takes over.
    //   k_off(F) = k_off_catch * exp(-F/F_catch) + k_off_slip * exp(F/F_slip)
    // When use_catch_slip = false, uses standard Bell: k_off * exp(F/F_crit)
    bool use_catch_slip = false;
    Real k_off_catch = 0.05;  // catch pathway dissociation rate
    Real F_catch = 0.02;      // catch force scale (bond strengthens under force)
    Real k_off_slip = 0.001;  // slip pathway dissociation rate
    Real F_slip = 0.01;       // slip force scale (bond weakens under force)

    // Cell-wall adhesion
    bool wall_adhesion = false;
    Real wall_k_on = 0.001;
    Real wall_k_off = 0.01;
    Real wall_k_bond = 0.05;
    Real wall_receptor_spacing = 2.0; // spacing between virtual wall receptors

    // Type-dependent: adhesion_matrix[i][j] = true means types i,j can bond
    // Empty = all types can bond with all types
    std::vector<std::vector<bool>> adhesion_matrix;
};

struct ScalarParams {
    bool enabled = false;
    int n_species = 1;                     // number of scalar species
    std::vector<Real> diffusivity = {0.01}; // diffusivity per species
    std::vector<Real> inlet_concentration = {0.0};  // fixed at inlet
    bool periodic_y = false;               // periodic scalar BC in y (default: no-flux)
    // Per capsule-type source/sink rates (set via Python)
    // release_rate[type] = concentration added at capsule nodes per step
    // absorption_rate[type] = concentration removed per step

    // --- Physics-based leaching kinetics (Gap A + C) ---
    // J_leach = k_leach[type] * (C_eq[type] - C_surface)
    // Empty = physics-based leaching disabled for that type (constant rate used instead)
    std::vector<Real> k_leach     = {};   // mass transfer coefficient per capsule type
    std::vector<Real> C_eq        = {};   // equilibrium/saturation concentration per type
    std::vector<Real> M_p_initial = {};   // initial chemical mass per capsule (0 = infinite)

    // --- Langmuir adsorption/desorption kinetics (Gap B) ---
    // dΓ/dt = k_adsorb * C_surface * (1 - Γ/Γ_max) - k_desorb * Γ
    std::vector<Real> k_adsorb    = {};   // adsorption rate k_a per capsule type
    std::vector<Real> k_desorb    = {};   // desorption rate k_d per capsule type
    std::vector<Real> Gamma_max   = {};   // max surface coverage Γ_max per capsule type
};

struct ShanChenParams {
    bool enabled = false;
    Real G = -5.0;           // interaction strength (negative = attraction → phase separation)
    Real rho0_sc = 1.0;      // reference density for psi function

    // Equation of State: 0 = exponential (original Shan-Chen)
    //                     1 = Carnahan-Starling (Yuan & Schaefer 2006)
    int eos_type = 0;
    Real cs_a = 1.0;         // C-S attractive parameter
    Real cs_b = 4.0;         // C-S covolume parameter
    Real cs_T = 0.06;        // C-S temperature
    Real cs_R = 1.0;         // C-S gas constant

    // Wetting / contact angle control
    Real wetting_angle = 90.0; // contact angle in degrees (90 = neutral)

    // Multi-component
    int n_components = 1;      // 1 = single component, 2 = binary mixture
    Real G_12 = -5.0;         // inter-component interaction strength
    Real tau_2 = 0.8;         // second component relaxation time
};

struct MLParams {
    bool enabled = false;
    int warmup_steps = 5000;       // full IBM steps before ML kicks in
    int retrain_interval = 2000;   // retrain every N steps
    Real error_threshold = 0.05;   // relative error threshold for validation
    int hidden_size = 64;          // hidden layer size
    Real learning_rate = 0.001;
    int training_epochs = 50;
    int buffer_size = 10000;       // training data buffer

    // Adam optimizer (Kingma & Ba 2015)
    Real adam_beta1 = 0.9;
    Real adam_beta2 = 0.999;

    // Physics-Informed Neural Network (PINN)
    bool use_pinn = false;
    Real pinn_lambda = 0.1;  // physics loss weight

    // Graph Neural Network
    bool use_gnn = false;
    Real gnn_cutoff = 10.0;  // neighbor cutoff for graph edges
    int gnn_layers = 2;      // message-passing rounds

    // Adaptive switching
    bool adaptive_switching = false;
    int validation_interval = 100; // validate ML vs full IBM every N steps
};

struct SimulationParams {
    // Domain
    int nx = 200;            // lattice width
    int ny = 50;             // lattice height

    // Timing
    Real dt = 1.0;           // timestep (lattice units, typically 1.0 for LBM)
    int num_steps = 10000;
    int output_interval = 100;

    // IBM coupling
    int ibm_iterations = 1;  // multi-direct forcing iterations (1 = standard, 3 = recommended)

    // Sub-components
    FluidParams fluid;
    MembraneParams membrane;
    RepulsionParams repulsion;
    LubricationParams lubrication;
    AdhesionParams adhesion;
    ScalarParams scalar;
    ShanChenParams shan_chen;
    MLParams ml;

    // Output
    std::string output_dir = "output";
    std::string output_format = "vtk"; // "vtk" or "csv"
    int checkpoint_interval = 0;       // 0 = disabled, N = save every N steps
    int metrics_interval = 0;          // 0 = disabled, N = compute metrics every N steps

    // Advanced output configuration
    int vtk_dump_every = 0;            // 0 = use output_interval
    int csv_dump_every = 0;            // 0 = use output_interval
    int probe_dump_every = 10;         // fluid probe dump interval
    int stats_dump_every = 100;        // global stats dump interval
    std::string vtk_format = "ascii";  // "ascii" or "binary"

    // Stability monitoring
    bool enable_stability_checks = true;
    Real max_density_deviation = 0.1;  // warn if |rho - rho0| / rho0 > this
    Real max_velocity = 0.1;           // warn if |u| > this (Mach number limit)
    int stability_check_interval = 1000;

    // Profiling
    bool enable_profiling = true;

    // Reproducibility: a single canonical seed from which every stochastic
    // sub-stream (random capsule placement, adhesion stochastics, ML init,
    // future insertion module) derives its own seed. See deriveSeed() in
    // run_manifest.cpp. Setting this to 0 is treated as "no seed set" and
    // the manifest will record a one-time entropy-derived seed instead.
    uint64_t rng_seed = 0;

    // IBM-LBM stability: hard cap on |F_lattice| applied just before
    // the LBM collision step (simulation.cpp). The historical default
    // 0.01 was hard-coded; it is exposed here so users can tune it (or
    // disable with a large value) and so its value is captured in the
    // run manifest. checkStability() reports the per-step number of
    // capped nodes when params.enable_stability_checks is true.
    Real max_lattice_force = 0.01;

    Real kinematicViscosity() const {
        return (fluid.tau - 0.5) / 3.0;
    }
};

} // namespace softflow
