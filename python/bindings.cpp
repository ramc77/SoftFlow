#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>

#include "../src/core/types.h"
#include "../src/core/parameters.h"
#include "../src/lbm/lattice_field.h"
#include "../src/lbm/lbm_solver.h"
#include "../src/lbm/bgk_collision.h"
#include "../src/lbm/regularized_bgk.h"
#include "../src/lbm/zou_he_boundary.h"
#include "../src/lbm/bounce_back.h"
#include "../src/lbm/moving_wall.h"
#include "../src/lbm/interpolated_bounce_back.h"
#include "../src/lbm/advection_diffusion.h"
#include "../src/lbm/shan_chen.h"
#include "../src/lbm/free_surface.h"
#include "../src/membrane/capsule.h"
#include "../src/membrane/capsule_system.h"
#include "../src/membrane/repulsion.h"
#include "../src/membrane/lubrication.h"
#include "../src/membrane/adhesion.h"
#include "../src/coupling/immersed_boundary.h"
#include "../src/coupling/viscosity_field.h"
#include "../src/geometry/obstacle.h"
#include "../src/geometry/channel_builder.h"
#include "../src/geometry/circle_obstacle.h"
#include "../src/geometry/rect_obstacle.h"
#include "../src/geometry/polygon_obstacle.h"
#include "../src/analysis/segregation_metrics.h"
#include "../src/ml/surrogate_model.h"
#include "../src/ml/simple_nn.h"
#include "../src/ml/training_data.h"
#include "../src/io/vtk_writer.h"
#include "../src/io/csv_writer.h"
#include "../src/io/checkpoint.h"
#include "../src/io/output_config.h"
#include "../src/io/profiler.h"
#include "../src/io/fluid_probe_writer.h"
#include "../src/io/global_stats_writer.h"
#include "../src/engine/simulation.h"

#include "../src/core/insertion/region.h"
#include "../src/core/insertion/image_mask_region.h"
#include "../src/core/insertion/size_distribution.h"
#include "../src/core/insertion/inserter.h"
#include "../src/core/insertion/lattice_inserter.h"
#include "../src/core/insertion/rsa_inserter.h"
#include "../src/core/insertion/dynamic_inserter.h"

namespace py = pybind11;
using namespace softflow;

PYBIND11_MODULE(softflow_core, m) {
    m.doc() = "SoftFlow: 2D LBM-IBM simulation of deformable capsules in fluid flow";

    // ═══════════════════════════════════════════════════════════════
    // Core types and enums
    // ═══════════════════════════════════════════════════════════════

    py::class_<Vec2d>(m, "Vec2d")
        .def(py::init<>())
        .def(py::init<Real, Real>(), py::arg("x"), py::arg("y"))
        .def_readwrite("x", &Vec2d::x)
        .def_readwrite("y", &Vec2d::y)
        .def("dot", &Vec2d::dot, py::arg("other"))
        .def("cross", &Vec2d::cross, py::arg("other"))
        .def("norm", &Vec2d::norm)
        .def("norm2", &Vec2d::norm2)
        .def("normalized", &Vec2d::normalized)
        .def("perp", &Vec2d::perp)
        .def("__add__", &Vec2d::operator+)
        .def("__sub__", &Vec2d::operator-)
        .def("__mul__", &Vec2d::operator*)
        .def("__truediv__", &Vec2d::operator/)
        .def("__iadd__", &Vec2d::operator+=)
        .def("__isub__", &Vec2d::operator-=)
        .def("__imul__", &Vec2d::operator*=)
        .def("__repr__", [](const Vec2d& v) {
            return "Vec2d(" + std::to_string(v.x) + ", " + std::to_string(v.y) + ")";
        });

    py::enum_<CellType>(m, "CellType")
        .value("FLUID", CellType::FLUID)
        .value("SOLID", CellType::SOLID)
        .value("INLET", CellType::INLET)
        .value("OUTLET", CellType::OUTLET)
        .value("EMPTY",  CellType::EMPTY)
        .export_values();

    py::enum_<BoundaryType>(m, "BoundaryType")
        .value("INLET_OUTLET", BoundaryType::INLET_OUTLET)
        .value("PERIODIC", BoundaryType::PERIODIC)
        .value("CLOSED", BoundaryType::CLOSED)
        .export_values();

    py::enum_<MembraneModel>(m, "MembraneModel")
        .value("HOOKEAN", MembraneModel::HOOKEAN)
        .value("NEO_HOOKEAN", MembraneModel::NEO_HOOKEAN)
        .value("SKALAK", MembraneModel::SKALAK)
        .value("WLC", MembraneModel::WLC)
        .export_values();

    py::enum_<CapsuleShape>(m, "CapsuleShape")
        .value("CIRCLE",    CapsuleShape::CIRCLE)
        .value("ELLIPSE",   CapsuleShape::ELLIPSE)
        .value("BICONCAVE", CapsuleShape::BICONCAVE)
        .value("FIBER",     CapsuleShape::FIBER)
        .export_values();

    py::enum_<CollisionModel>(m, "CollisionModel")
        .value("BGK", CollisionModel::BGK)
        .value("MRT", CollisionModel::MRT)
        .value("REGULARIZED", CollisionModel::REGULARIZED)
        .export_values();

    py::enum_<NonNewtonianModel>(m, "NonNewtonianModel")
        .value("NONE", NonNewtonianModel::NONE)
        .value("POWER_LAW", NonNewtonianModel::POWER_LAW)
        .value("CARREAU", NonNewtonianModel::CARREAU)
        .export_values();

    // ═══════════════════════════════════════════════════════════════
    // Parameter structs
    // ═══════════════════════════════════════════════════════════════

    py::class_<FluidParams>(m, "FluidParams")
        .def(py::init<>())
        .def_readwrite("tau", &FluidParams::tau)
        .def_readwrite("rho0", &FluidParams::rho0)
        .def_readwrite("inlet_velocity", &FluidParams::inlet_velocity)
        .def_readwrite("outlet_density", &FluidParams::outlet_density)
        .def_readwrite("boundary_type", &FluidParams::boundary_type)
        .def_readwrite("body_force_x", &FluidParams::body_force_x)
        .def_readwrite("body_force_y", &FluidParams::body_force_y)
        .def_readwrite("gravity_x", &FluidParams::gravity_x)
        .def_readwrite("gravity_y", &FluidParams::gravity_y)
        .def_readwrite("apply_gravity_to_fluid", &FluidParams::apply_gravity_to_fluid)
        .def_readwrite("apply_gravity_to_capsules", &FluidParams::apply_gravity_to_capsules)
        .def_readwrite("collision_model", &FluidParams::collision_model)
        .def_readwrite("use_mrt", &FluidParams::use_mrt)
        .def_readwrite("mrt_se", &FluidParams::mrt_se)
        .def_readwrite("mrt_s_eps", &FluidParams::mrt_s_eps)
        .def_readwrite("mrt_sq", &FluidParams::mrt_sq)
        .def_readwrite("viscosity_contrast", &FluidParams::viscosity_contrast)
        .def_readwrite("viscosity_update_interval", &FluidParams::viscosity_update_interval)
        .def_readwrite("periodic_y", &FluidParams::periodic_y)
        .def_readwrite("capsule_periodic_x", &FluidParams::capsule_periodic_x)
        .def_readwrite("top_wall_velocity", &FluidParams::top_wall_velocity)
        .def_readwrite("bottom_wall_velocity", &FluidParams::bottom_wall_velocity)
        .def_readwrite("use_interpolated_bb", &FluidParams::use_interpolated_bb)
        .def_readwrite("use_pressure_bc", &FluidParams::use_pressure_bc)
        .def_readwrite("inlet_pressure", &FluidParams::inlet_pressure)
        .def_readwrite("outlet_pressure", &FluidParams::outlet_pressure)
        .def_readwrite("use_open_boundary", &FluidParams::use_open_boundary)
        .def_readwrite("non_newtonian_model", &FluidParams::non_newtonian_model)
        .def_readwrite("nn_K", &FluidParams::nn_K)
        .def_readwrite("nn_n", &FluidParams::nn_n)
        .def_readwrite("nn_nu_0", &FluidParams::nn_nu_0)
        .def_readwrite("nn_nu_inf", &FluidParams::nn_nu_inf)
        .def_readwrite("nn_lambda", &FluidParams::nn_lambda)
        .def_readwrite("nn_tau_min", &FluidParams::nn_tau_min)
        .def_readwrite("nn_tau_max", &FluidParams::nn_tau_max);

    py::class_<MembraneParams>(m, "MembraneParams")
        .def(py::init<>())
        .def_readwrite("model", &MembraneParams::model)
        .def_readwrite("k_stretch", &MembraneParams::k_stretch)
        .def_readwrite("G_s", &MembraneParams::G_s)
        .def_readwrite("C_skalak", &MembraneParams::C_skalak)
        .def_readwrite("k_bend", &MembraneParams::k_bend)
        .def_readwrite("use_helfrich_bending", &MembraneParams::use_helfrich_bending)
        .def_readwrite("kappa_0", &MembraneParams::kappa_0)
        .def_readwrite("k_area", &MembraneParams::k_area)
        .def_readwrite("k_perimeter", &MembraneParams::k_perimeter)
        .def_readwrite("gamma_visc", &MembraneParams::gamma_visc)
        .def_readwrite("eta_membrane", &MembraneParams::eta_membrane)
        .def_readwrite("viscosity_ratio", &MembraneParams::viscosity_ratio)
        .def_readwrite("density", &MembraneParams::density)
        .def_readwrite("wlc_L_max_ratio", &MembraneParams::wlc_L_max_ratio)
        .def_readwrite("wlc_kBT_p", &MembraneParams::wlc_kBT_p)
        .def_readwrite("wlc_k_pow", &MembraneParams::wlc_k_pow)
        .def_readwrite("shape", &MembraneParams::shape)
        .def_readwrite("aspect_ratio", &MembraneParams::aspect_ratio)
        .def_readwrite("indent_depth", &MembraneParams::indent_depth)
        .def_readwrite("is_rigid", &MembraneParams::is_rigid);

    py::class_<RepulsionParams>(m, "RepulsionParams")
        .def(py::init<>())
        .def_readwrite("epsilon", &RepulsionParams::epsilon)
        .def_readwrite("sigma", &RepulsionParams::sigma)
        .def_readwrite("r_cut", &RepulsionParams::r_cut)
        .def_readwrite("power", &RepulsionParams::power)
        .def_readwrite("damping_normal", &RepulsionParams::damping_normal)
        .def_readwrite("friction_coeff", &RepulsionParams::friction_coeff);

    py::class_<LubricationParams>(m, "LubricationParams")
        .def(py::init<>())
        .def_readwrite("enabled", &LubricationParams::enabled)
        .def_readwrite("h_threshold", &LubricationParams::h_threshold)
        .def_readwrite("h_min", &LubricationParams::h_min);

    py::class_<AdhesionParams>(m, "AdhesionParams")
        .def(py::init<>())
        .def_readwrite("enabled", &AdhesionParams::enabled)
        .def_readwrite("k_on", &AdhesionParams::k_on)
        .def_readwrite("k_off", &AdhesionParams::k_off)
        .def_readwrite("k_bond", &AdhesionParams::k_bond)
        .def_readwrite("d_bond", &AdhesionParams::d_bond)
        .def_readwrite("F_crit", &AdhesionParams::F_crit)
        .def_readwrite("max_bonds_per_node", &AdhesionParams::max_bonds_per_node)
        .def_readwrite("wall_adhesion", &AdhesionParams::wall_adhesion)
        .def_readwrite("wall_k_on", &AdhesionParams::wall_k_on)
        .def_readwrite("wall_k_off", &AdhesionParams::wall_k_off)
        .def_readwrite("wall_k_bond", &AdhesionParams::wall_k_bond)
        .def_readwrite("wall_receptor_spacing", &AdhesionParams::wall_receptor_spacing)
        .def_readwrite("adhesion_matrix", &AdhesionParams::adhesion_matrix)
        .def_readwrite("use_catch_slip", &AdhesionParams::use_catch_slip)
        .def_readwrite("k_off_catch", &AdhesionParams::k_off_catch)
        .def_readwrite("F_catch", &AdhesionParams::F_catch)
        .def_readwrite("k_off_slip", &AdhesionParams::k_off_slip)
        .def_readwrite("F_slip", &AdhesionParams::F_slip);

    py::class_<ScalarParams>(m, "ScalarParams")
        .def(py::init<>())
        .def_readwrite("enabled", &ScalarParams::enabled)
        .def_readwrite("n_species", &ScalarParams::n_species)
        .def_readwrite("diffusivity", &ScalarParams::diffusivity)
        .def_readwrite("inlet_concentration", &ScalarParams::inlet_concentration)
        .def_readwrite("periodic_y", &ScalarParams::periodic_y)
        .def_readwrite("k_leach",     &ScalarParams::k_leach)
        .def_readwrite("C_eq",        &ScalarParams::C_eq)
        .def_readwrite("M_p_initial", &ScalarParams::M_p_initial)
        .def_readwrite("k_adsorb",    &ScalarParams::k_adsorb)
        .def_readwrite("k_desorb",    &ScalarParams::k_desorb)
        .def_readwrite("Gamma_max",   &ScalarParams::Gamma_max);

    py::class_<ShanChenParams>(m, "ShanChenParams")
        .def(py::init<>())
        .def_readwrite("enabled", &ShanChenParams::enabled)
        .def_readwrite("G", &ShanChenParams::G)
        .def_readwrite("rho0_sc", &ShanChenParams::rho0_sc)
        .def_readwrite("eos_type", &ShanChenParams::eos_type)
        .def_readwrite("cs_a", &ShanChenParams::cs_a)
        .def_readwrite("cs_b", &ShanChenParams::cs_b)
        .def_readwrite("cs_T", &ShanChenParams::cs_T)
        .def_readwrite("cs_R", &ShanChenParams::cs_R)
        .def_readwrite("wetting_angle", &ShanChenParams::wetting_angle)
        .def_readwrite("n_components", &ShanChenParams::n_components)
        .def_readwrite("G_12", &ShanChenParams::G_12)
        .def_readwrite("tau_2", &ShanChenParams::tau_2);

    py::class_<MLParams>(m, "MLParams")
        .def(py::init<>())
        .def_readwrite("enabled", &MLParams::enabled)
        .def_readwrite("warmup_steps", &MLParams::warmup_steps)
        .def_readwrite("retrain_interval", &MLParams::retrain_interval)
        .def_readwrite("error_threshold", &MLParams::error_threshold)
        .def_readwrite("hidden_size", &MLParams::hidden_size)
        .def_readwrite("learning_rate", &MLParams::learning_rate)
        .def_readwrite("training_epochs", &MLParams::training_epochs)
        .def_readwrite("buffer_size", &MLParams::buffer_size)
        .def_readwrite("adam_beta1", &MLParams::adam_beta1)
        .def_readwrite("adam_beta2", &MLParams::adam_beta2)
        .def_readwrite("use_pinn", &MLParams::use_pinn)
        .def_readwrite("pinn_lambda", &MLParams::pinn_lambda)
        .def_readwrite("use_gnn", &MLParams::use_gnn)
        .def_readwrite("gnn_cutoff", &MLParams::gnn_cutoff)
        .def_readwrite("gnn_layers", &MLParams::gnn_layers)
        .def_readwrite("adaptive_switching", &MLParams::adaptive_switching)
        .def_readwrite("validation_interval", &MLParams::validation_interval);

    py::class_<SimulationParams>(m, "SimulationParams")
        .def(py::init<>())
        .def_readwrite("nx", &SimulationParams::nx)
        .def_readwrite("ny", &SimulationParams::ny)
        .def_readwrite("dt", &SimulationParams::dt)
        .def_readwrite("num_steps", &SimulationParams::num_steps)
        .def_readwrite("output_interval", &SimulationParams::output_interval)
        .def_readwrite("ibm_iterations", &SimulationParams::ibm_iterations)
        .def_readwrite("fluid", &SimulationParams::fluid)
        .def_readwrite("membrane", &SimulationParams::membrane)
        .def_readwrite("repulsion", &SimulationParams::repulsion)
        .def_readwrite("lubrication", &SimulationParams::lubrication)
        .def_readwrite("adhesion", &SimulationParams::adhesion)
        .def_readwrite("scalar", &SimulationParams::scalar)
        .def_readwrite("shan_chen", &SimulationParams::shan_chen)
        .def_readwrite("ml", &SimulationParams::ml)
        .def_readwrite("output_dir", &SimulationParams::output_dir)
        .def_readwrite("output_format", &SimulationParams::output_format)
        .def_readwrite("checkpoint_interval", &SimulationParams::checkpoint_interval)
        .def_readwrite("metrics_interval", &SimulationParams::metrics_interval)
        .def_readwrite("vtk_dump_every", &SimulationParams::vtk_dump_every)
        .def_readwrite("csv_dump_every", &SimulationParams::csv_dump_every)
        .def_readwrite("probe_dump_every", &SimulationParams::probe_dump_every)
        .def_readwrite("stats_dump_every", &SimulationParams::stats_dump_every)
        .def_readwrite("vtk_format", &SimulationParams::vtk_format)
        .def_readwrite("enable_stability_checks", &SimulationParams::enable_stability_checks)
        .def_readwrite("max_density_deviation", &SimulationParams::max_density_deviation)
        .def_readwrite("max_velocity", &SimulationParams::max_velocity)
        .def_readwrite("stability_check_interval", &SimulationParams::stability_check_interval)
        .def_readwrite("enable_profiling", &SimulationParams::enable_profiling)
        // Phase 1: reproducibility seed and configurable IBM force cap.
        .def_readwrite("rng_seed",          &SimulationParams::rng_seed)
        .def_readwrite("max_lattice_force", &SimulationParams::max_lattice_force)
        .def("kinematicViscosity", &SimulationParams::kinematicViscosity);

    // ═══════════════════════════════════════════════════════════════
    // LBM: LatticeField with numpy array views
    // ═══════════════════════════════════════════════════════════════

    py::class_<LatticeField>(m, "LatticeField")
        .def(py::init<int, int>(), py::arg("nx"), py::arg("ny"))
        .def("getNx", &LatticeField::getNx)
        .def("getNy", &LatticeField::getNy)
        .def("size", &LatticeField::size)
        .def("density", [](LatticeField& f) {
            return py::array_t<Real>(
                {f.getNy(), f.getNx()},
                {f.getNx() * (int)sizeof(Real), (int)sizeof(Real)},
                f.rhoData(), py::cast(f));
        }, py::return_value_policy::reference_internal)
        .def("velocity_x", [](LatticeField& f) {
            return py::array_t<Real>(
                {f.getNy(), f.getNx()},
                {f.getNx() * (int)sizeof(Real), (int)sizeof(Real)},
                f.uxData(), py::cast(f));
        }, py::return_value_policy::reference_internal)
        .def("velocity_y", [](LatticeField& f) {
            return py::array_t<Real>(
                {f.getNy(), f.getNx()},
                {f.getNx() * (int)sizeof(Real), (int)sizeof(Real)},
                f.uyData(), py::cast(f));
        }, py::return_value_policy::reference_internal)
        .def("force_x", [](LatticeField& f) {
            return py::array_t<Real>(
                {f.getNy(), f.getNx()},
                {f.getNx() * (int)sizeof(Real), (int)sizeof(Real)},
                f.FxData(), py::cast(f));
        }, py::return_value_policy::reference_internal)
        .def("force_y", [](LatticeField& f) {
            return py::array_t<Real>(
                {f.getNy(), f.getNx()},
                {f.getNx() * (int)sizeof(Real), (int)sizeof(Real)},
                f.FyData(), py::cast(f));
        }, py::return_value_policy::reference_internal)
        .def("distributions", [](LatticeField& f) {
            // SoA layout: f[q * N + idx] → numpy shape [9, ny, nx]
            int N = f.getNx() * f.getNy();
            return py::array_t<Real>(
                {9, f.getNy(), f.getNx()},
                {N * (int)sizeof(Real), f.getNx() * (int)sizeof(Real), (int)sizeof(Real)},
                f.fData(), py::cast(f));
        }, py::return_value_policy::reference_internal)
        .def("getRho", &LatticeField::getRho, py::arg("x"), py::arg("y"))
        .def("getUx", &LatticeField::getUx, py::arg("x"), py::arg("y"))
        .def("getUy", &LatticeField::getUy, py::arg("x"), py::arg("y"))
        .def("getCellType", &LatticeField::getCellType, py::arg("x"), py::arg("y"))
        .def("setCellType", &LatticeField::setCellType, py::arg("x"), py::arg("y"), py::arg("ct"))
        .def("addExternalForce", &LatticeField::addExternalForce,
             py::arg("x"), py::arg("y"), py::arg("fx"), py::arg("fy"))
        .def("clearForces", &LatticeField::clearForces)
        .def("clearExternalForces", &LatticeField::clearExternalForces)
        .def("computeMacroscopic", &LatticeField::computeMacroscopic)
        .def("setEquilibrium", &LatticeField::setEquilibrium,
             py::arg("rho0"), py::arg("ux0"), py::arg("uy0"))
        .def("setRho", &LatticeField::setRho,
             py::arg("x"), py::arg("y"), py::arg("rho"))
        .def("setUx", &LatticeField::setUx,
             py::arg("x"), py::arg("y"), py::arg("ux"))
        .def("setUy", &LatticeField::setUy,
             py::arg("x"), py::arg("y"), py::arg("uy"))
        .def("initializeEquilibriumAt", &LatticeField::initializeEquilibriumAt,
             py::arg("x"), py::arg("y"))
        .def("initializeAllEquilibrium", &LatticeField::initializeAllEquilibrium)
        .def("swapBuffers", &LatticeField::swapBuffers)
        .def("f", [](const LatticeField& field, int x, int y, int q) {
            return field.f(x, y, q);
        }, py::arg("x"), py::arg("y"), py::arg("q"))
        .def("set_f", [](LatticeField& field, int x, int y, int q, Real val) {
            field.f(x, y, q) = val;
        }, py::arg("x"), py::arg("y"), py::arg("q"), py::arg("value"));

    // ═══════════════════════════════════════════════════════════════
    // LBM: Sub-solvers
    // ═══════════════════════════════════════════════════════════════

    py::class_<BGKCollision>(m, "BGKCollision")
        .def(py::init<Real>(), py::arg("tau"))
        .def("collide", &BGKCollision::collide, py::arg("field"))
        .def("setTau", &BGKCollision::setTau, py::arg("tau"))
        .def("getTau", &BGKCollision::getTau);

    py::class_<RegularizedBGK>(m, "RegularizedBGK")
        .def(py::init<Real>(), py::arg("tau"))
        .def("collide", &RegularizedBGK::collide, py::arg("field"));

    py::class_<ZouHeBoundary>(m, "ZouHeBoundary")
        .def(py::init<Real, Real>(), py::arg("inlet_velocity"), py::arg("outlet_density"))
        .def("apply", static_cast<void (ZouHeBoundary::*)(LatticeField&)>(&ZouHeBoundary::apply), py::arg("field"))
        .def("applyInletLeft", &ZouHeBoundary::applyInletLeft, py::arg("field"))
        .def("applyOutletRight", &ZouHeBoundary::applyOutletRight, py::arg("field"))
        .def("setInletVelocity", &ZouHeBoundary::setInletVelocity, py::arg("ux"))
        .def("setOutletDensity", &ZouHeBoundary::setOutletDensity, py::arg("rho"))
        .def("getInletVelocity", &ZouHeBoundary::getInletVelocity)
        .def("getOutletDensity", &ZouHeBoundary::getOutletDensity);

    py::class_<BounceBack>(m, "BounceBack")
        .def(py::init<>())
        .def("apply", static_cast<void (BounceBack::*)(LatticeField&)>(&BounceBack::apply), py::arg("field"));

    py::class_<MovingWall>(m, "MovingWall")
        .def(py::init<Real, Real>(), py::arg("top_velocity"), py::arg("bottom_velocity"))
        .def("apply", &MovingWall::apply, py::arg("field"));

    py::class_<ShanChen>(m, "ShanChen")
        .def(py::init<const ShanChenParams&>(), py::arg("params"))
        .def("computeForce", &ShanChen::computeForce, py::arg("field"))
        .def("getG", &ShanChen::getG);

    py::class_<AdvectionDiffusion>(m, "AdvectionDiffusion")
        .def(py::init<int, int, const ScalarParams&>(),
             py::arg("nx"), py::arg("ny"), py::arg("params"))
        .def("initialize", &AdvectionDiffusion::initialize,
             py::arg("initial_concentration") = 0.0)
        .def("setRegion", &AdvectionDiffusion::setRegion,
             py::arg("x0"), py::arg("y0"), py::arg("x1"), py::arg("y1"),
             py::arg("concentration"), py::arg("species") = 0)
        .def("setPoint", &AdvectionDiffusion::setPoint,
             py::arg("x"), py::arg("y"), py::arg("concentration"), py::arg("species") = 0)
        .def("step", &AdvectionDiffusion::step, py::arg("fluid_field"))
        .def("getConcentration", &AdvectionDiffusion::getConcentration,
             py::arg("x"), py::arg("y"), py::arg("species") = 0)
        .def("concentration", [](AdvectionDiffusion& ad, int species) {
            return py::array_t<Real>(
                {ad.getNy(), ad.getNx()},
                {ad.getNx() * (int)sizeof(Real), (int)sizeof(Real)},
                ad.concentrationData(species), py::cast(ad));
        }, py::arg("species") = 0, py::return_value_policy::reference_internal)
        .def("getNumSpecies", &AdvectionDiffusion::getNumSpecies)
        .def("getCapsuleReleased", &AdvectionDiffusion::getCapsuleReleased,
             py::arg("capsule_id"))
        .def("getCapsuleAbsorbed", &AdvectionDiffusion::getCapsuleAbsorbed,
             py::arg("capsule_id"))
        .def("numTrackedCapsules", &AdvectionDiffusion::numTrackedCapsules)
        .def("getCapsuleMp", &AdvectionDiffusion::getCapsuleMp,
             py::arg("capsule_id"),
             "Remaining particle chemical reservoir mass (−1 = infinite)")
        .def("getNodeGamma", &AdvectionDiffusion::getNodeGamma,
             py::arg("capsule_id"), py::arg("node_k"),
             "Surface coverage Γ at node k of capsule (Langmuir adsorption state)")
        .def("setNodeGamma", &AdvectionDiffusion::setNodeGamma,
             py::arg("capsule_id"), py::arg("node_k"), py::arg("gamma"));

    py::class_<LBMSolver>(m, "LBMSolver")
        .def(py::init<const SimulationParams&>(), py::arg("params"))
        .def("initialize", &LBMSolver::initialize)
        .def("step", &LBMSolver::step)
        .def("field", static_cast<LatticeField& (LBMSolver::*)()>(&LBMSolver::field),
             py::return_value_policy::reference_internal)
        .def("collision", &LBMSolver::collision, py::return_value_policy::reference_internal)
        .def("boundary", &LBMSolver::boundary, py::return_value_policy::reference_internal)
        .def("bounceBack", &LBMSolver::bounceBack, py::return_value_policy::reference_internal)
        .def("getNx", &LBMSolver::getNx)
        .def("getNy", &LBMSolver::getNy)
        .def("getStep", &LBMSolver::getStep)
        .def("isPeriodicX", &LBMSolver::isPeriodicX)
        .def("isPeriodicY", &LBMSolver::isPeriodicY)
        .def("setFluidRegion", &LBMSolver::setFluidRegion,
             py::arg("x0"), py::arg("y0"), py::arg("x1"), py::arg("y1"),
             py::arg("rho"), py::arg("ux"), py::arg("uy"))
        .def("rebuildBoundaryNodeLists", &LBMSolver::rebuildBoundaryNodeLists);

    // ═══════════════════════════════════════════════════════════════
    // Free-surface (wet-dry dam-break)
    // ═══════════════════════════════════════════════════════════════
    py::class_<FreeSurface>(m, "FreeSurface")
        .def(py::init<LBMSolver&, Real, Real>(),
             py::arg("solver"), py::arg("rho_atm") = 1.0, py::arg("threshold") = 0.002)
        .def("mark_empty",   &FreeSurface::markEmpty,
             py::arg("x0"), py::arg("y0"), py::arg("x1"), py::arg("y1"))
        .def("apply",        &FreeSurface::apply)
        .def("step",         &FreeSurface::step)
        .def("total_wetted", &FreeSurface::totalWetted)
        .def("is_originally_empty", &FreeSurface::isOriginallyEmpty,
             py::arg("x"), py::arg("y"));

    // ═══════════════════════════════════════════════════════════════
    // Membrane: Capsule and CapsuleSystem
    // ═══════════════════════════════════════════════════════════════

    py::class_<Capsule>(m, "Capsule")
        .def("nodePosition", &Capsule::nodePosition, py::arg("k"))
        .def("nodeVelocity", &Capsule::nodeVelocity, py::arg("k"))
        .def("nodeForce", &Capsule::nodeForce, py::arg("k"))
        .def("setNodeVelocity", &Capsule::setNodeVelocity, py::arg("k"), py::arg("v"))
        .def("centroid", &Capsule::centroid)
        .def("area", &Capsule::area)
        .def("perimeter", &Capsule::perimeter)
        .def("deformationIndex", &Capsule::deformationIndex)
        .def("inclinationAngle", &Capsule::inclinationAngle)
        .def("effectiveRadius", &Capsule::effectiveRadius)
        .def("numNodes", &Capsule::numNodes)
        .def("getId", &Capsule::getId)
        .def("getType", &Capsule::getType)
        .def("getTypeId", &Capsule::getTypeId)
        .def("clearForces", &Capsule::clearForces)
        .def("computeMembraneForces", &Capsule::computeMembraneForces)
        .def("moveNodes", &Capsule::moveNodes, py::arg("dt"))
        .def("setPeriodicX", &Capsule::setPeriodicX, py::arg("nx"))
        .def("getPeriodicX", &Capsule::getPeriodicX)
        .def("node_positions_array", [](const Capsule& c) {
            int n = c.numNodes();
            py::array_t<Real> arr({n, 2});
            auto buf = arr.mutable_unchecked<2>();
            for (int i = 0; i < n; ++i) {
                Vec2d p = c.nodePosition(i);
                buf(i, 0) = p.x;
                buf(i, 1) = p.y;
            }
            return arr;
        });

    py::class_<CapsuleSystem>(m, "CapsuleSystem")
        .def(py::init<>())
        .def("addCapsule", &CapsuleSystem::addCapsule,
             py::arg("center"), py::arg("radius"), py::arg("num_nodes") = 0,
             py::arg("params"), py::arg("type") = 0)
        .def("computeAllMembraneForces", &CapsuleSystem::computeAllMembraneForces)
        .def("clearAllForces", &CapsuleSystem::clearAllForces)
        .def("moveAllNodes", &CapsuleSystem::moveAllNodes, py::arg("dt"))
        .def("numCapsules", &CapsuleSystem::numCapsules)
        .def("totalNodes", &CapsuleSystem::totalNodes)
        .def("__getitem__", [](CapsuleSystem& cs, int i) -> Capsule& {
            if (i < 0 || i >= cs.numCapsules())
                throw py::index_error("Capsule index out of range");
            return cs[i];
        }, py::return_value_policy::reference_internal)
        .def("__len__", &CapsuleSystem::numCapsules);

    // ═══════════════════════════════════════════════════════════════
    // Membrane: RepulsionForce, Lubrication, Adhesion
    // ═══════════════════════════════════════════════════════════════

    py::class_<RepulsionForce>(m, "RepulsionForce")
        .def(py::init<const RepulsionParams&>(), py::arg("params"))
        .def("computeAll", &RepulsionForce::computeAll,
             py::arg("system"), py::arg("y_bottom"), py::arg("y_top"))
        .def("computeInterCapsule", &RepulsionForce::computeInterCapsule,
             py::arg("ci"), py::arg("cj"))
        .def("computeWallRepulsion", &RepulsionForce::computeWallRepulsion,
             py::arg("c"), py::arg("y_bottom"), py::arg("y_top"));

    py::class_<LubricationCorrection>(m, "LubricationCorrection")
        .def(py::init<const LubricationParams&, Real>(),
             py::arg("params"), py::arg("kinematic_viscosity"))
        .def("computeAll", &LubricationCorrection::computeAll,
             py::arg("capsules"), py::arg("ny"), py::arg("periodic_nx"));

    py::class_<Bond>(m, "Bond")
        .def_readonly("capsule_i", &Bond::capsule_i)
        .def_readonly("node_i", &Bond::node_i)
        .def_readonly("capsule_j", &Bond::capsule_j)
        .def_readonly("node_j", &Bond::node_j)
        .def_readonly("rest_length", &Bond::rest_length)
        .def_readonly("current_force", &Bond::current_force);

    py::class_<AdhesionModel>(m, "AdhesionModel")
        .def(py::init<const AdhesionParams&, unsigned>(),
             py::arg("params"), py::arg("seed") = 42)
        .def("update", &AdhesionModel::update,
             py::arg("capsules"), py::arg("dt"), py::arg("ny"), py::arg("periodic_nx"))
        .def("getBonds", &AdhesionModel::getBonds, py::return_value_policy::reference_internal)
        .def("getNumBonds", &AdhesionModel::getNumBonds)
        .def("getClusterIds", &AdhesionModel::getClusterIds, py::return_value_policy::reference_internal)
        .def("getClusterSizes", &AdhesionModel::getClusterSizes, py::return_value_policy::reference_internal)
        .def("getNumClusters", &AdhesionModel::getNumClusters)
        .def("getBondsForCapsule", &AdhesionModel::getBondsForCapsule, py::arg("capsule_id"));

    // ═══════════════════════════════════════════════════════════════
    // Coupling: ImmersedBoundary, ViscosityField
    // ═══════════════════════════════════════════════════════════════

    py::class_<ImmersedBoundary>(m, "ImmersedBoundary")
        .def(py::init<>())
        .def("spreadForces", &ImmersedBoundary::spreadForces,
             py::arg("capsules"), py::arg("field"))
        .def("interpolateVelocity", &ImmersedBoundary::interpolateVelocity,
             py::arg("field"), py::arg("capsules"))
        .def("multiDirectForcing", &ImmersedBoundary::multiDirectForcing,
             py::arg("capsules"), py::arg("field"), py::arg("iterations"));

    py::class_<ViscosityField>(m, "ViscosityField")
        .def(py::init<int, int, Real>(),
             py::arg("nx"), py::arg("ny"), py::arg("tau_out"))
        .def("update", &ViscosityField::update, py::arg("capsules"))
        .def("getTauLocal", &ViscosityField::getTauLocal,
             py::arg("x"), py::arg("y"));

    // ═══════════════════════════════════════════════════════════════
    // Geometry: Obstacles and ChannelBuilder
    // ═══════════════════════════════════════════════════════════════

    py::class_<CircleObstacle, std::shared_ptr<CircleObstacle>>(m, "CircleObstacle")
        .def(py::init<Real, Real, Real>(), py::arg("cx"), py::arg("cy"), py::arg("radius"))
        .def("getCx", &CircleObstacle::getCx)
        .def("getCy", &CircleObstacle::getCy)
        .def("getRadius", &CircleObstacle::getRadius)
        .def("contains", &CircleObstacle::contains, py::arg("x"), py::arg("y"))
        .def("signedDistance", &CircleObstacle::signedDistance, py::arg("x"), py::arg("y"))
        .def("nearestPoint", &CircleObstacle::nearestPoint, py::arg("x"), py::arg("y"))
        .def("normalAt", &CircleObstacle::normalAt, py::arg("x"), py::arg("y"));

    py::class_<RectObstacle, std::shared_ptr<RectObstacle>>(m, "RectObstacle")
        .def(py::init<Real, Real, Real, Real>(),
             py::arg("x0"), py::arg("y0"), py::arg("x1"), py::arg("y1"))
        .def("contains", &RectObstacle::contains, py::arg("x"), py::arg("y"))
        .def("signedDistance", &RectObstacle::signedDistance, py::arg("x"), py::arg("y"))
        .def("nearestPoint", &RectObstacle::nearestPoint, py::arg("x"), py::arg("y"))
        .def("normalAt", &RectObstacle::normalAt, py::arg("x"), py::arg("y"))
        .def("getX0", &RectObstacle::getX0)
        .def("getY0", &RectObstacle::getY0)
        .def("getX1", &RectObstacle::getX1)
        .def("getY1", &RectObstacle::getY1);

    py::class_<PolygonObstacle, std::shared_ptr<PolygonObstacle>>(m, "PolygonObstacle")
        .def(py::init<const std::vector<Vec2d>&>(), py::arg("vertices"))
        .def("contains", &PolygonObstacle::contains, py::arg("x"), py::arg("y"))
        .def("signedDistance", &PolygonObstacle::signedDistance, py::arg("x"), py::arg("y"))
        .def("nearestPoint", &PolygonObstacle::nearestPoint, py::arg("x"), py::arg("y"))
        .def("normalAt", &PolygonObstacle::normalAt, py::arg("x"), py::arg("y"));

    py::class_<ChannelBuilder>(m, "ChannelBuilder")
        .def(py::init<int, int>(), py::arg("nx"), py::arg("ny"))
        .def("addWalls", &ChannelBuilder::addWalls,
             py::return_value_policy::reference_internal)
        .def("addCirclePillar", &ChannelBuilder::addCirclePillar,
             py::arg("cx"), py::arg("cy"), py::arg("radius"),
             py::return_value_policy::reference_internal)
        .def("addRectPillar", &ChannelBuilder::addRectPillar,
             py::arg("x0"), py::arg("y0"), py::arg("x1"), py::arg("y1"),
             py::return_value_policy::reference_internal)
        .def("setBoundaryType", &ChannelBuilder::setBoundaryType, py::arg("bt"),
             py::return_value_policy::reference_internal)
        .def("getBoundaryType", &ChannelBuilder::getBoundaryType)
        .def("hasTopWall", &ChannelBuilder::hasTopWall)
        .def("hasBottomWall", &ChannelBuilder::hasBottomWall)
        .def("getNx", &ChannelBuilder::getNx)
        .def("getNy", &ChannelBuilder::getNy)
        .def("wallBottom", &ChannelBuilder::wallBottom)
        .def("wallTop", &ChannelBuilder::wallTop);

    // ═══════════════════════════════════════════════════════════════
    // Analysis: Segregation metrics
    // ═══════════════════════════════════════════════════════════════

    py::class_<SegregationResults>(m, "SegregationResults")
        .def(py::init<>())
        .def_readwrite("lateral_distribution", &SegregationResults::lateral_distribution)
        .def_readwrite("margination_parameter", &SegregationResults::margination_parameter)
        .def_readwrite("mixing_entropy", &SegregationResults::mixing_entropy)
        .def_readwrite("separation_efficiency", &SegregationResults::separation_efficiency)
        .def_readwrite("cfl_bottom", &SegregationResults::cfl_bottom)
        .def_readwrite("cfl_top", &SegregationResults::cfl_top)
        .def_readwrite("num_clusters", &SegregationResults::num_clusters)
        .def_readwrite("mean_cluster_size", &SegregationResults::mean_cluster_size)
        .def_readwrite("max_cluster_size", &SegregationResults::max_cluster_size)
        .def_readwrite("mean_deformation", &SegregationResults::mean_deformation)
        .def_readwrite("std_deformation", &SegregationResults::std_deformation)
        .def_readwrite("velocity_profile", &SegregationResults::velocity_profile)
        .def_readwrite("rdf_r", &SegregationResults::rdf_r)
        .def_readwrite("rdf_g", &SegregationResults::rdf_g);

    py::class_<SegregationMetrics>(m, "SegregationMetrics")
        .def(py::init<int, int>(), py::arg("ny"), py::arg("n_bins") = 20)
        .def("compute", &SegregationMetrics::compute,
             py::arg("capsules"), py::arg("adhesion") = nullptr)
        .def("writeCSV", &SegregationMetrics::writeCSV,
             py::arg("filename"), py::arg("results"), py::arg("step"), py::arg("time"));

    // ═══════════════════════════════════════════════════════════════
    // ML: Features, training data, and neural network
    // ═══════════════════════════════════════════════════════════════

    py::class_<MLFeatures>(m, "MLFeatures")
        .def(py::init<>())
        .def_readwrite("rel_vel_x", &MLFeatures::rel_vel_x)
        .def_readwrite("rel_vel_y", &MLFeatures::rel_vel_y)
        .def_readwrite("Re_p", &MLFeatures::Re_p)
        .def_readwrite("local_rho", &MLFeatures::local_rho)
        .def_readwrite("grad_rho_x", &MLFeatures::grad_rho_x)
        .def_readwrite("grad_rho_y", &MLFeatures::grad_rho_y)
        .def_readwrite("wall_distance", &MLFeatures::wall_distance)
        .def_readwrite("radius_normalized", &MLFeatures::radius_normalized)
        .def_readwrite("local_solid_fraction", &MLFeatures::local_solid_fraction)
        .def("toVector", &MLFeatures::toVector);

    py::class_<TrainingData>(m, "TrainingData")
        .def(py::init<int>(), py::arg("max_size") = 10000)
        .def("addSample", &TrainingData::addSample, py::arg("features"), py::arg("force"))
        .def("clear", &TrainingData::clear)
        .def("size", &TrainingData::size)
        .def("isFull", &TrainingData::isFull)
        .def("getInputs", &TrainingData::getInputs, py::return_value_policy::reference_internal)
        .def("getTargets", &TrainingData::getTargets, py::return_value_policy::reference_internal);

    py::class_<SimpleNN>(m, "SimpleNN")
        .def(py::init<const MLParams&>(), py::arg("params"))
        .def("predictForce", &SimpleNN::predictForce, py::arg("features"))
        .def("train", &SimpleNN::train, py::arg("inputs"), py::arg("targets"))
        .def("isReady", &SimpleNN::isReady)
        .def("saveWeights", &SimpleNN::saveWeights, py::arg("path"))
        .def("loadWeights", &SimpleNN::loadWeights, py::arg("path"));

    // ═══════════════════════════════════════════════════════════════
    // I/O: Output configuration structs
    // ═══════════════════════════════════════════════════════════════

    py::class_<FluidOutputFields>(m, "FluidOutputFields")
        .def(py::init<>())
        .def_readwrite("density", &FluidOutputFields::density)
        .def_readwrite("velocity", &FluidOutputFields::velocity)
        .def_readwrite("pressure", &FluidOutputFields::pressure)
        .def_readwrite("vorticity", &FluidOutputFields::vorticity)
        .def_readwrite("strain_rate", &FluidOutputFields::strain_rate)
        .def_readwrite("ibm_force", &FluidOutputFields::ibm_force)
        .def_readwrite("node_type", &FluidOutputFields::node_type)
        .def_readwrite("velocity_mag", &FluidOutputFields::velocity_mag)
        .def_readwrite("component_density", &FluidOutputFields::component_density)
        .def_readwrite("concentration", &FluidOutputFields::concentration);

    py::class_<ParticleVTKFields>(m, "ParticleVTKFields")
        .def(py::init<>())
        .def_readwrite("velocity", &ParticleVTKFields::velocity)
        .def_readwrite("force", &ParticleVTKFields::force)
        .def_readwrite("particle_id", &ParticleVTKFields::particle_id)
        .def_readwrite("particle_type", &ParticleVTKFields::particle_type)
        .def_readwrite("particle_group", &ParticleVTKFields::particle_group)
        .def_readwrite("local_strain", &ParticleVTKFields::local_strain)
        .def_readwrite("local_curvature", &ParticleVTKFields::local_curvature);

    py::class_<ParticleOutputFields>(m, "ParticleOutputFields")
        .def(py::init<>())
        .def_readwrite("position", &ParticleOutputFields::position)
        .def_readwrite("velocity", &ParticleOutputFields::velocity)
        .def_readwrite("force", &ParticleOutputFields::force)
        .def_readwrite("diameter", &ParticleOutputFields::diameter)
        .def_readwrite("type", &ParticleOutputFields::type)
        .def_readwrite("group", &ParticleOutputFields::group)
        .def_readwrite("deformation", &ParticleOutputFields::deformation)
        .def_readwrite("area_volume", &ParticleOutputFields::area_volume)
        .def_readwrite("orientation", &ParticleOutputFields::orientation)
        .def_readwrite("angular_vel", &ParticleOutputFields::angular_vel)
        .def_readwrite("com_position", &ParticleOutputFields::com_position)
        .def_readwrite("com_velocity", &ParticleOutputFields::com_velocity);

    py::enum_<ParticleFilter::FilterType>(m, "FilterType")
        .value("ALL", ParticleFilter::ALL)
        .value("BY_ID", ParticleFilter::BY_ID)
        .value("BY_TYPE", ParticleFilter::BY_TYPE)
        .value("BY_GROUP", ParticleFilter::BY_GROUP)
        .value("BY_REGION", ParticleFilter::BY_REGION)
        .value("COMBINED", ParticleFilter::COMBINED)
        .export_values();

    py::class_<ParticleFilter>(m, "ParticleFilter")
        .def(py::init<>())
        .def_readwrite("filter_type", &ParticleFilter::filter_type)
        .def_readwrite("selected_ids", &ParticleFilter::selected_ids)
        .def_readwrite("selected_types", &ParticleFilter::selected_types)
        .def_readwrite("selected_groups", &ParticleFilter::selected_groups)
        .def_readwrite("region_xmin", &ParticleFilter::region_xmin)
        .def_readwrite("region_xmax", &ParticleFilter::region_xmax)
        .def_readwrite("region_ymin", &ParticleFilter::region_ymin)
        .def_readwrite("region_ymax", &ParticleFilter::region_ymax)
        .def("passes", &ParticleFilter::passes,
             py::arg("id"), py::arg("type_id"), py::arg("group"),
             py::arg("cx"), py::arg("cy"));

    py::class_<FluidProbe>(m, "FluidProbe")
        .def(py::init<>())
        .def_readwrite("i", &FluidProbe::i)
        .def_readwrite("j", &FluidProbe::j)
        .def_readwrite("label", &FluidProbe::label);

    py::class_<OutputConfig>(m, "OutputConfig")
        .def(py::init<>())
        .def_readwrite("output_dir", &OutputConfig::output_dir)
        .def_readwrite("vtk_dump_every", &OutputConfig::vtk_dump_every)
        .def_readwrite("vtk_format", &OutputConfig::vtk_format)
        .def_readwrite("csv_dump_every", &OutputConfig::csv_dump_every)
        .def_readwrite("csv_format", &OutputConfig::csv_format)
        .def_readwrite("csv_append", &OutputConfig::csv_append)
        .def_readwrite("probe_dump_every", &OutputConfig::probe_dump_every)
        .def_readwrite("stats_dump_every", &OutputConfig::stats_dump_every)
        .def_readwrite("fluid_fields", &OutputConfig::fluid_fields)
        .def_readwrite("particle_vtk_fields", &OutputConfig::particle_vtk_fields)
        .def_readwrite("csv_fields", &OutputConfig::csv_fields)
        .def_readwrite("csv_filter", &OutputConfig::csv_filter)
        .def_readwrite("probes", &OutputConfig::probes);

    // ═══════════════════════════════════════════════════════════════
    // I/O: Profiler
    // ═══════════════════════════════════════════════════════════════

    py::class_<Profiler>(m, "Profiler")
        .def("printReport", &Profiler::printReport, py::arg("total_steps"))
        .def("reset", &Profiler::reset);

    // ═══════════════════════════════════════════════════════════════
    // I/O: VTK, CSV writers, Checkpoint
    // ═══════════════════════════════════════════════════════════════

    py::class_<VTKWriter>(m, "VTKWriter")
        .def(py::init<const std::string&>(), py::arg("output_dir"))
        .def("setFluidFields", &VTKWriter::setFluidFields, py::arg("fields"))
        .def("setParticleFields", &VTKWriter::setParticleFields, py::arg("fields"))
        .def("setFormat", &VTKWriter::setFormat, py::arg("format"))
        .def("setPeriodicX", &VTKWriter::setPeriodicX, py::arg("nx"))
        .def("writeFluidField", &VTKWriter::writeFluidField,
             py::arg("field"), py::arg("step"))
        .def("writeCapsules", &VTKWriter::writeCapsules,
             py::arg("capsules"), py::arg("step"))
        .def("writePVDFiles", &VTKWriter::writePVDFiles)
        .def("setLegacyMode", &VTKWriter::setLegacyMode, py::arg("legacy"))
        .def("recordTimestep", &VTKWriter::recordTimestep,
             py::arg("step"), py::arg("time"));

    py::class_<CSVWriter>(m, "CSVWriter")
        .def(py::init<const std::string&, const std::string&, const std::string&, bool>(),
             py::arg("output_dir"),
             py::arg("filename") = "particle_data.csv",
             py::arg("format") = "csv",
             py::arg("append") = true)
        .def("setFields", &CSVWriter::setFields, py::arg("fields"))
        .def("setFilter", &CSVWriter::setFilter, py::arg("filter"))
        .def("writeTimestep", &CSVWriter::writeTimestep,
             py::arg("capsules"), py::arg("step"), py::arg("time"))
        .def("close", &CSVWriter::close);

    // ═══════════════════════════════════════════════════════════════
    // Engine: Top-level Simulation
    // ═══════════════════════════════════════════════════════════════

    py::class_<Simulation>(m, "Simulation")
        .def(py::init<const SimulationParams&>(), py::arg("params"))
        .def("setChannelBuilder", &Simulation::setChannelBuilder, py::arg("builder"))
        .def("addCapsule", &Simulation::addCapsule,
             py::arg("center"), py::arg("radius"), py::arg("num_nodes") = 0,
             py::arg("mparams"), py::arg("type") = 0)
        .def("addCapsuleRandom", &Simulation::addCapsuleRandom,
             py::arg("count"), py::arg("x0"), py::arg("y0"),
             py::arg("x1"), py::arg("y1"),
             py::arg("radius_min"), py::arg("radius_max"),
             py::arg("num_nodes") = 0, py::arg("mparams"), py::arg("type") = 0,
             py::arg("seed") = 12345, py::arg("min_gap") = 1.0,
             py::arg("max_attempts") = 0)
        .def("initialize", &Simulation::initialize)
        .def("step", &Simulation::step)
        .def("finalize", &Simulation::finalize)
        .def("run", &Simulation::run, py::arg("num_steps"),
             py::call_guard<py::gil_scoped_release>())
        .def("currentStep", &Simulation::currentStep)
        .def("setCurrentStep", &Simulation::setCurrentStep, py::arg("step"))
        .def("saveCheckpoint", &Simulation::saveCheckpoint, py::arg("filename"))
        .def("loadCheckpoint", &Simulation::loadCheckpoint, py::arg("filename"))
        .def("getMaxSpeed", &Simulation::getMaxSpeed)
        .def("lbmSolver", static_cast<LBMSolver& (Simulation::*)()>(&Simulation::lbmSolver),
             py::return_value_policy::reference_internal)
        .def("field", [](Simulation& sim) -> LatticeField& {
            return sim.lbmSolver().field();
        }, py::return_value_policy::reference_internal)
        .def("capsules", [](Simulation& sim) -> CapsuleSystem& {
            return sim.capsules();
        }, py::return_value_policy::reference_internal)
        .def("params", [](const Simulation& sim) -> const SimulationParams& {
            return sim.params();
        }, py::return_value_policy::reference_internal)
        // New feature accessors
        .def("advectionDiffusion", [](Simulation& sim) -> AdvectionDiffusion* {
            return sim.advectionDiffusion();
        }, py::return_value_policy::reference_internal)
        .def("adhesion", [](Simulation& sim) -> AdhesionModel* {
            return sim.adhesion();
        }, py::return_value_policy::reference_internal)
        .def("lastSegregationResults", &Simulation::lastSegregationResults,
             py::return_value_policy::reference_internal)
        .def("setFluidRegion", &Simulation::setFluidRegion,
             py::arg("x0"), py::arg("y0"), py::arg("x1"), py::arg("y1"),
             py::arg("rho"), py::arg("ux"), py::arg("uy"))
        .def("enableFreeSurface", &Simulation::enableFreeSurface,
             py::arg("rho_atm") = 1.0, py::arg("threshold") = 0.002)
        .def("setEmptyRegion", &Simulation::setEmptyRegion,
             py::arg("x0"), py::arg("y0"), py::arg("x1"), py::arg("y1"))
        .def("freeSurface", [](Simulation& s) -> FreeSurface* {
             return s.freeSurface();
         }, py::return_value_policy::reference_internal)
        .def("setScalarRegion", &Simulation::setScalarRegion,
             py::arg("x0"), py::arg("y0"), py::arg("x1"), py::arg("y1"),
             py::arg("concentration"), py::arg("species") = 0)
        .def("setScalarReleaseRate", &Simulation::setScalarReleaseRate,
             py::arg("capsule_type"), py::arg("rate"))
        .def("setScalarAbsorptionRate", &Simulation::setScalarAbsorptionRate,
             py::arg("capsule_type"), py::arg("rate"))
        .def("setLeachingParams", &Simulation::setLeachingParams,
             py::arg("capsule_type"), py::arg("k_leach"), py::arg("C_eq"),
             "Physics-based Fick leaching: J = k_leach*(C_eq - C_surface)")
        .def("setAdsorptionParams", &Simulation::setAdsorptionParams,
             py::arg("capsule_type"), py::arg("k_a"), py::arg("k_d"), py::arg("Gamma_max"),
             "Langmuir adsorption/desorption: dΓ/dt = k_a*C*(1-Γ/Γ_max) - k_d*Γ")
        .def("setParticleMass", &Simulation::setParticleMass,
             py::arg("capsule_type"), py::arg("Mp0"),
             "Initial particle chemical reservoir mass (0 = infinite)")
        .def("saveCheckpoint", &Simulation::saveCheckpoint, py::arg("filename"))
        .def("loadCheckpoint", &Simulation::loadCheckpoint, py::arg("filename"))
        .def("setStepCallback", [](Simulation& sim, py::function cb) {
            sim.setStepCallback([cb](Simulation& s, int step) {
                py::gil_scoped_acquire acquire;
                cb(py::cast(s, py::return_value_policy::reference), step);
            });
        }, py::arg("callback"))
        // Output configuration
        .def("setOutputConfig", &Simulation::setOutputConfig, py::arg("config"))
        .def("setVTKOutput", &Simulation::setVTKOutput,
             py::arg("dump_every"), py::arg("format"),
             py::arg("fluid_fields"), py::arg("particle_fields"))
        .def("setCSVOutput", &Simulation::setCSVOutput,
             py::arg("dump_every"), py::arg("format"),
             py::arg("fields"), py::arg("filter"),
             py::arg("append") = true)
        .def("addExtraCSVOutput", &Simulation::addExtraCSVOutput,
             py::arg("filename"), py::arg("dump_every"),
             py::arg("fields"), py::arg("filter"))
        .def("addFluidProbe", &Simulation::addFluidProbe,
             py::arg("i"), py::arg("j"), py::arg("label"))
        // Viscosity field access
        .def("viscosityField", [](Simulation& sim) -> ViscosityField* {
            return sim.viscosityField();
        }, py::return_value_policy::reference_internal)
        // Segregation metrics access
        .def("segregationMetrics", [](Simulation& sim) -> SegregationMetrics* {
            return sim.segregationMetrics();
        }, py::return_value_policy::reference_internal)
        // Profiler access
        .def("profiler", &Simulation::profiler,
             py::return_value_policy::reference_internal)

        // ── Insertion API (Phase 2) ──
        // Static fill: builds an InsertionContext from the current
        // sim state, drives the inserter once, and adds the resulting
        // capsules. Returns the count actually placed.
        .def("insertCapsules",
             [](Simulation& sim, insertion::IInserter& inserter,
                const MembraneParams& mp, int type, Real min_gap,
                int num_nodes, const std::string& seed_tag) {
                 return sim.insertCapsules(inserter, mp, type, min_gap,
                                           num_nodes, seed_tag);
             },
             py::arg("inserter"), py::arg("mparams"),
             py::arg("type") = 0, py::arg("min_gap") = 1.0,
             py::arg("num_nodes") = 0,
             py::arg("seed_tag") = std::string("default"))

        // Dynamic registration. The simulation shares ownership of
        // the inserter; the user-side Python handle can go out of
        // scope freely after registration.
        .def("registerDynamicInserter",
             [](Simulation& sim,
                std::shared_ptr<insertion::IDynamicInserter> ins,
                const MembraneParams& mp, int type, Real min_gap,
                int num_nodes, const std::string& seed_tag) {
                 sim.registerDynamicInserter(std::move(ins), mp, type,
                                              min_gap, num_nodes,
                                              seed_tag);
             },
             py::arg("inserter"), py::arg("mparams"),
             py::arg("type") = 0, py::arg("min_gap") = 1.0,
             py::arg("num_nodes") = 0,
             py::arg("seed_tag") = std::string("dynamic"))
        .def("numDynamicInserters", &Simulation::numDynamicInserters);

    // ═══════════════════════════════════════════════════════════════
    //  Insertion submodule (Phase 2)
    // ═══════════════════════════════════════════════════════════════
    //
    // Layout: softflow_core.insertion exposes Region (abstract) plus
    // concrete RectRegion / CircleRegion / PolygonRegion /
    // ImageMaskRegion; SizeDistribution (abstract) plus Monodisperse /
    // Bidisperse / Lognormal / UserDiscrete; Inserter (abstract) plus
    // SquareLattice / HexagonalLattice / RSA / PoissonDisk; and
    // DynamicInserter (abstract) plus PoissonStochastic / ConstantFlux
    // / Conveyor.
    //
    // All classes are wrapped in std::shared_ptr (for the static-fill
    // and region/size hierarchies) or std::unique_ptr (for dynamic
    // inserters, which the Simulation takes ownership of). This
    // matches the C++ side: `IInserter`s are intended to be passed by
    // shared_ptr because regions and size distributions are aliased
    // across multiple inserters.
    auto ins_mod = m.def_submodule("insertion",
        "Particle insertion module (Phase 2). Static layouts: "
        "SquareLattice, HexagonalLattice, RSA, PoissonDisk. Dynamic: "
        "PoissonStochastic, ConstantFlux, Conveyor. Regions: Rect, "
        "Circle, Polygon, ImageMask.");

    // ── Regions ─────────────────────────────────────────────────────
    py::class_<insertion::IRegion, std::shared_ptr<insertion::IRegion>>(
        ins_mod, "Region")
        .def("contains", &insertion::IRegion::contains, py::arg("point"))
        .def("bbox",     &insertion::IRegion::bbox)
        .def("area",     &insertion::IRegion::area);

    py::class_<insertion::RectRegion, insertion::IRegion,
               std::shared_ptr<insertion::RectRegion>>(ins_mod, "RectRegion")
        .def(py::init<Real, Real, Real, Real>(),
             py::arg("x0"), py::arg("x1"), py::arg("y0"), py::arg("y1"))
        .def_property_readonly("x0", &insertion::RectRegion::x0)
        .def_property_readonly("x1", &insertion::RectRegion::x1)
        .def_property_readonly("y0", &insertion::RectRegion::y0)
        .def_property_readonly("y1", &insertion::RectRegion::y1);

    py::class_<insertion::CircleRegion, insertion::IRegion,
               std::shared_ptr<insertion::CircleRegion>>(ins_mod, "CircleRegion")
        .def(py::init<Vec2d, Real>(), py::arg("center"), py::arg("radius"))
        .def_property_readonly("center", &insertion::CircleRegion::center)
        .def_property_readonly("radius", &insertion::CircleRegion::radius);

    py::class_<insertion::PolygonRegion, insertion::IRegion,
               std::shared_ptr<insertion::PolygonRegion>>(ins_mod, "PolygonRegion")
        .def(py::init<std::vector<Vec2d>>(), py::arg("vertices"))
        .def_property_readonly("vertices", &insertion::PolygonRegion::vertices);

    py::class_<insertion::ImageMaskRegion, insertion::IRegion,
               std::shared_ptr<insertion::ImageMaskRegion>>(ins_mod, "ImageMaskRegion")
        .def(py::init<std::vector<std::uint8_t>, int, int, Vec2d, Real, std::uint8_t>(),
             py::arg("pixels"), py::arg("width"), py::arg("height"),
             py::arg("origin"), py::arg("scale"), py::arg("threshold") = 127)
        .def_static("fromPGM", &insertion::ImageMaskRegion::fromPGM,
             py::arg("path"), py::arg("origin"), py::arg("scale"),
             py::arg("threshold") = 127);

    // ── Size distributions ─────────────────────────────────────────
    py::class_<insertion::ISizeDistribution,
               std::shared_ptr<insertion::ISizeDistribution>>(
        ins_mod, "SizeDistribution")
        .def("minRadius", &insertion::ISizeDistribution::minRadius)
        .def("maxRadius", &insertion::ISizeDistribution::maxRadius);

    py::class_<insertion::Monodisperse, insertion::ISizeDistribution,
               std::shared_ptr<insertion::Monodisperse>>(ins_mod, "Monodisperse")
        .def(py::init<Real>(), py::arg("radius"));

    py::class_<insertion::Bidisperse, insertion::ISizeDistribution,
               std::shared_ptr<insertion::Bidisperse>>(ins_mod, "Bidisperse")
        .def(py::init<Real, Real, Real>(),
             py::arg("r_small"), py::arg("r_large"),
             py::arg("fraction_small"));

    py::class_<insertion::Lognormal, insertion::ISizeDistribution,
               std::shared_ptr<insertion::Lognormal>>(ins_mod, "Lognormal")
        .def(py::init<Real, Real, Real, Real>(),
             py::arg("mu_log"), py::arg("sigma_log"),
             py::arg("r_min"), py::arg("r_max"));

    py::class_<insertion::UserDiscrete, insertion::ISizeDistribution,
               std::shared_ptr<insertion::UserDiscrete>>(ins_mod, "UserDiscrete")
        .def(py::init<std::vector<Real>, std::vector<Real>>(),
             py::arg("radii"), py::arg("weights"));

    // ── Static inserters (passed by reference to Simulation::insert) ─
    py::class_<insertion::IInserter,
               std::shared_ptr<insertion::IInserter>>(ins_mod, "Inserter");

    py::class_<insertion::SquareLatticeInserter, insertion::IInserter,
               std::shared_ptr<insertion::SquareLatticeInserter>>(
        ins_mod, "SquareLatticeInserter")
        .def(py::init<std::shared_ptr<insertion::IRegion>, Real, Real,
                      std::shared_ptr<insertion::ISizeDistribution>, Real>(),
             py::arg("region"), py::arg("spacing_x"), py::arg("spacing_y"),
             py::arg("sizes"), py::arg("jitter") = 0.0);

    py::class_<insertion::HexagonalLatticeInserter, insertion::IInserter,
               std::shared_ptr<insertion::HexagonalLatticeInserter>>(
        ins_mod, "HexagonalLatticeInserter")
        .def(py::init<std::shared_ptr<insertion::IRegion>, Real,
                      std::shared_ptr<insertion::ISizeDistribution>, Real>(),
             py::arg("region"), py::arg("spacing"),
             py::arg("sizes"), py::arg("jitter") = 0.0);

    py::class_<insertion::RSAInserter, insertion::IInserter,
               std::shared_ptr<insertion::RSAInserter>>(ins_mod, "RSAInserter")
        .def(py::init<std::shared_ptr<insertion::IRegion>, int,
                      std::shared_ptr<insertion::ISizeDistribution>, int>(),
             py::arg("region"), py::arg("target_count"),
             py::arg("sizes"), py::arg("max_attempts") = 0);

    py::class_<insertion::PoissonDiskInserter, insertion::IInserter,
               std::shared_ptr<insertion::PoissonDiskInserter>>(
        ins_mod, "PoissonDiskInserter")
        .def(py::init<std::shared_ptr<insertion::IRegion>, Real,
                      std::shared_ptr<insertion::ISizeDistribution>, int>(),
             py::arg("region"), py::arg("r_min"),
             py::arg("sizes"), py::arg("k") = 30)
        .def("lastMinSeparation",
             &insertion::PoissonDiskInserter::lastMinSeparation);

    // ── Dynamic inserters ──
    //
    // Same shared_ptr holder convention as static inserters: the
    // Simulation registers them via shared_ptr so pybind11 can
    // round-trip ownership without the unique_ptr corner cases that
    // bite cross-hierarchy casting.
    py::class_<insertion::IDynamicInserter,
               std::shared_ptr<insertion::IDynamicInserter>>(
        ins_mod, "DynamicInserter");

    py::class_<insertion::PoissonStochasticInserter, insertion::IDynamicInserter,
               std::shared_ptr<insertion::PoissonStochasticInserter>>(
        ins_mod, "PoissonStochasticInserter")
        .def(py::init<std::shared_ptr<insertion::IRegion>, Real,
                      std::shared_ptr<insertion::ISizeDistribution>, int>(),
             py::arg("region"), py::arg("rate"),
             py::arg("sizes"), py::arg("attempts_per_event") = 16);

    py::class_<insertion::ConstantFluxInserter, insertion::IDynamicInserter,
               std::shared_ptr<insertion::ConstantFluxInserter>>(
        ins_mod, "ConstantFluxInserter")
        .def(py::init<std::shared_ptr<insertion::IRegion>, Real,
                      std::shared_ptr<insertion::ISizeDistribution>, int, int>(),
             py::arg("region"), py::arg("target_phi"),
             py::arg("sizes"),
             py::arg("max_per_step") = 4,
             py::arg("attempts_per_event") = 32);

    py::class_<insertion::ConveyorInserter, insertion::IDynamicInserter,
               std::shared_ptr<insertion::ConveyorInserter>>(
        ins_mod, "ConveyorInserter")
        .def(py::init<std::shared_ptr<insertion::IRegion>, int,
                      std::shared_ptr<insertion::ISizeDistribution>, int, int>(),
             py::arg("region"), py::arg("target_count"),
             py::arg("sizes"),
             py::arg("max_per_step") = 4,
             py::arg("attempts_per_event") = 32);
}
