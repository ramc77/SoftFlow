// Smoke test for the run_manifest.json writer.
//
// Drives a 5-step LBM-only simulation in a temp directory, then reads
// back the generated run_manifest.json and asserts that the build
// provenance, RNG seed, force cap, and a few canonical fluid fields
// landed in the output. A future commit should add a JSON parser-based
// schema check; for now we use raw string-search to keep the test self-
// contained (no nlohmann/json dependency in tests/).

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "engine/simulation.h"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

namespace fs = std::filesystem;
using namespace softflow;

namespace {

std::string readFile(const fs::path& p) {
    std::ifstream f(p);
    REQUIRE(f.is_open());
    std::ostringstream os;
    os << f.rdbuf();
    return os.str();
}

}  // namespace

TEST_CASE("run_manifest.json captures build provenance + full params") {
    // Use a unique tmp dir per test invocation so parallel ctest is safe.
    fs::path tmp = fs::temp_directory_path() /
                   ("softflow_manifest_" + std::to_string(::getpid()));
    fs::remove_all(tmp);
    fs::create_directories(tmp);

    SimulationParams p;
    p.nx = 16;
    p.ny = 8;
    p.dt = 1.0;
    p.num_steps = 5;
    p.fluid.boundary_type   = BoundaryType::PERIODIC;
    p.fluid.body_force_x    = 1e-6;
    p.fluid.collision_model = CollisionModel::BGK;
    p.output_dir            = tmp.string();
    p.vtk_format            = "ascii";  // modern layout: config/ subdir
    p.rng_seed              = 0xC0FFEEull;
    p.max_lattice_force     = 0.005;
    p.enable_profiling      = false;

    Simulation sim(p);
    sim.initialize();
    for (int i = 0; i < p.num_steps; ++i) sim.step();
    sim.finalize();

    fs::path manifest = tmp / "config" / "run_manifest.json";
    REQUIRE(fs::exists(manifest));

    std::string m = readFile(manifest);

    // Build provenance block exists and includes the keys we need to
    // trace a result back to its compile.
    CHECK(m.find("\"build\":")            != std::string::npos);
    CHECK(m.find("\"git_sha\":")          != std::string::npos);
    CHECK(m.find("\"git_dirty\":")        != std::string::npos);
    CHECK(m.find("\"compiler_id\":")      != std::string::npos);
    CHECK(m.find("\"compiler_version\":") != std::string::npos);
    CHECK(m.find("\"cxx_standard\":")     != std::string::npos);
    CHECK(m.find("\"cxx_flags\":")        != std::string::npos);
    CHECK(m.find("\"openmp_threads\":")   != std::string::npos);

    // Reproducibility seed is the canonical params_.rng_seed.
    CHECK(m.find("\"rng_seed\": 12648430") != std::string::npos);

    // Force cap is captured.
    CHECK(m.find("\"max_lattice_force\":") != std::string::npos);

    // Enums are serialized as string names, not integer codes.
    CHECK(m.find("\"boundary_type\": \"PERIODIC\"")    != std::string::npos);
    CHECK(m.find("\"collision_model\": \"BGK\"")       != std::string::npos);
    CHECK(m.find("\"membrane\":")                      != std::string::npos);

    // Cleanup
    fs::remove_all(tmp);
}
