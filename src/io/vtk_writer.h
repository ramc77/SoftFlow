#pragma once
#include "../core/types.h"
#include "output_config.h"
#include <string>
#include <vector>
#include <cstdio>

namespace softflow {

class LatticeField;
class CapsuleSystem;
class Capsule;
class AdvectionDiffusion;

class VTKWriter {
public:
    explicit VTKWriter(const std::string& output_dir);

    // Set periodic wrapping for capsule output
    void setPeriodicX(int nx) { periodic_nx_ = nx; }

    // Configure output fields
    void setFluidFields(const FluidOutputFields& fields) { fluid_fields_ = fields; }
    void setParticleFields(const ParticleVTKFields& fields) { particle_fields_ = fields; }
    void setFormat(const std::string& fmt) { format_ = fmt; } // "ascii" or "binary"

    // Write fluid field as VTK ImageData (.vti) with XML format
    void writeFluidField(const LatticeField& field, int step);

    // Write capsule data as VTK PolyData (.vtp)
    void writeCapsules(const CapsuleSystem& capsules, int step);

    // Legacy VTK format (.vtk) — works with all ParaView versions
    void writeFluidFieldLegacy(const LatticeField& field, int step);
    void writeCapsulesLegacy(const CapsuleSystem& capsules, int step);

    // Set legacy mode (outputs .vtk instead of .vti/.vtp)
    void setLegacyMode(bool legacy) { legacy_mode_ = legacy; }

    // Set advection-diffusion solver for concentration output
    void setAdvectionDiffusion(const AdvectionDiffusion* ad) { advection_diffusion_ = ad; }

    // Write .pvd collection files for ParaView time-series
    void writePVDFiles() const;

    // Record timestep for PVD file
    void recordTimestep(int step, double time);

private:
    std::string output_dir_;
    int periodic_nx_ = 0;
    std::string format_ = "ascii";
    bool legacy_mode_ = false;
    FluidOutputFields fluid_fields_;
    ParticleVTKFields particle_fields_;
    const AdvectionDiffusion* advection_diffusion_ = nullptr;

    // Recorded timesteps for PVD files
    std::vector<int> recorded_steps_;
    std::vector<double> recorded_times_;

    mutable bool dirs_created_ = false;
    void ensureDirs() const;

    // Vorticity computation (2D: scalar = duy/dx - dux/dy)
    static Real computeVorticity(const LatticeField& field, int x, int y);

    // Strain rate magnitude: |gamma_dot| = sqrt(2*S_ij*S_ij)
    static Real computeStrainRateMagnitude(const LatticeField& field, int x, int y);
};

} // namespace softflow
