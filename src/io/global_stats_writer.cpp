#include "global_stats_writer.h"
#include "../lbm/lattice_field.h"
#include "../lbm/advection_diffusion.h"
#include "../membrane/capsule_system.h"
#include "../membrane/capsule.h"
#include <sys/stat.h>
#include <cmath>
#include <algorithm>

namespace softflow {

GlobalStatsWriter::GlobalStatsWriter(const std::string& output_dir)
    : output_dir_(output_dir)
{
    mkdir(output_dir_.c_str(), 0755);
}

GlobalStatsWriter::~GlobalStatsWriter() {
    close();
}

void GlobalStatsWriter::openFile() {
    if (fp_) return;
    std::string path = output_dir_ + "/global_stats.csv";
    fp_ = fopen(path.c_str(), "a");
}

void GlobalStatsWriter::writeHeader(bool has_scalar) {
    if (!fp_ || header_written_) return;
    fprintf(fp_, "timestep,time,mean_rho,max_velocity,mean_velocity,"
                 "total_kinetic_energy,n_active_particles,"
                 "mean_deformation,max_deformation,"
                 "min_rho,max_rho");
    if (has_scalar)
        fprintf(fp_, ",total_dissolved_mass,total_particle_mass,"
                     "total_adsorbed_mass");
    fprintf(fp_, "\n");
    header_written_ = true;
}

void GlobalStatsWriter::writeTimestep(const LatticeField& field,
                                       const CapsuleSystem& capsules,
                                       int step, Real time,
                                       const AdvectionDiffusion* ad) {
    openFile();
    if (!fp_) return;
    writeHeader(ad != nullptr);

    int nx = field.getNx();
    int ny = field.getNy();

    // Compute fluid statistics
    Real sum_rho = 0.0, min_rho = 1e30, max_rho = -1e30;
    Real max_vel2 = 0.0, sum_vel = 0.0;
    Real total_ke = 0.0;
    int fluid_count = 0;

    for (int y = 0; y < ny; ++y) {
        for (int x = 0; x < nx; ++x) {
            if (field.getCellType(x, y) == CellType::SOLID) continue;
            Real rho = field.getRho(x, y);
            Real ux = field.getUx(x, y);
            Real uy = field.getUy(x, y);
            Real v2 = ux * ux + uy * uy;

            sum_rho += rho;
            if (rho < min_rho) min_rho = rho;
            if (rho > max_rho) max_rho = rho;
            if (v2 > max_vel2) max_vel2 = v2;
            sum_vel += std::sqrt(v2);
            total_ke += 0.5 * rho * v2;
            fluid_count++;
        }
    }

    Real mean_rho = (fluid_count > 0) ? sum_rho / fluid_count : 0.0;
    Real max_vel = std::sqrt(max_vel2);
    Real mean_vel = (fluid_count > 0) ? sum_vel / fluid_count : 0.0;

    // Compute capsule statistics
    int ncaps = capsules.numCapsules();
    Real mean_def = 0.0, max_def = 0.0;
    for (int c = 0; c < ncaps; ++c) {
        Real D = capsules[c].deformationIndex();
        mean_def += D;
        if (D > max_def) max_def = D;
    }
    if (ncaps > 0) mean_def /= ncaps;

    fprintf(fp_, "%d,%.6e,%.6e,%.6e,%.6e,%.6e,%d,%.6e,%.6e,%.6e,%.6e",
            step, time, mean_rho, max_vel, mean_vel,
            total_ke, ncaps, mean_def, max_def, min_rho, max_rho);

    if (ad) {
        // Total dissolved mass: sum C over all fluid cells and species
        Real total_dissolved = 0.0;
        int nx = field.getNx();
        int ny = field.getNy();
        for (int s = 0; s < ad->getNumSpecies(); ++s) {
            const Real* C = ad->concentrationData(s);
            if (!C) continue;
            for (int y = 0; y < ny; ++y)
                for (int x = 0; x < nx; ++x)
                    total_dissolved += C[y * nx + x];
        }
        // Total particle chemical reservoir mass remaining
        Real total_particle_mass = 0.0;
        for (int c = 0; c < ncaps; ++c) {
            Real mp = ad->getCapsuleMp(c);
            if (mp >= 0.0) total_particle_mass += mp;
        }
        // Total adsorbed mass: sum Γ over all capsule nodes
        Real total_adsorbed = 0.0;
        for (int c = 0; c < ncaps; ++c)
            for (int k = 0; k < capsules[c].numNodes(); ++k)
                total_adsorbed += ad->getNodeGamma(c, k);

        fprintf(fp_, ",%.6e,%.6e,%.6e", total_dissolved, total_particle_mass, total_adsorbed);
    }

    fprintf(fp_, "\n");

    fflush(fp_);
}

void GlobalStatsWriter::close() {
    if (fp_) {
        fclose(fp_);
        fp_ = nullptr;
    }
}

} // namespace softflow
