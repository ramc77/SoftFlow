#include "vtk_writer.h"
#include "../lbm/lattice_field.h"
#include "../lbm/advection_diffusion.h"
#include "../membrane/capsule_system.h"
#include "../membrane/capsule.h"
#include <fstream>
#include <iomanip>
#include <sstream>
#include <sys/stat.h>
#include <cmath>
#include <cstdio>
#include <algorithm>

namespace softflow {

VTKWriter::VTKWriter(const std::string& output_dir)
    : output_dir_(output_dir) {
    // Only create the base directory here; subdirectories are created on demand
    mkdir(output_dir_.c_str(), 0755);
}

void VTKWriter::ensureDirs() const {
    if (dirs_created_) return;
    mkdir(output_dir_.c_str(), 0755);
    if (!legacy_mode_) {
        // XML mode uses subdirectories for fluid and particles
        mkdir((output_dir_ + "/fluid").c_str(), 0755);
        mkdir((output_dir_ + "/particles").c_str(), 0755);
    }
    // Legacy mode: all files in flat output directory — no subdirectories
    dirs_created_ = true;
}

void VTKWriter::recordTimestep(int step, double time) {
    recorded_steps_.push_back(step);
    recorded_times_.push_back(time);
}

Real VTKWriter::computeVorticity(const LatticeField& field, int x, int y) {
    int nx = field.getNx();
    int ny = field.getNy();
    // Central differences with boundary clamping
    int xp = std::min(x + 1, nx - 1), xm = std::max(x - 1, 0);
    int yp = std::min(y + 1, ny - 1), ym = std::max(y - 1, 0);

    Real duy_dx = (field.getUy(xp, y) - field.getUy(xm, y)) / static_cast<Real>(xp - xm);
    Real dux_dy = (field.getUx(x, yp) - field.getUx(x, ym)) / static_cast<Real>(yp - ym);

    return duy_dx - dux_dy;
}

Real VTKWriter::computeStrainRateMagnitude(const LatticeField& field, int x, int y) {
    int nx = field.getNx();
    int ny = field.getNy();
    int xp = std::min(x + 1, nx - 1), xm = std::max(x - 1, 0);
    int yp = std::min(y + 1, ny - 1), ym = std::max(y - 1, 0);

    Real dux_dx = (field.getUx(xp, y) - field.getUx(xm, y)) / static_cast<Real>(xp - xm);
    Real duy_dy = (field.getUy(x, yp) - field.getUy(x, ym)) / static_cast<Real>(yp - ym);
    Real dux_dy = (field.getUx(x, yp) - field.getUx(x, ym)) / static_cast<Real>(yp - ym);
    Real duy_dx = (field.getUy(xp, y) - field.getUy(xm, y)) / static_cast<Real>(xp - xm);

    // S_ij = 0.5 * (du_i/dx_j + du_j/dx_i)
    Real s11 = dux_dx;
    Real s22 = duy_dy;
    Real s12 = 0.5 * (dux_dy + duy_dx);

    // |gamma_dot| = sqrt(2 * S_ij * S_ij)
    return std::sqrt(2.0 * (s11 * s11 + s22 * s22 + 2.0 * s12 * s12));
}

// ═══════════════════════════════════════════════════════════════════
// Fluid VTK output — VTK ImageData (.vti) XML format
// ═══════════════════════════════════════════════════════════════════
void VTKWriter::writeFluidField(const LatticeField& field, int step) {
    ensureDirs();
    if (legacy_mode_) { writeFluidFieldLegacy(field, step); return; }

    char filename[512];
    snprintf(filename, sizeof(filename), "%s/fluid/fluid_%06d.vti",
             output_dir_.c_str(), step);

    FILE* fp = fopen(filename, "w");
    if (!fp) return;

    int nx = field.getNx();
    int ny = field.getNy();

    fprintf(fp, "<?xml version=\"1.0\"?>\n");
    fprintf(fp, "<VTKFile type=\"ImageData\" version=\"1.0\" byte_order=\"LittleEndian\">\n");
    fprintf(fp, "<ImageData WholeExtent=\"0 %d 0 %d 0 0\" Origin=\"0 0 0\" Spacing=\"1 1 1\">\n",
            nx - 1, ny - 1);
    fprintf(fp, "<Piece Extent=\"0 %d 0 %d 0 0\">\n", nx - 1, ny - 1);

    // Determine default scalars and vectors attributes
    fprintf(fp, "<PointData Scalars=\"density\" Vectors=\"velocity\">\n");

    // ── Density ──
    if (fluid_fields_.density) {
        fprintf(fp, "<DataArray type=\"Float64\" Name=\"density\" format=\"ascii\">\n");
        for (int y = 0; y < ny; ++y)
            for (int x = 0; x < nx; ++x)
                fprintf(fp, "%.6e\n", field.getRho(x, y));
        fprintf(fp, "</DataArray>\n");
    }

    // ── Velocity vector ──
    if (fluid_fields_.velocity) {
        fprintf(fp, "<DataArray type=\"Float64\" Name=\"velocity\" NumberOfComponents=\"3\" format=\"ascii\">\n");
        for (int y = 0; y < ny; ++y)
            for (int x = 0; x < nx; ++x)
                fprintf(fp, "%.6e %.6e 0\n", field.getUx(x, y), field.getUy(x, y));
        fprintf(fp, "</DataArray>\n");
    }

    // ── Velocity magnitude ──
    if (fluid_fields_.velocity_mag) {
        fprintf(fp, "<DataArray type=\"Float64\" Name=\"velocity_magnitude\" format=\"ascii\">\n");
        for (int y = 0; y < ny; ++y)
            for (int x = 0; x < nx; ++x) {
                Real ux = field.getUx(x, y);
                Real uy = field.getUy(x, y);
                fprintf(fp, "%.6e\n", std::sqrt(ux * ux + uy * uy));
            }
        fprintf(fp, "</DataArray>\n");
    }

    // ── Pressure (p = rho * cs^2 = rho / 3) ──
    if (fluid_fields_.pressure) {
        fprintf(fp, "<DataArray type=\"Float64\" Name=\"pressure\" format=\"ascii\">\n");
        for (int y = 0; y < ny; ++y)
            for (int x = 0; x < nx; ++x) {
                bool is_fluid = (field.getCellType(x, y) == CellType::FLUID);
                fprintf(fp, "%.6e\n", is_fluid ? field.getRho(x, y) / 3.0 : 0.0);
            }
        fprintf(fp, "</DataArray>\n");

        fprintf(fp, "<DataArray type=\"Float64\" Name=\"pressure_fluctuation\" format=\"ascii\">\n");
        for (int y = 0; y < ny; ++y)
            for (int x = 0; x < nx; ++x) {
                bool is_fluid = (field.getCellType(x, y) == CellType::FLUID);
                fprintf(fp, "%.6e\n", is_fluid ? (field.getRho(x, y) - 1.0) / 3.0 : 0.0);
            }
        fprintf(fp, "</DataArray>\n");
    }

    // ── Vorticity (2D scalar: omega_z = duy/dx - dux/dy) ──
    if (fluid_fields_.vorticity) {
        fprintf(fp, "<DataArray type=\"Float64\" Name=\"vorticity\" format=\"ascii\">\n");
        for (int y = 0; y < ny; ++y)
            for (int x = 0; x < nx; ++x)
                fprintf(fp, "%.6e\n", computeVorticity(field, x, y));
        fprintf(fp, "</DataArray>\n");
    }

    // ── Strain rate magnitude ──
    if (fluid_fields_.strain_rate) {
        fprintf(fp, "<DataArray type=\"Float64\" Name=\"strain_rate\" format=\"ascii\">\n");
        for (int y = 0; y < ny; ++y)
            for (int x = 0; x < nx; ++x)
                fprintf(fp, "%.6e\n", computeStrainRateMagnitude(field, x, y));
        fprintf(fp, "</DataArray>\n");
    }

    // ── IBM force on lattice ──
    if (fluid_fields_.ibm_force) {
        fprintf(fp, "<DataArray type=\"Float64\" Name=\"ibm_force\" NumberOfComponents=\"3\" format=\"ascii\">\n");
        for (int y = 0; y < ny; ++y)
            for (int x = 0; x < nx; ++x) {
                int n = y * nx + x;
                fprintf(fp, "%.6e %.6e 0\n", field.FxData()[n], field.FyData()[n]);
            }
        fprintf(fp, "</DataArray>\n");
    }

    // ── Node type ──
    if (fluid_fields_.node_type) {
        fprintf(fp, "<DataArray type=\"Int32\" Name=\"node_type\" format=\"ascii\">\n");
        for (int y = 0; y < ny; ++y)
            for (int x = 0; x < nx; ++x)
                fprintf(fp, "%d\n", static_cast<int>(field.getCellType(x, y)));
        fprintf(fp, "</DataArray>\n");
    }

    // ── Concentration (scalar transport) ──
    if (fluid_fields_.concentration && advection_diffusion_) {
        int n_species = advection_diffusion_->getNumSpecies();
        for (int s = 0; s < n_species; ++s) {
            if (n_species == 1)
                fprintf(fp, "<DataArray type=\"Float64\" Name=\"concentration\" format=\"ascii\">\n");
            else
                fprintf(fp, "<DataArray type=\"Float64\" Name=\"concentration_%d\" format=\"ascii\">\n", s);
            for (int y = 0; y < ny; ++y)
                for (int x = 0; x < nx; ++x)
                    fprintf(fp, "%.6e\n", advection_diffusion_->getConcentration(x, y, s));
            fprintf(fp, "</DataArray>\n");
        }
    }

    fprintf(fp, "</PointData>\n");
    fprintf(fp, "</Piece>\n</ImageData>\n</VTKFile>\n");
    fflush(fp);
    fclose(fp);
}

// ═══════════════════════════════════════════════════════════════════
// Capsule VTK output — VTK PolyData (.vtp) XML format
// ═══════════════════════════════════════════════════════════════════

// Helper: build unwrapped (coherent) node positions for a capsule.
static std::vector<Vec2d> unwrapNodes(const Capsule& cap, int periodic_nx) {
    int N = cap.numNodes();
    std::vector<Vec2d> pts(N);
    pts[0] = cap.nodePosition(0);
    if (periodic_nx > 0) {
        Real Lx = static_cast<Real>(periodic_nx);
        for (int k = 1; k < N; ++k) {
            Vec2d d = cap.nodePosition(k) - cap.nodePosition(k - 1);
            if (d.x >  0.5 * Lx) d.x -= Lx;
            if (d.x < -0.5 * Lx) d.x += Lx;
            pts[k] = pts[k - 1] + d;
        }
    } else {
        for (int k = 1; k < N; ++k)
            pts[k] = cap.nodePosition(k);
    }
    return pts;
}

void VTKWriter::writeCapsules(const CapsuleSystem& capsules, int step) {
    ensureDirs();
    if (legacy_mode_) { writeCapsulesLegacy(capsules, step); return; }

    int ncaps = capsules.numCapsules();
    if (ncaps == 0) return;

    char filename[512];
    snprintf(filename, sizeof(filename), "%s/particles/particles_%06d.vtp",
             output_dir_.c_str(), step);

    FILE* fp = fopen(filename, "w");
    if (!fp) return;

    Real Lx = static_cast<Real>(periodic_nx_);

    // ── Build polygon copies (handle periodic wrapping) ──────────
    struct PolyCopy {
        int capsule_idx;
        std::vector<Vec2d> pts;
    };
    std::vector<PolyCopy> copies;
    copies.reserve(ncaps * 2);

    for (int c = 0; c < ncaps; ++c) {
        const Capsule& cap = capsules[c];
        std::vector<Vec2d> uw = unwrapNodes(cap, periodic_nx_);

        if (periodic_nx_ <= 0) {
            copies.push_back({c, std::move(uw)});
            continue;
        }

        Real xmin = uw[0].x, xmax = uw[0].x;
        for (auto& p : uw) {
            if (p.x < xmin) xmin = p.x;
            if (p.x > xmax) xmax = p.x;
        }

        bool exits_right = (xmax >= Lx);
        bool exits_left  = (xmin < 0.0);

        if (!exits_right && !exits_left) {
            copies.push_back({c, std::move(uw)});
        } else {
            std::vector<Vec2d> copy1(uw.size());
            for (size_t k = 0; k < uw.size(); ++k) {
                Real cx = std::clamp(uw[k].x, 0.0, Lx);
                copy1[k] = Vec2d{cx, uw[k].y};
            }
            copies.push_back({c, std::move(copy1)});

            Real shift = exits_right ? -Lx : Lx;
            std::vector<Vec2d> copy2(uw.size());
            for (size_t k = 0; k < uw.size(); ++k) {
                Real cx = std::clamp(uw[k].x + shift, 0.0, Lx);
                copy2[k] = Vec2d{cx, uw[k].y};
            }
            copies.push_back({c, std::move(copy2)});
        }
    }

    // ── Count totals ──
    int total_nodes = 0;
    int total_polys = static_cast<int>(copies.size());
    for (auto& cp : copies) total_nodes += static_cast<int>(cp.pts.size());

    // ── Write VTP XML format ──
    fprintf(fp, "<?xml version=\"1.0\"?>\n");
    fprintf(fp, "<VTKFile type=\"PolyData\" version=\"1.0\" byte_order=\"LittleEndian\">\n");
    fprintf(fp, "<PolyData>\n");
    fprintf(fp, "<Piece NumberOfPoints=\"%d\" NumberOfVerts=\"0\" NumberOfLines=\"0\" "
                "NumberOfStrips=\"0\" NumberOfPolys=\"%d\">\n",
            total_nodes, total_polys);

    // ── Points ──
    fprintf(fp, "<Points>\n");
    fprintf(fp, "<DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"ascii\">\n");
    for (auto& cp : copies)
        for (auto& p : cp.pts)
            fprintf(fp, "%.6e %.6e 0\n", p.x, p.y);
    fprintf(fp, "</DataArray>\n</Points>\n");

    // ── Polygons ──
    fprintf(fp, "<Polys>\n");
    // Connectivity
    fprintf(fp, "<DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">\n");
    int offset_acc = 0;
    for (auto& cp : copies) {
        int N = static_cast<int>(cp.pts.size());
        for (int k = 0; k < N; ++k)
            fprintf(fp, "%d ", offset_acc + k);
        fprintf(fp, "\n");
        offset_acc += N;
    }
    fprintf(fp, "</DataArray>\n");
    // Offsets
    fprintf(fp, "<DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">\n");
    int poly_offset = 0;
    for (auto& cp : copies) {
        poly_offset += static_cast<int>(cp.pts.size());
        fprintf(fp, "%d ", poly_offset);
    }
    fprintf(fp, "\n</DataArray>\n");
    fprintf(fp, "</Polys>\n");

    // ── PointData ──
    fprintf(fp, "<PointData>\n");

    // Capsule ID
    if (particle_fields_.particle_id) {
        fprintf(fp, "<DataArray type=\"Int32\" Name=\"capsule_id\" format=\"ascii\">\n");
        for (auto& cp : copies)
            for (size_t k = 0; k < cp.pts.size(); ++k)
                fprintf(fp, "%d ", cp.capsule_idx);
        fprintf(fp, "\n</DataArray>\n");
    }

    // Capsule type
    if (particle_fields_.particle_type) {
        fprintf(fp, "<DataArray type=\"Int32\" Name=\"capsule_type\" format=\"ascii\">\n");
        for (auto& cp : copies) {
            int tid = capsules[cp.capsule_idx].getTypeId();
            for (size_t k = 0; k < cp.pts.size(); ++k)
                fprintf(fp, "%d ", tid);
        }
        fprintf(fp, "\n</DataArray>\n");
    }

    // Node velocity
    if (particle_fields_.velocity) {
        fprintf(fp, "<DataArray type=\"Float64\" Name=\"velocity\" NumberOfComponents=\"3\" format=\"ascii\">\n");
        for (auto& cp : copies) {
            const Capsule& cap = capsules[cp.capsule_idx];
            for (int k = 0; k < cap.numNodes(); ++k) {
                Vec2d v = cap.nodeVelocity(k);
                fprintf(fp, "%.6e %.6e 0\n", v.x, v.y);
            }
        }
        fprintf(fp, "</DataArray>\n");
    }

    // Node force
    if (particle_fields_.force) {
        fprintf(fp, "<DataArray type=\"Float64\" Name=\"force\" NumberOfComponents=\"3\" format=\"ascii\">\n");
        for (auto& cp : copies) {
            const Capsule& cap = capsules[cp.capsule_idx];
            for (int k = 0; k < cap.numNodes(); ++k) {
                Vec2d f = cap.nodeForce(k);
                fprintf(fp, "%.6e %.6e 0\n", f.x, f.y);
            }
        }
        fprintf(fp, "</DataArray>\n");
    }

    // Surface coverage Γ per node (Langmuir adsorption state)
    if (advection_diffusion_) {
        fprintf(fp, "<DataArray type=\"Float64\" Name=\"gamma\" format=\"ascii\">\n");
        for (auto& cp : copies) {
            const Capsule& cap = capsules[cp.capsule_idx];
            for (int k = 0; k < cap.numNodes(); ++k)
                fprintf(fp, "%.6e\n", advection_diffusion_->getNodeGamma(cp.capsule_idx, k));
        }
        fprintf(fp, "</DataArray>\n");
    }

    fprintf(fp, "</PointData>\n");

    // ── CellData ──
    fprintf(fp, "<CellData>\n");
    // Per-polygon capsule ID
    fprintf(fp, "<DataArray type=\"Int32\" Name=\"cell_capsule_id\" format=\"ascii\">\n");
    for (auto& cp : copies)
        fprintf(fp, "%d ", cp.capsule_idx);
    fprintf(fp, "\n</DataArray>\n");

    // Per-polygon capsule type
    fprintf(fp, "<DataArray type=\"Int32\" Name=\"cell_capsule_type\" format=\"ascii\">\n");
    for (auto& cp : copies)
        fprintf(fp, "%d ", capsules[cp.capsule_idx].getTypeId());
    fprintf(fp, "\n</DataArray>\n");

    // Per-polygon deformation index
    fprintf(fp, "<DataArray type=\"Float64\" Name=\"deformation_index\" format=\"ascii\">\n");
    for (auto& cp : copies)
        fprintf(fp, "%.6e ", capsules[cp.capsule_idx].deformationIndex());
    fprintf(fp, "\n</DataArray>\n");

    // Per-polygon capsule speed
    fprintf(fp, "<DataArray type=\"Float64\" Name=\"capsule_speed\" format=\"ascii\">\n");
    for (auto& cp : copies) {
        const Capsule& cap = capsules[cp.capsule_idx];
        int Nn = cap.numNodes();
        Vec2d avg_vel{0, 0};
        for (int k = 0; k < Nn; ++k) {
            Vec2d v = cap.nodeVelocity(k);
            avg_vel.x += v.x; avg_vel.y += v.y;
        }
        avg_vel.x /= Nn; avg_vel.y /= Nn;
        fprintf(fp, "%.6e ", avg_vel.norm());
    }
    fprintf(fp, "\n</DataArray>\n");

    // Per-polygon chemical reservoir mass M_p (-1 = infinite)
    if (advection_diffusion_) {
        fprintf(fp, "<DataArray type=\"Float64\" Name=\"Mp\" format=\"ascii\">\n");
        for (auto& cp : copies)
            fprintf(fp, "%.6e ", advection_diffusion_->getCapsuleMp(cp.capsule_idx));
        fprintf(fp, "\n</DataArray>\n");

        // Per-polygon mean surface coverage Γ
        fprintf(fp, "<DataArray type=\"Float64\" Name=\"gamma_mean\" format=\"ascii\">\n");
        for (auto& cp : copies) {
            const Capsule& cap = capsules[cp.capsule_idx];
            int Nn = cap.numNodes();
            double sum = 0.0;
            for (int k = 0; k < Nn; ++k)
                sum += advection_diffusion_->getNodeGamma(cp.capsule_idx, k);
            fprintf(fp, "%.6e ", Nn > 0 ? sum / Nn : 0.0);
        }
        fprintf(fp, "\n</DataArray>\n");

        // Cumulative released / absorbed scalars
        fprintf(fp, "<DataArray type=\"Float64\" Name=\"cumulative_released\" format=\"ascii\">\n");
        for (auto& cp : copies)
            fprintf(fp, "%.6e ", advection_diffusion_->getCapsuleReleased(cp.capsule_idx));
        fprintf(fp, "\n</DataArray>\n");

        fprintf(fp, "<DataArray type=\"Float64\" Name=\"cumulative_absorbed\" format=\"ascii\">\n");
        for (auto& cp : copies)
            fprintf(fp, "%.6e ", advection_diffusion_->getCapsuleAbsorbed(cp.capsule_idx));
        fprintf(fp, "\n</DataArray>\n");
    }

    fprintf(fp, "</CellData>\n");
    fprintf(fp, "</Piece>\n</PolyData>\n</VTKFile>\n");
    fflush(fp);
    fclose(fp);
}

// ═══════════════════════════════════════════════════════════════════
// PVD collection files for ParaView time-series
// ═══════════════════════════════════════════════════════════════════
void VTKWriter::writePVDFiles() const {
    if (recorded_steps_.empty()) return;

    if (legacy_mode_) {
        // Legacy mode: .vtk files in flat output directory
        // Fluid PVD
        {
            char filename[512];
            snprintf(filename, sizeof(filename), "%s/fluid.pvd", output_dir_.c_str());
            FILE* fp = fopen(filename, "w");
            if (fp) {
                fprintf(fp, "<?xml version=\"1.0\"?>\n");
                fprintf(fp, "<VTKFile type=\"Collection\" version=\"0.1\">\n<Collection>\n");
                for (size_t d = 0; d < recorded_steps_.size(); ++d)
                    fprintf(fp, "<DataSet timestep=\"%.6e\" file=\"fluid_%06d.vtk\"/>\n",
                            recorded_times_[d], recorded_steps_[d]);
                fprintf(fp, "</Collection>\n</VTKFile>\n");
                fclose(fp);
            }
        }
        // Particles PVD
        {
            char filename[512];
            snprintf(filename, sizeof(filename), "%s/particles.pvd", output_dir_.c_str());
            FILE* fp = fopen(filename, "w");
            if (fp) {
                fprintf(fp, "<?xml version=\"1.0\"?>\n");
                fprintf(fp, "<VTKFile type=\"Collection\" version=\"0.1\">\n<Collection>\n");
                for (size_t d = 0; d < recorded_steps_.size(); ++d)
                    fprintf(fp, "<DataSet timestep=\"%.6e\" file=\"particles_%06d.vtk\"/>\n",
                            recorded_times_[d], recorded_steps_[d]);
                fprintf(fp, "</Collection>\n</VTKFile>\n");
                fclose(fp);
            }
        }
        return;
    }

    // XML mode: .vti/.vtp files in subdirectories
    // Fluid PVD
    {
        char filename[512];
        snprintf(filename, sizeof(filename), "%s/fluid/fluid.pvd", output_dir_.c_str());
        FILE* fp = fopen(filename, "w");
        if (fp) {
            fprintf(fp, "<?xml version=\"1.0\"?>\n");
            fprintf(fp, "<VTKFile type=\"Collection\" version=\"0.1\">\n<Collection>\n");
            for (size_t d = 0; d < recorded_steps_.size(); ++d)
                fprintf(fp, "<DataSet timestep=\"%.6e\" file=\"fluid_%06d.vti\"/>\n",
                        recorded_times_[d], recorded_steps_[d]);
            fprintf(fp, "</Collection>\n</VTKFile>\n");
            fclose(fp);
        }
    }

    // Particles PVD
    {
        char filename[512];
        snprintf(filename, sizeof(filename), "%s/particles/particles.pvd", output_dir_.c_str());
        FILE* fp = fopen(filename, "w");
        if (fp) {
            fprintf(fp, "<?xml version=\"1.0\"?>\n");
            fprintf(fp, "<VTKFile type=\"Collection\" version=\"0.1\">\n<Collection>\n");
            for (size_t d = 0; d < recorded_steps_.size(); ++d)
                fprintf(fp, "<DataSet timestep=\"%.6e\" file=\"particles_%06d.vtp\"/>\n",
                        recorded_times_[d], recorded_steps_[d]);
            fprintf(fp, "</Collection>\n</VTKFile>\n");
            fclose(fp);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// Legacy VTK format (.vtk) — compatible with all ParaView versions
// ═══════════════════════════════════════════════════════════════════

void VTKWriter::writeFluidFieldLegacy(const LatticeField& field, int step) {
    char filename[512];
    snprintf(filename, sizeof(filename), "%s/fluid_%06d.vtk",
             output_dir_.c_str(), step);

    FILE* fp = fopen(filename, "w");
    if (!fp) return;

    int nx = field.getNx();
    int ny = field.getNy();
    int npoints = nx * ny;

    // VTK legacy header
    fprintf(fp, "# vtk DataFile Version 3.0\n");
    fprintf(fp, "SoftFlow fluid field step %d\n", step);
    fprintf(fp, "ASCII\n");
    fprintf(fp, "DATASET STRUCTURED_POINTS\n");
    fprintf(fp, "DIMENSIONS %d %d 1\n", nx, ny);
    fprintf(fp, "ORIGIN 0 0 0\n");
    fprintf(fp, "SPACING 1 1 1\n");
    fprintf(fp, "POINT_DATA %d\n", npoints);

    // Density
    if (fluid_fields_.density) {
        fprintf(fp, "SCALARS density double 1\n");
        fprintf(fp, "LOOKUP_TABLE default\n");
        for (int y = 0; y < ny; ++y)
            for (int x = 0; x < nx; ++x)
                fprintf(fp, "%.6e\n", field.getRho(x, y));
    }

    // Velocity
    if (fluid_fields_.velocity) {
        fprintf(fp, "VECTORS velocity double\n");
        for (int y = 0; y < ny; ++y)
            for (int x = 0; x < nx; ++x)
                fprintf(fp, "%.6e %.6e 0\n", field.getUx(x, y), field.getUy(x, y));
    }

    // Velocity magnitude
    if (fluid_fields_.velocity_mag) {
        fprintf(fp, "SCALARS velocity_magnitude double 1\n");
        fprintf(fp, "LOOKUP_TABLE default\n");
        for (int y = 0; y < ny; ++y)
            for (int x = 0; x < nx; ++x) {
                Real ux = field.getUx(x, y);
                Real uy = field.getUy(x, y);
                fprintf(fp, "%.6e\n", std::sqrt(ux * ux + uy * uy));
            }
    }

    // Pressure (absolute: p = rho * c_s^2 = rho/3)
    if (fluid_fields_.pressure) {
        fprintf(fp, "SCALARS pressure double 1\n");
        fprintf(fp, "LOOKUP_TABLE default\n");
        for (int y = 0; y < ny; ++y)
            for (int x = 0; x < nx; ++x) {
                bool is_fluid = (field.getCellType(x, y) == CellType::FLUID);
                fprintf(fp, "%.6e\n", is_fluid ? field.getRho(x, y) / 3.0 : 0.0);
            }

        // Pressure fluctuation: dp = (rho - 1) / 3
        // Zero mean — shows the actual pressure variation clearly in ParaView
        // Solid nodes output 0 so they don't distort the color range
        fprintf(fp, "SCALARS pressure_fluctuation double 1\n");
        fprintf(fp, "LOOKUP_TABLE default\n");
        for (int y = 0; y < ny; ++y)
            for (int x = 0; x < nx; ++x) {
                bool is_fluid = (field.getCellType(x, y) == CellType::FLUID);
                fprintf(fp, "%.6e\n", is_fluid ? (field.getRho(x, y) - 1.0) / 3.0 : 0.0);
            }
    }

    // Vorticity
    if (fluid_fields_.vorticity) {
        fprintf(fp, "SCALARS vorticity double 1\n");
        fprintf(fp, "LOOKUP_TABLE default\n");
        for (int y = 0; y < ny; ++y)
            for (int x = 0; x < nx; ++x)
                fprintf(fp, "%.6e\n", computeVorticity(field, x, y));
    }

    // Strain rate magnitude
    if (fluid_fields_.strain_rate) {
        fprintf(fp, "SCALARS strain_rate double 1\n");
        fprintf(fp, "LOOKUP_TABLE default\n");
        for (int y = 0; y < ny; ++y)
            for (int x = 0; x < nx; ++x)
                fprintf(fp, "%.6e\n", computeStrainRateMagnitude(field, x, y));
    }

    // IBM force (fluid-structure coupling force)
    if (fluid_fields_.ibm_force) {
        fprintf(fp, "VECTORS ibm_force double\n");
        for (int y = 0; y < ny; ++y)
            for (int x = 0; x < nx; ++x) {
                int n = y * nx + x;
                fprintf(fp, "%.6e %.6e 0\n", field.FxData()[n], field.FyData()[n]);
            }
    }

    // Node type
    if (fluid_fields_.node_type) {
        fprintf(fp, "SCALARS node_type int 1\n");
        fprintf(fp, "LOOKUP_TABLE default\n");
        for (int y = 0; y < ny; ++y)
            for (int x = 0; x < nx; ++x)
                fprintf(fp, "%d\n", static_cast<int>(field.getCellType(x, y)));
    }

    // Concentration (scalar transport — chemical leaching from MPs)
    if (fluid_fields_.concentration && advection_diffusion_) {
        int n_species = advection_diffusion_->getNumSpecies();
        for (int s = 0; s < n_species; ++s) {
            if (n_species == 1)
                fprintf(fp, "SCALARS concentration double 1\n");
            else
                fprintf(fp, "SCALARS concentration_%d double 1\n", s);
            fprintf(fp, "LOOKUP_TABLE default\n");
            for (int y = 0; y < ny; ++y)
                for (int x = 0; x < nx; ++x)
                    fprintf(fp, "%.6e\n", advection_diffusion_->getConcentration(x, y, s));
        }
    }

    fflush(fp);
    fclose(fp);
}

void VTKWriter::writeCapsulesLegacy(const CapsuleSystem& capsules, int step) {
    int ncaps = capsules.numCapsules();
    if (ncaps == 0) return;

    char filename[512];
    snprintf(filename, sizeof(filename), "%s/particles_%06d.vtk",
             output_dir_.c_str(), step);

    FILE* fp = fopen(filename, "w");
    if (!fp) return;

    Real Lx = static_cast<Real>(periodic_nx_);

    // ── Build polygon copies with periodic boundary splitting ──
    // Same logic as the XML writer: when a capsule straddles the
    // periodic boundary, create TWO clipped copies so it appears
    // to exit one side and enter the other simultaneously.
    struct PolyCopy {
        int capsule_idx;
        std::vector<Vec2d> pts;
    };
    std::vector<PolyCopy> copies;
    copies.reserve(ncaps * 2);

    for (int c = 0; c < ncaps; ++c) {
        const Capsule& cap = capsules[c];
        std::vector<Vec2d> uw = unwrapNodes(cap, periodic_nx_);

        if (periodic_nx_ <= 0) {
            copies.push_back({c, std::move(uw)});
            continue;
        }

        Real xmin = uw[0].x, xmax = uw[0].x;
        for (auto& p : uw) {
            if (p.x < xmin) xmin = p.x;
            if (p.x > xmax) xmax = p.x;
        }

        bool exits_right = (xmax >= Lx);
        bool exits_left  = (xmin < 0.0);

        if (!exits_right && !exits_left) {
            // Capsule fully inside domain — single copy
            copies.push_back({c, std::move(uw)});
        } else {
            // Capsule straddles boundary — create two clipped copies
            std::vector<Vec2d> copy1(uw.size());
            for (size_t k = 0; k < uw.size(); ++k) {
                Real cx = std::clamp(uw[k].x, 0.0, Lx);
                copy1[k] = Vec2d{cx, uw[k].y};
            }
            copies.push_back({c, std::move(copy1)});

            Real shift = exits_right ? -Lx : Lx;
            std::vector<Vec2d> copy2(uw.size());
            for (size_t k = 0; k < uw.size(); ++k) {
                Real cx = std::clamp(uw[k].x + shift, 0.0, Lx);
                copy2[k] = Vec2d{cx, uw[k].y};
            }
            copies.push_back({c, std::move(copy2)});
        }
    }

    // ── Count totals ──
    int total_nodes = 0;
    int total_polys = static_cast<int>(copies.size());
    for (auto& cp : copies) total_nodes += static_cast<int>(cp.pts.size());

    // VTK legacy header
    fprintf(fp, "# vtk DataFile Version 3.0\n");
    fprintf(fp, "SoftFlow particles step %d\n", step);
    fprintf(fp, "ASCII\n");
    fprintf(fp, "DATASET POLYDATA\n");

    // Points
    fprintf(fp, "POINTS %d double\n", total_nodes);
    for (auto& cp : copies)
        for (auto& p : cp.pts)
            fprintf(fp, "%.6e %.6e 0\n", p.x, p.y);

    // Polygons
    int poly_entries = total_polys + total_nodes;
    fprintf(fp, "POLYGONS %d %d\n", total_polys, poly_entries);
    int offset = 0;
    for (auto& cp : copies) {
        int nn = static_cast<int>(cp.pts.size());
        fprintf(fp, "%d", nn);
        for (int k = 0; k < nn; ++k)
            fprintf(fp, " %d", offset + k);
        fprintf(fp, "\n");
        offset += nn;
    }

    // Point data: per-node attributes
    fprintf(fp, "POINT_DATA %d\n", total_nodes);

    fprintf(fp, "SCALARS capsule_id int 1\n");
    fprintf(fp, "LOOKUP_TABLE default\n");
    for (auto& cp : copies)
        for (size_t k = 0; k < cp.pts.size(); ++k)
            fprintf(fp, "%d\n", cp.capsule_idx);

    fprintf(fp, "SCALARS capsule_type int 1\n");
    fprintf(fp, "LOOKUP_TABLE default\n");
    for (auto& cp : copies) {
        int tid = capsules[cp.capsule_idx].getTypeId();
        for (size_t k = 0; k < cp.pts.size(); ++k)
            fprintf(fp, "%d\n", tid);
    }

    // Node velocity
    fprintf(fp, "VECTORS node_velocity double\n");
    for (auto& cp : copies) {
        const Capsule& cap = capsules[cp.capsule_idx];
        for (int k = 0; k < cap.numNodes(); ++k) {
            Vec2d v = cap.nodeVelocity(k);
            fprintf(fp, "%.6e %.6e 0\n", v.x, v.y);
        }
    }

    // Node force (membrane + IBM)
    fprintf(fp, "VECTORS node_force double\n");
    for (auto& cp : copies) {
        const Capsule& cap = capsules[cp.capsule_idx];
        for (int k = 0; k < cap.numNodes(); ++k) {
            Vec2d f = cap.nodeForce(k);
            fprintf(fp, "%.6e %.6e 0\n", f.x, f.y);
        }
    }

    // Per-node surface coverage Γ (Langmuir adsorption state, 0 when inactive)
    if (advection_diffusion_) {
        fprintf(fp, "SCALARS gamma double 1\n");
        fprintf(fp, "LOOKUP_TABLE default\n");
        for (auto& cp : copies) {
            const Capsule& cap = capsules[cp.capsule_idx];
            for (int k = 0; k < cap.numNodes(); ++k)
                fprintf(fp, "%.6e\n",
                        advection_diffusion_->getNodeGamma(cp.capsule_idx, k));
        }
    }

    // Cell data: per-polygon (per capsule copy) — particle-level quantities
    fprintf(fp, "CELL_DATA %d\n", total_polys);

    fprintf(fp, "SCALARS capsule_id_cell int 1\n");
    fprintf(fp, "LOOKUP_TABLE default\n");
    for (auto& cp : copies)
        fprintf(fp, "%d\n", cp.capsule_idx);

    fprintf(fp, "SCALARS cell_type int 1\n");
    fprintf(fp, "LOOKUP_TABLE default\n");
    for (auto& cp : copies)
        fprintf(fp, "%d\n", capsules[cp.capsule_idx].getTypeId());

    fprintf(fp, "SCALARS deformation_index double 1\n");
    fprintf(fp, "LOOKUP_TABLE default\n");
    for (auto& cp : copies)
        fprintf(fp, "%.6e\n", capsules[cp.capsule_idx].deformationIndex());

    // Per-particle centroid velocity (average of node velocities)
    fprintf(fp, "VECTORS particle_velocity double\n");
    for (auto& cp : copies) {
        const Capsule& cap = capsules[cp.capsule_idx];
        int Nn = cap.numNodes();
        Vec2d avg_vel{0, 0};
        for (int k = 0; k < Nn; ++k) {
            Vec2d v = cap.nodeVelocity(k);
            avg_vel.x += v.x; avg_vel.y += v.y;
        }
        avg_vel.x /= Nn; avg_vel.y /= Nn;
        fprintf(fp, "%.6e %.6e 0\n", avg_vel.x, avg_vel.y);
    }

    // Per-particle total force (sum of node forces)
    fprintf(fp, "VECTORS particle_force double\n");
    for (auto& cp : copies) {
        const Capsule& cap = capsules[cp.capsule_idx];
        int Nn = cap.numNodes();
        Vec2d total_f{0, 0};
        for (int k = 0; k < Nn; ++k) {
            Vec2d f = cap.nodeForce(k);
            total_f.x += f.x; total_f.y += f.y;
        }
        fprintf(fp, "%.6e %.6e 0\n", total_f.x, total_f.y);
    }

    // Per-particle speed (magnitude of centroid velocity)
    fprintf(fp, "SCALARS particle_speed double 1\n");
    fprintf(fp, "LOOKUP_TABLE default\n");
    for (auto& cp : copies) {
        const Capsule& cap = capsules[cp.capsule_idx];
        int Nn = cap.numNodes();
        Vec2d avg_vel{0, 0};
        for (int k = 0; k < Nn; ++k) {
            Vec2d v = cap.nodeVelocity(k);
            avg_vel.x += v.x; avg_vel.y += v.y;
        }
        avg_vel.x /= Nn; avg_vel.y /= Nn;
        fprintf(fp, "%.6e\n", avg_vel.norm());
    }

    // Per-particle area
    fprintf(fp, "SCALARS particle_area double 1\n");
    fprintf(fp, "LOOKUP_TABLE default\n");
    for (auto& cp : copies)
        fprintf(fp, "%.6e\n", capsules[cp.capsule_idx].area());

    // Per-particle perimeter
    fprintf(fp, "SCALARS particle_perimeter double 1\n");
    fprintf(fp, "LOOKUP_TABLE default\n");
    for (auto& cp : copies)
        fprintf(fp, "%.6e\n", capsules[cp.capsule_idx].perimeter());

    // Per-particle chemistry fields (always written when scalar transport enabled)
    if (advection_diffusion_) {
        // Cumulative scalar released/absorbed (constant-rate or Fick leaching)
        fprintf(fp, "SCALARS scalar_released double 1\n");
        fprintf(fp, "LOOKUP_TABLE default\n");
        for (auto& cp : copies)
            fprintf(fp, "%.6e\n", advection_diffusion_->getCapsuleReleased(cp.capsule_idx));

        fprintf(fp, "SCALARS scalar_absorbed double 1\n");
        fprintf(fp, "LOOKUP_TABLE default\n");
        for (auto& cp : copies)
            fprintf(fp, "%.6e\n", advection_diffusion_->getCapsuleAbsorbed(cp.capsule_idx));

        // Chemical reservoir mass M_p (-1 = infinite / not a leaching particle)
        fprintf(fp, "SCALARS Mp double 1\n");
        fprintf(fp, "LOOKUP_TABLE default\n");
        for (auto& cp : copies)
            fprintf(fp, "%.6e\n", advection_diffusion_->getCapsuleMp(cp.capsule_idx));

        // Mean surface coverage Γ per capsule (Langmuir adsorption state)
        fprintf(fp, "SCALARS gamma_mean double 1\n");
        fprintf(fp, "LOOKUP_TABLE default\n");
        for (auto& cp : copies) {
            const Capsule& cap = capsules[cp.capsule_idx];
            int Nn = cap.numNodes();
            double sum = 0.0;
            for (int k = 0; k < Nn; ++k)
                sum += advection_diffusion_->getNodeGamma(cp.capsule_idx, k);
            fprintf(fp, "%.6e\n", Nn > 0 ? sum / Nn : 0.0);
        }
    }

    fflush(fp);
    fclose(fp);
}

} // namespace softflow
