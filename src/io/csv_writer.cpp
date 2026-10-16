#include "csv_writer.h"
#include "../membrane/capsule_system.h"
#include "../membrane/capsule.h"
#include <sys/stat.h>
#include <cmath>

namespace softflow {

CSVWriter::CSVWriter(const std::string& output_dir,
                     const std::string& filename,
                     const std::string& format,
                     bool append)
    : output_dir_(output_dir), filename_(filename), format_(format), append_(append)
{
    mkdir(output_dir_.c_str(), 0755);
}

CSVWriter::~CSVWriter() {
    close();
}

void CSVWriter::openFile() {
    if (fp_) return;
    std::string path = output_dir_ + "/" + filename_;
    fp_ = fopen(path.c_str(), append_ ? "a" : "w");
}

void CSVWriter::writeHeader() {
    if (!fp_ || header_written_) return;
    char sep = separator();

    fprintf(fp_, "timestep%ctime%ccapsule_id", sep, sep);

    if (fields_.type) fprintf(fp_, "%ctype", sep);
    if (fields_.group) fprintf(fp_, "%cgroup", sep);
    if (fields_.com_position) fprintf(fp_, "%ccx%ccy", sep, sep);
    if (fields_.com_velocity) fprintf(fp_, "%cvx%cvy", sep, sep);
    if (fields_.force) fprintf(fp_, "%cfx%cfy", sep, sep);
    if (fields_.diameter) fprintf(fp_, "%cdiameter", sep);
    if (fields_.deformation) fprintf(fp_, "%cdeformation_index", sep);
    if (fields_.area_volume) fprintf(fp_, "%carea%carea_ratio%cperimeter%cperimeter_ratio", sep, sep, sep, sep);
    if (fields_.orientation) fprintf(fp_, "%corientation_angle", sep);
    if (fields_.angular_vel) fprintf(fp_, "%cangular_velocity", sep);

    fprintf(fp_, "\n");
    header_written_ = true;
}

void CSVWriter::writeTimestep(const CapsuleSystem& capsules, int step, Real time) {
    openFile();
    if (!fp_) return;
    writeHeader();

    char sep = separator();
    int ncaps = capsules.numCapsules();

    for (int c = 0; c < ncaps; ++c) {
        const Capsule& cap = capsules[c];
        Vec2d cen = cap.centroid();
        int id = cap.getId();
        int type_id = cap.getTypeId();
        int group = 0; // group not yet in Capsule class, use 0

        // Apply filter
        if (!filter_.passes(id, type_id, group, cen.x, cen.y))
            continue;

        // Compute centroid velocity
        Vec2d vel{0, 0};
        int Nn = cap.numNodes();
        for (int k = 0; k < Nn; ++k) {
            Vec2d v = cap.nodeVelocity(k);
            vel.x += v.x; vel.y += v.y;
        }
        vel.x /= Nn; vel.y /= Nn;

        fprintf(fp_, "%d%c%.6e%c%d", step, sep, time, sep, id);

        if (fields_.type) fprintf(fp_, "%c%d", sep, type_id);
        if (fields_.group) fprintf(fp_, "%c%d", sep, group);
        if (fields_.com_position) fprintf(fp_, "%c%.6e%c%.6e", sep, cen.x, sep, cen.y);
        if (fields_.com_velocity) fprintf(fp_, "%c%.6e%c%.6e", sep, vel.x, sep, vel.y);

        if (fields_.force) {
            Vec2d F{0, 0};
            for (int k = 0; k < Nn; ++k) {
                Vec2d f = cap.nodeForce(k);
                F.x += f.x; F.y += f.y;
            }
            fprintf(fp_, "%c%.6e%c%.6e", sep, F.x, sep, F.y);
        }

        if (fields_.diameter) {
            fprintf(fp_, "%c%.6e", sep, 2.0 * cap.effectiveRadius());
        }

        if (fields_.deformation) {
            fprintf(fp_, "%c%.6e", sep, cap.deformationIndex());
        }

        if (fields_.area_volume) {
            Real A = cap.area();
            Real P = cap.perimeter();
            Real A0 = PI * cap.effectiveRadius() * cap.effectiveRadius();
            Real P0 = 2.0 * PI * cap.effectiveRadius();
            fprintf(fp_, "%c%.6e%c%.6e%c%.6e%c%.6e",
                    sep, A, sep, (A0 > 1e-15 ? A / A0 : 0.0),
                    sep, P, sep, (P0 > 1e-15 ? P / P0 : 0.0));
        }

        if (fields_.orientation) {
            // Orientation: angle of major axis from inertia tensor
            Real Ixx = 0, Iyy = 0, Ixy = 0;
            for (int k = 0; k < Nn; ++k) {
                Vec2d p = cap.nodePosition(k);
                Real dx = p.x - cen.x;
                Real dy = p.y - cen.y;
                Ixx += dy * dy;
                Iyy += dx * dx;
                Ixy -= dx * dy;
            }
            Real angle = 0.5 * std::atan2(2.0 * Ixy, Ixx - Iyy);
            fprintf(fp_, "%c%.6e", sep, angle);
        }

        if (fields_.angular_vel) {
            // Angular velocity: omega = sum(r x v) / sum(r^2)
            Real num = 0, den = 0;
            for (int k = 0; k < Nn; ++k) {
                Vec2d p = cap.nodePosition(k);
                Vec2d v = cap.nodeVelocity(k);
                Real dx = p.x - cen.x;
                Real dy = p.y - cen.y;
                num += dx * (v.y - vel.y) - dy * (v.x - vel.x);
                den += dx * dx + dy * dy;
            }
            Real omega = (den > 1e-15) ? num / den : 0.0;
            fprintf(fp_, "%c%.6e", sep, omega);
        }

        fprintf(fp_, "\n");
    }

    fflush(fp_);
}

void CSVWriter::close() {
    if (fp_) {
        fclose(fp_);
        fp_ = nullptr;
    }
}

} // namespace softflow
