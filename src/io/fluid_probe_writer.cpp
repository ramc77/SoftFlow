#include "fluid_probe_writer.h"
#include "../lbm/lattice_field.h"
#include <sys/stat.h>

namespace softflow {

FluidProbeWriter::FluidProbeWriter(const std::string& output_dir)
    : output_dir_(output_dir)
{
    mkdir(output_dir_.c_str(), 0755);
}

FluidProbeWriter::~FluidProbeWriter() {
    close();
}

void FluidProbeWriter::addProbe(int i, int j, const std::string& label) {
    probes_.push_back({i, j, label});
    // Reset header since probe list changed
    header_written_ = false;
}

void FluidProbeWriter::openFile() {
    if (fp_) return;
    std::string path = output_dir_ + "/fluid_probes.csv";
    fp_ = fopen(path.c_str(), "a");
}

void FluidProbeWriter::writeHeader() {
    if (!fp_ || header_written_ || probes_.empty()) return;

    fprintf(fp_, "timestep,time");
    for (const auto& p : probes_) {
        fprintf(fp_, ",%s_rho,%s_ux,%s_uy,%s_p",
                p.label.c_str(), p.label.c_str(),
                p.label.c_str(), p.label.c_str());
    }
    fprintf(fp_, "\n");
    header_written_ = true;
}

void FluidProbeWriter::writeTimestep(const LatticeField& field, int step, Real time) {
    if (probes_.empty()) return;

    openFile();
    if (!fp_) return;
    writeHeader();

    int nx = field.getNx();
    int ny = field.getNy();

    fprintf(fp_, "%d,%.6e", step, time);
    for (const auto& p : probes_) {
        int x = std::clamp(p.i, 0, nx - 1);
        int y = std::clamp(p.j, 0, ny - 1);
        Real rho = field.getRho(x, y);
        Real ux = field.getUx(x, y);
        Real uy = field.getUy(x, y);
        Real pressure = rho / 3.0;
        fprintf(fp_, ",%.6e,%.6e,%.6e,%.6e", rho, ux, uy, pressure);
    }
    fprintf(fp_, "\n");
    fflush(fp_);
}

void FluidProbeWriter::close() {
    if (fp_) {
        fclose(fp_);
        fp_ = nullptr;
    }
}

} // namespace softflow
