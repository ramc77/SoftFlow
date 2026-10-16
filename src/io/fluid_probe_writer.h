#pragma once
#include "../core/types.h"
#include "output_config.h"
#include <vector>
#include <string>
#include <cstdio>

namespace softflow {

class LatticeField;

// Writes time-series data at user-defined probe points
class FluidProbeWriter {
public:
    explicit FluidProbeWriter(const std::string& output_dir);
    ~FluidProbeWriter();

    void addProbe(int i, int j, const std::string& label);

    // Write one timestep of probe data
    void writeTimestep(const LatticeField& field, int step, Real time);

    void close();

private:
    std::string output_dir_;
    std::vector<FluidProbe> probes_;
    FILE* fp_ = nullptr;
    bool header_written_ = false;

    void openFile();
    void writeHeader();
};

} // namespace softflow
