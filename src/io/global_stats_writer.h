#pragma once
#include "../core/types.h"
#include <string>
#include <cstdio>

namespace softflow {

class LatticeField;
class CapsuleSystem;
class AdvectionDiffusion;

// Writes per-timestep global statistics to CSV
class GlobalStatsWriter {
public:
    explicit GlobalStatsWriter(const std::string& output_dir);
    ~GlobalStatsWriter();

    void writeTimestep(const LatticeField& field, const CapsuleSystem& capsules,
                       int step, Real time,
                       const AdvectionDiffusion* ad = nullptr);

    void close();

private:
    std::string output_dir_;
    FILE* fp_ = nullptr;
    bool header_written_ = false;

    void openFile();
    void writeHeader(bool has_scalar);
};

} // namespace softflow
