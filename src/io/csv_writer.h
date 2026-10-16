#pragma once
#include "../core/types.h"
#include "output_config.h"
#include <string>
#include <cstdio>

namespace softflow {

class CapsuleSystem;

// Flexible CSV writer with field selection and particle filtering
class CSVWriter {
public:
    CSVWriter(const std::string& output_dir,
              const std::string& filename = "particle_data.csv",
              const std::string& format = "csv",
              bool append = true);

    ~CSVWriter();

    // Configure which fields to output
    void setFields(const ParticleOutputFields& fields) { fields_ = fields; }

    // Configure particle filter
    void setFilter(const ParticleFilter& filter) { filter_ = filter; }

    // Write data for one timestep (appends to file)
    void writeTimestep(const CapsuleSystem& capsules, int step, Real time);

    // Close the file
    void close();

private:
    std::string output_dir_;
    std::string filename_;
    std::string format_;
    bool append_;
    FILE* fp_ = nullptr;
    bool header_written_ = false;
    ParticleOutputFields fields_;
    ParticleFilter filter_;

    void openFile();
    void writeHeader();
    char separator() const { return (format_ == "dat" || format_ == "DAT") ? '\t' : ','; }
};

} // namespace softflow
