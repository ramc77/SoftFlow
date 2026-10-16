// Optional CSV dump of verification measurements, for the figures in the
// paper.  Inactive unless the environment variable SOFTFLOW_VV_DIR names
// a directory; the pass/fail checks are unaffected.
#pragma once

#include <cstdio>
#include <cstdlib>
#include <string>

namespace vv {

inline FILE* open(const std::string& name, const char* header) {
    const char* dir = std::getenv("SOFTFLOW_VV_DIR");
    if (!dir) return nullptr;
    FILE* f = std::fopen((std::string(dir) + "/" + name + ".csv").c_str(), "w");
    if (f) std::fprintf(f, "%s\n", header);
    return f;
}

inline void close(FILE* f) { if (f) std::fclose(f); }

} // namespace vv
