#pragma once
#include "../core/types.h"
#include <vector>
#include <string>
#include <cmath>

namespace softflow {

// ── Selectable fluid fields for VTK output ────────────────────────
struct FluidOutputFields {
    bool density       = true;
    bool velocity      = true;
    bool pressure      = true;
    bool vorticity     = false;
    bool strain_rate   = false;
    bool ibm_force     = false;
    bool node_type     = true;
    bool velocity_mag  = true;
    bool component_density = true;  // for multi-component
    bool concentration = true;      // scalar transport concentration field
};

// ── Selectable particle fields for VTK output ─────────────────────
struct ParticleVTKFields {
    bool velocity       = true;
    bool force          = true;
    bool particle_id    = true;
    bool particle_type  = true;
    bool particle_group = true;
    bool local_strain   = false;
    bool local_curvature = false;
};

// ── Selectable particle fields for CSV output ─────────────────────
struct ParticleOutputFields {
    bool position      = true;
    bool velocity      = true;
    bool force         = false;
    bool diameter      = false;
    bool type          = true;
    bool group         = false;
    bool deformation   = false;
    bool area_volume   = false;
    bool orientation   = false;
    bool angular_vel   = false;
    bool com_position  = true;
    bool com_velocity  = true;
};

// ── Particle filter for CSV output ────────────────────────────────
struct ParticleFilter {
    enum FilterType { ALL, BY_ID, BY_TYPE, BY_GROUP, BY_REGION, COMBINED };
    FilterType filter_type = ALL;

    std::vector<int> selected_ids;
    std::vector<int> selected_types;
    std::vector<int> selected_groups;

    double region_xmin = -1e30, region_xmax = 1e30;
    double region_ymin = -1e30, region_ymax = 1e30;

    bool passes(int id, int type_id, int group, double cx, double cy) const {
        switch (filter_type) {
            case ALL: return true;
            case BY_ID:
                for (int sid : selected_ids) if (sid == id) return true;
                return false;
            case BY_TYPE:
                for (int st : selected_types) if (st == type_id) return true;
                return false;
            case BY_GROUP:
                for (int sg : selected_groups) if (sg == group) return true;
                return false;
            case BY_REGION:
                return cx >= region_xmin && cx <= region_xmax &&
                       cy >= region_ymin && cy <= region_ymax;
            case COMBINED: {
                // Must pass all non-empty filters
                if (!selected_types.empty()) {
                    bool found = false;
                    for (int st : selected_types) if (st == type_id) { found = true; break; }
                    if (!found) return false;
                }
                if (!selected_ids.empty()) {
                    bool found = false;
                    for (int sid : selected_ids) if (sid == id) { found = true; break; }
                    if (!found) return false;
                }
                if (!selected_groups.empty()) {
                    bool found = false;
                    for (int sg : selected_groups) if (sg == group) { found = true; break; }
                    if (!found) return false;
                }
                if (cx < region_xmin || cx > region_xmax ||
                    cy < region_ymin || cy > region_ymax)
                    return false;
                return true;
            }
        }
        return true;
    }
};

// ── Fluid probe definition ────────────────────────────────────────
struct FluidProbe {
    int i, j;
    std::string label;
};

// ── Extra CSV output configuration ────────────────────────────────
struct ExtraCSVConfig {
    std::string filename;
    int dump_every = 100;
    ParticleOutputFields fields;
    ParticleFilter filter;
    bool append = true;
};

// ── Master output configuration ───────────────────────────────────
struct OutputConfig {
    std::string output_dir = "output";

    // VTK dump interval
    int vtk_dump_every = 1000;
    std::string vtk_format = "ascii";  // "ascii" or "binary"

    // CSV dump interval
    int csv_dump_every = 100;
    std::string csv_format = "csv";    // "csv" or "dat"
    bool csv_append = true;

    // Probe dump interval
    int probe_dump_every = 10;

    // Stats dump interval
    int stats_dump_every = 100;

    // VTK fluid field toggles
    FluidOutputFields fluid_fields;

    // VTK particle field toggles
    ParticleVTKFields particle_vtk_fields;

    // CSV particle field selection
    ParticleOutputFields csv_fields;
    ParticleFilter csv_filter;

    // Extra CSV outputs (different filters/fields)
    std::vector<ExtraCSVConfig> extra_csv;

    // Fluid probes
    std::vector<FluidProbe> probes;
};

} // namespace softflow
