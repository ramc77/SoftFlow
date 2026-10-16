#include "checkpoint.h"
#include "../engine/simulation.h"
#include "../lbm/advection_diffusion.h"
#include <fstream>
#include <cstring>
#include <iostream>

namespace softflow {

// Binary checkpoint format (version 3):
//
// HEADER:
//   magic:   "SFCK"  (4 bytes)
//   version: int     (= 3)
//   step:    int     (current timestep)
//   nx:      int     (lattice width)
//   ny:      int     (lattice height)
//   ncaps:   int     (number of capsules)
//
// LBM STATE:
//   f_data:  Real[nx * ny * 9]   (distribution functions)
//   rho:     Real[nx * ny]       (density)
//   ux:      Real[nx * ny]       (x-velocity)
//   uy:      Real[nx * ny]       (y-velocity)
//   flags:   CellType[nx * ny]   (cell type flags: FLUID, SOLID, etc.)
//
// CAPSULE STATE (for each capsule i = 0..ncaps-1):
//   id:          int
//   type_id:     int
//   num_nodes:   int
//   positions:   Real[num_nodes * 2]   (x,y pairs)
//   velocities:  Real[num_nodes * 2]   (vx,vy pairs)
//   rest_area:   Real
//   rest_perim:  Real
//   rest_lengths: Real[num_nodes]
//   periodic_nx: int
//
// ADHESION BONDS (if any):
//   nbonds:  int
//   bonds:   {capsule_i, node_i, capsule_j, node_j, rest_length, current_force} * nbonds
//
// SCALAR STATE (version 3+, present even when no scalar solver — has_scalar=0):
//   has_scalar:  int   (1 if scalar solver is active, else 0)
//   if has_scalar == 1:
//     n_species:    int
//     For each species s = 0..n_species-1:
//       g_data:  Real[nx * ny * 9]   (scalar distribution functions)
//       C_data:  Real[nx * ny]       (macroscopic concentration field)
//     n_tracked:    int              (size of capsule release/absorb arrays)
//     released:     Real[n_tracked]  (cumulative scalar released per capsule)
//     absorbed:     Real[n_tracked]  (cumulative scalar absorbed per capsule)

// Helper to write raw data
template<typename T>
static void writeVal(std::ofstream& out, const T& val) {
    out.write(reinterpret_cast<const char*>(&val), sizeof(T));
}

template<typename T>
static void writeArray(std::ofstream& out, const T* data, size_t count) {
    out.write(reinterpret_cast<const char*>(data), count * sizeof(T));
}

template<typename T>
static bool readVal(std::ifstream& in, T& val) {
    in.read(reinterpret_cast<char*>(&val), sizeof(T));
    return in.good();
}

template<typename T>
static bool readArray(std::ifstream& in, T* data, size_t count) {
    in.read(reinterpret_cast<char*>(data), count * sizeof(T));
    return in.good();
}

bool Checkpoint::save(const Simulation& sim, const std::string& filename) {
    std::ofstream out(filename, std::ios::binary);
    if (!out.is_open()) {
        std::cerr << "Checkpoint: cannot open " << filename << " for writing\n";
        return false;
    }

    const auto& params = sim.params();
    int nx = params.nx;
    int ny = params.ny;
    int step = sim.currentStep();
    const auto& field = sim.lbmSolver().field();
    const auto& caps = sim.capsules();
    int ncaps = caps.numCapsules();

    // ── Header ──
    out.write("SFCK", 4);
    int version = 4;
    writeVal(out, version);
    writeVal(out, step);
    writeVal(out, nx);
    writeVal(out, ny);
    writeVal(out, ncaps);

    // ── LBM distributions (the critical state) ──
    int nxy = nx * ny;
    writeArray(out, field.fData(), nxy * D2Q9::Q);

    // ── Macroscopic fields ──
    writeArray(out, field.rhoData(), nxy);
    writeArray(out, field.uxData(), nxy);
    writeArray(out, field.uyData(), nxy);

    // ── Cell type flags ──
    writeArray(out, field.cellTypeData(), nxy);

    // ── Capsule state ──
    for (int c = 0; c < ncaps; ++c) {
        const auto& cap = caps[c];
        int id = cap.getId();
        int tid = cap.getTypeId();
        int nn = cap.numNodes();
        int pnx = cap.getPeriodicX();

        writeVal(out, id);
        writeVal(out, tid);
        writeVal(out, nn);

        // Positions
        const auto& pos = cap.positions();
        for (int k = 0; k < nn; ++k) {
            writeVal(out, pos[k].x);
            writeVal(out, pos[k].y);
        }

        // Velocities
        const auto& vel = cap.velocities();
        for (int k = 0; k < nn; ++k) {
            writeVal(out, vel[k].x);
            writeVal(out, vel[k].y);
        }

        // Rest state
        Real rarea = cap.area();   // We'll store current area as proxy
        Real rperim = cap.perimeter();
        writeVal(out, rarea);
        writeVal(out, rperim);
        writeVal(out, pnx);
    }

    // ── Adhesion bonds ──
    const auto* adh = sim.adhesion();
    if (adh) {
        const auto& bonds = adh->getBonds();
        int nbonds = static_cast<int>(bonds.size());
        writeVal(out, nbonds);
        for (const auto& b : bonds) {
            writeVal(out, b.capsule_i);
            writeVal(out, b.node_i);
            writeVal(out, b.capsule_j);
            writeVal(out, b.node_j);
            writeVal(out, b.rest_length);
            writeVal(out, b.current_force);
        }
    } else {
        int nbonds = 0;
        writeVal(out, nbonds);
    }

    // ── Scalar transport state ──
    const AdvectionDiffusion* ad = sim.advectionDiffusion();
    int has_scalar = (ad != nullptr) ? 1 : 0;
    writeVal(out, has_scalar);
    if (has_scalar) {
        int n_species = ad->getNumSpecies();
        writeVal(out, n_species);
        for (int s = 0; s < n_species; ++s) {
            writeArray(out, ad->gData(s), ad->gSize(s));           // g distributions
            writeArray(out, ad->concentrationData(s), nxy);        // C field
        }
        int n_tracked = ad->numCapsuleTracked();
        writeVal(out, n_tracked);
        writeArray(out, ad->capsuleReleasedData(), n_tracked);     // cumulative released
        writeArray(out, ad->capsuleAbsorbedData(), n_tracked);     // cumulative absorbed

        // Version 4: particle chemical mass M_p + surface coverage Γ
        int n_mp = ad->numMpTracked();
        writeVal(out, n_mp);
        if (n_mp > 0)
            writeArray(out, ad->capsuleMpData(), n_mp);
        int n_gamma = ad->totalGammaNodes();
        writeVal(out, n_gamma);
        if (n_gamma > 0)
            writeArray(out, ad->gammaData(), n_gamma);
    }

    out.close();
    std::cout << "Checkpoint saved: step " << step << ", " << ncaps
              << " capsules" << (has_scalar ? ", scalar state (v4)" : "") << " → " << filename << std::endl;
    return true;
}

bool Checkpoint::load(Simulation& sim, const std::string& filename) {
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
        std::cerr << "Checkpoint: cannot open " << filename << " for reading\n";
        return false;
    }

    // ── Header ──
    char magic[4];
    in.read(magic, 4);
    if (std::strncmp(magic, "SFCK", 4) != 0) {
        std::cerr << "Checkpoint: invalid magic (not a SoftFlow checkpoint)\n";
        return false;
    }

    int version;
    readVal(in, version);
    if (version < 2 || version > 4) {
        std::cerr << "Checkpoint: unsupported version " << version
                  << " (expected 2, 3, or 4)\n";
        return false;
    }
    if (version == 2) {
        std::cerr << "Checkpoint: loading version 2 file — scalar transport state "
                     "will be reset to initial values (no scalar data in v2 files)\n";
    }

    int step, nx, ny, ncaps;
    readVal(in, step);
    readVal(in, nx);
    readVal(in, ny);
    readVal(in, ncaps);

    // Validate against current simulation
    const auto& params = sim.params();
    if (nx != params.nx || ny != params.ny) {
        std::cerr << "Checkpoint: domain mismatch. File has " << nx << "x" << ny
                  << " but simulation is " << params.nx << "x" << params.ny << "\n";
        return false;
    }

    int sim_ncaps = sim.capsules().numCapsules();
    if (ncaps != sim_ncaps) {
        std::cerr << "Checkpoint: capsule count mismatch. File has " << ncaps
                  << " but simulation has " << sim_ncaps << "\n";
        return false;
    }

    // ── Restore step counter ──
    sim.setCurrentStep(step);

    // ── Restore LBM distributions ──
    auto& field = sim.lbmSolver().field();
    int nxy = nx * ny;
    readArray(in, field.fData(), nxy * D2Q9::Q);

    // ── Restore macroscopic fields ──
    readArray(in, field.rhoData(), nxy);
    readArray(in, field.uxData(), nxy);
    readArray(in, field.uyData(), nxy);

    // ── Restore cell type flags ──
    readArray(in, field.cellTypeData(), nxy);

    // ── Restore capsule state ──
    auto& caps = sim.capsules();
    for (int c = 0; c < ncaps; ++c) {
        auto& cap = caps[c];
        int id, tid, nn, pnx;
        readVal(in, id);
        readVal(in, tid);
        readVal(in, nn);

        if (nn != cap.numNodes()) {
            std::cerr << "Checkpoint: node count mismatch for capsule " << c
                      << ". File has " << nn << ", sim has " << cap.numNodes() << "\n";
            return false;
        }

        // Positions
        auto& pos = cap.positions();
        for (int k = 0; k < nn; ++k) {
            readVal(in, pos[k].x);
            readVal(in, pos[k].y);
        }

        // Velocities
        auto& vel = cap.velocities();
        for (int k = 0; k < nn; ++k) {
            readVal(in, vel[k].x);
            readVal(in, vel[k].y);
        }

        // Skip rest state fields (we keep the original rest state)
        Real dummy_area, dummy_perim;
        int dummy_pnx;
        readVal(in, dummy_area);
        readVal(in, dummy_perim);
        readVal(in, dummy_pnx);
    }

    // ── Restore adhesion bonds ──
    int nbonds;
    readVal(in, nbonds);
    auto* adh = sim.adhesion();
    if (adh && nbonds > 0) {
        auto& bonds = adh->getBondsMutable();
        bonds.clear();
        bonds.resize(nbonds);
        for (int b = 0; b < nbonds; ++b) {
            readVal(in, bonds[b].capsule_i);
            readVal(in, bonds[b].node_i);
            readVal(in, bonds[b].capsule_j);
            readVal(in, bonds[b].node_j);
            readVal(in, bonds[b].rest_length);
            readVal(in, bonds[b].current_force);
        }
    }

    // ── Restore scalar transport state (version 3 only) ──
    if (version >= 3) {
        int has_scalar = 0;
        readVal(in, has_scalar);
        AdvectionDiffusion* ad = sim.advectionDiffusion();

        if (has_scalar && ad) {
            int n_species = 0;
            readVal(in, n_species);
            if (n_species != ad->getNumSpecies()) {
                std::cerr << "Checkpoint: scalar species count mismatch. "
                          << "File has " << n_species << ", sim has "
                          << ad->getNumSpecies() << "\n";
                return false;
            }
            for (int s = 0; s < n_species; ++s) {
                readArray(in, ad->gData(s), ad->gSize(s));          // g distributions
                readArray(in, ad->concentrationData(s), nxy);       // C field
            }
            int n_tracked = 0;
            readVal(in, n_tracked);
            // Resize to match checkpoint (capsule_released_ may be 0 before first step)
            if (n_tracked > 0) {
                ad->prepareCapsuleTrackersForLoad(n_tracked);
                readArray(in, ad->capsuleReleasedData(), n_tracked);    // cumulative released
                readArray(in, ad->capsuleAbsorbedData(), n_tracked);    // cumulative absorbed
            }

            // Version 4: particle mass M_p + surface coverage Γ
            if (version >= 4) {
                int n_mp = 0;
                readVal(in, n_mp);
                if (n_mp > 0)
                    readArray(in, ad->prepareMpForLoad(n_mp), n_mp);
                int n_gamma = 0;
                readVal(in, n_gamma);
                if (n_gamma > 0) {
                    Real* gptr = ad->prepareGammaFlatForLoad(n_gamma);
                    readArray(in, gptr, n_gamma);
                    ad->syncGammaFromFlat(sim.capsules());
                }
            }
        } else if (has_scalar && !ad) {
            std::cerr << "Checkpoint: file contains scalar state but sim has no "
                         "scalar solver — scalar state will be ignored\n";
        }
    }

    in.close();
    std::cout << "Checkpoint loaded: step " << step << ", " << ncaps
              << " capsules ← " << filename << std::endl;
    return true;
}

} // namespace softflow
