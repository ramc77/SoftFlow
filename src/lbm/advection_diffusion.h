#pragma once
#include "lattice_field.h"
#include "../core/parameters.h"
#include <vector>

namespace softflow {

class CapsuleSystem;

/// LBM-based advection-diffusion solver for passive scalar transport.
/// Uses a separate distribution function g_q on the D2Q9 lattice,
/// advected by the fluid velocity field.
///
/// Applications: glucose/sugar concentration, drug delivery,
/// virus particle transport, chemokine gradients.
class AdvectionDiffusion {
public:
    AdvectionDiffusion(int nx, int ny, const ScalarParams& params);

    /// Initialize concentration field
    void initialize(Real initial_concentration = 0.0);

    /// Set concentration in a rectangular region
    void setRegion(int x0, int y0, int x1, int y1, Real concentration,
                   int species = 0);

    /// Set concentration at a single point
    void setPoint(int x, int y, Real concentration, int species = 0);

    /// Perform one timestep: collision + streaming (uses fluid velocity)
    void step(const LatticeField& fluid_field);

    /// Apply source/sink terms at capsule locations (constant-rate, backward-compatible)
    void applySourceSink(const CapsuleSystem& capsules,
                         const std::vector<Real>& release_rates,
                         const std::vector<Real>& absorption_rates);

    /// Physics-based chemistry: concentration-dependent leaching (Fick), Langmuir
    /// adsorption/desorption, particle mass M_p tracking, Peskin-spread fluxes.
    /// Falls back to constant-rate logic per type when physics params are absent.
    void applyChemistry(const CapsuleSystem& capsules,
                        const std::vector<Real>& release_rates,
                        const std::vector<Real>& absorption_rates,
                        const ScalarParams& sp);

    /// Get concentration at a point
    Real getConcentration(int x, int y, int species = 0) const;

    /// Get full concentration field pointer (for VTK output)
    const Real* concentrationData(int species = 0) const;
    Real* concentrationData(int species = 0);

    int getNx() const { return nx_; }
    int getNy() const { return ny_; }
    int getNumSpecies() const { return n_species_; }

    /// Raw distribution data for checkpoint save/load
    const Real* gData(int species = 0) const { return g_[species].data(); }
    Real*       gData(int species = 0)       { return g_[species].data(); }
    size_t      gSize(int species = 0) const { return g_[species].size(); }

    /// Raw capsule tracking arrays for checkpoint save/load
    const Real* capsuleReleasedData() const { return capsule_released_.data(); }
    Real*       capsuleReleasedData()       { return capsule_released_.data(); }
    const Real* capsuleAbsorbedData() const { return capsule_absorbed_.data(); }
    Real*       capsuleAbsorbedData()       { return capsule_absorbed_.data(); }
    int         numCapsuleTracked()   const { return static_cast<int>(capsule_released_.size()); }
    /// Resize released/absorbed arrays for checkpoint load (avoids lazy-init mismatch)
    void        prepareCapsuleTrackersForLoad(int n) {
        capsule_released_.assign(n, 0.0);
        capsule_absorbed_.assign(n, 0.0);
    }

    /// Per-capsule cumulative released / absorbed amount
    Real getCapsuleReleased(int capsule_id) const {
        return (capsule_id >= 0 && capsule_id < static_cast<int>(capsule_released_.size()))
               ? capsule_released_[capsule_id] : 0.0;
    }
    Real getCapsuleAbsorbed(int capsule_id) const {
        return (capsule_id >= 0 && capsule_id < static_cast<int>(capsule_absorbed_.size()))
               ? capsule_absorbed_[capsule_id] : 0.0;
    }
    int numTrackedCapsules() const { return static_cast<int>(capsule_released_.size()); }

    /// Particle chemical reservoir mass (−1 = infinite)
    Real getCapsuleMp(int capsule_id) const {
        return (capsule_id >= 0 && capsule_id < static_cast<int>(capsule_Mp_.size()))
               ? capsule_Mp_[capsule_id] : -1.0;
    }
    const Real* capsuleMpData() const { return capsule_Mp_.data(); }
    Real*       capsuleMpData()       { return capsule_Mp_.data(); }
    int         numMpTracked()  const { return static_cast<int>(capsule_Mp_.size()); }
    /// Resize M_p array for checkpoint load (-1 = infinite reservoir default)
    Real*       prepareMpForLoad(int n) {
        capsule_Mp_.assign(n, -1.0); return capsule_Mp_.data();
    }

    /// Surface coverage Γ per capsule node (Langmuir adsorption state)
    Real getNodeGamma(int capsule_id, int node_k) const {
        if (capsule_id < 0 || capsule_id >= static_cast<int>(gamma_nodes_.size())) return 0.0;
        const auto& g = gamma_nodes_[capsule_id];
        return (node_k >= 0 && node_k < static_cast<int>(g.size())) ? g[node_k] : 0.0;
    }
    void setNodeGamma(int capsule_id, int node_k, Real gamma) {
        if (capsule_id >= 0 && capsule_id < static_cast<int>(gamma_nodes_.size())) {
            auto& g = gamma_nodes_[capsule_id];
            if (node_k >= 0 && node_k < static_cast<int>(g.size()))
                g[node_k] = gamma;
        }
    }

    /// Flat Γ array for checkpoint I/O
    int         totalGammaNodes() const;
    const Real* gammaData() const;   // syncs flat buffer (mutable cache), returns pointer
    /// Resize flat buffer and return writable pointer (used by checkpoint load)
    Real*       prepareGammaFlatForLoad(int n) {
        gamma_flat_.assign(n, 0.0); return gamma_flat_.data();
    }
    void        syncGammaFromFlat(const CapsuleSystem& capsules);

private:
    int nx_, ny_;
    int n_species_;
    bool periodic_y_;                 // periodic BC in y (else no-flux)
    std::vector<Real> diffusivity_;   // per species
    std::vector<Real> tau_s_;         // tau_s = 3*D + 0.5

    // Distribution functions: g_[species][idx*Q + q]
    std::vector<std::vector<Real>> g_;
    std::vector<std::vector<Real>> g_tmp_;

    // Macroscopic concentration: C_[species][idx]
    std::vector<std::vector<Real>> C_;

    // Inlet concentration BCs
    std::vector<Real> inlet_concentration_;

    // Per-capsule cumulative scalar tracking
    std::vector<Real> capsule_released_;
    std::vector<Real> capsule_absorbed_;

    // Physics-based chemistry state (Gap B + C)
    std::vector<Real>              capsule_Mp_;      // particle chemical reservoir (−1=infinite)
    std::vector<std::vector<Real>> gamma_nodes_;     // surface coverage Γ[capsule][node]
    mutable std::vector<Real>      gamma_flat_;      // flat copy for checkpoint I/O (mutable cache)

    int idx(int x, int y) const { return y * nx_ + x; }

    // Three-pass D2Q9 BGK for the scalar:
    //   1. collide()         — read g_, write post-collision into g_tmp_
    //   2. streamWithBC()    — pull from g_tmp_ into g_, with halfway
    //                          bounce-back for SOLID upstream cells and
    //                          periodic wrap in x (and optional y).
    //   3. computeConcentration() — accumulate C_ from streamed g_.
    //
    // The pre-Phase-1 implementation fused collide+stream into one pass
    // and applied SOLID bounce-back by writing into g_ from inside an
    // OpenMP parallel region that read g_ via a `src` pointer — a real
    // race. The split below removes the race entirely (all writers and
    // readers are disjoint within each pass) and matches the textbook
    // D2Q9-ADR scheme (Krüger 2017 Ch.8).
    void collide(int species, const LatticeField& fluid_field);
    void streamWithBC(int species, const LatticeField& fluid_field);
    void computeConcentration(int species);
};

} // namespace softflow
