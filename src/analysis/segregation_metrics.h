#pragma once
#include "../core/types.h"
#include "../core/parameters.h"
#include <vector>
#include <string>

namespace softflow {

class CapsuleSystem;
class AdhesionModel;

/// Quantitative segregation and biomedical metrics computed during simulation.
struct SegregationResults {
    // Per-type lateral distributions
    std::vector<std::vector<Real>> lateral_distribution; // [type][y_bin]

    // Margination parameter: fraction of type in near-wall region
    std::vector<Real> margination_parameter; // per type

    // Mixing entropy: 0 = fully segregated, ln(N_bins) = perfectly mixed
    Real mixing_entropy = 0.0;

    // Separation efficiency (normalized margination)
    std::vector<Real> separation_efficiency; // per type

    // Cell-free layer thickness (from each wall)
    Real cfl_bottom = 0.0;
    Real cfl_top = 0.0;

    // Cluster statistics (if adhesion enabled)
    int num_clusters = 0;
    Real mean_cluster_size = 0.0;
    int max_cluster_size = 0;

    // Deformation index statistics per type
    std::vector<Real> mean_deformation; // per type
    std::vector<Real> std_deformation;  // per type

    // Velocity profiles per type
    std::vector<std::vector<Real>> velocity_profile; // [type][y_bin]

    // Radial distribution function g(r)
    std::vector<Real> rdf_r;    // bin centers
    std::vector<Real> rdf_g;    // g(r) values
};

/// Compute segregation metrics from current capsule state.
class SegregationMetrics {
public:
    SegregationMetrics(int ny, int n_bins = 20);

    /// Compute all metrics
    SegregationResults compute(const CapsuleSystem& capsules,
                               const AdhesionModel* adhesion = nullptr);

    /// Write metrics to CSV file
    void writeCSV(const std::string& filename,
                  const SegregationResults& results,
                  int step, Real time);

private:
    int ny_;
    int n_bins_;
    Real cfl_threshold_; // fraction of H for CFL measurement

    void computeLateralDistribution(const CapsuleSystem& capsules,
                                     SegregationResults& results);
    void computeMargination(const CapsuleSystem& capsules,
                            SegregationResults& results);
    void computeMixingEntropy(SegregationResults& results);
    void computeCFL(const CapsuleSystem& capsules,
                    SegregationResults& results);
    void computeDeformationStats(const CapsuleSystem& capsules,
                                  SegregationResults& results);
    void computeVelocityProfiles(const CapsuleSystem& capsules,
                                  SegregationResults& results);
    void computeRDF(const CapsuleSystem& capsules,
                    SegregationResults& results);
};

} // namespace softflow
