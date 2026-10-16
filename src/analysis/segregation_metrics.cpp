#include "segregation_metrics.h"
#include "../membrane/capsule_system.h"
#include "../membrane/adhesion.h"
#include <cmath>
#include <fstream>
#include <algorithm>
#include <numeric>

namespace softflow {

SegregationMetrics::SegregationMetrics(int ny, int n_bins)
    : ny_(ny), n_bins_(n_bins), cfl_threshold_(0.15)
{}

SegregationResults SegregationMetrics::compute(
    const CapsuleSystem& capsules,
    const AdhesionModel* adhesion) {

    SegregationResults results;

    computeLateralDistribution(capsules, results);
    computeMargination(capsules, results);
    computeMixingEntropy(results);
    computeCFL(capsules, results);
    computeDeformationStats(capsules, results);
    computeVelocityProfiles(capsules, results);
    computeRDF(capsules, results);

    // Cluster statistics from adhesion model
    if (adhesion) {
        results.num_clusters = adhesion->getNumClusters();
        const auto& sizes = adhesion->getClusterSizes();
        if (!sizes.empty()) {
            results.max_cluster_size = *std::max_element(sizes.begin(), sizes.end());
            Real sum = std::accumulate(sizes.begin(), sizes.end(), 0.0);
            results.mean_cluster_size = sum / sizes.size();
        }
    }

    return results;
}

void SegregationMetrics::computeLateralDistribution(
    const CapsuleSystem& capsules, SegregationResults& results) {

    int ncaps = capsules.numCapsules();
    if (ncaps == 0) return;

    // Find max type
    int max_type = 0;
    for (int c = 0; c < ncaps; ++c) {
        max_type = std::max(max_type, capsules[c].getType());
    }
    int n_types = max_type + 1;

    results.lateral_distribution.resize(n_types, std::vector<Real>(n_bins_, 0.0));

    Real H = static_cast<Real>(ny_ - 2); // channel height (excluding walls)
    Real dy = H / n_bins_;

    for (int c = 0; c < ncaps; ++c) {
        Vec2d cen = capsules[c].centroid();
        int type = capsules[c].getType();
        Real y_norm = (cen.y - 1.0) / H; // normalize to [0, 1]
        int bin = static_cast<int>(y_norm * n_bins_);
        bin = std::max(0, std::min(n_bins_ - 1, bin));
        results.lateral_distribution[type][bin] += 1.0;
    }

    // Normalize to probability
    for (int t = 0; t < n_types; ++t) {
        Real total = 0.0;
        for (int b = 0; b < n_bins_; ++b) total += results.lateral_distribution[t][b];
        if (total > 0) {
            for (int b = 0; b < n_bins_; ++b) {
                results.lateral_distribution[t][b] /= (total * dy);
            }
        }
    }
}

void SegregationMetrics::computeMargination(
    const CapsuleSystem& capsules, SegregationResults& results) {

    int ncaps = capsules.numCapsules();
    if (ncaps == 0) return;

    int max_type = 0;
    for (int c = 0; c < ncaps; ++c)
        max_type = std::max(max_type, capsules[c].getType());
    int n_types = max_type + 1;

    Real H = static_cast<Real>(ny_ - 2);
    Real wall_region = cfl_threshold_ * H;

    results.margination_parameter.resize(n_types, 0.0);
    results.separation_efficiency.resize(n_types, 0.0);

    std::vector<int> near_wall(n_types, 0);
    std::vector<int> total(n_types, 0);

    for (int c = 0; c < ncaps; ++c) {
        int type = capsules[c].getType();
        Vec2d cen = capsules[c].centroid();
        Real y_rel = cen.y - 1.0; // distance from bottom wall
        total[type]++;

        if (y_rel < wall_region || y_rel > H - wall_region) {
            near_wall[type]++;
        }
    }

    Real M_random = 2.0 * cfl_threshold_; // expected fraction near wall if random
    for (int t = 0; t < n_types; ++t) {
        if (total[t] > 0) {
            results.margination_parameter[t] =
                static_cast<Real>(near_wall[t]) / total[t];
            Real M = results.margination_parameter[t];
            results.separation_efficiency[t] =
                (M_random < 1.0) ? (M - M_random) / (1.0 - M_random) : 0.0;
        }
    }
}

void SegregationMetrics::computeMixingEntropy(SegregationResults& results) {
    if (results.lateral_distribution.empty()) return;
    int n_types = static_cast<int>(results.lateral_distribution.size());

    Real S = 0.0;
    for (int b = 0; b < n_bins_; ++b) {
        Real total_in_bin = 0.0;
        for (int t = 0; t < n_types; ++t) {
            total_in_bin += results.lateral_distribution[t][b];
        }
        if (total_in_bin < 1e-15) continue;

        for (int t = 0; t < n_types; ++t) {
            Real p = results.lateral_distribution[t][b] / total_in_bin;
            if (p > 1e-15) {
                S -= p * std::log(p);
            }
        }
    }
    results.mixing_entropy = S / n_bins_; // average over bins
}

void SegregationMetrics::computeCFL(
    const CapsuleSystem& capsules, SegregationResults& results) {

    int ncaps = capsules.numCapsules();
    if (ncaps == 0) return;

    Real min_y = 1e10, max_y = -1e10;
    for (int c = 0; c < ncaps; ++c) {
        Vec2d cen = capsules[c].centroid();
        min_y = std::min(min_y, cen.y);
        max_y = std::max(max_y, cen.y);
    }

    Real H = static_cast<Real>(ny_ - 2);
    results.cfl_bottom = (min_y - 1.0) / H;
    results.cfl_top = (static_cast<Real>(ny_ - 1) - max_y) / H;
}

void SegregationMetrics::computeDeformationStats(
    const CapsuleSystem& capsules, SegregationResults& results) {

    int ncaps = capsules.numCapsules();
    if (ncaps == 0) return;

    int max_type = 0;
    for (int c = 0; c < ncaps; ++c)
        max_type = std::max(max_type, capsules[c].getType());
    int n_types = max_type + 1;

    std::vector<std::vector<Real>> D_vals(n_types);
    for (int c = 0; c < ncaps; ++c) {
        D_vals[capsules[c].getType()].push_back(capsules[c].deformationIndex());
    }

    results.mean_deformation.resize(n_types, 0.0);
    results.std_deformation.resize(n_types, 0.0);

    for (int t = 0; t < n_types; ++t) {
        if (D_vals[t].empty()) continue;
        Real mean = std::accumulate(D_vals[t].begin(), D_vals[t].end(), 0.0)
                    / D_vals[t].size();
        Real var = 0.0;
        for (Real d : D_vals[t]) var += (d - mean) * (d - mean);
        var /= D_vals[t].size();
        results.mean_deformation[t] = mean;
        results.std_deformation[t] = std::sqrt(var);
    }
}

void SegregationMetrics::computeVelocityProfiles(
    const CapsuleSystem& capsules, SegregationResults& results) {

    int ncaps = capsules.numCapsules();
    if (ncaps == 0) return;

    int max_type = 0;
    for (int c = 0; c < ncaps; ++c)
        max_type = std::max(max_type, capsules[c].getType());
    int n_types = max_type + 1;

    results.velocity_profile.resize(n_types, std::vector<Real>(n_bins_, 0.0));
    std::vector<std::vector<int>> count(n_types, std::vector<int>(n_bins_, 0));

    Real H = static_cast<Real>(ny_ - 2);

    for (int c = 0; c < ncaps; ++c) {
        Vec2d cen = capsules[c].centroid();
        int type = capsules[c].getType();
        Real y_norm = (cen.y - 1.0) / H;
        int bin = static_cast<int>(y_norm * n_bins_);
        bin = std::max(0, std::min(n_bins_ - 1, bin));

        // Average velocity from all nodes
        Real vx_avg = 0.0;
        for (int ni = 0; ni < capsules[c].numNodes(); ++ni) {
            vx_avg += capsules[c].nodeVelocity(ni).x;
        }
        vx_avg /= capsules[c].numNodes();

        results.velocity_profile[type][bin] += vx_avg;
        count[type][bin]++;
    }

    for (int t = 0; t < n_types; ++t) {
        for (int b = 0; b < n_bins_; ++b) {
            if (count[t][b] > 0)
                results.velocity_profile[t][b] /= count[t][b];
        }
    }
}

void SegregationMetrics::computeRDF(
    const CapsuleSystem& capsules, SegregationResults& results) {

    int ncaps = capsules.numCapsules();
    if (ncaps < 2) return;

    int n_rdf_bins = 50;
    Real r_max = static_cast<Real>(ny_) * 0.5;
    Real dr = r_max / n_rdf_bins;

    results.rdf_r.resize(n_rdf_bins, 0.0);
    results.rdf_g.resize(n_rdf_bins, 0.0);
    std::vector<int> counts(n_rdf_bins, 0);

    for (int b = 0; b < n_rdf_bins; ++b) {
        results.rdf_r[b] = (b + 0.5) * dr;
    }

    for (int ci = 0; ci < ncaps; ++ci) {
        Vec2d cI = capsules[ci].centroid();
        for (int cj = ci + 1; cj < ncaps; ++cj) {
            Vec2d cJ = capsules[cj].centroid();
            Real r = (cJ - cI).norm();
            int bin = static_cast<int>(r / dr);
            if (bin >= 0 && bin < n_rdf_bins) {
                counts[bin]++;
            }
        }
    }

    // Normalize: g(r) = counts / (N * rho * 2*pi*r*dr)
    Real rho_avg = static_cast<Real>(ncaps) / (static_cast<Real>(ny_ - 2) * 100.0);
    for (int b = 0; b < n_rdf_bins; ++b) {
        Real r = results.rdf_r[b];
        Real shell_area = 2.0 * PI * r * dr;
        Real expected = 0.5 * ncaps * rho_avg * shell_area;
        results.rdf_g[b] = (expected > 1e-15) ? counts[b] / expected : 0.0;
    }
}

void SegregationMetrics::writeCSV(const std::string& filename,
                                   const SegregationResults& results,
                                   int step, Real time) {
    std::ofstream out(filename);
    if (!out.is_open()) return;

    out << "step,time,mixing_entropy,cfl_bottom,cfl_top,"
        << "num_clusters,mean_cluster_size,max_cluster_size\n";
    out << step << "," << time << ","
        << results.mixing_entropy << ","
        << results.cfl_bottom << ","
        << results.cfl_top << ","
        << results.num_clusters << ","
        << results.mean_cluster_size << ","
        << results.max_cluster_size << "\n";

    // Per-type margination
    out << "\ntype,margination,separation_efficiency,mean_D,std_D\n";
    for (int t = 0; t < static_cast<int>(results.margination_parameter.size()); ++t) {
        out << t << ","
            << results.margination_parameter[t] << ","
            << results.separation_efficiency[t] << ","
            << results.mean_deformation[t] << ","
            << results.std_deformation[t] << "\n";
    }

    out.close();
}

} // namespace softflow
