#pragma once

#include "inserter.h"
#include "region.h"
#include "size_distribution.h"

#include <memory>

namespace softflow::insertion {

/// Random sequential addition (Widom 1966): repeatedly pick a uniform
/// candidate centre in the region and a radius from `sizes`, accept
/// if it passes `isPlacementValid`, reject otherwise. Stops when
/// `target_count` placements have landed or `max_attempts` candidates
/// have been drawn (whichever comes first). Returns whatever was
/// placed — the inserter never spins forever.
///
/// For monodisperse disks the long-time saturation density approaches
/// the 2-D RSA jamming limit φ_J ≈ 0.547. The classical sampling-
/// efficiency literature (Talbot–Tarjus–Schaaf, *Adv. Colloid
/// Interface Sci.* 2000) recommends ~ 100 × target_count attempts to
/// approach within a few percent of jamming.
class RSAInserter final : public IInserter {
public:
    RSAInserter(std::shared_ptr<IRegion>           region,
                int                                target_count,
                std::shared_ptr<ISizeDistribution> sizes,
                int                                max_attempts = 0);

    std::vector<Placement> generate(const InsertionContext& ctx,
                                    std::mt19937_64& rng) override;

private:
    std::shared_ptr<IRegion>           region_;
    int                                target_;
    std::shared_ptr<ISizeDistribution> sizes_;
    int                                max_attempts_;
};

/// Bridson (2007) Poisson-disk sampling. Maintains a background grid
/// of cell size `r_min / sqrt(2)` so that each cell holds at most one
/// sample, and an "active list" of recently-placed points. For each
/// active point we attempt up to `k` new candidates in the annulus
/// [r_min, 2·r_min] around it; accepted ones go on the active list.
/// Output is a quasi-uniform "blue-noise" point set with every pair
/// at least `r_min + min_gap` apart.
///
/// `r_min` is the minimum centre-to-centre separation (must satisfy
/// `r_min ≥ 2 · maxRadius(sizes) + ctx.min_gap` for non-overlapping
/// disks). The classic Bridson default is `k = 30` candidates per
/// active site, which gives near-optimal coverage with rejection
/// probability < 0.1 % per cell.
class PoissonDiskInserter final : public IInserter {
public:
    PoissonDiskInserter(std::shared_ptr<IRegion>           region,
                        Real                               r_min,
                        std::shared_ptr<ISizeDistribution> sizes,
                        int                                k = 30);

    std::vector<Placement> generate(const InsertionContext& ctx,
                                    std::mt19937_64& rng) override;

    /// Tightest separation actually achieved between any two
    /// placements in the most recent generate() call. Useful for
    /// V&V; reset to +∞ at the start of each generate().
    Real lastMinSeparation() const { return last_min_sep_; }

private:
    std::shared_ptr<IRegion>           region_;
    Real                               r_min_;
    std::shared_ptr<ISizeDistribution> sizes_;
    int                                k_;
    Real                               last_min_sep_ = 0.0;
};

} // namespace softflow::insertion
