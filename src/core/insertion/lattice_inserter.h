#pragma once

#include "inserter.h"
#include "region.h"
#include "size_distribution.h"

#include <memory>

namespace softflow::insertion {

/// Square-grid layout. Walks the region's bounding box on a regular
/// (sx, sy) lattice, samples a radius from `sizes`, optionally jitters
/// the centre by `jitter * spacing`, and accepts the placement only if
/// it (a) lies inside the region and (b) passes `isPlacementValid`.
///
/// `jitter ∈ [0, 0.5)` is the fractional offset relative to the
/// spacing. Pure 0 gives a perfect grid; values around 0.1 break the
/// degenerate symmetry that can stall the LBM at low Reynolds.
class SquareLatticeInserter final : public IInserter {
public:
    SquareLatticeInserter(std::shared_ptr<IRegion>           region,
                          Real                               spacing_x,
                          Real                               spacing_y,
                          std::shared_ptr<ISizeDistribution> sizes,
                          Real                               jitter = 0.0);

    std::vector<Placement> generate(const InsertionContext& ctx,
                                    std::mt19937_64& rng) override;

private:
    std::shared_ptr<IRegion>           region_;
    Real                               sx_, sy_;
    std::shared_ptr<ISizeDistribution> sizes_;
    Real                               jitter_;
};

/// Close-packed hexagonal lattice. Rows are spaced by sqrt(3)/2 * s,
/// odd rows offset by s/2. This is the densest disk packing in 2-D
/// (φ_max ≈ 0.9069 for monodisperse touching disks at s = 2r).
///
/// `spacing` is the centre-to-centre distance between nearest neighbours.
/// For a touching arrangement, set `spacing = 2 * radius`. For looser
/// fills, use `spacing > 2 * radius`.
class HexagonalLatticeInserter final : public IInserter {
public:
    HexagonalLatticeInserter(std::shared_ptr<IRegion>           region,
                             Real                               spacing,
                             std::shared_ptr<ISizeDistribution> sizes,
                             Real                               jitter = 0.0);

    std::vector<Placement> generate(const InsertionContext& ctx,
                                    std::mt19937_64& rng) override;

private:
    std::shared_ptr<IRegion>           region_;
    Real                               s_;
    std::shared_ptr<ISizeDistribution> sizes_;
    Real                               jitter_;
};

} // namespace softflow::insertion
