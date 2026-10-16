#include "lattice_inserter.h"

#include <cmath>
#include <stdexcept>

namespace softflow::insertion {

namespace {

// Standard sentinel for the jitter parameter. We deliberately leave
// 0.5 unreachable because at jitter=0.5 a placement can land exactly
// at its neighbour's lattice site, defeating the "structured layout"
// guarantee.
constexpr Real kMaxJitter = 0.499;

// Apply a uniform [-jitter*scale, +jitter*scale] perturbation to a
// scalar coordinate. Uses a single mt19937_64 stream so that repeated
// calls during a single generate() produce a deterministic, seed-
// dependent jitter pattern.
Real jittered(std::mt19937_64& rng, Real x, Real scale, Real jitter) {
    if (jitter <= 0.0) return x;
    std::uniform_real_distribution<Real> dist(-jitter * scale, jitter * scale);
    return x + dist(rng);
}

}  // namespace

// ─── Square lattice ──────────────────────────────────────────────────

SquareLatticeInserter::SquareLatticeInserter(
    std::shared_ptr<IRegion>           region,
    Real                               spacing_x,
    Real                               spacing_y,
    std::shared_ptr<ISizeDistribution> sizes,
    Real                               jitter)
    : region_(std::move(region)),
      sx_(spacing_x),
      sy_(spacing_y),
      sizes_(std::move(sizes)),
      jitter_(jitter)
{
    if (!region_) throw std::invalid_argument("SquareLatticeInserter: null region");
    if (!sizes_)  throw std::invalid_argument("SquareLatticeInserter: null sizes");
    if (!(sx_ > 0.0 && sy_ > 0.0)) {
        throw std::invalid_argument("SquareLatticeInserter: spacings must be > 0");
    }
    if (jitter_ < 0.0 || jitter_ >= kMaxJitter) {
        throw std::invalid_argument("SquareLatticeInserter: jitter must be in [0, 0.5)");
    }
}

std::vector<Placement> SquareLatticeInserter::generate(
    const InsertionContext& ctx, std::mt19937_64& rng)
{
    std::vector<Placement> out;

    const auto [lo, hi] = region_->bbox();

    // Step half a spacing into the region to centre the first node.
    const Real x_start = lo.x + 0.5 * sx_;
    const Real y_start = lo.y + 0.5 * sy_;

    for (Real y = y_start; y <= hi.y; y += sy_) {
        for (Real x = x_start; x <= hi.x; x += sx_) {
            const Vec2d c{
                jittered(rng, x, sx_, jitter_),
                jittered(rng, y, sy_, jitter_)
            };
            if (!region_->contains(c)) continue;
            const Real r = sizes_->sample(rng);
            if (!isPlacementValid(ctx, c, r)) continue;
            out.push_back({c, r});
        }
    }

    return out;
}

// ─── Hexagonal close-packed lattice ─────────────────────────────────

HexagonalLatticeInserter::HexagonalLatticeInserter(
    std::shared_ptr<IRegion>           region,
    Real                               spacing,
    std::shared_ptr<ISizeDistribution> sizes,
    Real                               jitter)
    : region_(std::move(region)),
      s_(spacing),
      sizes_(std::move(sizes)),
      jitter_(jitter)
{
    if (!region_) throw std::invalid_argument("HexagonalLatticeInserter: null region");
    if (!sizes_)  throw std::invalid_argument("HexagonalLatticeInserter: null sizes");
    if (!(s_ > 0.0)) {
        throw std::invalid_argument("HexagonalLatticeInserter: spacing must be > 0");
    }
    if (jitter_ < 0.0 || jitter_ >= kMaxJitter) {
        throw std::invalid_argument("HexagonalLatticeInserter: jitter must be in [0, 0.5)");
    }
}

std::vector<Placement> HexagonalLatticeInserter::generate(
    const InsertionContext& ctx, std::mt19937_64& rng)
{
    std::vector<Placement> out;

    const auto [lo, hi] = region_->bbox();

    // Row spacing for hex close-packing: dy = sqrt(3)/2 · s.
    const Real dy   = 0.5 * std::sqrt(static_cast<Real>(3.0)) * s_;
    const Real dx_o = 0.5 * s_;     // odd-row offset

    int row = 0;
    for (Real y = lo.y + 0.5 * dy; y <= hi.y; y += dy) {
        const Real x_offset = (row & 1) ? dx_o : 0.0;
        for (Real x = lo.x + 0.5 * s_ + x_offset; x <= hi.x; x += s_) {
            const Vec2d c{
                jittered(rng, x, s_, jitter_),
                jittered(rng, y, dy, jitter_)
            };
            if (!region_->contains(c)) continue;
            const Real r = sizes_->sample(rng);
            if (!isPlacementValid(ctx, c, r)) continue;
            out.push_back({c, r});
        }
        ++row;
    }

    return out;
}

} // namespace softflow::insertion
