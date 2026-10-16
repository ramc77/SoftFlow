#include "rsa_inserter.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace softflow::insertion {

// ─── RSA ──────────────────────────────────────────────────────────────

RSAInserter::RSAInserter(std::shared_ptr<IRegion>           region,
                         int                                target_count,
                         std::shared_ptr<ISizeDistribution> sizes,
                         int                                max_attempts)
    : region_(std::move(region)),
      target_(target_count),
      sizes_(std::move(sizes)),
      max_attempts_(max_attempts)
{
    if (!region_) throw std::invalid_argument("RSAInserter: null region");
    if (!sizes_)  throw std::invalid_argument("RSAInserter: null sizes");
    if (target_  < 0) throw std::invalid_argument("RSAInserter: target_count < 0");
    if (max_attempts_ < 0) {
        throw std::invalid_argument("RSAInserter: max_attempts < 0");
    }
}

std::vector<Placement> RSAInserter::generate(const InsertionContext& ctx,
                                              std::mt19937_64& rng)
{
    std::vector<Placement> out;
    if (target_ == 0) return out;

    // Allow callers to leave the budget at 0 and use a sensible default.
    // Talbot et al. recommend ~100× target for near-jamming density;
    // we use 200× which is conservative but still bounded.
    const int budget = (max_attempts_ > 0) ? max_attempts_ : (200 * target_);

    const auto [lo, hi] = region_->bbox();
    std::uniform_real_distribution<Real> ux(lo.x, hi.x);
    std::uniform_real_distribution<Real> uy(lo.y, hi.y);

    // Make a working copy of the context so we can inject newly-
    // placed Placements as "existing" for subsequent overlap tests.
    InsertionContext local = ctx;
    local.existing_centers.reserve(local.existing_centers.size() + target_);
    local.existing_radii.reserve(local.existing_radii.size() + target_);

    int attempts = 0;
    int placed   = 0;
    while (placed < target_ && attempts < budget) {
        ++attempts;
        const Vec2d c{ ux(rng), uy(rng) };
        if (!region_->contains(c)) continue;
        const Real r = sizes_->sample(rng);
        if (!isPlacementValid(local, c, r)) continue;
        out.push_back({c, r});
        local.existing_centers.push_back(c);
        local.existing_radii.push_back(r);
        ++placed;
    }
    return out;
}

// ─── Poisson-disk (Bridson 2007) ─────────────────────────────────────

PoissonDiskInserter::PoissonDiskInserter(std::shared_ptr<IRegion>           region,
                                          Real                               r_min,
                                          std::shared_ptr<ISizeDistribution> sizes,
                                          int                                k)
    : region_(std::move(region)),
      r_min_(r_min),
      sizes_(std::move(sizes)),
      k_(k)
{
    if (!region_) throw std::invalid_argument("PoissonDiskInserter: null region");
    if (!sizes_)  throw std::invalid_argument("PoissonDiskInserter: null sizes");
    if (!(r_min_ > 0.0)) {
        throw std::invalid_argument("PoissonDiskInserter: r_min must be > 0");
    }
    if (k_ <= 0) throw std::invalid_argument("PoissonDiskInserter: k must be > 0");
}

std::vector<Placement> PoissonDiskInserter::generate(
    const InsertionContext& ctx, std::mt19937_64& rng)
{
    // We intentionally avoid std::numeric_limits<Real>::infinity() here:
    // SoftFlow compiles with -ffast-math (cmake/CompilerFlags.cmake)
    // which lets the optimizer assume `-ffinite-math-only`, and
    // `infinity()` then collapses to undefined behaviour (in practice,
    // 0 on AppleClang). the design review / Phase-1 follow-up tracks the
    // broader cleanup of -ffast-math hostile code. The largest finite
    // double is more than enough as a "no pair seen yet" sentinel.
    last_min_sep_ = std::numeric_limits<Real>::max();

    std::vector<Placement> out;

    const auto [lo, hi] = region_->bbox();
    const Real Lx = hi.x - lo.x;
    const Real Ly = hi.y - lo.y;
    if (Lx <= 0.0 || Ly <= 0.0) return out;

    // Bridson's grid: each cell holds at most one sample. Cell size
    // r_min / sqrt(2) guarantees that any two samples in the same or
    // neighbouring cells satisfy the r_min separation iff they are
    // pairwise tested against neighbours within a 5×5 stencil.
    const Real cell = r_min_ / std::sqrt(static_cast<Real>(2.0));
    const int  gx   = std::max(1, static_cast<int>(std::ceil(Lx / cell)));
    const int  gy   = std::max(1, static_cast<int>(std::ceil(Ly / cell)));
    std::vector<int> grid(static_cast<std::size_t>(gx) * gy, -1);

    auto gridIndex = [&](Vec2d p) {
        const int ix = std::clamp(static_cast<int>((p.x - lo.x) / cell), 0, gx - 1);
        const int iy = std::clamp(static_cast<int>((p.y - lo.y) / cell), 0, gy - 1);
        return iy * gx + ix;
    };

    // Working overlap context (includes our own placements).
    InsertionContext local = ctx;

    auto tryAcceptCandidate = [&](Vec2d c, Real r, int /*requesterIdx*/) -> int {
        if (!region_->contains(c)) return -1;
        // Bridson's separation check via the 5×5 grid stencil.
        const int ix = std::clamp(static_cast<int>((c.x - lo.x) / cell), 0, gx - 1);
        const int iy = std::clamp(static_cast<int>((c.y - lo.y) / cell), 0, gy - 1);
        for (int dj = -2; dj <= 2; ++dj) {
            for (int di = -2; di <= 2; ++di) {
                const int jx = ix + di;
                const int jy = iy + dj;
                if (jx < 0 || jx >= gx || jy < 0 || jy >= gy) continue;
                const int idx = grid[jy * gx + jx];
                if (idx < 0) continue;
                const Vec2d& q = out[static_cast<std::size_t>(idx)].center;
                if (distance(local, c, q) < r_min_) return -1;
            }
        }
        if (!isPlacementValid(local, c, r)) return -1;
        // Accept.
        const int new_idx = static_cast<int>(out.size());
        out.push_back({c, r});
        local.existing_centers.push_back(c);
        local.existing_radii.push_back(r);
        grid[static_cast<std::size_t>(iy) * gx + ix] = new_idx;
        return new_idx;
    };

    // Seed with one accepted point, drawn uniformly until one passes
    // the validity tests. Bound the seeding loop so a degenerate
    // region (e.g. surrounded by obstacles) doesn't cause infinite
    // spin.
    const int seed_attempts = 64;
    {
        std::uniform_real_distribution<Real> ux(lo.x, hi.x);
        std::uniform_real_distribution<Real> uy(lo.y, hi.y);
        bool seeded = false;
        for (int i = 0; i < seed_attempts && !seeded; ++i) {
            const Vec2d c{ ux(rng), uy(rng) };
            const Real  r = sizes_->sample(rng);
            if (tryAcceptCandidate(c, r, -1) >= 0) seeded = true;
        }
        if (!seeded) return out;
    }

    // Active list: indices into `out` that still want to spawn.
    std::vector<int> active;
    active.push_back(0);

    std::uniform_real_distribution<Real> uang(0.0, 2.0 * M_PI);
    std::uniform_real_distribution<Real> ufrac(0.0, 1.0);

    while (!active.empty()) {
        std::uniform_int_distribution<int> pick(0, static_cast<int>(active.size()) - 1);
        const int      slot = pick(rng);
        const Vec2d&   src  = out[static_cast<std::size_t>(active[slot])].center;
        bool spawned = false;
        for (int i = 0; i < k_ && !spawned; ++i) {
            // Bridson 2007 Eq. (1): annulus [r_min, 2 r_min].
            const Real theta = uang(rng);
            const Real radius = r_min_ * std::sqrt(1.0 + 3.0 * ufrac(rng));
            const Vec2d c{
                src.x + radius * std::cos(theta),
                src.y + radius * std::sin(theta),
            };
            const Real r = sizes_->sample(rng);
            const int new_idx = tryAcceptCandidate(c, r, active[slot]);
            if (new_idx >= 0) {
                active.push_back(new_idx);
                spawned = true;
            }
        }
        if (!spawned) {
            // Exhausted this site — remove it from the active list.
            active[slot] = active.back();
            active.pop_back();
        }
    }

    // Diagnostic: tightest pairwise separation actually achieved.
    // Useful for V&V — if Bridson is correct, `last_min_sep_` must be
    // ≥ r_min on return.
    for (std::size_t i = 0; i < out.size(); ++i) {
        for (std::size_t j = i + 1; j < out.size(); ++j) {
            const Real d = distance(ctx, out[i].center, out[j].center);
            if (d < last_min_sep_) last_min_sep_ = d;
        }
    }
    if (out.size() < 2) last_min_sep_ = std::numeric_limits<Real>::max();

    return out;
}

} // namespace softflow::insertion
