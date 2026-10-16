#include "dynamic_inserter.h"

#include <cmath>
#include <stdexcept>

namespace softflow::insertion {

namespace {

// Try `attempts` uniform placements in `region`, returning the
// accepted ones. Used by all three dynamic inserters; differs from
// RSAInserter only in that we do not maintain a target_count — the
// caller decides how many to draw.
std::vector<Placement> rsaDrawN(const InsertionContext&  ctx_in,
                                IRegion&                 region,
                                ISizeDistribution&       sizes,
                                int                      target_count,
                                int                      attempts,
                                std::mt19937_64&         rng)
{
    std::vector<Placement> out;
    if (target_count <= 0 || attempts <= 0) return out;

    const auto [lo, hi] = region.bbox();
    std::uniform_real_distribution<Real> ux(lo.x, hi.x);
    std::uniform_real_distribution<Real> uy(lo.y, hi.y);

    InsertionContext local = ctx_in;
    local.existing_centers.reserve(local.existing_centers.size() + target_count);
    local.existing_radii.reserve(local.existing_radii.size() + target_count);

    int placed = 0, tries = 0;
    while (placed < target_count && tries < attempts) {
        ++tries;
        const Vec2d c{ ux(rng), uy(rng) };
        if (!region.contains(c)) continue;
        const Real r = sizes.sample(rng);
        if (!isPlacementValid(local, c, r)) continue;
        out.push_back({c, r});
        local.existing_centers.push_back(c);
        local.existing_radii.push_back(r);
        ++placed;
    }
    return out;
}

// Estimate the current area fraction inside `region` from the list of
// existing capsules. We approximate every disk as being entirely in
// the region if its centre is in the region — exact for fully
// contained disks, optimistic for disks straddling the region
// boundary. Inserters use this to detect deficits, not to compute
// publication-grade densities; the segregation analysis (Phase 3)
// uses a better estimator.
Real currentAreaFraction(const InsertionContext& ctx, const IRegion& region) {
    Real disk_area_sum = 0.0;
    const std::size_t n = ctx.existing_centers.size();
    for (std::size_t i = 0; i < n; ++i) {
        if (!region.contains(ctx.existing_centers[i])) continue;
        const Real r = (i < ctx.existing_radii.size()) ? ctx.existing_radii[i] : 0.0;
        disk_area_sum += M_PI * r * r;
    }
    return disk_area_sum / region.area();
}

int currentCount(const InsertionContext& ctx, const IRegion& region) {
    int n = 0;
    for (const auto& c : ctx.existing_centers) if (region.contains(c)) ++n;
    return n;
}

}  // namespace

// ─── PoissonStochasticInserter ───────────────────────────────────────

PoissonStochasticInserter::PoissonStochasticInserter(
    std::shared_ptr<IRegion>           region,
    Real                               rate,
    std::shared_ptr<ISizeDistribution> sizes,
    int                                attempts_per_event)
    : region_(std::move(region)),
      rate_(rate),
      sizes_(std::move(sizes)),
      attempts_(attempts_per_event)
{
    if (!region_) throw std::invalid_argument("PoissonStochasticInserter: null region");
    if (!sizes_)  throw std::invalid_argument("PoissonStochasticInserter: null sizes");
    if (rate_ < 0.0) throw std::invalid_argument("PoissonStochasticInserter: rate must be ≥ 0");
    if (attempts_ <= 0) {
        throw std::invalid_argument("PoissonStochasticInserter: attempts_per_event must be > 0");
    }
}

std::vector<Placement> PoissonStochasticInserter::step(
    const InsertionContext& ctx, Real dt, std::mt19937_64& rng)
{
    const Real lambda = rate_ * dt;
    if (!(lambda > 0.0)) return {};
    std::poisson_distribution<int> p(lambda);
    const int n_target = p(rng);
    if (n_target <= 0) return {};
    return rsaDrawN(ctx, *region_, *sizes_, n_target,
                    attempts_ * n_target, rng);
}

// ─── ConstantFluxInserter ────────────────────────────────────────────

ConstantFluxInserter::ConstantFluxInserter(
    std::shared_ptr<IRegion>           region,
    Real                               target_phi,
    std::shared_ptr<ISizeDistribution> sizes,
    int                                max_per_step,
    int                                attempts_per_event)
    : region_(std::move(region)),
      target_phi_(target_phi),
      sizes_(std::move(sizes)),
      max_per_step_(max_per_step),
      attempts_(attempts_per_event)
{
    if (!region_) throw std::invalid_argument("ConstantFluxInserter: null region");
    if (!sizes_)  throw std::invalid_argument("ConstantFluxInserter: null sizes");
    if (!(target_phi_ >= 0.0 && target_phi_ <= 1.0)) {
        throw std::invalid_argument("ConstantFluxInserter: target_phi must be in [0, 1]");
    }
    if (max_per_step_ <= 0 || attempts_ <= 0) {
        throw std::invalid_argument("ConstantFluxInserter: max_per_step / attempts_per_event must be > 0");
    }
}

std::vector<Placement> ConstantFluxInserter::step(
    const InsertionContext& ctx, Real /*dt*/, std::mt19937_64& rng)
{
    const Real phi = currentAreaFraction(ctx, *region_);
    if (phi >= target_phi_) return {};
    return rsaDrawN(ctx, *region_, *sizes_,
                    max_per_step_, attempts_ * max_per_step_, rng);
}

// ─── ConveyorInserter ────────────────────────────────────────────────

ConveyorInserter::ConveyorInserter(
    std::shared_ptr<IRegion>           region,
    int                                target_count,
    std::shared_ptr<ISizeDistribution> sizes,
    int                                max_per_step,
    int                                attempts_per_event)
    : region_(std::move(region)),
      target_count_(target_count),
      sizes_(std::move(sizes)),
      max_per_step_(max_per_step),
      attempts_(attempts_per_event)
{
    if (!region_) throw std::invalid_argument("ConveyorInserter: null region");
    if (!sizes_)  throw std::invalid_argument("ConveyorInserter: null sizes");
    if (target_count_ < 0) {
        throw std::invalid_argument("ConveyorInserter: target_count must be ≥ 0");
    }
    if (max_per_step_ <= 0 || attempts_ <= 0) {
        throw std::invalid_argument("ConveyorInserter: max_per_step / attempts_per_event must be > 0");
    }
}

std::vector<Placement> ConveyorInserter::step(
    const InsertionContext& ctx, Real /*dt*/, std::mt19937_64& rng)
{
    const int have = currentCount(ctx, *region_);
    const int need = target_count_ - have;
    if (need <= 0) return {};
    const int n = std::min(need, max_per_step_);
    return rsaDrawN(ctx, *region_, *sizes_, n, attempts_ * n, rng);
}

} // namespace softflow::insertion
