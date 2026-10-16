#pragma once

#include "inserter.h"
#include "region.h"
#include "size_distribution.h"

#include <memory>

namespace softflow::insertion {

/// Base class for inserters that fire on every timestep rather than
/// once at setup. Each registered dynamic inserter is invoked from
/// `Simulation::step()` after capsule motion and before metrics
/// (`simulation.cpp` step 10/11), with an `InsertionContext` that
/// already includes every existing capsule. The returned placements
/// are added immediately.
///
/// `dt` is the physical timestep in lattice units. `rng` is the
/// inserter's own deterministic sub-stream, derived from
/// `params.rng_seed` and the registration tag (see
/// `Simulation::registerDynamicInserter`).
///
/// Implementations must be safe to call thousands of times per run
/// with no allocation churn — the existing-capsule field grows over
/// time, but the per-call work should remain O(N · k) for small k.
class IDynamicInserter {
public:
    virtual ~IDynamicInserter() = default;

    virtual std::vector<Placement> step(
        const InsertionContext& ctx,
        Real dt,
        std::mt19937_64& rng) = 0;
};

/// Poisson process: each timestep draws N ~ Poisson(rate · dt) and
/// attempts that many uniform placements in `region`. Placements that
/// fail overlap or wall checks are silently dropped — the realised
/// rate is therefore ≤ `rate`, asymptotically equal under low
/// occupancy. The natural choice when the user wants a stochastic
/// "trickle" of new particles, e.g. injecting drug carriers at an
/// inlet.
class PoissonStochasticInserter final : public IDynamicInserter {
public:
    PoissonStochasticInserter(std::shared_ptr<IRegion>           region,
                              Real                               rate,
                              std::shared_ptr<ISizeDistribution> sizes,
                              int                                attempts_per_event = 16);

    std::vector<Placement> step(const InsertionContext& ctx,
                                Real dt,
                                std::mt19937_64& rng) override;

private:
    std::shared_ptr<IRegion>           region_;
    Real                               rate_;       // events per unit lattice time
    std::shared_ptr<ISizeDistribution> sizes_;
    int                                attempts_;   // RSA budget per event
};

/// Constant-flux feeder: keeps the area fraction φ in the region at
/// or above `target_phi`. Each step we (a) measure the current φ in
/// `region` from the capsules already in `ctx.existing_centers`, and
/// (b) if there is a deficit, attempt up to `max_per_step` RSA
/// placements to close it. The realised φ in steady state is just
/// below `target_phi` (because each placement is one capsule, and we
/// stop as soon as the fraction crosses).
///
/// the project specification: "Constant flux insertion at an inlet region
/// (target volume fraction φ)". The "flux" name is historical — we
/// don't impose a flux directly; we maintain φ, which gives a
/// flux at steady state equal to (φ · u̅) / region_length.
class ConstantFluxInserter final : public IDynamicInserter {
public:
    ConstantFluxInserter(std::shared_ptr<IRegion>           region,
                         Real                               target_phi,
                         std::shared_ptr<ISizeDistribution> sizes,
                         int                                max_per_step = 4,
                         int                                attempts_per_event = 32);

    std::vector<Placement> step(const InsertionContext& ctx,
                                Real dt,
                                std::mt19937_64& rng) override;

private:
    std::shared_ptr<IRegion>           region_;
    Real                               target_phi_;
    std::shared_ptr<ISizeDistribution> sizes_;
    int                                max_per_step_;
    int                                attempts_;
};

/// Conveyor: maintains a target capsule *count* inside `region`. Each
/// step we count how many existing capsules have their centroid in
/// the region; if that drops below `target_count`, we replenish (up
/// to `max_per_step` placements). The natural way to keep a periodic
/// channel "fed" — as particles drift past the inlet zone, new ones
/// take their place.
class ConveyorInserter final : public IDynamicInserter {
public:
    ConveyorInserter(std::shared_ptr<IRegion>           region,
                     int                                target_count,
                     std::shared_ptr<ISizeDistribution> sizes,
                     int                                max_per_step = 4,
                     int                                attempts_per_event = 32);

    std::vector<Placement> step(const InsertionContext& ctx,
                                Real dt,
                                std::mt19937_64& rng) override;

private:
    std::shared_ptr<IRegion>           region_;
    int                                target_count_;
    std::shared_ptr<ISizeDistribution> sizes_;
    int                                max_per_step_;
    int                                attempts_;
};

} // namespace softflow::insertion
