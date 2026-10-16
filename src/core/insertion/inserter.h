#pragma once

#include "../types.h"
#include "../../geometry/obstacle.h"

#include <memory>
#include <random>
#include <vector>

namespace softflow::insertion {

/// One placement decision produced by an inserter: a centre and a
/// radius. The Simulation translates each Placement into a call to
/// `addCapsule(...)` with the membrane parameters and type supplied
/// at the insertion call site. Inserters never mutate Simulation
/// state directly — they return placements, the caller commits them.
struct Placement {
    Vec2d center;
    Real  radius;
};

/// Information every inserter needs to make safe placement decisions.
/// Walls, obstacles, and the existing capsule field are all hard
/// constraints; min_gap is a soft envelope around every existing
/// circle and the channel walls. periodic_nx > 0 enables the
/// streamwise minimum-image distance test.
struct InsertionContext {
    int     nx           = 0;
    int     ny           = 0;
    Real    wall_y_bottom = 0.0;
    Real    wall_y_top    = 0.0;
    int     periodic_nx   = 0;            // 0 = non-periodic
    Real    min_gap       = 1.0;
    int     max_attempts  = 0;            // 0 = strategy-specific default

    std::vector<std::shared_ptr<Obstacle>> obstacles;
    std::vector<Vec2d>                     existing_centers;
    std::vector<Real>                      existing_radii;
};

/// Static-fill inserter base. `generate()` is called once at setup
/// and returns the placements; the Simulation walks the result and
/// calls `addCapsule` for each. Implementations must:
///
///   - respect `ctx.min_gap` against walls, obstacles, and every
///     existing capsule (overlap helpers below do the right thing);
///   - never spin forever — saturate gracefully and return what was
///     placed if the region cannot accommodate the request;
///   - be deterministic in `rng` so seeded runs reproduce bit-exact.
class IInserter {
public:
    virtual ~IInserter() = default;

    virtual std::vector<Placement> generate(
        const InsertionContext& ctx,
        std::mt19937_64& rng) = 0;
};

// ─── shared safety helpers, used by every concrete inserter ──────────

/// Streamwise minimum-image distance. dx is wrapped through
/// periodic_nx if it exceeds half the domain, matching the convention
/// the rest of the engine uses (see `Capsule::minImageDiff`). When
/// periodic_nx == 0 the wrap is a no-op.
Real minImageDx(Real dx, int periodic_nx);

/// Distance between two points under the streamwise periodic-x
/// convention encoded in `ctx`. Strictly positive.
Real distance(const InsertionContext& ctx, Vec2d a, Vec2d b);

/// Fast-rejection overlap test: a candidate placement at `center`
/// with `radius` is acceptable if it stays at least `ctx.min_gap`
/// from every existing capsule, both walls, and every obstacle.
bool isPlacementValid(const InsertionContext& ctx,
                      Vec2d center,
                      Real  radius);

} // namespace softflow::insertion
