#include "inserter.h"

#include <cmath>

namespace softflow::insertion {

Real minImageDx(Real dx, int periodic_nx) {
    if (periodic_nx <= 0) return dx;
    const Real Lx = static_cast<Real>(periodic_nx);
    if (dx >  0.5 * Lx) dx -= Lx;
    if (dx < -0.5 * Lx) dx += Lx;
    return dx;
}

Real distance(const InsertionContext& ctx, Vec2d a, Vec2d b) {
    const Real dx = minImageDx(a.x - b.x, ctx.periodic_nx);
    const Real dy = a.y - b.y;            // y is never periodic in SoftFlow
    return std::sqrt(dx * dx + dy * dy);
}

bool isPlacementValid(const InsertionContext& ctx,
                      Vec2d center,
                      Real  radius)
{
    // Wall envelope. We require the disk plus min_gap to stay between
    // the wall lines. wall_y_bottom and wall_y_top are the y-coords of
    // the two solid wall rows; the fluid extent is (bottom, top).
    if (center.y - radius < ctx.wall_y_bottom + ctx.min_gap) return false;
    if (center.y + radius > ctx.wall_y_top    - ctx.min_gap) return false;

    // Obstacle envelope. signedDistance > 0 outside the obstacle, < 0
    // inside. We require strictly outside by at least radius + min_gap.
    for (const auto& obs : ctx.obstacles) {
        if (!obs) continue;
        const Real sd = obs->signedDistance(center.x, center.y);
        if (sd < radius + ctx.min_gap) return false;
    }

    // Inter-capsule overlap. Worst case: existing capsule of radius
    // r_e at distance d. Required: d > radius + r_e + min_gap.
    const auto&    centers = ctx.existing_centers;
    const auto&    radii   = ctx.existing_radii;
    const std::size_t n    = centers.size();
    for (std::size_t i = 0; i < n; ++i) {
        const Real r_e   = (i < radii.size()) ? radii[i] : radius;
        const Real d     = distance(ctx, center, centers[i]);
        if (d < radius + r_e + ctx.min_gap) return false;
    }

    return true;
}

} // namespace softflow::insertion
