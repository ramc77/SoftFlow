#include "region.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace softflow::insertion {

// ─── Rectangle ───────────────────────────────────────────────────────

RectRegion::RectRegion(Real x0, Real x1, Real y0, Real y1)
    : x0_(std::min(x0, x1)), x1_(std::max(x0, x1)),
      y0_(std::min(y0, y1)), y1_(std::max(y0, y1))
{
    // Reject degenerate (zero-area) rectangles up front so dynamic
    // inserters don't divide-by-zero when computing area fractions.
    if (x1_ <= x0_ || y1_ <= y0_) {
        throw std::invalid_argument("RectRegion: zero-area or inverted rectangle");
    }
}

bool RectRegion::contains(Vec2d p) const {
    return p.x >= x0_ && p.x <= x1_ && p.y >= y0_ && p.y <= y1_;
}

std::pair<Vec2d, Vec2d> RectRegion::bbox() const {
    return { Vec2d{x0_, y0_}, Vec2d{x1_, y1_} };
}

Real RectRegion::area() const {
    return (x1_ - x0_) * (y1_ - y0_);
}

// ─── Circle ──────────────────────────────────────────────────────────

CircleRegion::CircleRegion(Vec2d center, Real radius)
    : center_(center), r_(radius)
{
    if (!(radius > 0.0)) {
        throw std::invalid_argument("CircleRegion: radius must be > 0");
    }
}

bool CircleRegion::contains(Vec2d p) const {
    const Real dx = p.x - center_.x;
    const Real dy = p.y - center_.y;
    return (dx * dx + dy * dy) <= r_ * r_;
}

std::pair<Vec2d, Vec2d> CircleRegion::bbox() const {
    return {
        Vec2d{ center_.x - r_, center_.y - r_ },
        Vec2d{ center_.x + r_, center_.y + r_ }
    };
}

Real CircleRegion::area() const {
    return M_PI * r_ * r_;
}

// ─── Polygon ─────────────────────────────────────────────────────────

PolygonRegion::PolygonRegion(std::vector<Vec2d> vertices)
    : verts_(std::move(vertices))
{
    if (verts_.size() < 3) {
        throw std::invalid_argument("PolygonRegion: need at least 3 vertices");
    }

    bbox_lo_ = bbox_hi_ = verts_[0];
    for (const auto& v : verts_) {
        bbox_lo_.x = std::min(bbox_lo_.x, v.x);
        bbox_lo_.y = std::min(bbox_lo_.y, v.y);
        bbox_hi_.x = std::max(bbox_hi_.x, v.x);
        bbox_hi_.y = std::max(bbox_hi_.y, v.y);
    }

    // Shoelace area; absolute value so the result is orientation-
    // independent (CW vs CCW vertices both give a positive area).
    Real twice = 0.0;
    const std::size_t n = verts_.size();
    for (std::size_t i = 0; i < n; ++i) {
        const Vec2d& a = verts_[i];
        const Vec2d& b = verts_[(i + 1) % n];
        twice += (a.x * b.y) - (b.x * a.y);
    }
    area_ = 0.5 * std::abs(twice);
    if (!(area_ > 0.0)) {
        throw std::invalid_argument("PolygonRegion: degenerate (zero-area) polygon");
    }
}

bool PolygonRegion::contains(Vec2d p) const {
    // Crossing-number / ray-casting (Sutherland 1974). Cast a
    // half-line from `p` along +x and count intersections with
    // polygon edges; an odd count means inside. The standard
    // reference implementation handles edges parallel to the ray
    // implicitly via the strict y comparisons.
    bool inside = false;
    const std::size_t n = verts_.size();
    for (std::size_t i = 0, j = n - 1; i < n; j = i++) {
        const Vec2d& a = verts_[i];
        const Vec2d& b = verts_[j];
        if ((a.y > p.y) != (b.y > p.y)) {
            const Real x_cross = (b.x - a.x) * (p.y - a.y) / (b.y - a.y) + a.x;
            if (p.x < x_cross) inside = !inside;
        }
    }
    return inside;
}

std::pair<Vec2d, Vec2d> PolygonRegion::bbox() const {
    return { bbox_lo_, bbox_hi_ };
}

Real PolygonRegion::area() const {
    return area_;
}

} // namespace softflow::insertion
