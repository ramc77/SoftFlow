#pragma once
#include "obstacle.h"
#include <algorithm>

namespace softflow {

class RectObstacle : public Obstacle {
public:
    RectObstacle(Real x0, Real y0, Real x1, Real y1)
        : x0_(std::min(x0, x1)), y0_(std::min(y0, y1)),
          x1_(std::max(x0, x1)), y1_(std::max(y0, y1)) {}

    bool contains(Real x, Real y) const override {
        return x >= x0_ && x <= x1_ && y >= y0_ && y <= y1_;
    }

    Real signedDistance(Real x, Real y) const override {
        Real dx = std::max({x0_ - x, x - x1_, Real(0)});
        Real dy = std::max({y0_ - y, y - y1_, Real(0)});
        Real outside = std::sqrt(dx * dx + dy * dy);
        if (outside > 1e-15) return outside;
        // Inside: negative distance to nearest edge
        Real dmin = std::min({x - x0_, x1_ - x, y - y0_, y1_ - y});
        return -dmin;
    }

    Vec2d nearestPoint(Real x, Real y) const override {
        Real cx = std::clamp(x, x0_, x1_);
        Real cy = std::clamp(y, y0_, y1_);
        if (cx == x && cy == y) {
            // Point is inside: find nearest edge
            Real dl = x - x0_, dr = x1_ - x;
            Real db = y - y0_, dt = y1_ - y;
            Real dmin = std::min({dl, dr, db, dt});
            if (dmin == dl) return {x0_, y};
            if (dmin == dr) return {x1_, y};
            if (dmin == db) return {x, y0_};
            return {x, y1_};
        }
        return {cx, cy};
    }

    Vec2d normalAt(Real x, Real y) const override {
        Vec2d np = nearestPoint(x, y);
        Real dx = x - np.x;
        Real dy = y - np.y;
        Real d = std::sqrt(dx * dx + dy * dy);
        if (d < 1e-15) {
            // On the surface: use nearest edge normal
            Real dl = std::abs(x - x0_), dr = std::abs(x - x1_);
            Real db = std::abs(y - y0_), dt = std::abs(y - y1_);
            Real dmin = std::min({dl, dr, db, dt});
            if (dmin == dl) return {-1.0, 0.0};
            if (dmin == dr) return {1.0, 0.0};
            if (dmin == db) return {0.0, -1.0};
            return {0.0, 1.0};
        }
        return {dx / d, dy / d};
    }

    Real getX0() const { return x0_; }
    Real getY0() const { return y0_; }
    Real getX1() const { return x1_; }
    Real getY1() const { return y1_; }

private:
    Real x0_, y0_, x1_, y1_;
};

} // namespace softflow
