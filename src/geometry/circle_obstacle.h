#pragma once
#include "obstacle.h"

namespace softflow {

class CircleObstacle : public Obstacle {
public:
    CircleObstacle(Real cx, Real cy, Real radius)
        : cx_(cx), cy_(cy), radius_(radius) {}

    bool contains(Real x, Real y) const override {
        Real dx = x - cx_;
        Real dy = y - cy_;
        return (dx * dx + dy * dy) <= radius_ * radius_;
    }

    Real signedDistance(Real x, Real y) const override {
        Real dx = x - cx_;
        Real dy = y - cy_;
        return std::sqrt(dx * dx + dy * dy) - radius_;
    }

    Vec2d nearestPoint(Real x, Real y) const override {
        Real dx = x - cx_;
        Real dy = y - cy_;
        Real d = std::sqrt(dx * dx + dy * dy);
        if (d < 1e-15) return {cx_ + radius_, cy_};
        return {cx_ + radius_ * dx / d, cy_ + radius_ * dy / d};
    }

    Vec2d normalAt(Real x, Real y) const override {
        Real dx = x - cx_;
        Real dy = y - cy_;
        Real d = std::sqrt(dx * dx + dy * dy);
        if (d < 1e-15) return {1.0, 0.0};
        return {dx / d, dy / d};
    }

    Real getCx() const { return cx_; }
    Real getCy() const { return cy_; }
    Real getRadius() const { return radius_; }

private:
    Real cx_, cy_, radius_;
};

} // namespace softflow
