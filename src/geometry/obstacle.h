#pragma once
#include "../core/types.h"

namespace softflow {

class Obstacle {
public:
    virtual ~Obstacle() = default;
    virtual bool contains(Real x, Real y) const = 0;
    virtual Real signedDistance(Real x, Real y) const = 0;
    virtual Vec2d nearestPoint(Real x, Real y) const = 0;
    virtual Vec2d normalAt(Real x, Real y) const = 0;
};

} // namespace softflow
