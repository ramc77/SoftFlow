#pragma once
#include "obstacle.h"
#include <vector>

namespace softflow {

/// Arbitrary polygon obstacle defined by a list of vertices.
/// Uses ray-casting for point-in-polygon test and edge projection
/// for signed distance computation.
class PolygonObstacle : public Obstacle {
public:
    explicit PolygonObstacle(const std::vector<Vec2d>& vertices);

    bool contains(Real x, Real y) const override;
    Real signedDistance(Real x, Real y) const override;
    Vec2d nearestPoint(Real x, Real y) const override;
    Vec2d normalAt(Real x, Real y) const override;

private:
    std::vector<Vec2d> vertices_;
    int n_verts_;

    /// Nearest point on edge (v0, v1) to point p
    Vec2d nearestPointOnEdge(const Vec2d& p, const Vec2d& v0,
                              const Vec2d& v1) const;
};

} // namespace softflow
