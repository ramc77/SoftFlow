#include "polygon_obstacle.h"
#include <cmath>
#include <algorithm>
#include <limits>

namespace softflow {

PolygonObstacle::PolygonObstacle(const std::vector<Vec2d>& vertices)
    : vertices_(vertices), n_verts_(static_cast<int>(vertices.size()))
{}

bool PolygonObstacle::contains(Real x, Real y) const {
    // Ray-casting algorithm
    bool inside = false;
    for (int i = 0, j = n_verts_ - 1; i < n_verts_; j = i++) {
        Real xi = vertices_[i].x, yi = vertices_[i].y;
        Real xj = vertices_[j].x, yj = vertices_[j].y;

        if (((yi > y) != (yj > y)) &&
            (x < (xj - xi) * (y - yi) / (yj - yi) + xi)) {
            inside = !inside;
        }
    }
    return inside;
}

Vec2d PolygonObstacle::nearestPointOnEdge(const Vec2d& p, const Vec2d& v0,
                                            const Vec2d& v1) const {
    Vec2d edge = v1 - v0;
    Real len2 = edge.x * edge.x + edge.y * edge.y;
    if (len2 < 1e-30) return v0;

    Real t = ((p.x - v0.x) * edge.x + (p.y - v0.y) * edge.y) / len2;
    t = std::max(0.0, std::min(1.0, t));
    return Vec2d{v0.x + t * edge.x, v0.y + t * edge.y};
}

Vec2d PolygonObstacle::nearestPoint(Real x, Real y) const {
    Vec2d p{x, y};
    Real min_dist2 = std::numeric_limits<Real>::max();
    Vec2d closest{0, 0};

    for (int i = 0; i < n_verts_; ++i) {
        int j = (i + 1) % n_verts_;
        Vec2d np = nearestPointOnEdge(p, vertices_[i], vertices_[j]);
        Real dx = np.x - x, dy = np.y - y;
        Real dist2 = dx * dx + dy * dy;
        if (dist2 < min_dist2) {
            min_dist2 = dist2;
            closest = np;
        }
    }
    return closest;
}

Real PolygonObstacle::signedDistance(Real x, Real y) const {
    Vec2d np = nearestPoint(x, y);
    Real dist = std::sqrt((np.x - x) * (np.x - x) + (np.y - y) * (np.y - y));
    return contains(x, y) ? -dist : dist;
}

Vec2d PolygonObstacle::normalAt(Real x, Real y) const {
    Vec2d np = nearestPoint(x, y);
    Real dx = x - np.x;
    Real dy = y - np.y;
    Real len = std::sqrt(dx * dx + dy * dy);
    if (len < 1e-15) return Vec2d{0, 1};
    Vec2d n{dx / len, dy / len};
    // If point is inside polygon, normal points inward — flip it
    // so repulsion always pushes outward
    if (contains(x, y)) {
        n.x = -n.x;
        n.y = -n.y;
    }
    return n;
}

} // namespace softflow
