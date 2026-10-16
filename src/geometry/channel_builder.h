#pragma once
#include "../core/types.h"
#include "obstacle.h"
#include "circle_obstacle.h"
#include "rect_obstacle.h"
#include <vector>
#include <memory>

namespace softflow {

class LatticeField;

// Fluent API for building channel geometry
class ChannelBuilder {
public:
    ChannelBuilder(int nx, int ny) : nx_(nx), ny_(ny) {}

    // Add top and bottom walls (always present in a channel)
    ChannelBuilder& addWalls() {
        has_top_wall_ = true;
        has_bottom_wall_ = true;
        return *this;
    }

    ChannelBuilder& addCirclePillar(Real cx, Real cy, Real radius) {
        obstacles_.push_back(std::make_shared<CircleObstacle>(cx, cy, radius));
        return *this;
    }

    ChannelBuilder& addRectPillar(Real x0, Real y0, Real x1, Real y1) {
        obstacles_.push_back(std::make_shared<RectObstacle>(x0, y0, x1, y1));
        return *this;
    }

    ChannelBuilder& setBoundaryType(BoundaryType bt) {
        boundary_type_ = bt;
        return *this;
    }
    BoundaryType getBoundaryType() const { return boundary_type_; }

    // Apply the geometry to a LatticeField (set cell flags)
    void applyToField(LatticeField& field) const;

    bool hasTopWall() const { return has_top_wall_; }
    bool hasBottomWall() const { return has_bottom_wall_; }
    int getNx() const { return nx_; }
    int getNy() const { return ny_; }

    const std::vector<std::shared_ptr<Obstacle>>& getObstacles() const {
        return obstacles_;
    }

    Real wallBottom() const { return 0.0; }
    Real wallTop() const { return static_cast<Real>(ny_ - 1); }

private:
    int nx_, ny_;
    bool has_top_wall_ = false;
    bool has_bottom_wall_ = false;
    BoundaryType boundary_type_ = BoundaryType::INLET_OUTLET;
    std::vector<std::shared_ptr<Obstacle>> obstacles_;
};

} // namespace softflow
