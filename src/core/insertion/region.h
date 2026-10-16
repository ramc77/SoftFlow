#pragma once

#include "../types.h"

#include <utility>
#include <vector>

namespace softflow::insertion {

/// Abstract 2-D region used by inserters to bound where placements may
/// land. Concrete regions encapsulate point-membership, an axis-aligned
/// bounding box for sampling, and the geometric area (used by density
/// estimators in dynamic inserters).
///
/// Subclasses are declared `final` to make the class hierarchy closed
/// from the C++ side; the Python side reaches the same set of regions
/// via factory functions in pysoftflow.insertion. Custom regions in
/// user code subclass IRegion directly.
class IRegion {
public:
    virtual ~IRegion() = default;

    /// True if the point lies inside the region. The boundary is
    /// considered part of the region.
    virtual bool contains(Vec2d p) const = 0;

    /// Axis-aligned bounding box. Returned as `(min_corner, max_corner)`.
    /// Inserters use this to draw uniform candidate points.
    virtual std::pair<Vec2d, Vec2d> bbox() const = 0;

    /// Geometric area of the region in lattice units squared.
    /// Used by dynamic inserters to relate count to volume fraction
    /// (or, in 2D, area fraction).
    virtual Real area() const = 0;
};

/// Axis-aligned rectangular region. The convention matches the project specification
/// §7.1: `RectRegion(x=(x0, x1), y=(y0, y1))`. Both ends are inclusive
/// — a point on the boundary is in the region.
class RectRegion final : public IRegion {
public:
    RectRegion(Real x0, Real x1, Real y0, Real y1);

    bool contains(Vec2d p) const override;
    std::pair<Vec2d, Vec2d> bbox() const override;
    Real area() const override;

    Real x0() const { return x0_; }
    Real x1() const { return x1_; }
    Real y0() const { return y0_; }
    Real y1() const { return y1_; }

private:
    Real x0_, x1_, y0_, y1_;
};

/// Disk: every point within `radius` of `center`. Boundary inclusive.
/// `area = π r²`. The bbox is the inscribed square `[cx ± r, cy ± r]`,
/// which inserters use for uniform candidate sampling — points
/// outside the disk but inside the bbox are rejected by `contains()`,
/// so density per attempt is reduced by π/4 vs a square region.
class CircleRegion final : public IRegion {
public:
    CircleRegion(Vec2d center, Real radius);

    bool contains(Vec2d p) const override;
    std::pair<Vec2d, Vec2d> bbox() const override;
    Real area() const override;

    Vec2d center() const { return center_; }
    Real  radius() const { return r_; }

private:
    Vec2d center_;
    Real  r_;
};

/// Simple (non-self-intersecting) polygon. Membership uses the
/// crossing-number test (Sutherland 1974); area uses the Shoelace
/// formula and is reported as the absolute value (orientation-
/// independent). The polygon is automatically closed: the final
/// vertex implicitly connects back to the first.
///
/// SoftFlow already has a `geometry::PolygonObstacle` for the LBM
/// side. We deliberately do not reuse it here because (a) the
/// inserter side never asks for `signedDistance` or `nearestPoint`,
/// only `contains` / `bbox` / `area`, and (b) keeping insertion's
/// region hierarchy independent lets users write Python regions
/// without depending on the geometry hierarchy.
class PolygonRegion final : public IRegion {
public:
    explicit PolygonRegion(std::vector<Vec2d> vertices);

    bool contains(Vec2d p) const override;
    std::pair<Vec2d, Vec2d> bbox() const override;
    Real area() const override;

    const std::vector<Vec2d>& vertices() const { return verts_; }

private:
    std::vector<Vec2d> verts_;
    Vec2d              bbox_lo_, bbox_hi_;
    Real               area_;
};

} // namespace softflow::insertion
