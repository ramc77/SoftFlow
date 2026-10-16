#pragma once

#include "region.h"

#include <cstdint>
#include <string>
#include <vector>

namespace softflow::insertion {

/// 2-D bitmap mask used as an inserter region. Each pixel maps to a
/// rectangular cell in lattice coordinates; a query point is "inside"
/// the region iff its enclosing pixel's grayscale value exceeds
/// `threshold`. The intended use is irregular inlet geometries
/// (vessel cross-sections, microfluidic device masks) that are too
/// involved to express as polygons.
///
/// Mapping convention. The image's pixel `(i, j)` (0-indexed, with
/// `j = 0` at the **top** of the image as is conventional for graymaps)
/// covers the lattice rectangle
///
///   x ∈ [origin.x + i · scale,           origin.x + (i + 1) · scale]
///   y ∈ [origin.y + (height - 1 - j)·scale, origin.y + (height - j)·scale]
///
/// — i.e. the image y-axis is flipped so that "up" in the image
/// matches "up" in the lattice (positive y).
///
/// `area()` is computed once at construction by counting the pixels
/// that pass the threshold and multiplying by `scale²`.
class ImageMaskRegion final : public IRegion {
public:
    ImageMaskRegion(std::vector<std::uint8_t> pixels,
                    int                       width,
                    int                       height,
                    Vec2d                     origin,
                    Real                      scale,
                    std::uint8_t              threshold = 127);

    /// Read a PGM file from disk (P2 ASCII or P5 binary, max value ≤
    /// 255). Throws std::runtime_error on a malformed file or an
    /// unsupported variant (PBM, 16-bit, etc.). Throws
    /// std::invalid_argument for non-positive scale.
    static ImageMaskRegion fromPGM(const std::string& path,
                                   Vec2d              origin,
                                   Real               scale,
                                   std::uint8_t       threshold = 127);

    bool contains(Vec2d p) const override;
    std::pair<Vec2d, Vec2d> bbox() const override;
    Real area() const override { return area_; }

    int          width()  const { return width_; }
    int          height() const { return height_; }
    Vec2d        origin() const { return origin_; }
    Real         scale()  const { return scale_; }
    std::uint8_t threshold() const { return threshold_; }

private:
    std::vector<std::uint8_t> pixels_;     // size width_ * height_, row-major
    int                       width_, height_;
    Vec2d                     origin_;
    Real                      scale_;
    std::uint8_t              threshold_;
    Real                      area_;
};

} // namespace softflow::insertion
