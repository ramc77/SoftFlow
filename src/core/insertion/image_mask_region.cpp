#include "image_mask_region.h"

#include <cctype>
#include <fstream>
#include <stdexcept>
#include <string>

namespace softflow::insertion {

namespace {

// Read the next whitespace-delimited token from a stream, treating
// PGM `#` comments as line comments. Throws on EOF.
std::string nextPgmToken(std::istream& in, const std::string& path) {
    std::string tok;
    while (in.good()) {
        int c = in.peek();
        if (c == EOF) break;
        if (std::isspace(static_cast<unsigned char>(c))) {
            in.get();
            continue;
        }
        if (c == '#') {
            // skip comment to end of line
            std::string line;
            std::getline(in, line);
            continue;
        }
        // accumulate the token until next whitespace
        while (in.good()) {
            c = in.peek();
            if (c == EOF || std::isspace(static_cast<unsigned char>(c))) break;
            tok.push_back(static_cast<char>(in.get()));
        }
        return tok;
    }
    throw std::runtime_error("ImageMaskRegion::fromPGM: unexpected EOF in '"
                             + path + "'");
}

} // namespace

ImageMaskRegion::ImageMaskRegion(std::vector<std::uint8_t> pixels,
                                 int                       width,
                                 int                       height,
                                 Vec2d                     origin,
                                 Real                      scale,
                                 std::uint8_t              threshold)
    : pixels_(std::move(pixels)),
      width_(width),
      height_(height),
      origin_(origin),
      scale_(scale),
      threshold_(threshold)
{
    if (width <= 0 || height <= 0) {
        throw std::invalid_argument("ImageMaskRegion: width/height must be > 0");
    }
    if (!(scale > 0.0)) {
        throw std::invalid_argument("ImageMaskRegion: scale must be > 0");
    }
    if (pixels_.size() != static_cast<std::size_t>(width) * height) {
        throw std::invalid_argument(
            "ImageMaskRegion: pixel buffer size != width * height");
    }

    // Precompute the area: number of "inside" pixels times pixel area.
    std::size_t inside = 0;
    for (auto px : pixels_) if (px > threshold_) ++inside;
    if (inside == 0) {
        throw std::invalid_argument(
            "ImageMaskRegion: every pixel is below threshold "
            "(empty region — no placement is possible)");
    }
    area_ = static_cast<Real>(inside) * scale_ * scale_;
}

bool ImageMaskRegion::contains(Vec2d p) const {
    // Map lattice → image coords with the y-flip described in the
    // header so that "up in the image" matches "up in lattice y".
    const int i = static_cast<int>((p.x - origin_.x) / scale_);
    const int j_lattice = static_cast<int>((p.y - origin_.y) / scale_);
    if (i < 0 || i >= width_) return false;
    if (j_lattice < 0 || j_lattice >= height_) return false;
    const int j = (height_ - 1) - j_lattice;     // flip
    const std::uint8_t px = pixels_[static_cast<std::size_t>(j) * width_ + i];
    return px > threshold_;
}

std::pair<Vec2d, Vec2d> ImageMaskRegion::bbox() const {
    return {
        Vec2d{ origin_.x,                              origin_.y                              },
        Vec2d{ origin_.x + width_  * scale_,           origin_.y + height_ * scale_           }
    };
}

ImageMaskRegion ImageMaskRegion::fromPGM(const std::string& path,
                                         Vec2d              origin,
                                         Real               scale,
                                         std::uint8_t       threshold)
{
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("ImageMaskRegion::fromPGM: cannot open '"
                                 + path + "'");
    }

    const std::string magic = nextPgmToken(in, path);
    const bool is_p2 = (magic == "P2");
    const bool is_p5 = (magic == "P5");
    if (!is_p2 && !is_p5) {
        throw std::runtime_error(
            "ImageMaskRegion::fromPGM: '" + path
            + "' is not a P2/P5 PGM (got '" + magic + "')");
    }

    const int width  = std::stoi(nextPgmToken(in, path));
    const int height = std::stoi(nextPgmToken(in, path));
    const int maxval = std::stoi(nextPgmToken(in, path));
    if (width <= 0 || height <= 0) {
        throw std::runtime_error("ImageMaskRegion::fromPGM: invalid dimensions");
    }
    if (maxval <= 0 || maxval > 255) {
        throw std::runtime_error(
            "ImageMaskRegion::fromPGM: only maxval ≤ 255 is supported "
            "(got " + std::to_string(maxval) + ")");
    }

    std::vector<std::uint8_t> pixels(static_cast<std::size_t>(width) * height);

    if (is_p5) {
        // P5: a single whitespace separates header from raw bytes.
        // Standard says it's exactly one whitespace; we have just
        // consumed `maxval`'s trailing whitespace inside nextPgmToken
        // (it stops at the first whitespace char without consuming
        // it), so we must consume one more byte before raw data.
        // Actually nextPgmToken consumes the token chars but leaves
        // the trailing whitespace; we need to skip exactly one
        // whitespace byte to reach the raw payload.
        in.get();    // skip the single whitespace byte after maxval
        in.read(reinterpret_cast<char*>(pixels.data()),
                static_cast<std::streamsize>(pixels.size()));
        if (!in) {
            throw std::runtime_error(
                "ImageMaskRegion::fromPGM: short read of P5 payload in '"
                + path + "'");
        }
    } else {
        // P2: ASCII pixel values, separated by whitespace.
        for (std::size_t k = 0; k < pixels.size(); ++k) {
            int v;
            if (!(in >> v)) {
                throw std::runtime_error(
                    "ImageMaskRegion::fromPGM: short read of P2 payload in '"
                    + path + "'");
            }
            if (v < 0 || v > maxval) {
                throw std::runtime_error(
                    "ImageMaskRegion::fromPGM: pixel out of range in '"
                    + path + "'");
            }
            pixels[k] = static_cast<std::uint8_t>(v);
        }
    }

    return ImageMaskRegion(std::move(pixels), width, height,
                           origin, scale, threshold);
}

} // namespace softflow::insertion
