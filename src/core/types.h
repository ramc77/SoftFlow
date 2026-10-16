#pragma once
#include <cmath>
#include <cstddef>
#include <vector>
#include <string>
#include <stdexcept>
#include <algorithm>
#include <numeric>
#include <functional>
#include <memory>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cassert>

namespace softflow {

using Real = double;
constexpr Real PI = 3.14159265358979323846;

struct Vec2d {
    Real x = 0.0, y = 0.0;

    Vec2d() = default;
    Vec2d(Real x_, Real y_) : x(x_), y(y_) {}

    Vec2d operator+(const Vec2d& o) const { return {x + o.x, y + o.y}; }
    Vec2d operator-(const Vec2d& o) const { return {x - o.x, y - o.y}; }
    Vec2d operator*(Real s) const { return {x * s, y * s}; }
    Vec2d operator/(Real s) const { return {x / s, y / s}; }
    Vec2d& operator+=(const Vec2d& o) { x += o.x; y += o.y; return *this; }
    Vec2d& operator-=(const Vec2d& o) { x -= o.x; y -= o.y; return *this; }
    Vec2d& operator*=(Real s) { x *= s; y *= s; return *this; }

    Real dot(const Vec2d& o) const { return x * o.x + y * o.y; }
    Real cross(const Vec2d& o) const { return x * o.y - y * o.x; }
    Real norm() const { return std::sqrt(x * x + y * y); }
    Real norm2() const { return x * x + y * y; }
    Vec2d normalized() const {
        Real n = norm();
        return n > 1e-15 ? Vec2d{x / n, y / n} : Vec2d{0, 0};
    }
    Vec2d perp() const { return {-y, x}; }
};

inline Vec2d operator*(Real s, const Vec2d& v) { return {s * v.x, s * v.y}; }

enum class CellType : int {
    FLUID  = 0,
    SOLID  = 1,
    INLET  = 2,
    OUTLET = 3,
    EMPTY  = 4   // Free-surface: empty/gas cell. Treated as solid (bounce-back)
                 // until adjacent fluid pressure exceeds atmospheric threshold,
                 // then converted to FLUID. Used for dam-break / gravity-driven flow.
};

enum class BoundaryType : int {
    INLET_OUTLET = 0,   // Zou-He inlet (left) + outlet (right)
    PERIODIC = 1,       // Periodic in x-direction, flow driven by body force
    CLOSED = 2          // Solid walls on all four sides (closed box)
};

} // namespace softflow
