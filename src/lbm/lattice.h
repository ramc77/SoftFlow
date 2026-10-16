#pragma once
#include "../core/types.h"

namespace softflow {

// D2Q9 lattice constants
namespace D2Q9 {

constexpr int Q = 9;

// Lattice weights
constexpr Real w[Q] = {
    4.0 / 9.0,                                         // 0: rest
    1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0,      // 1-4: axis-aligned
    1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0    // 5-8: diagonals
};

// Lattice velocity x-components
//  Direction layout:
//    6  2  5
//    3  0  1
//    7  4  8
constexpr int cx[Q] = { 0,  1,  0, -1,  0,  1, -1, -1,  1 };
constexpr int cy[Q] = { 0,  0,  1,  0, -1,  1,  1, -1, -1 };

// Opposite direction indices (bounce-back partner)
constexpr int opp[Q] = { 0,  3,  4,  1,  2,  7,  8,  5,  6 };

// Speed of sound squared: cs^2 = 1/3
constexpr Real cs2 = 1.0 / 3.0;
constexpr Real cs4 = cs2 * cs2;  // 1/9

// Compute equilibrium distribution for direction q
// feq_q = w_q * rho * (1 + (e_q . u)/cs2 + (e_q . u)^2/(2*cs4) - (u . u)/(2*cs2))
inline Real feq(int q, Real rho, Real ux, Real uy) {
    Real eu = static_cast<Real>(cx[q]) * ux + static_cast<Real>(cy[q]) * uy;
    Real uu = ux * ux + uy * uy;
    return w[q] * rho * (1.0 + eu / cs2 + (eu * eu) / (2.0 * cs4) - uu / (2.0 * cs2));
}

} // namespace D2Q9
} // namespace softflow
