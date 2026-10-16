#pragma once
#include <cmath>
#include "../core/types.h"

namespace softflow {

// Peskin's 4-point regularized delta function for Immersed Boundary Method
// Reference: Peskin, Acta Numerica (2002)
namespace PeskinDelta {

    // Support radius: delta is nonzero for |r| < 2
    constexpr int SUPPORT = 2;

    // ---------- Lookup table for phi (avoids sqrt per call) ----------
    constexpr int TABLE_SIZE = 1024;

    // Exact formula (used to build the table and as reference)
    inline Real phi_exact(Real r) {
        Real ar = std::abs(r);
        if (ar <= 1.0) {
            return 0.125 * (3.0 - 2.0 * ar + std::sqrt(1.0 + 4.0 * ar - 4.0 * ar * ar));
        } else if (ar <= 2.0) {
            return 0.125 * (5.0 - 2.0 * ar - std::sqrt(-7.0 + 12.0 * ar - 4.0 * ar * ar));
        }
        return 0.0;
    }

    // Static table initialized on first use (thread-safe in C++11+)
    struct PhiTable {
        Real data[TABLE_SIZE + 1];
        PhiTable() {
            for (int i = 0; i <= TABLE_SIZE; ++i) {
                Real r = 2.0 * static_cast<Real>(i) / TABLE_SIZE; // [0, 2]
                data[i] = phi_exact(r);
            }
        }
    };

    inline const PhiTable& getTable() {
        static const PhiTable table;
        return table;
    }

    // Fast 1D delta via linear interpolation of lookup table
    inline Real phi(Real r) {
        Real ar = std::abs(r);
        if (ar >= 2.0) return 0.0;
        const auto& tbl = getTable().data;
        Real idx = ar * (TABLE_SIZE * 0.5); // maps [0,2] → [0, TABLE_SIZE]
        int i = static_cast<int>(idx);
        if (i >= TABLE_SIZE) return 0.0;
        Real frac = idx - static_cast<Real>(i);
        return tbl[i] * (1.0 - frac) + tbl[i + 1] * frac;
    }

    // 2D delta function: product of 1D components
    inline Real delta2d(Real rx, Real ry) {
        return phi(rx) * phi(ry);
    }

} // namespace PeskinDelta

} // namespace softflow
