#include "bgk_collision.h"
#include "lattice.h"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace softflow {

BGKCollision::BGKCollision(Real tau)
    : tau_(tau), omega_(1.0 / tau)
{
}

void BGKCollision::collide(LatticeField& field) {
    const int N  = field.size();

    // SoA: gather pointers to each q-plane
    Real* fp[D2Q9::Q];
    for (int q = 0; q < D2Q9::Q; ++q) {
        fp[q] = field.fData() + q * N;
    }

    const Real* __restrict__ rhoPtr = field.rhoData();
    const Real* __restrict__ uxPtr  = field.uxData();
    const Real* __restrict__ uyPtr  = field.uyData();
    const Real* __restrict__ FxPtr  = field.FxData();
    const Real* __restrict__ FyPtr  = field.FyData();
    const CellType* __restrict__ flags = field.cellTypeData();

    const Real om  = omega_;
    const Real om1 = 1.0 - om;          // 1 - 1/tau
    const Real guo_prefactor = 1.0 - 0.5 * om;  // (1 - 1/(2*tau))

    // Precompute inverse cs2/cs4 to avoid divisions in inner loop
    constexpr Real inv_cs2 = 3.0;       // 1/cs2 = 3
    constexpr Real inv_cs4 = 9.0;       // 1/cs4 = 9

#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int n = 0; n < N; ++n) {
        if (flags[n] != CellType::FLUID) continue;

        const Real r  = rhoPtr[n];
        const Real vx = uxPtr[n];
        const Real vy = uyPtr[n];
        const Real fx = FxPtr[n];
        const Real fy = FyPtr[n];

        // Precompute velocity-squared term (used by all 9 directions)
        const Real uu = vx * vx + vy * vy;
        const Real uu_term = -0.5 * inv_cs2 * uu;  // -u·u / (2*cs^2)

        // ── Local register array: gather from SoA planes ────
        Real f_local[D2Q9::Q];
        for (int q = 0; q < D2Q9::Q; ++q) {
            f_local[q] = fp[q][n];
        }

        // ── Collision + Guo forcing for all 9 directions ────
        for (int q = 0; q < D2Q9::Q; ++q) {
            const Real eqx = static_cast<Real>(D2Q9::cx[q]);
            const Real eqy = static_cast<Real>(D2Q9::cy[q]);
            const Real eu = eqx * vx + eqy * vy;

            // Equilibrium: f_eq = w_q * rho * (1 + eu/cs2 + eu^2/(2*cs4) - uu/(2*cs2))
            Real f_eq = D2Q9::w[q] * r * (1.0 + eu * inv_cs2
                        + 0.5 * eu * eu * inv_cs4 + uu_term);

            // Guo forcing term:
            // F_i = (1 - 1/(2*tau)) * w_q * [(e-u)/cs2 + (e·u)/cs4 * e] · F
            Real term_x = (eqx - vx) * inv_cs2 + (eu * inv_cs4) * eqx;
            Real term_y = (eqy - vy) * inv_cs2 + (eu * inv_cs4) * eqy;
            Real F_i = guo_prefactor * D2Q9::w[q] * (term_x * fx + term_y * fy);

            // BGK: f = (1-omega)*f + omega*f_eq + F_i
            f_local[q] = om1 * f_local[q] + om * f_eq + F_i;
        }

        // ── Scatter back to SoA planes ──────────────────────
        for (int q = 0; q < D2Q9::Q; ++q) {
            fp[q][n] = f_local[q];
        }
    }
}

} // namespace softflow
