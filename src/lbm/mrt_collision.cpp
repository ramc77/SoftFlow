#include "mrt_collision.h"
#include "lattice.h"
#include "lattice_field.h"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace softflow {

MRTCollision::MRTCollision(Real tau, Real se, Real s_eps, Real sq)
    : tau_(tau), s_nu_(1.0 / tau), se_(se), s_eps_(s_eps), sq_(sq)
{
}

// D2Q9 MRT collision with Guo forcing in moment space.
//
// Transformation matrix M (Lallemand & Luo 2000):
//   Row 0 (rho):    1  1  1  1  1  1  1  1  1
//   Row 1 (e):     -4 -1 -1 -1 -1  2  2  2  2
//   Row 2 (eps):    4 -2 -2 -2 -2  1  1  1  1
//   Row 3 (jx):     0  1  0 -1  0  1 -1 -1  1
//   Row 4 (qx):     0 -2  0  2  0  1 -1 -1  1
//   Row 5 (jy):     0  0  1  0 -1  1  1 -1 -1
//   Row 6 (qy):     0  0 -2  0  2  1  1 -1 -1
//   Row 7 (pxx):    0  1 -1  1 -1  0  0  0  0
//   Row 8 (pxy):    0  0  0  0  0  1 -1  1 -1
//
// Relaxation rates: S = diag(0, se, s_eps, 0, sq, 0, sq, s_nu, s_nu)
//
void MRTCollision::collide(LatticeField& field) {
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

    const Real sn = s_nu_;   // stress relaxation = 1/tau
    const Real s1 = se_;
    const Real s2 = s_eps_;
    const Real s4 = sq_;
    const Real s6 = sq_;

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

        // ── Gather from SoA planes into local registers ─────
        Real f0 = fp[0][n];
        Real f1 = fp[1][n];
        Real f2 = fp[2][n];
        Real f3 = fp[3][n];
        Real f4 = fp[4][n];
        Real f5 = fp[5][n];
        Real f6 = fp[6][n];
        Real f7 = fp[7][n];
        Real f8 = fp[8][n];

        // ── Forward transform: m = M * f ────────────────────
        Real m0 = f0 + f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8;
        Real m1 = -4*f0 - f1 - f2 - f3 - f4 + 2*f5 + 2*f6 + 2*f7 + 2*f8;
        Real m2 = 4*f0 - 2*f1 - 2*f2 - 2*f3 - 2*f4 + f5 + f6 + f7 + f8;
        Real m3 = f1 - f3 + f5 - f6 - f7 + f8;
        Real m4 = -2*f1 + 2*f3 + f5 - f6 - f7 + f8;
        Real m5 = f2 - f4 + f5 + f6 - f7 - f8;
        Real m6 = -2*f2 + 2*f4 + f5 + f6 - f7 - f8;
        Real m7 = f1 - f2 + f3 - f4;
        Real m8 = f5 - f6 + f7 - f8;

        // ── Equilibrium moments ─────────────────────────────
        Real uu = vx * vx + vy * vy;
        Real meq1 = -2.0 * r + 3.0 * r * uu;
        Real meq2 = r - 3.0 * r * uu;
        Real meq4 = -r * vx;
        Real meq6 = -r * vy;
        Real meq7 = r * (vx * vx - vy * vy);
        Real meq8 = r * vx * vy;

        // ── Guo forcing source in moment space ──────────────
        Real uF = vx * fx + vy * fy;
        Real Sb1 = 6.0 * uF;
        Real Sb2 = -6.0 * uF;
        Real Sb3 = fx;
        Real Sb4 = -fx;
        Real Sb5 = fy;
        Real Sb6 = -fy;
        Real Sb7 = 2.0 * (vx * fx - vy * fy);
        Real Sb8 = vx * fy + vy * fx;

        // ── Relax + forcing: m* = m - s*(m - meq) + (1 - s/2)*Sb ──
        // m0 conserved (s=0), no forcing on density — left unchanged
        m1 = m1 - s1 * (m1 - meq1) + (1.0 - s1 * 0.5) * Sb1;
        m2 = m2 - s2 * (m2 - meq2) + (1.0 - s2 * 0.5) * Sb2;
        m3 = m3 + Sb3;  // conserved (s=0)
        m4 = m4 - s4 * (m4 - meq4) + (1.0 - s4 * 0.5) * Sb4;
        m5 = m5 + Sb5;  // conserved (s=0)
        m6 = m6 - s6 * (m6 - meq6) + (1.0 - s6 * 0.5) * Sb6;
        m7 = m7 - sn * (m7 - meq7) + (1.0 - sn * 0.5) * Sb7;
        m8 = m8 - sn * (m8 - meq8) + (1.0 - sn * 0.5) * Sb8;

        // ── Inverse transform: f = M^{-1} * m ──────────────
        f0 = m0 / 9.0 - m1 / 9.0 + m2 / 9.0;
        f1 = m0 / 9.0 - m1 / 36.0 - m2 / 18.0 + m3 / 6.0 - m4 / 6.0 + m7 / 4.0;
        f2 = m0 / 9.0 - m1 / 36.0 - m2 / 18.0 + m5 / 6.0 - m6 / 6.0 - m7 / 4.0;
        f3 = m0 / 9.0 - m1 / 36.0 - m2 / 18.0 - m3 / 6.0 + m4 / 6.0 + m7 / 4.0;
        f4 = m0 / 9.0 - m1 / 36.0 - m2 / 18.0 - m5 / 6.0 + m6 / 6.0 - m7 / 4.0;
        f5 = m0 / 9.0 + m1 / 18.0 + m2 / 36.0 + m3 / 6.0 + m4 / 12.0 + m5 / 6.0 + m6 / 12.0 + m8 / 4.0;
        f6 = m0 / 9.0 + m1 / 18.0 + m2 / 36.0 - m3 / 6.0 - m4 / 12.0 + m5 / 6.0 + m6 / 12.0 - m8 / 4.0;
        f7 = m0 / 9.0 + m1 / 18.0 + m2 / 36.0 - m3 / 6.0 - m4 / 12.0 - m5 / 6.0 - m6 / 12.0 + m8 / 4.0;
        f8 = m0 / 9.0 + m1 / 18.0 + m2 / 36.0 + m3 / 6.0 + m4 / 12.0 - m5 / 6.0 - m6 / 12.0 - m8 / 4.0;

        // ── Scatter back to SoA planes ──────────────────────
        fp[0][n] = f0;
        fp[1][n] = f1;
        fp[2][n] = f2;
        fp[3][n] = f3;
        fp[4][n] = f4;
        fp[5][n] = f5;
        fp[6][n] = f6;
        fp[7][n] = f7;
        fp[8][n] = f8;
    }
}

} // namespace softflow
