#pragma once
#include "../core/types.h"
#include "../core/aligned_memory.h"
#include "lattice.h"
#include <cstring>

namespace softflow {

// LatticeField: all LBM data in aligned SoA (Structure of Arrays) layout.
//
// Distribution layout: f_[q * N + idx]  (q-outermost, spatial-innermost)
//   — This is SoA: each velocity direction is a contiguous array over all nodes.
//   — Enables vectorization: the inner loop over spatial nodes is contiguous.
//   — Streaming accesses one q-plane at a time → single contiguous memcpy-like.
//
// All arrays are 64-byte aligned for optimal cache-line and SIMD performance.
// ZERO heap allocations occur after construction.
class LatticeField {
public:
    LatticeField(int nx, int ny);

    // ── Dimensions ──────────────────────────────────────────
    int getNx() const { return nx_; }
    int getNy() const { return ny_; }
    int size()  const { return N_; }

    // ── Flat index helpers ──────────────────────────────────
    int idx(int x, int y) const { return y * nx_ + x; }

    // ── Distribution function access (SoA: q-outermost) ────
    //  Layout: f[q * N + idx(x,y)]
    Real& f(int x, int y, int q)       { return f_[q * N_ + idx(x, y)]; }
    Real  f(int x, int y, int q) const { return f_[q * N_ + idx(x, y)]; }

    Real& fTmp(int x, int y, int q)       { return f_tmp_[q * N_ + idx(x, y)]; }
    Real  fTmp(int x, int y, int q) const { return f_tmp_[q * N_ + idx(x, y)]; }

    // Raw pointer access to full f arrays
    // Layout: Q planes of N elements each. Total size = Q * N.
    Real* fData()       { return f_.data(); }
    Real* fTmpData()    { return f_tmp_.data(); }
    const Real* fData()    const { return f_.data(); }
    const Real* fTmpData() const { return f_tmp_.data(); }

    // Pointer to start of q-th distribution plane (N contiguous Reals)
    Real*       fPlane(int q)       { return f_.data() + q * N_; }
    const Real* fPlane(int q) const { return f_.data() + q * N_; }
    Real*       fTmpPlane(int q)       { return f_tmp_.data() + q * N_; }
    const Real* fTmpPlane(int q) const { return f_tmp_.data() + q * N_; }

    // ── Macroscopic fields (contiguous aligned arrays) ──────
    Real& rho(int x, int y)       { return rho_[idx(x, y)]; }
    Real  rho(int x, int y) const { return rho_[idx(x, y)]; }
    Real& ux(int x, int y)        { return ux_[idx(x, y)]; }
    Real  ux(int x, int y) const  { return ux_[idx(x, y)]; }
    Real& uy(int x, int y)        { return uy_[idx(x, y)]; }
    Real  uy(int x, int y) const  { return uy_[idx(x, y)]; }

    Real* rhoData()  { return rho_.data(); }
    Real* uxData()   { return ux_.data(); }
    Real* uyData()   { return uy_.data(); }
    const Real* rhoData() const { return rho_.data(); }
    const Real* uxData()  const { return ux_.data(); }
    const Real* uyData()  const { return uy_.data(); }

    // ── External force arrays (IBM + Shan-Chen) ────────────
    Real& Fx(int x, int y)       { return Fx_[idx(x, y)]; }
    Real  Fx(int x, int y) const { return Fx_[idx(x, y)]; }
    Real& Fy(int x, int y)       { return Fy_[idx(x, y)]; }
    Real  Fy(int x, int y) const { return Fy_[idx(x, y)]; }

    Real* FxData()  { return Fx_.data(); }
    Real* FyData()  { return Fy_.data(); }
    const Real* FxData() const { return Fx_.data(); }
    const Real* FyData() const { return Fy_.data(); }

    void clearForces();

    // ── Cell type flags ─────────────────────────────────────
    CellType& cellType(int x, int y)       { return flags_[idx(x, y)]; }
    CellType  cellType(int x, int y) const { return flags_[idx(x, y)]; }
    CellType* cellTypeData() { return flags_.data(); }
    const CellType* cellTypeData() const { return flags_.data(); }

    // Convenience aliases used by IBM, Shan-Chen, I/O, geometry
    Real getRho(int x, int y) const { return rho_[idx(x, y)]; }
    Real getUx(int x, int y) const  { return ux_[idx(x, y)]; }
    Real getUy(int x, int y) const  { return uy_[idx(x, y)]; }
    CellType getCellType(int x, int y) const { return flags_[idx(x, y)]; }
    void setCellType(int x, int y, CellType ct) { flags_[idx(x, y)] = ct; }
    void setRho(int x, int y, Real val) { rho_[idx(x, y)] = val; }
    void setUx(int x, int y, Real val)  { ux_[idx(x, y)] = val; }
    void setUy(int x, int y, Real val)  { uy_[idx(x, y)] = val; }
    void clearExternalForces() { clearForces(); }
    void addExternalForce(int x, int y, Real fx, Real fy) {
        int n = idx(x, y);
        Fx_[n] += fx;
        Fy_[n] += fy;
    }

    // Compute / set macroscopic ───────────────────────────
    void computeMacroscopic();
    void setEquilibrium(Real rho0, Real ux0, Real uy0);
    void initializeEquilibriumAt(int x, int y);
    void initializeAllEquilibrium();

    // ── Swap buffers after streaming (O(1) pointer swap) ────
    void swapBuffers();

private:
    int nx_, ny_, N_;

    // All arrays 64-byte aligned via AlignedArray
    AlignedArray<Real>     f_;       // Q * N (SoA: q-outermost)
    AlignedArray<Real>     f_tmp_;   // Q * N
    AlignedArray<Real>     rho_;     // N
    AlignedArray<Real>     ux_;      // N
    AlignedArray<Real>     uy_;      // N
    AlignedArray<Real>     Fx_;      // N
    AlignedArray<Real>     Fy_;      // N
    AlignedArray<CellType> flags_;   // N
};

} // namespace softflow
