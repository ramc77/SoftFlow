#pragma once
#include "../core/types.h"

namespace softflow {

class LatticeField;

// Multiple Relaxation Time (MRT) collision operator for D2Q9.
// References:
//   Lallemand & Luo, Phys. Rev. E 61, 6546 (2000)
//   Li et al., Phys. Rev. E 86, 016709 (2012) — MRT pseudopotential
//
// Uses Guo forcing in moment space for correct Navier-Stokes recovery.
// Significantly more stable than BGK for multiphase and high-Re flows.
class MRTCollision {
public:
    MRTCollision(Real tau, Real se = 1.4, Real s_eps = 1.4, Real sq = 1.4);

    void collide(LatticeField& field);

    Real getTau() const { return tau_; }
    void setTau(Real tau) { tau_ = tau; s_nu_ = 1.0 / tau; }

private:
    Real tau_;
    Real s_nu_;     // viscosity relaxation rate = 1/tau  (for stress modes s7, s8)
    Real se_;       // energy mode relaxation rate (s1)
    Real s_eps_;    // energy-squared mode relaxation rate (s2)
    Real sq_;       // energy-flux mode relaxation rate (s4, s6)
};

} // namespace softflow
