#pragma once
#include "../core/types.h"
#include "../core/parameters.h"

namespace softflow {

class LatticeField;

// Shan-Chen pseudopotential model for multiphase LBM.
//
// Supports two equations of state:
//   EOS 0 (original): psi(rho) = rho0 * (1 - exp(-rho/rho0))
//       Limited density ratio (~10:1). Good for surface tension.
//   EOS 1 (Carnahan-Starling): psi = sqrt(2*(p_CS - rho*cs2)/(G*cs2))
//       Yuan & Schaefer, Phys. Fluids 18, 042101 (2006)
//       High density ratio (up to 100:1+). Realistic multiphase.
//
// References:
//   Shan & Chen, PRE 47 (1993) — original model
//   Yuan & Schaefer, Phys. Fluids 18 (2006) — realistic EOS
//   Li et al., Phys. Rev. E 86 (2012) — MRT pseudopotential
class ShanChen {
public:
    explicit ShanChen(const ShanChenParams& params,
                      BoundaryType bc = BoundaryType::PERIODIC);

    // Compute Shan-Chen interaction force and add to external force arrays
    void computeForce(LatticeField& field);

    Real getG() const { return G_; }

private:
    Real G_;
    Real rho0_;
    BoundaryType bc_;

    // EOS parameters
    int eos_type_;
    Real cs_a_, cs_b_, cs_T_, cs_R_;

    // Pseudopotential function
    Real psi(Real rho) const;

    // Carnahan-Starling equation of state pressure
    Real p_carnahan_starling(Real rho) const;
};

} // namespace softflow
