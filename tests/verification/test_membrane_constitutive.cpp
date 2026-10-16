// V&V: membrane tension laws, measured from the assembled nodal forces.
//
// A circular ring of N nodes is stretched uniformly to lambda * R.  Each
// edge then carries the same tension T, and every node feels a net inward
// force of magnitude 2 T sin(pi/N), so the tension the code applies is
//
//     T_meas = |F_node| / (2 sin(pi/N)).
//
// It is compared with the plane-strain (lambda_2 = 1) principal tensions
// of Barthes-Biesel, Diaz & Dhenin, J. Fluid Mech. 460 (2002) 211:
//
//   Hookean      T = k_s (lambda - 1)
//   Neo-Hookean  T = G_s (lambda - lambda^-3)
//   Skalak       T = G_s lambda (lambda^2 - 1) (1 + C)
//
// over compression and extension, 0.8 <= lambda <= 1.6.  Small-strain
// moduli: 4 G_s (Neo-Hookean) and 2 G_s (1 + C) (Skalak), equal for C = 1.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "membrane/capsule.h"

#include <cmath>
#include <functional>
#include <string>
#include "vv_dump.h"

using namespace softflow;

namespace {

Real measuredTension(MembraneParams mp, Real lambda, int N = 128, Real R = 10.0) {
    mp.k_bend = 0.0;
    mp.use_helfrich_bending = false;
    mp.k_area = 0.0;
    mp.k_perimeter = 0.0;
    mp.gamma_visc = 0.0;
    mp.eta_membrane = 0.0;
    const Vec2d c{50.0, 50.0};
    Capsule cap(0, c, R, N, mp);
    for (auto& p : cap.positions()) p = c + (p - c) * lambda;
    cap.clearForces();
    cap.computeMembraneForces();
    Real fsum = 0.0;
    for (const auto& f : cap.forces()) fsum += f.norm();
    return (fsum / N) / (2.0 * std::sin(PI / N));
}

} // namespace

TEST_CASE("membrane tension laws match the plane-strain references") {
    struct Law {
        std::string name;
        MembraneParams mp;
        std::function<Real(Real)> T;
    };
    const Real Gs = 0.01, ks = 0.04;
    MembraneParams hk; hk.model = MembraneModel::HOOKEAN;     hk.k_stretch = ks;
    MembraneParams nh; nh.model = MembraneModel::NEO_HOOKEAN; nh.G_s = Gs;
    MembraneParams s1; s1.model = MembraneModel::SKALAK; s1.G_s = Gs; s1.C_skalak = 1.0;
    MembraneParams s10 = s1; s10.C_skalak = 10.0;
    const Law laws[] = {
        {"hookean",     hk,  [&](Real l) { return ks * (l - 1.0); }},
        {"neo_hookean", nh,  [&](Real l) { return Gs * (l - 1.0 / (l * l * l)); }},
        {"skalak_C1",   s1,  [&](Real l) { return Gs * l * (l * l - 1.0) * 2.0; }},
        {"skalak_C10",  s10, [&](Real l) { return Gs * l * (l * l - 1.0) * 11.0; }},
    };

    FILE* out = vv::open("membrane_tension", "law,lambda,T_measured,T_reference");
    for (const auto& law : laws) {
        Real worst = 0.0;
        for (int i = 0; i <= 16; ++i) {
            const Real lam = 0.8 + 0.05 * i;
            if (std::abs(lam - 1.0) < 1e-9) continue;
            const Real Tm = std::abs(measuredTension(law.mp, lam));
            const Real Tr = std::abs(law.T(lam));
            worst = std::max(worst, std::abs(Tm - Tr) / Tr);
            if (out) std::fprintf(out, "%s,%.3f,%.9e,%.9e\n", law.name.c_str(),
                                  lam, Tm, Tr);
        }
        INFO(law.name << ": worst relative error " << worst);
        CHECK(worst < 1e-10);
    }
    vv::close(out);
}
