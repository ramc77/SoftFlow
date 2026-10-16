#include "advection_diffusion.h"
#include "lattice.h"
#include "../membrane/capsule_system.h"
#include "../coupling/delta_function.h"
#include <cmath>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace softflow {

AdvectionDiffusion::AdvectionDiffusion(int nx, int ny,
                                       const ScalarParams& params)
    : nx_(nx), ny_(ny),
      n_species_(params.n_species),
      periodic_y_(params.periodic_y)
{
    diffusivity_.resize(n_species_);
    tau_s_.resize(n_species_);
    inlet_concentration_.resize(n_species_, 0.0);

    for (int s = 0; s < n_species_; ++s) {
        diffusivity_[s] = (s < static_cast<int>(params.diffusivity.size()))
                          ? params.diffusivity[s] : 0.01;
        tau_s_[s] = 3.0 * diffusivity_[s] + 0.5;

        if (s < static_cast<int>(params.inlet_concentration.size())) {
            inlet_concentration_[s] = params.inlet_concentration[s];
        }
    }

    int N = nx * ny;
    g_.resize(n_species_);
    g_tmp_.resize(n_species_);
    C_.resize(n_species_);

    for (int s = 0; s < n_species_; ++s) {
        g_[s].resize(N * D2Q9::Q, 0.0);
        g_tmp_[s].resize(N * D2Q9::Q, 0.0);
        C_[s].resize(N, 0.0);
    }
}

void AdvectionDiffusion::initialize(Real initial_concentration) {
    int N = nx_ * ny_;
    for (int s = 0; s < n_species_; ++s) {
        for (int n = 0; n < N; ++n) {
            C_[s][n] = initial_concentration;
            // Set to equilibrium
            for (int q = 0; q < D2Q9::Q; ++q) {
                g_[s][n * D2Q9::Q + q] = D2Q9::w[q] * initial_concentration;
            }
        }
    }
}

void AdvectionDiffusion::setRegion(int x0, int y0, int x1, int y1,
                                    Real concentration, int species) {
    if (species < 0 || species >= n_species_) return;
    for (int y = std::max(0, y0); y < std::min(ny_, y1); ++y) {
        for (int x = std::max(0, x0); x < std::min(nx_, x1); ++x) {
            int n = idx(x, y);
            C_[species][n] = concentration;
            for (int q = 0; q < D2Q9::Q; ++q) {
                g_[species][n * D2Q9::Q + q] = D2Q9::w[q] * concentration;
            }
        }
    }
}

void AdvectionDiffusion::setPoint(int x, int y, Real concentration,
                                   int species) {
    if (x < 0 || x >= nx_ || y < 0 || y >= ny_) return;
    if (species < 0 || species >= n_species_) return;
    int n = idx(x, y);
    C_[species][n] = concentration;
    for (int q = 0; q < D2Q9::Q; ++q) {
        g_[species][n * D2Q9::Q + q] = D2Q9::w[q] * concentration;
    }
}

void AdvectionDiffusion::step(const LatticeField& fluid_field) {
    for (int s = 0; s < n_species_; ++s) {
        collide(s, fluid_field);
        streamWithBC(s, fluid_field);
        computeConcentration(s);
    }
}

// Pass 1 — collision into g_tmp_.
//
// Reads g_[species], writes g_tmp_[species]. No cross-cell writes, so
// the OpenMP loop is race-free (this is the principal correctness fix
// over the pre-Phase-1 implementation, which wrote into g_[species]
// from inside the parallel region).
//
// Equilibrium (first-order in u, Krüger 2017 Eq. 8.32):
//
//   g_eq_q(C, u) = w_q C ( 1 + (e_q · u) / cs² )
//
// First-order is the textbook-correct form for D2Q9 BGK ADR: the
// recovered macroscopic equation is ∂_t C + ∇·(uC) = D ∇²C with
// D = (τ_s − ½) cs². Higher-order velocity terms in g_eq do *not*
// extend the stable Péclet range — they introduce u² corrections to
// the recovered diffusivity that break the Aris–Taylor result. For
// genuinely high-Pe extension, the right tools are MRT-ADR
// (Yoshida & Nagaoka 2010) or grid refinement, which we leave to a
// later phase.
void AdvectionDiffusion::collide(int species, const LatticeField& fluid_field) {
    const Real omega_s = 1.0 / tau_s_[species];
    const Real C_inlet = inlet_concentration_[species];
    const Real* src   = g_[species].data();
    Real*       dst   = g_tmp_[species].data();
    constexpr Real inv_cs2 = 1.0 / D2Q9::cs2;       // 3

#ifdef _OPENMP
    #pragma omp parallel for collapse(2) schedule(static)
#endif
    for (int y = 0; y < ny_; ++y) {
        for (int x = 0; x < nx_; ++x) {
            const int n = idx(x, y);
            const int qbase = n * D2Q9::Q;

            const CellType ct = fluid_field.cellType(x, y);
            if (ct == CellType::SOLID) {
                // SOLID cells hold no scalar; copy g unchanged so the
                // streaming pass can read predictable values from them
                // (streamWithBC() never pulls from SOLID — it bounces
                // back instead — but we keep the buffer in a consistent
                // state for output writers and checkpoints).
                for (int q = 0; q < D2Q9::Q; ++q) dst[qbase + q] = src[qbase + q];
                continue;
            }

            // Inlet Dirichlet: re-impose C_inlet before computing g_eq.
            // (A dedicated anti-bounce-back inlet would be more accurate
            // but we keep the existing semantics for backward compat;
            // see TODO in docs/theory/adr.md.)
            const Real C  = (ct == CellType::INLET) ? C_inlet : C_[species][n];
            const Real ux = fluid_field.ux(x, y);
            const Real uy = fluid_field.uy(x, y);

            for (int q = 0; q < D2Q9::Q; ++q) {
                const Real eu = static_cast<Real>(D2Q9::cx[q]) * ux
                              + static_cast<Real>(D2Q9::cy[q]) * uy;
                const Real geq = D2Q9::w[q] * C * (1.0 + eu * inv_cs2);
                dst[qbase + q] = src[qbase + q] - omega_s * (src[qbase + q] - geq);
            }
        }
    }
}

// Pass 2 — pull-streaming with periodic wrap and halfway bounce-back.
//
// For each fluid (or inlet/outlet) cell (x,y) and each direction q:
//   upstream = (x − cx[q], y − cy[q]) with periodic wrap as configured.
//   if upstream is in-domain and not SOLID:
//       g_[n*Q + q] = g_tmp_[upstream*Q + q]                 (regular pull)
//   else (out-of-domain wall, or upstream SOLID):
//       g_[n*Q + q] = g_tmp_[n*Q + opp[q]]                   (halfway BB)
//
// The bounce-back term uses g_tmp_ (post-collision distribution at the
// fluid cell itself), so the boundary condition is zero-flux Neumann
// regardless of what the SOLID cell's distribution is. This is the
// standard ADR wall treatment (Krüger 2017 §8.5.4).
void AdvectionDiffusion::streamWithBC(int species, const LatticeField& fluid_field) {
    const Real* src = g_tmp_[species].data();
    Real*       dst = g_[species].data();

#ifdef _OPENMP
    #pragma omp parallel for collapse(2) schedule(static)
#endif
    for (int y = 0; y < ny_; ++y) {
        for (int x = 0; x < nx_; ++x) {
            const int n     = idx(x, y);
            const int qbase = n * D2Q9::Q;

            if (fluid_field.cellType(x, y) == CellType::SOLID) {
                // Solid cells are inert; leave their g_ alone. computeConcentration
                // will still read them but they were never streamed from upstream
                // fluid (we use halfway BB at fluid sites instead).
                continue;
            }

            for (int q = 0; q < D2Q9::Q; ++q) {
                int xs = x - D2Q9::cx[q];
                int ys = y - D2Q9::cy[q];

                // Periodic-x is always on (matches the project specification streamwise PBC).
                xs = ((xs % nx_) + nx_) % nx_;
                if (periodic_y_) ys = ((ys % ny_) + ny_) % ny_;

                const bool y_oob = (ys < 0 || ys >= ny_);
                if (y_oob) {
                    dst[qbase + q] = src[qbase + D2Q9::opp[q]];   // halfway BB at y-wall
                    continue;
                }

                const int n_up = idx(xs, ys);
                if (fluid_field.cellType(xs, ys) == CellType::SOLID) {
                    dst[qbase + q] = src[qbase + D2Q9::opp[q]];   // halfway BB at obstacle
                } else {
                    dst[qbase + q] = src[n_up * D2Q9::Q + q];     // regular pull
                }
            }
        }
    }
}

void AdvectionDiffusion::computeConcentration(int species) {
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int n = 0; n < nx_ * ny_; ++n) {
        Real C = 0.0;
        for (int q = 0; q < D2Q9::Q; ++q) {
            C += g_[species][n * D2Q9::Q + q];
        }
        C_[species][n] = C;
    }
}

void AdvectionDiffusion::applySourceSink(
    const CapsuleSystem& capsules,
    const std::vector<Real>& release_rates,
    const std::vector<Real>& absorption_rates) {
    int ncaps = capsules.numCapsules();

    // Resize per-capsule tracking arrays if needed
    if (static_cast<int>(capsule_released_.size()) < ncaps) {
        capsule_released_.resize(ncaps, 0.0);
        capsule_absorbed_.resize(ncaps, 0.0);
    }

    // For each capsule, add/remove concentration at its node positions
    for (int c = 0; c < ncaps; ++c) {
        const auto& cap = capsules[c];
        int type = cap.getType();

        Real release = (type < static_cast<int>(release_rates.size()))
                       ? release_rates[type] : 0.0;
        Real absorb = (type < static_cast<int>(absorption_rates.size()))
                      ? absorption_rates[type] : 0.0;

        for (int ni = 0; ni < cap.numNodes(); ++ni) {
            Vec2d pos = cap.nodePosition(ni);
            int ix = static_cast<int>(pos.x + 0.5);
            int iy = static_cast<int>(pos.y + 0.5);
            ix = std::max(0, std::min(nx_ - 1, ix));
            iy = std::max(0, std::min(ny_ - 1, iy));

            for (int s = 0; s < n_species_; ++s) {
                int n = idx(ix, iy);

                // Release
                C_[s][n] += release;
                capsule_released_[c] += release;

                // Absorption (only absorb what's available)
                Real actual_absorb = std::min(absorb, C_[s][n]);
                C_[s][n] -= actual_absorb;
                capsule_absorbed_[c] += actual_absorb;

                // Update distributions to match new concentration
                for (int q = 0; q < D2Q9::Q; ++q) {
                    g_[s][n * D2Q9::Q + q] = D2Q9::w[q] * C_[s][n];
                }
            }
        }
    }
}

Real AdvectionDiffusion::getConcentration(int x, int y, int species) const {
    if (x < 0 || x >= nx_ || y < 0 || y >= ny_) return 0.0;
    if (species < 0 || species >= n_species_) return 0.0;
    return C_[species][idx(x, y)];
}

const Real* AdvectionDiffusion::concentrationData(int species) const {
    if (species < 0 || species >= n_species_) return nullptr;
    return C_[species].data();
}

Real* AdvectionDiffusion::concentrationData(int species) {
    if (species < 0 || species >= n_species_) return nullptr;
    return C_[species].data();
}

// ── Physics-based chemistry: leaching + Langmuir adsorption ──────────────────
void AdvectionDiffusion::applyChemistry(
    const CapsuleSystem& capsules,
    const std::vector<Real>& release_rates,
    const std::vector<Real>& absorption_rates,
    const ScalarParams& sp)
{
    int ncaps = capsules.numCapsules();

    // Lazy-init all per-capsule tracking arrays
    if (static_cast<int>(capsule_released_.size()) < ncaps) {
        capsule_released_.resize(ncaps, 0.0);
        capsule_absorbed_.resize(ncaps, 0.0);
    }
    if (static_cast<int>(capsule_Mp_.size()) < ncaps) {
        int old_sz = static_cast<int>(capsule_Mp_.size());
        capsule_Mp_.resize(ncaps, -1.0);  // -1 = infinite reservoir
        // Apply M_p_initial for new capsules
        for (int c = old_sz; c < ncaps; ++c) {
            int type = capsules[c].getType();
            if (type < static_cast<int>(sp.M_p_initial.size()) &&
                sp.M_p_initial[type] > 0.0)
                capsule_Mp_[c] = sp.M_p_initial[type];
        }
    }
    if (static_cast<int>(gamma_nodes_.size()) < ncaps) {
        gamma_nodes_.resize(ncaps);
    }

    const int Lx = nx_;  // periodic wrap in x only (same as IBM spreading)

    for (int c = 0; c < ncaps; ++c) {
        const auto& cap = capsules[c];
        int type = cap.getType();
        int Nn   = cap.numNodes();

        // Fetch physics params for this capsule type
        Real k_L   = (type < static_cast<int>(sp.k_leach.size()))   ? sp.k_leach[type]   : 0.0;
        Real Ceq   = (type < static_cast<int>(sp.C_eq.size()))      ? sp.C_eq[type]      : 0.0;
        Real k_a   = (type < static_cast<int>(sp.k_adsorb.size()))  ? sp.k_adsorb[type]  : 0.0;
        Real k_d   = (type < static_cast<int>(sp.k_desorb.size()))  ? sp.k_desorb[type]  : 0.0;
        Real Gmax  = (type < static_cast<int>(sp.Gamma_max.size())) ? sp.Gamma_max[type] : 1.0;

        // Constant-rate fallback (backward compat, used when no physics params)
        Real const_release = (type < static_cast<int>(release_rates.size()))
                             ? release_rates[type] : 0.0;
        Real const_absorb  = (type < static_cast<int>(absorption_rates.size()))
                             ? absorption_rates[type] : 0.0;

        // Ensure gamma array is allocated for this capsule
        if (static_cast<int>(gamma_nodes_[c].size()) != Nn)
            gamma_nodes_[c].assign(Nn, 0.0);

        for (int k = 0; k < Nn; ++k) {
            Vec2d pos = cap.nodePosition(k);

            // Arc-length segment ds (average of adjacent half-edges)
            int kp = (k + 1) % Nn;
            int km = (k - 1 + Nn) % Nn;
            Vec2d pp = cap.nodePosition(kp);
            Vec2d pm = cap.nodePosition(km);
            Real dxp = pp.x - pos.x, dyp = pp.y - pos.y;
            Real dxm = pos.x - pm.x, dym = pos.y - pm.y;
            Real ds = 0.5 * (std::sqrt(dxp*dxp + dyp*dyp) +
                             std::sqrt(dxm*dxm + dym*dym));
            if (ds < 1e-12) ds = 1.0;

            // ── Step A: Interpolate C_surface via 4×4 Peskin kernel ──────────
            int ix0 = static_cast<int>(std::floor(pos.x));
            int iy0 = static_cast<int>(std::floor(pos.y));
            Real C_surf = 0.0;
            for (int s = 0; s < n_species_; ++s) {
                C_surf = 0.0;
                for (int dy = -1; dy <= 2; ++dy) {
                    int jy = iy0 + dy;
                    if (jy < 0 || jy >= ny_) continue;
                    for (int dx = -1; dx <= 2; ++dx) {
                        int jx = ix0 + dx;
                        // Periodic wrap in x
                        jx = ((jx % Lx) + Lx) % Lx;
                        Real rx = pos.x - static_cast<Real>(ix0 + dx);
                        Real ry = pos.y - static_cast<Real>(jy);
                        Real w = PeskinDelta::delta2d(rx, ry);
                        C_surf += C_[s][idx(jx, jy)] * w;
                    }
                }
            }

            // ── Step B: Compute leaching flux dC_leach ────────────────────────
            //
            // dC_leach has units of mass per (arc-length × time) — it is a
            // *surface flux density*. The actual mass leaving M_p through
            // this node in one timestep is therefore dC_leach × ds (the
            // local arc-length segment of the membrane). The same factor
            // appears in the Eulerian spread below where the flux is
            // weighted by w · ds — so for global conservation we must
            // drain M_p by dC_leach × ds, not by dC_leach alone.
            // (Bug fix for; see
            // tests/verification/test_chemistry_mass_balance.cpp.)
            Real dC_leach = 0.0;
            if (k_L > 0.0) {
                // Physics-based Fick-type leaching
                dC_leach = k_L * (Ceq - C_surf);  // positive = release, negative = uptake
                // Enforce finite particle reservoir (Gap C)
                if (capsule_Mp_[c] >= 0.0) {
                    if (dC_leach > 0.0) {
                        // Cap the surface flux density so that the actual
                        // mass leaving (dC_leach × ds) does not exceed the
                        // remaining reservoir.
                        Real available_per_ds = capsule_Mp_[c] / ds;
                        dC_leach = std::min(dC_leach, available_per_ds);
                        capsule_Mp_[c] -= dC_leach * ds;
                        if (capsule_Mp_[c] < 0.0) capsule_Mp_[c] = 0.0;
                    }
                    // When M_p = 0, leaching stops; uptake still allowed
                }
            } else {
                // Constant-rate fallback (backward compatible)
                Real absorb_now = std::min(const_absorb, C_surf);
                dC_leach = const_release - absorb_now;
            }

            // ── Step C: Langmuir adsorption/desorption ────────────────────────
            Real dC_adsorb = 0.0;
            if (k_a > 0.0 || k_d > 0.0) {
                Real& Gamma = gamma_nodes_[c][k];
                Real dGamma = k_a * C_surf * (1.0 - Gamma / Gmax) - k_d * Gamma;
                Gamma = std::max(0.0, std::min(Gmax, Gamma + dGamma));
                dC_adsorb = -dGamma;  // adsorption removes from fluid
            }

            Real total_flux = dC_leach + dC_adsorb;
            if (total_flux == 0.0) continue;

            // ── Step D: Spread flux to Eulerian grid via Peskin kernel ─────────
            for (int s = 0; s < n_species_; ++s) {
                for (int dy = -1; dy <= 2; ++dy) {
                    int jy = iy0 + dy;
                    if (jy < 0 || jy >= ny_) continue;
                    for (int dx = -1; dx <= 2; ++dx) {
                        int jx = ix0 + dx;
                        jx = ((jx % Lx) + Lx) % Lx;
                        Real rx = pos.x - static_cast<Real>(ix0 + dx);
                        Real ry = pos.y - static_cast<Real>(jy);
                        Real w = PeskinDelta::delta2d(rx, ry);
                        if (w < 1e-15) continue;
                        int n = idx(jx, jy);
                        C_[s][n] += total_flux * w * ds;
                        if (C_[s][n] < 0.0) C_[s][n] = 0.0;
                        // Reset g_ to equilibrium with new concentration
                        for (int q = 0; q < D2Q9::Q; ++q)
                            g_[s][n * D2Q9::Q + q] = D2Q9::w[q] * C_[s][n];
                    }
                }

                // Update cumulative tracking
                if (total_flux > 0.0)
                    capsule_released_[c] += total_flux * ds;
                else
                    capsule_absorbed_[c] -= total_flux * ds;
            }
        }
    }
}

// ── Gamma flat-buffer helpers (for checkpoint I/O) ──────────────────────────
int AdvectionDiffusion::totalGammaNodes() const {
    int total = 0;
    for (const auto& g : gamma_nodes_) total += static_cast<int>(g.size());
    return total;
}

const Real* AdvectionDiffusion::gammaData() const {
    // Sync flat buffer from nested vector
    gamma_flat_.clear();
    gamma_flat_.reserve(totalGammaNodes());
    for (const auto& g : gamma_nodes_)
        gamma_flat_.insert(gamma_flat_.end(), g.begin(), g.end());
    return gamma_flat_.data();
}

void AdvectionDiffusion::syncGammaFromFlat(const CapsuleSystem& capsules) {
    int ncaps = capsules.numCapsules();
    gamma_nodes_.resize(ncaps);
    int offset = 0;
    for (int c = 0; c < ncaps; ++c) {
        int nn = capsules[c].numNodes();
        gamma_nodes_[c].resize(nn, 0.0);
        for (int k = 0; k < nn && offset < static_cast<int>(gamma_flat_.size()); ++k)
            gamma_nodes_[c][k] = gamma_flat_[offset++];
    }
}

} // namespace softflow
