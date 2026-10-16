#include "capsule.h"
#include <algorithm>
#include <cmath>

namespace softflow {

Capsule::Capsule(int id, Vec2d center, Real radius, int num_nodes,
                 const MembraneParams& params, int type)
    : num_nodes_(num_nodes > 0 ? num_nodes : computeOptimalNodes(radius)),
      model_(params.model),
      k_stretch_(params.k_stretch),
      G_s_(params.G_s),
      C_skalak_(params.C_skalak),
      k_bend_(params.k_bend),
      use_helfrich_(params.use_helfrich_bending),
      kappa_0_(params.kappa_0),
      k_area_(params.k_area),
      k_perimeter_(params.k_perimeter),
      gamma_visc_(params.gamma_visc),
      eta_membrane_(params.eta_membrane),
      viscosity_ratio_(params.viscosity_ratio),
      density_(params.density),
      shape_(params.shape),
      aspect_ratio_(params.aspect_ratio),
      indent_depth_(params.indent_depth),
      is_rigid_(params.is_rigid),
      type_id_(type),
      id_(id),
      wlc_L_max_ratio_(params.wlc_L_max_ratio),
      wlc_kBT_p_(params.wlc_kBT_p),
      wlc_k_pow_(params.wlc_k_pow)
{
    nodes_.resize(num_nodes_);
    velocities_.resize(num_nodes_, Vec2d{0.0, 0.0});
    forces_.resize(num_nodes_, Vec2d{0.0, 0.0});
    rest_lengths_.resize(num_nodes_);

    // Place nodes around the rest shape at equal arc length.  (Equal
    // polar-angle spacing would give a biconcave or elongated shape rest
    // edges that differ by a factor of 3 or more; the short edges are then
    // much stiffer under the explicit update, and tank-treading forces large
    // edge strains as material moves between the dimple and the rim.)
    auto shapePoint = [&](Real theta) -> Vec2d {
        if (shape_ == CapsuleShape::ELLIPSE || shape_ == CapsuleShape::FIBER)
            return {radius * std::cos(theta), radius * aspect_ratio_ * std::sin(theta)};
        if (shape_ == CapsuleShape::BICONCAVE) {
            Real r = radius * (1.0 - indent_depth_ * std::sin(theta) * std::sin(theta));
            return {r * std::cos(theta), r * std::sin(theta)};
        }
        return {radius * std::cos(theta), radius * std::sin(theta)};
    };
    if (shape_ == CapsuleShape::CIRCLE) {
        for (int i = 0; i < num_nodes_; ++i) {
            Vec2d r = shapePoint(2.0 * PI * i / num_nodes_);
            nodes_[i] = Vec2d{center.x + r.x, center.y + r.y};
        }
    } else {
        const int M = 256 * num_nodes_;           // fine sampling of the curve
        std::vector<Vec2d> pts(M + 1);
        std::vector<Real> cum(M + 1, 0.0);
        for (int m = 0; m <= M; ++m) {
            pts[m] = shapePoint(2.0 * PI * m / M);
            if (m > 0) cum[m] = cum[m - 1] + (pts[m] - pts[m - 1]).norm();
        }
        const Real total = cum[M];
        int m = 0;
        for (int i = 0; i < num_nodes_; ++i) {
            const Real target = total * i / num_nodes_;
            while (m < M - 1 && cum[m + 1] < target) ++m;
            const Real seg = cum[m + 1] - cum[m];
            const Real w = (seg > 1e-15) ? (target - cum[m]) / seg : 0.0;
            Vec2d r = pts[m] + (pts[m + 1] - pts[m]) * w;
            nodes_[i] = Vec2d{center.x + r.x, center.y + r.y};
        }
    }

    // Compute rest lengths for each spring link
    for (int i = 0; i < num_nodes_; ++i) {
        Vec2d diff = nodes_[next(i)] - nodes_[i];
        rest_lengths_[i] = diff.norm();
    }

    // Initialize previous edge lengths for Kelvin-Voigt model
    prev_lengths_ = rest_lengths_;
    has_prev_lengths_ = false;

    // Compute rest area and perimeter from the initial configuration
    rest_area_ = area();
    rest_perimeter_ = perimeter();

    // Auto-compute WLC parameters if needed
    if (model_ == MembraneModel::WLC) {
        // Fedosov et al., Biophys J 2010:
        // WLC attractive force:  F_wlc = (kBT/p) * [1/(4(1-x)^2) - 1/4 + x]
        // Power-law repulsive:   F_pow = -k_pow / L^m  (m=1 in 2D)
        // At rest (x = L0/L_max), net force = 0 → k_pow determined
        //
        // If wlc_kBT_p == 0, auto-compute from G_s so that the
        // linearized spring constant at rest matches G_s:
        //   dF/dL|_{L=L0} ≈ G_s / L0  (match Skalak at small deformation)
        if (rest_lengths_.size() > 0) {
            Real L0 = rest_lengths_[0];
            Real L_max = wlc_L_max_ratio_ * L0;
            Real x0 = L0 / L_max;  // rest extension ratio

            if (wlc_kBT_p_ <= 0.0) {
                // dF_wlc/dx = (kBT/p) * [1/(2(1-x)^3) + 1]
                // dF/dL = (dF/dx) * (1/L_max)
                // Set dF/dL = G_s/L0 → kBT/p = G_s * L_max / (L0 * [1/(2(1-x0)^3) + 1])
                Real denom = 1.0 / (2.0 * (1.0-x0)*(1.0-x0)*(1.0-x0)) + 1.0;
                wlc_kBT_p_ = G_s_ * L_max / (L0 * denom);
            }

            if (wlc_k_pow_ <= 0.0) {
                // At rest: F_wlc(x0) + F_pow = 0
                // F_wlc(x0) = (kBT/p) * [1/(4(1-x0)^2) - 1/4 + x0]
                Real F_wlc_rest = wlc_kBT_p_ * (
                    1.0/(4.0*(1.0-x0)*(1.0-x0)) - 0.25 + x0);
                // F_pow = -k_pow / L0  →  k_pow = F_wlc_rest * L0
                wlc_k_pow_ = F_wlc_rest * L0;
            }
        }
    }

    // Rest turning angles for the discrete-elastica bending energy.  The
    // rest shape is stress-free: theta0_i is the turning angle of the
    // initial configuration, unless a uniform spontaneous curvature kappa_0
    // is given, in which case theta0_i = kappa_0 * l0_i.
    if (use_helfrich_) {
        rest_angles_.resize(num_nodes_);
        rest_ds_.resize(num_nodes_);
        for (int i = 0; i < num_nodes_; ++i) {
            Vec2d e_prev = minImageDiff(nodes_[i], nodes_[prev(i)]);
            Vec2d e_next = minImageDiff(nodes_[next(i)], nodes_[i]);
            rest_ds_[i] = 0.5 * (e_prev.norm() + e_next.norm());
            rest_angles_[i] = (kappa_0_ != 0.0)
                ? kappa_0_ * rest_ds_[i]
                : std::atan2(e_prev.cross(e_next), e_prev.dot(e_next));
        }
    }
}

void Capsule::clearForces() {
    for (int i = 0; i < num_nodes_; ++i) {
        forces_[i] = Vec2d{0.0, 0.0};
    }
}

void Capsule::computeMembraneForces() {
    if (is_rigid_) return;  // rigid body: no internal deformation forces
    computeStretchingForces();
    computeBendingForces();
    computeAreaForces();
    computePerimeterForces();
    computeViscousDamping();
    computeMembraneViscosity();  // Kelvin-Voigt viscoelastic damping
}

Vec2d Capsule::centroid() const {
    if (periodic_nx_ > 0 && num_nodes_ > 0) {
        Vec2d ref = nodes_[0];
        Vec2d sum = ref;
        for (int i = 1; i < num_nodes_; ++i) {
            sum += ref + minImageDiff(nodes_[i], ref);
        }
        return sum / static_cast<Real>(num_nodes_);
    }
    Vec2d c{0.0, 0.0};
    for (int i = 0; i < num_nodes_; ++i) {
        c += nodes_[i];
    }
    return c / static_cast<Real>(num_nodes_);
}

Real Capsule::area() const {
    if (periodic_nx_ > 0 && num_nodes_ > 0) {
        std::vector<Vec2d> unwrapped(num_nodes_);
        unwrapped[0] = nodes_[0];
        for (int i = 1; i < num_nodes_; ++i) {
            unwrapped[i] = unwrapped[i-1] + minImageDiff(nodes_[i], nodes_[i-1]);
        }
        Real A = 0.0;
        for (int i = 0; i < num_nodes_; ++i) {
            int j = (i + 1) % num_nodes_;
            A += unwrapped[i].cross(unwrapped[j]);
        }
        return 0.5 * std::abs(A);
    }
    Real A = 0.0;
    for (int i = 0; i < num_nodes_; ++i) {
        int j = next(i);
        A += nodes_[i].cross(nodes_[j]);
    }
    return 0.5 * std::abs(A);
}

Real Capsule::perimeter() const {
    Real P = 0.0;
    for (int i = 0; i < num_nodes_; ++i) {
        Vec2d diff = minImageDiff(nodes_[next(i)], nodes_[i]);
        P += diff.norm();
    }
    return P;
}

// Central second moments of area of the membrane polygon (Green's theorem),
// on nodes unwrapped across the periodic x boundary.  For an ellipse with
// semi-axes a >= b the eigenvalues are J1 = pi a^3 b / 4, J2 = pi a b^3 / 4.
void Capsule::secondMoments(Real& mxx, Real& myy, Real& mxy) const {
    std::vector<Vec2d> p(num_nodes_);
    p[0] = nodes_[0];
    for (int i = 1; i < num_nodes_; ++i)
        p[i] = p[i-1] + minImageDiff(nodes_[i], nodes_[i-1]);

    Real A = 0.0, cx = 0.0, cy = 0.0;
    for (int i = 0; i < num_nodes_; ++i) {
        const Vec2d& a = p[i];
        const Vec2d& b = p[(i + 1) % num_nodes_];
        Real cr = a.x * b.y - b.x * a.y;
        A  += cr;
        cx += (a.x + b.x) * cr;
        cy += (a.y + b.y) * cr;
    }
    A *= 0.5;
    mxx = myy = mxy = 0.0;
    if (std::abs(A) < 1e-15) return;
    cx /= (6.0 * A);
    cy /= (6.0 * A);

    for (int i = 0; i < num_nodes_; ++i) {
        Real x0 = p[i].x - cx, y0 = p[i].y - cy;
        const Vec2d& q = p[(i + 1) % num_nodes_];
        Real x1 = q.x - cx, y1 = q.y - cy;
        Real cr = x0 * y1 - x1 * y0;
        mxx += (x0 * x0 + x0 * x1 + x1 * x1) * cr;
        myy += (y0 * y0 + y0 * y1 + y1 * y1) * cr;
        mxy += (x0 * y1 + 2.0 * x0 * y0 + 2.0 * x1 * y1 + x1 * y0) * cr;
    }
    const Real sgn = (A > 0.0) ? 1.0 : -1.0;    // node ordering
    mxx *= sgn / 12.0;
    myy *= sgn / 12.0;
    mxy *= sgn / 24.0;
}

// Taylor deformation parameter D = (L - B)/(L + B) of the ellipse with the
// same second moments of area (Ramanujan & Pozrikidis 1998): L/B is
// sqrt(J1/J2).  Independent of orientation, unlike a bounding-box measure.
Real Capsule::deformationIndex() const {
    if (num_nodes_ < 3) return 0.0;
    Real mxx, myy, mxy;
    secondMoments(mxx, myy, mxy);
    Real tr = 0.5 * (mxx + myy);
    Real dd = std::sqrt(0.25 * (mxx - myy) * (mxx - myy) + mxy * mxy);
    Real J1 = tr + dd, J2 = std::max(tr - dd, 0.0);
    Real s1 = std::sqrt(J1), s2 = std::sqrt(J2);
    return (s1 + s2 > 1e-15) ? (s1 - s2) / (s1 + s2) : 0.0;
}

// Angle of the major principal axis to the x axis, in (-pi/2, pi/2].
Real Capsule::inclinationAngle() const {
    if (num_nodes_ < 3) return 0.0;
    Real mxx, myy, mxy;
    secondMoments(mxx, myy, mxy);
    return 0.5 * std::atan2(2.0 * mxy, mxx - myy);
}

void Capsule::moveNodes(Real dt) {
    if (is_rigid_) {
        // Project IBM-interpolated node velocities onto rigid body motion:
        // v_i = v_cm + omega × r_i  (pure translation + rotation, no deformation)
        Vec2d ctr = centroid();
        Vec2d v_cm{0.0, 0.0};
        for (int i = 0; i < num_nodes_; ++i) v_cm += velocities_[i];
        v_cm *= (1.0 / num_nodes_);
        Real omega_num = 0.0, omega_den = 0.0;
        for (int i = 0; i < num_nodes_; ++i) {
            Vec2d r = nodes_[i] - ctr;
            omega_num += r.x * velocities_[i].y - r.y * velocities_[i].x;
            omega_den += r.x * r.x + r.y * r.y;
        }
        Real omega = (omega_den > 1e-10) ? omega_num / omega_den : 0.0;
        for (int i = 0; i < num_nodes_; ++i) {
            Vec2d r = nodes_[i] - ctr;
            velocities_[i] = Vec2d{v_cm.x - omega * r.y, v_cm.y + omega * r.x};
        }
    }
    for (int i = 0; i < num_nodes_; ++i) {
        nodes_[i] += velocities_[i] * dt;
    }
    if (periodic_nx_ > 0) {
        Real Lx = static_cast<Real>(periodic_nx_);
        for (int i = 0; i < num_nodes_; ++i) {
            while (nodes_[i].x < 0.0)  nodes_[i].x += Lx;
            while (nodes_[i].x >= Lx)  nodes_[i].x -= Lx;
        }
    }
}

// ---------------------------------------------------------------------------
// Stretching forces — dispatch by model
// ---------------------------------------------------------------------------
//
// A 2-D capsule here is a closed Lagrangian ring of N nodes connected by
// straight edges; it is NOT a continuous-surface FEM membrane. Every
// constitutive law below (Hookean, Neo-Hookean, Skalak, WLC) is implemented
// as the **one-dimensional restriction of the surface law to the edge**:
// for an edge with current length L and rest length L0, we set the in-plane
// stretch lambda_1 = L/L0 and the transverse stretch lambda_2 = 1, then
// evaluate the edge tension T(lambda_1) and apply it along the edge as an
// equal-and-opposite pair of nodal forces. Surface-area incompressibility
// (the second invariant) is NOT inherited from the constitutive law in this
// reduction; it is enforced by the separate area-penalty term
// `computeAreaConservation()` with stiffness `k_area`. Bending follows the
// discrete-curvature Helfrich form `computeBendingForcesHelfrich()` (or the
// cheaper discrete-Laplacian default). This decomposition matches the
// closed-chain "spring + area + bending" recipe of Pozrikidis 2003 and is
// the standard 2-D LBM-IBM reduction of Skalak/Neo-Hookean used in e.g.
// Krüger 2017 §11; the manuscript text should describe it as such, not as
// a surface-strain-invariant formulation.
void Capsule::computeStretchingForces() {
    switch (model_) {
        case MembraneModel::NEO_HOOKEAN:
            computeStretchingForcesNeoHookean();
            break;
        case MembraneModel::SKALAK:
            computeStretchingForcesSkalak();
            break;
        case MembraneModel::WLC:
            computeStretchingForcesWLC();
            break;
        case MembraneModel::HOOKEAN:
        default:
            computeStretchingForcesHookean();
            break;
    }
}

void Capsule::computeStretchingForcesHookean() {
    for (int i = 0; i < num_nodes_; ++i) {
        int j = next(i);
        Vec2d diff = minImageDiff(nodes_[j], nodes_[i]);
        Real L = diff.norm();
        if (L < 1e-15) continue;

        Real L0 = rest_lengths_[i];
        Real F_mag = k_stretch_ * (L / L0 - 1.0);

        Vec2d dir = diff / L;
        Vec2d F = dir * F_mag;

        forces_[i] += F;
        forces_[j] -= F;
    }
}

void Capsule::computeStretchingForcesNeoHookean() {
    // Neo-Hookean law for 2D membranes (Barthes-Biesel 2016, Sui et al. 2008)
    // Strain energy: W = G_s/2 * (I1 - 1 + 1/(I1+1))
    // In 1D (edge): lambda = L/L0, incompressible transverse: lambda_perp = 1/lambda
    // Tension: T = G_s * (lambda^2 - 1/lambda^2) / lambda
    // Force per edge: F = T * L0 = G_s * (lambda - 1/lambda^3)
    for (int i = 0; i < num_nodes_; ++i) {
        int j = next(i);
        Vec2d diff = minImageDiff(nodes_[j], nodes_[i]);
        Real L = diff.norm();
        if (L < 1e-15) continue;

        Real L0 = rest_lengths_[i];
        Real lambda = L / L0;
        Real lambda_inv3 = 1.0 / (lambda * lambda * lambda);

        // Force magnitude: F = G_s * (lambda - 1/lambda^3)
        Real F_mag = G_s_ * (lambda - lambda_inv3);

        Vec2d dir = diff / L;
        Vec2d F = dir * F_mag;

        forces_[i] += F;
        forces_[j] -= F;
    }
}

void Capsule::computeStretchingForcesSkalak() {
    // Skalak law (Skalak, Tozeren, Zarda & Chien, Biophys J 1973)
    // 2D membrane with shear elasticity and area dilation resistance.
    // In 1D ring discretization:
    //   lambda_1 = L/L0 (stretch along edge)
    //   lambda_2 ≈ 1 (transverse, handled by area penalty)
    //   I1 = lambda_1^2 + lambda_2^2 - 2 = lambda^2 - 1
    //   I2 = lambda_1^2 * lambda_2^2 - 1 = lambda^2 - 1
    // Tension: T = G_s * lambda * [(lambda^2 - 1) + C * (lambda^2 - 1)]
    //            = G_s * (1 + C) * lambda * (lambda^2 - 1)
    // Force: F = T / lambda = G_s * (1 + C) * (lambda^2 - 1)
    //
    // More precise form keeping lambda_2 = 1:
    // T = G_s * [lambda * (lambda^2 - 1) + C * lambda * (lambda^2 * 1 - 1)]
    //   = G_s * lambda * (lambda^2 - 1) * (1 + C)
    for (int i = 0; i < num_nodes_; ++i) {
        int j = next(i);
        Vec2d diff = minImageDiff(nodes_[j], nodes_[i]);
        Real L = diff.norm();
        if (L < 1e-15) continue;

        Real L0 = rest_lengths_[i];
        Real lambda = L / L0;
        Real lambda2 = lambda * lambda;

        // Skalak tension: T = G_s * lambda * (lambda^2 - 1) * (1 + C)
        // Force magnitude: F = T (since we apply along the edge)
        Real F_mag = G_s_ * lambda * (lambda2 - 1.0) * (1.0 + C_skalak_);

        Vec2d dir = diff / L;
        Vec2d F = dir * F_mag;

        forces_[i] += F;
        forces_[j] -= F;
    }
}

void Capsule::computeStretchingForcesWLC() {
    // Worm-Like Chain model (Fedosov, Caswell & Karniadakis, Biophys J 2010)
    //
    // Combines WLC attractive force with power-law repulsive force:
    //
    //   F_wlc(L) = (kBT/p) * [ 1/(4(1-x)^2) - 1/4 + x ]
    //   F_pow(L) = -k_pow / L
    //   F_total  = F_wlc + F_pow
    //
    // where x = L / L_max is the fractional extension.
    //
    // Properties:
    //   - F → -∞ as L → 0  (repulsive, prevents edge collapse)
    //   - F → +∞ as L → L_max  (attractive, diverges at max extension)
    //   - F = 0 at L = L0 (rest configuration)
    //   - Strain-hardening at large extensions (realistic for cytoskeleton)
    //
    // Parameters:
    //   wlc_kBT_p_: kBT/p (thermal energy / persistence length)
    //   wlc_L_max_ratio_: L_max / L0
    //   wlc_k_pow_: repulsive power-law coefficient

    for (int i = 0; i < num_nodes_; ++i) {
        int j = next(i);
        Vec2d diff = minImageDiff(nodes_[j], nodes_[i]);
        Real L = diff.norm();
        if (L < 1e-15) continue;

        Real L0 = rest_lengths_[i];
        Real L_max = wlc_L_max_ratio_ * L0;
        Real x = L / L_max;

        // Clamp x to avoid singularity at x=1
        if (x > 0.99) x = 0.99;

        // WLC attractive force (positive = tension)
        Real F_wlc = wlc_kBT_p_ * (
            1.0 / (4.0 * (1.0 - x) * (1.0 - x)) - 0.25 + x);

        // Power-law repulsive force (negative = compression)
        Real F_pow = -wlc_k_pow_ / L;

        Real F_mag = F_wlc + F_pow;

        Vec2d dir = diff / L;
        Vec2d F = dir * F_mag;

        forces_[i] += F;
        forces_[j] -= F;
    }
}

// ---------------------------------------------------------------------------
// Bending forces — dispatch by model
// ---------------------------------------------------------------------------

void Capsule::computeBendingForces() {
    if (use_helfrich_) {
        computeBendingForcesHelfrich();
    } else {
        computeBendingForcesLaplacian();
    }
}

void Capsule::computeBendingForcesLaplacian() {
    // Original discrete Laplacian bending:
    // F_i = k_bend * (minImageDiff(prev, i) + minImageDiff(next, i))
    for (int i = 0; i < num_nodes_; ++i) {
        Vec2d d_prev = minImageDiff(nodes_[prev(i)], nodes_[i]);
        Vec2d d_next = minImageDiff(nodes_[next(i)], nodes_[i]);
        Vec2d lap = d_prev + d_next;
        forces_[i] += lap * k_bend_;
    }
}

// Discrete elastica (two-dimensional Helfrich) bending:
//
//   E_b = (k_b / 2) * sum_i (theta_i - theta0_i)^2 / l0_i
//       ~ (k_b / 2) * closed-integral (kappa - kappa_0)^2 ds,
//
// with theta_i the turning angle at node i (kappa_i ~ theta_i / l_i) and
// l0_i the rest length associated with node i.  The force is the exact
// gradient F_j = -dE_b/dx_j, using
//
//   d theta_i / d x_{i+1} =  perp(e_next) / |e_next|^2,
//   d theta_i / d x_{i-1} =  perp(e_prev) / |e_prev|^2,
//   d theta_i / d x_i     = -(sum of the two),     perp(a) = (-a_y, a_x),
//
// so the rest shape is force-free and the bending force carries no net
// force or torque.
Real Capsule::bendingEnergy() const {
    if (!use_helfrich_) return 0.0;
    Real E = 0.0;
    for (int i = 0; i < num_nodes_; ++i) {
        Vec2d e_prev = minImageDiff(nodes_[i], nodes_[prev(i)]);
        Vec2d e_next = minImageDiff(nodes_[next(i)], nodes_[i]);
        Real dth = std::atan2(e_prev.cross(e_next), e_prev.dot(e_next))
                 - rest_angles_[i];
        E += 0.5 * k_bend_ * dth * dth / rest_ds_[i];
    }
    return E;
}

void Capsule::computeBendingForcesHelfrich() {
    for (int i = 0; i < num_nodes_; ++i) {
        const int ip = prev(i), in = next(i);
        Vec2d e_prev = minImageDiff(nodes_[i], nodes_[ip]);
        Vec2d e_next = minImageDiff(nodes_[in], nodes_[i]);
        Real lp2 = e_prev.dot(e_prev), ln2 = e_next.dot(e_next);
        if (lp2 < 1e-30 || ln2 < 1e-30) continue;

        Real dth = std::atan2(e_prev.cross(e_next), e_prev.dot(e_next))
                 - rest_angles_[i];
        Real c = -k_bend_ * dth / rest_ds_[i];        // -dE/dtheta_i

        Vec2d g_next = Vec2d{-e_next.y, e_next.x} / ln2;   // d theta / d x_{i+1}
        Vec2d g_prev = Vec2d{-e_prev.y, e_prev.x} / lp2;   // d theta / d x_{i-1}
        forces_[in] += g_next * c;
        forces_[ip] += g_prev * c;
        forces_[i]  -= (g_next + g_prev) * c;
    }
}

// ---------------------------------------------------------------------------
// Area and perimeter conservation forces (unchanged)
// ---------------------------------------------------------------------------

void Capsule::computeAreaForces() {
    Real A = area();
    Real A0 = rest_area_;
    if (A0 < 1e-15) return;

    Real coeff = -k_area_ * (A - A0) / A0;

    Real signed_area = 0.0;
    if (periodic_nx_ > 0) {
        std::vector<Vec2d> uw(num_nodes_);
        uw[0] = nodes_[0];
        for (int k = 1; k < num_nodes_; ++k) {
            uw[k] = uw[k-1] + minImageDiff(nodes_[k], nodes_[k-1]);
        }
        for (int k = 0; k < num_nodes_; ++k) {
            int kn = (k + 1) % num_nodes_;
            signed_area += uw[k].cross(uw[kn]);
        }
    } else {
        for (int k = 0; k < num_nodes_; ++k) {
            signed_area += nodes_[k].cross(nodes_[next(k)]);
        }
    }
    Real sign = (signed_area >= 0.0) ? 1.0 : -1.0;

    for (int i = 0; i < num_nodes_; ++i) {
        Vec2d e_prev = minImageDiff(nodes_[i], nodes_[prev(i)]);
        Vec2d e_next = minImageDiff(nodes_[next(i)], nodes_[i]);

        Vec2d n_prev = Vec2d{e_prev.y, -e_prev.x};
        Vec2d n_next = Vec2d{e_next.y, -e_next.x};

        Vec2d outward = (n_prev + n_next) * 0.5;
        Real len = outward.norm();
        if (len < 1e-15) continue;
        outward = outward / len * sign;

        forces_[i] += outward * coeff;
    }
}

void Capsule::computePerimeterForces() {
    Real P = perimeter();
    Real P0 = rest_perimeter_;
    if (P0 < 1e-15) return;

    Real coeff = -k_perimeter_ * (P - P0) / P0;

    for (int i = 0; i < num_nodes_; ++i) {
        int j = next(i);
        Vec2d diff = minImageDiff(nodes_[j], nodes_[i]);
        Real L = diff.norm();
        if (L < 1e-15) continue;
        Vec2d tangent = diff / L;
        forces_[i] -= tangent * coeff;
        forces_[j] += tangent * coeff;
    }
}

void Capsule::computeViscousDamping() {
    for (int i = 0; i < num_nodes_; ++i) {
        forces_[i] -= velocities_[i] * gamma_visc_;
    }
}

void Capsule::computeMembraneViscosity() {
    // Kelvin-Voigt viscoelastic membrane model
    // (Barthes-Biesel & Sgaier 1985, Yazdani & Bagchi 2013)
    //
    // Total edge force = F_elastic(strain) + F_viscous(strain_rate)
    //
    // The viscous component resists RATE OF DEFORMATION:
    //   F_visc = eta_m * (dλ/dt)
    //          = eta_m * (L_current - L_previous) / (L0 * dt)
    //
    // This is physically correct because:
    //   - Slow deformation (squeeze through gap) → small dλ/dt → weak resistance
    //   - Rapid deformation (collision) → large dλ/dt → strong resistance
    //   - Prevents blow-up while allowing large static deformation
    //
    // Parameters:
    //   eta_membrane_: membrane viscosity coefficient (units: force * time / length)
    //                  Higher = more viscous, slower deformation response
    //                  Typical: 0.01-0.1 in lattice units
    //
    // Reference: Fedosov et al., Biophys J 2010 (membrane viscosity for RBCs)

    if (eta_membrane_ <= 0.0) return;

    // First call: initialize prev_lengths_ from current state
    if (!has_prev_lengths_) {
        for (int i = 0; i < num_nodes_; ++i) {
            Vec2d diff = minImageDiff(nodes_[next(i)], nodes_[i]);
            prev_lengths_[i] = diff.norm();
        }
        has_prev_lengths_ = true;
        return; // no viscous force on first call (no rate yet)
    }

    for (int i = 0; i < num_nodes_; ++i) {
        int j = next(i);
        Vec2d diff = minImageDiff(nodes_[j], nodes_[i]);
        Real L = diff.norm();
        if (L < 1e-15) continue;

        Real L0 = rest_lengths_[i];
        Real L_prev = prev_lengths_[i];

        // Strain rate: d(lambda)/dt ≈ (L - L_prev) / (L0 * dt)
        // With dt = 1 in LBM lattice units:
        Real strain_rate = (L - L_prev) / L0;

        // Viscous force along edge direction (opposes stretch/compression rate)
        Real F_visc = eta_membrane_ * strain_rate;

        Vec2d dir = diff / L;
        Vec2d F = dir * F_visc;

        forces_[i] += F;
        forces_[j] -= F;

        // Update previous length for next timestep
        prev_lengths_[i] = L;
    }
}

void Capsule::applyBuoyancyForce(Real gx, Real gy, Real rho_fluid) {
    // Net buoyancy: F = (rho_capsule - rho_fluid) * A * g
    // Distributed equally across membrane nodes, then spread to fluid via IBM.
    // When density_ == rho_fluid (default 1.0), force is zero (massless IBM).
    // Reference: Feng & Michaelides, J. Comput. Phys. 195, 602-628 (2004)
    Real delta_rho = density_ - rho_fluid;
    if (std::abs(delta_rho) < 1e-12) return;

    Real A = area();
    Real inv_n = 1.0 / static_cast<Real>(num_nodes_);
    Real fx = delta_rho * A * gx * inv_n;
    Real fy = delta_rho * A * gy * inv_n;

    for (int i = 0; i < num_nodes_; ++i) {
        forces_[i].x += fx;
        forces_[i].y += fy;
    }
}

} // namespace softflow
