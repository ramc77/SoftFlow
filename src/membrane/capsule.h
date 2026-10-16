#pragma once
#include "../core/types.h"
#include "../core/parameters.h"
#include <vector>
#include <cmath>

namespace softflow {

class Capsule {
public:
    /// Compute optimal number of membrane nodes for IBM coupling.
    /// Uses ds ≈ 0.75 Δx  (Peskin, Acta Numerica 2002: ds ≤ Δx for 4-pt delta).
    /// Minimum 12 nodes to maintain polygon quality.
    static int computeOptimalNodes(Real radius, Real target_ds = 0.75) {
        int n = static_cast<int>(std::ceil(2.0 * PI * radius / target_ds));
        return std::max(n, 12);
    }

    /// If num_nodes <= 0, automatically computed from radius.
    Capsule(int id, Vec2d center, Real radius, int num_nodes,
            const MembraneParams& params, int type = 0);

    void clearForces();
    void computeMembraneForces();
    Vec2d centroid() const;
    Real area() const;
    Real perimeter() const;
    Real deformationIndex() const;   // equivalent-ellipse Taylor D
    Real inclinationAngle() const;   // major axis vs x, (-pi/2, pi/2]
    Real bendingEnergy() const;      // discrete elastica, 0 if not enabled
    void moveNodes(Real dt);

    /// Enable per-node periodic wrapping in x.
    void setPeriodicX(int nx) { periodic_nx_ = nx; }
    int  getPeriodicX() const { return periodic_nx_; }

    int numNodes() const { return num_nodes_; }
    int getId() const { return id_; }
    int getType() const { return type_id_; }
    int getTypeId() const { return type_id_; }

    // Per-node accessors (used by IBM coupling and I/O)
    Vec2d nodePosition(int k) const { return nodes_[k]; }
    Vec2d nodeVelocity(int k) const { return velocities_[k]; }
    Vec2d nodeForce(int k) const { return forces_[k]; }
    void setNodeVelocity(int k, const Vec2d& v) { velocities_[k] = v; }

    // Effective radius (from rest area: A = pi*r^2)
    Real effectiveRadius() const {
        return std::sqrt(rest_area_ / PI);
    }

    // Membrane model accessor
    MembraneModel getMembraneModel() const { return model_; }

    // Viscosity ratio for this capsule
    Real getViscosityRatio() const { return viscosity_ratio_; }

    // Density and buoyancy (Feng & Michaelides 2004, Aidun & Clausen 2010)
    void setDensity(Real d) { density_ = d; }
    Real getDensity() const { return density_; }
    Real enclosedArea() const { return area(); }  // Shoelace formula (reuses area())
    Real mass() const { return density_ * area(); }
    void applyBuoyancyForce(Real gx, Real gy, Real rho_fluid);

    // Direct access to node data vectors
    std::vector<Vec2d>& positions() { return nodes_; }
    const std::vector<Vec2d>& positions() const { return nodes_; }
    std::vector<Vec2d>& velocities() { return velocities_; }
    const std::vector<Vec2d>& velocities() const { return velocities_; }
    std::vector<Vec2d>& forces() { return forces_; }
    const std::vector<Vec2d>& forces() const { return forces_; }

private:
    void secondMoments(Real& mxx, Real& myy, Real& mxy) const;

    std::vector<Vec2d> nodes_;
    std::vector<Vec2d> velocities_;
    std::vector<Vec2d> forces_;
    int num_nodes_;
    Real rest_area_;
    Real rest_perimeter_;
    std::vector<Real> rest_lengths_;
    std::vector<Real> rest_angles_;     // discrete-elastica rest turning angles
    std::vector<Real> rest_ds_;         // rest length associated with each node

    // Membrane model
    MembraneModel model_;
    Real k_stretch_;       // Hookean spring constant
    Real G_s_;             // surface shear modulus (Neo-Hookean / Skalak)
    Real C_skalak_;        // Skalak area dilation ratio

    // Bending
    Real k_bend_;
    bool use_helfrich_;
    Real kappa_0_;         // spontaneous curvature

    // Conservation / damping
    Real k_area_;
    Real k_perimeter_;
    Real gamma_visc_;       // translational damping
    Real eta_membrane_;     // Kelvin-Voigt membrane viscosity (strain-rate damping)

    // Previous edge lengths for Kelvin-Voigt viscoelastic model
    std::vector<Real> prev_lengths_;
    bool has_prev_lengths_ = false;

    // Viscosity contrast
    Real viscosity_ratio_;

    // Density (1.0 = neutrally buoyant)
    Real density_;

    // Shape
    CapsuleShape shape_;
    Real         aspect_ratio_;
    Real         indent_depth_;
    bool         is_rigid_;

    int type_id_;
    int id_;
    int periodic_nx_ = 0;  // 0 = no periodic wrapping

    // Helpers for wrapping indices on the closed ring
    int prev(int i) const { return (i - 1 + num_nodes_) % num_nodes_; }
    int next(int i) const { return (i + 1) % num_nodes_; }

    // Minimum-image difference vector in x (for periodic BC)
    Vec2d minImageDiff(const Vec2d& a, const Vec2d& b) const {
        Vec2d d = a - b;
        if (periodic_nx_ > 0) {
            Real Lx = static_cast<Real>(periodic_nx_);
            if (d.x >  0.5 * Lx) d.x -= Lx;
            if (d.x < -0.5 * Lx) d.x += Lx;
        }
        return d;
    }

    // WLC parameters (computed at construction)
    Real wlc_L_max_ratio_;
    Real wlc_kBT_p_;     // kBT / persistence length
    Real wlc_k_pow_;     // repulsive power-law coefficient

    // Force computation methods
    void computeStretchingForces();
    void computeStretchingForcesHookean();
    void computeStretchingForcesNeoHookean();
    void computeStretchingForcesSkalak();
    void computeStretchingForcesWLC();
    void computeBendingForces();
    void computeBendingForcesLaplacian();
    void computeBendingForcesHelfrich();
    void computeAreaForces();
    void computePerimeterForces();
    void computeViscousDamping();
    void computeMembraneViscosity();  // Kelvin-Voigt viscoelastic damping
};

} // namespace softflow
