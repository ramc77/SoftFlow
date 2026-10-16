#pragma once
#include "../core/types.h"
#include "delta_function.h"

namespace softflow {

class LatticeField;
class CapsuleSystem;
class Capsule;

// Immersed Boundary Method coupling between LBM fluid and deformable capsules
// Two operations:
// 1. Spread: distribute membrane forces to fluid lattice nodes
// 2. Interpolate: compute fluid velocity at membrane node positions
class ImmersedBoundary {
public:
    // Spread forces from all capsule membrane nodes onto the LBM lattice
    // F_fluid(x) = sum_k f_k * delta(x - X_k) * ds_k
    // Uses thread-local buffers for OpenMP parallelism.
    void spreadForces(const CapsuleSystem& capsules, LatticeField& field);

    // Interpolate fluid velocity to all membrane node positions
    // U(X_k) = sum_x u(x) * delta(x - X_k) * dx^2
    // OpenMP parallel over capsules (each writes its own velocities).
    void interpolateVelocity(const LatticeField& field, CapsuleSystem& capsules);

    // Multi-direct forcing IBM (Luo et al., Phys. Rev. E 2007)
    // Iteratively corrects forces to better enforce no-slip at membrane.
    // iterations=1 is equivalent to standard IBM.
    void multiDirectForcing(CapsuleSystem& capsules, LatticeField& field,
                            int iterations);

private:
    // Spread force from a single node onto the lattice (direct write)
    void spreadNodeForce(const Vec2d& node_pos, const Vec2d& force, Real ds,
                         LatticeField& field);

    // Spread force from a single node into thread-local buffers
    void spreadNodeForceLocal(const Vec2d& node_pos, const Vec2d& force, Real ds,
                              int nx, int ny,
                              Real* local_Fx, Real* local_Fy);

    // Interpolate velocity at a single point
    Vec2d interpolateAtPoint(const Vec2d& pos, const LatticeField& field);
};

} // namespace softflow
