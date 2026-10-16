#pragma once
#include "../core/types.h"
#include <vector>

namespace softflow {

// Input features for ML force prediction per particle
struct MLFeatures {
    Real rel_vel_x, rel_vel_y;  // particle-fluid relative velocity
    Real Re_p;                   // particle Reynolds number
    Real local_rho;              // local fluid density
    Real grad_rho_x, grad_rho_y; // density gradient
    Real wall_distance;           // distance to nearest wall
    Real radius_normalized;       // radius / channel width
    Real local_solid_fraction;    // nearby particle density

    std::vector<Real> toVector() const {
        return {rel_vel_x, rel_vel_y, Re_p, local_rho,
                grad_rho_x, grad_rho_y, wall_distance,
                radius_normalized, local_solid_fraction};
    }
    static constexpr int SIZE = 9;
};

class SurrogateModel {
public:
    virtual ~SurrogateModel() = default;
    virtual Vec2d predictForce(const MLFeatures& features) = 0;
    virtual void train(const std::vector<MLFeatures>& inputs,
                       const std::vector<Vec2d>& targets) = 0;
    virtual bool isReady() const = 0;
};

} // namespace softflow
