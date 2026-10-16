#pragma once

#include "../types.h"

#include <random>
#include <vector>

namespace softflow::insertion {

/// Sample radii for inserters. Decoupled from the placement strategy
/// so that, e.g., a hex-lattice + bidisperse fill is the same code
/// path as a hex-lattice + log-normal fill — only the SizeDistribution
/// changes. minRadius() and maxRadius() are used by inserters to set
/// safety margins (no-overlap and no-wall checks must use the
/// worst-case radius).
class ISizeDistribution {
public:
    virtual ~ISizeDistribution() = default;

    /// Draw one radius. Implementations must be deterministic in the
    /// rng state so seeded runs reproduce bit-exact.
    virtual Real sample(std::mt19937_64& rng) const = 0;

    /// Tightest lower / upper bound on values that sample() can
    /// return. For unbounded distributions (log-normal), implementers
    /// should clamp internally and report the clamp values here.
    virtual Real minRadius() const = 0;
    virtual Real maxRadius() const = 0;
};

/// All draws return the same radius. Useful for V&V and the simplest
/// default.
class Monodisperse final : public ISizeDistribution {
public:
    explicit Monodisperse(Real radius);

    Real sample(std::mt19937_64& rng) const override;
    Real minRadius() const override { return r_; }
    Real maxRadius() const override { return r_; }

private:
    Real r_;
};

/// Two discrete sizes mixed with a prescribed fraction. Each draw
/// returns r_small with probability `fraction_small`, r_large
/// otherwise. The classic substrate for segregation studies — see
/// `examples/02_bidisperse_segregation/`.
class Bidisperse final : public ISizeDistribution {
public:
    Bidisperse(Real r_small, Real r_large, Real fraction_small);

    Real sample(std::mt19937_64& rng) const override;
    Real minRadius() const override;
    Real maxRadius() const override;

private:
    Real r_small_, r_large_, fraction_small_;
};

/// Log-normal distribution (Aitchison & Brown 1957). Parameterised by
/// the underlying Gaussian's mean μ and standard deviation σ in
/// log-space, plus hard truncation bounds [r_min, r_max] so that
/// inserters can rely on a finite worst-case radius for safety
/// margins. Draws outside the bounds are resampled.
class Lognormal final : public ISizeDistribution {
public:
    Lognormal(Real mu_log, Real sigma_log, Real r_min, Real r_max);

    Real sample(std::mt19937_64& rng) const override;
    Real minRadius() const override { return r_min_; }
    Real maxRadius() const override { return r_max_; }

private:
    Real mu_log_, sigma_log_, r_min_, r_max_;
};

/// User-supplied discrete distribution: a sorted vector of (radius,
/// weight) pairs. Weights need not be normalised; the sampler
/// constructs a `std::discrete_distribution` internally. This covers
/// the "arbitrary user PDF" case (the project specification) for any
/// distribution the user can quantise into bins; a continuous
/// callable PDF requires the Python binding (step 8).
class UserDiscrete final : public ISizeDistribution {
public:
    UserDiscrete(std::vector<Real> radii, std::vector<Real> weights);

    Real sample(std::mt19937_64& rng) const override;
    Real minRadius() const override { return r_min_; }
    Real maxRadius() const override { return r_max_; }

private:
    std::vector<Real>             radii_;
    Real                          r_min_, r_max_;
    // mutable because std::discrete_distribution::operator() is non-const
    mutable std::discrete_distribution<int> dist_;
};

} // namespace softflow::insertion
