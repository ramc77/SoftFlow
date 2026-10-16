#include "size_distribution.h"

#include <algorithm>
#include <stdexcept>

namespace softflow::insertion {

// ─── Monodisperse ────────────────────────────────────────────────────

Monodisperse::Monodisperse(Real radius) : r_(radius) {
    if (!(radius > 0.0)) {
        throw std::invalid_argument("Monodisperse: radius must be > 0");
    }
}

Real Monodisperse::sample(std::mt19937_64& /*rng*/) const {
    return r_;
}

// ─── Bidisperse ──────────────────────────────────────────────────────

Bidisperse::Bidisperse(Real r_small, Real r_large, Real fraction_small)
    : r_small_(r_small), r_large_(r_large), fraction_small_(fraction_small)
{
    if (!(r_small > 0.0))    throw std::invalid_argument("Bidisperse: r_small must be > 0");
    if (!(r_large > 0.0))    throw std::invalid_argument("Bidisperse: r_large must be > 0");
    if (!(fraction_small >= 0.0 && fraction_small <= 1.0)) {
        throw std::invalid_argument("Bidisperse: fraction_small must be in [0, 1]");
    }
}

Real Bidisperse::sample(std::mt19937_64& rng) const {
    std::uniform_real_distribution<Real> u(0.0, 1.0);
    return (u(rng) < fraction_small_) ? r_small_ : r_large_;
}

Real Bidisperse::minRadius() const { return std::min(r_small_, r_large_); }
Real Bidisperse::maxRadius() const { return std::max(r_small_, r_large_); }

// ─── Lognormal with hard truncation ─────────────────────────────────

Lognormal::Lognormal(Real mu_log, Real sigma_log, Real r_min, Real r_max)
    : mu_log_(mu_log), sigma_log_(sigma_log), r_min_(r_min), r_max_(r_max)
{
    if (!(sigma_log > 0.0)) throw std::invalid_argument("Lognormal: sigma_log must be > 0");
    if (!(r_min > 0.0))     throw std::invalid_argument("Lognormal: r_min must be > 0");
    if (!(r_max > r_min))   throw std::invalid_argument("Lognormal: r_max must exceed r_min");
}

Real Lognormal::sample(std::mt19937_64& rng) const {
    // std::lognormal_distribution is parameterised by (μ, σ) of the
    // underlying normal, matching our constructor convention. Reject-
    // and-resample is fine here because in practice users set
    // (r_min, r_max) generously around the median.
    std::lognormal_distribution<Real> dist(mu_log_, sigma_log_);
    constexpr int kMaxAttempts = 100;
    for (int i = 0; i < kMaxAttempts; ++i) {
        const Real r = dist(rng);
        if (r >= r_min_ && r <= r_max_) return r;
    }
    // Truncation bounds too tight — clamp the most recent draw rather
    // than spinning forever. The distribution is no longer exactly
    // log-normal at the bounds but the run still proceeds.
    return std::clamp(dist(rng), r_min_, r_max_);
}

// ─── UserDiscrete ────────────────────────────────────────────────────

UserDiscrete::UserDiscrete(std::vector<Real> radii, std::vector<Real> weights)
    : radii_(std::move(radii))
{
    if (radii_.empty()) {
        throw std::invalid_argument("UserDiscrete: radii must be non-empty");
    }
    if (radii_.size() != weights.size()) {
        throw std::invalid_argument(
            "UserDiscrete: radii and weights must have the same size");
    }
    for (Real r : radii_) {
        if (!(r > 0.0)) throw std::invalid_argument("UserDiscrete: every radius must be > 0");
    }
    Real wsum = 0.0;
    for (Real w : weights) {
        if (w < 0.0) throw std::invalid_argument("UserDiscrete: weights must be ≥ 0");
        wsum += w;
    }
    if (!(wsum > 0.0)) {
        throw std::invalid_argument("UserDiscrete: at least one weight must be > 0");
    }

    r_min_ = *std::min_element(radii_.begin(), radii_.end());
    r_max_ = *std::max_element(radii_.begin(), radii_.end());
    dist_  = std::discrete_distribution<int>(weights.begin(), weights.end());
}

Real UserDiscrete::sample(std::mt19937_64& rng) const {
    return radii_[static_cast<std::size_t>(dist_(rng))];
}

} // namespace softflow::insertion
