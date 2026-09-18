#ifndef PSUM_MAGNET_NOZZLE_2D_CONFIG_OPS_HPP
#define PSUM_MAGNET_NOZZLE_2D_CONFIG_OPS_HPP

#include "types.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace magnet_nozzle_2d {

struct DerivedParameters2D {
    double gamma = 1.0;
    double epsilon = epsilon_0;
    double lambda = 0.0;
    double plasma_period_scale = 0.0;
    size_t I = 0;
    size_t J = 0;
    double dx = 0.0;
    double dr = 0.0;
    double particle_weight = 0.0;
    double electron_gyro_period = 0.0;
    double dt = 0.0;
    double ion_injection_expectation = 0.0;
};

inline double effective_epsilon_0(double gamma) {
    return gamma * gamma * epsilon_0;
}

inline double load_gamma(psum::serialization::json_loader& param) {
    const double gamma = param.contains<double>("gamma") ? param.obj<double>("gamma") : 1.0;
    if (gamma <= 0.0) throw std::runtime_error("gamma must be positive");
    return gamma;
}

inline bool checkpoint_output_enabled(int save_step, int iter) {
    return save_step > 0 && iter % save_step == 0;
}

inline DerivedParameters2D compute_derived_parameters(
    double gamma,
    double Te,
    double Ti,
    double reference,
    double ppc,
    double time_coef,
    double grid_coef,
    double start_pos,
    double end_pos,
    double r_length,
    double r_inlet,
    double B_max
) {
    if (gamma <= 0.0 || Te <= 0.0 || Ti <= 0.0 || reference <= 0.0 || ppc <= 0.0
        || time_coef <= 0.0 || grid_coef <= 0.0 || start_pos >= end_pos
        || r_length <= 0.0 || r_inlet <= 0.0 || B_max <= 0.0) {
        throw std::runtime_error("invalid 2D derived-parameter input");
    }

    DerivedParameters2D d;
    d.gamma = gamma;
    d.epsilon = effective_epsilon_0(gamma);
    d.lambda = std::sqrt(d.epsilon * Te / reference / Q);
    d.plasma_period_scale = std::sqrt(d.epsilon * m_e / reference / (Q * Q))
        / std::sqrt(2.0);
    d.I = static_cast<size_t>((end_pos - start_pos) / (0.5 * d.lambda) * grid_coef);
    d.J = static_cast<size_t>(r_length / (0.5 * d.lambda) * grid_coef);
    d.I = std::max(d.I, static_cast<size_t>(2));
    d.J = std::max(d.J, static_cast<size_t>(2));
    d.dx = (end_pos - start_pos) / static_cast<double>(d.I);
    d.dr = r_length / static_cast<double>(d.J);

    // The 2D macro-particle represents the inlet cross-section swept over one x cell.
    d.particle_weight = reference * d.dx * M_PI * r_inlet * r_inlet / ppc;
    d.electron_gyro_period = 2.0 * M_PI * m_e / Q / B_max;
    d.dt = time_coef * 0.25 * std::min(d.plasma_period_scale, d.electron_gyro_period);
    d.ion_injection_expectation = 0.5 * reference
        * std::sqrt(8.0 * Q * Ti / M_PI / m_H)
        * M_PI * r_inlet * r_inlet
        * d.dt / d.particle_weight;
    return d;
}

}

#endif
