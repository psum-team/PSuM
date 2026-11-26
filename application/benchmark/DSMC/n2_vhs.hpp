#pragma once

#include <cmath>

namespace n2 {
inline double vhs_cross_section(double relative_speed_squared) {
    if (relative_speed_squared == 0) return 0;
    constexpr double boltzmann = 1.380649e-23;
    constexpr double reduced_mass = 0.5 * 4.652e-26;
    constexpr double diameter = 4.17e-10;
    constexpr double reference_temperature = 273;
    constexpr double omega = 0.74;
    return std::acos(-1.0) * diameter * diameter
        * std::pow(2 * boltzmann * reference_temperature / (reduced_mass * relative_speed_squared), omega - 0.5)
        / std::tgamma(2.5 - omega);
}
}
