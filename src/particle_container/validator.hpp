#ifndef PSUM_PARTICLE_CONTAINER_VALIDATOR_HPP
#define PSUM_PARTICLE_CONTAINER_VALIDATOR_HPP

#include <concepts>
#include <sycl/sycl.hpp>
#include "../tag/property.hpp"

namespace psum {

namespace particle_container {
    
    template <typename Validator, typename Particle>
    concept is_validator = requires(const Validator &v, Particle p) {  
        { Validator::make_invalid(p) } -> std::same_as<void>;  
        { Validator::is_valid(std::declval<const Particle&>()) } -> std::convertible_to<bool>;
    };
    
    template<typename Particle>
    struct pos_x_nan_is_invalid {
        static void make_invalid(Particle &in) {
            tag::get<tag::property::position>(in).x() = std::numeric_limits<double>::quiet_NaN();
        }
        static bool is_valid(const Particle &in) {
            return std::isfinite(tag::get<tag::property::position>(in).x());
        }
    };

}

}

#endif