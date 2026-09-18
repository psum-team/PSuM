#ifndef PSUM_TAG_PROPERTY_HPP
#define PSUM_TAG_PROPERTY_HPP

#include <cstdint>
#include "tagged_struct.hpp"

namespace psum {

namespace tag {

namespace tag_concepts {
    template <typename Type>
    concept convertible_arithmetic = 
        std::convertible_to<Type, long long> ||
        std::convertible_to<Type, unsigned long long> ||
        std::convertible_to<Type, double>;

    template <typename V>
    concept has_floating_x = requires(V v) {
        { v.x() } -> std::convertible_to<double>;
    };

    template <typename V>
    concept has_floating_functor = requires(V v) {
        { v() } -> std::convertible_to<double>;
    };
}

namespace property {
    
    struct mass: foundation::abstract_tag {
        inline const static std::string tag_name = "mass";
        template <typename Type>
        static constexpr bool check() {
            return std::convertible_to<Type, double>;
        }
    };

    struct charge: foundation::abstract_tag {
        inline const static std::string tag_name = "charge";
        template <typename Type>
        static constexpr bool check() {
            return std::convertible_to<Type, double>;
        }
    };

    struct weight: foundation::abstract_tag {
        inline const static std::string tag_name = "weight";
        template <typename Type>
        static constexpr bool check() {
            return tag_concepts::convertible_arithmetic<Type>;
        }
    };

    struct position: foundation::abstract_tag {
        inline const static std::string tag_name = "position";
        template <typename Type>
        static constexpr bool check() {
            static_assert(tag_concepts::has_floating_x<Type>, "Position tag requires a type with x() method.");
            return tag_concepts::has_floating_x<Type>;
        }
    };

    struct velocity: foundation::abstract_tag {
        inline const static std::string tag_name = "velocity";
        template <typename Type>
        static constexpr bool check() {
            static_assert(tag_concepts::has_floating_x<Type>, "Velocity tag requires a type with x() method.");
            return tag_concepts::has_floating_x<Type>;
        }
    };

    struct acceleration: foundation::abstract_tag {
        inline const static std::string tag_name = "acceleration";
        template <typename Type>
        static constexpr bool check() {
            static_assert(tag_concepts::has_floating_x<Type>, "Acceleration tag requires a type with x() method.");
            return tag_concepts::has_floating_x<Type>;
        }
    };

    struct random_seed: foundation::abstract_tag {
        inline const static std::string tag_name = "random_generator";
        template <typename Type>
        static constexpr bool check() {
            return std::is_same_v<Type,uint32_t> || std::is_same_v<Type,uint64_t> || std::is_same_v<Type, double>;
        }
    };

    struct species_id: foundation::abstract_tag {
        inline const static std::string tag_name = "species_id";
        template <typename Type>
        static constexpr bool check() {
            return std::is_enum_v<Type> || std::is_integral_v<Type>;
        }
    };

    struct internal_energy: foundation::abstract_tag {
        inline const static std::string tag_name = "internal_energy";
        template <typename Type>
        static constexpr bool check() {
            return std::convertible_to<Type, double>;
        }
    };

    struct lock_var: foundation::abstract_tag {
        inline const static std::string tag_name = "lock_var";
    };

}

}

}

#endif