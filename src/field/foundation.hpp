#ifndef PSUM_FIELD_FOUNDATION_HPP
#define PSUM_FIELD_FOUNDATION_HPP

#include <concepts>
#include <array>
#include <ostream>
#include <Eigen/Dense>
#include <type_traits>

namespace psum {

namespace field {

    enum class var_loc {
        cellCentered,
        faceCentered,
        edgeCentered,
        nodeCentered
    };

    enum class grid_element {
        cell,
        face,
        edge,
        node
    };

    constexpr grid_element map_vloc_to_gele(var_loc loc) {
        switch (loc) {
            case var_loc::cellCentered: return grid_element::cell;
            case var_loc::faceCentered: return grid_element::face;
            case var_loc::edgeCentered: return grid_element::edge;
            case var_loc::nodeCentered: return grid_element::node;
            default: return grid_element::node;
        }
    }

    enum class boundary_direction_1d {
        L, R,
        Count
    };

    enum class boundary_direction_2d {
        N, S, E, W,
        Count
    };
    
    enum class boundary_direction_3d {
        Xpos, Xneg, Ypos, Yneg, Zpos, Zneg,
        Count
    };
    
    template <int Dim> struct boundary_direction_traits;
    template <> struct boundary_direction_traits<1> { using type = boundary_direction_1d; };
    template <> struct boundary_direction_traits<2> { using type = boundary_direction_2d; };
    template <> struct boundary_direction_traits<3> { using type = boundary_direction_3d; };

    namespace foundation {
    
        // array_like matches std::array, Eigen::Vector, and Eigen::RowVector
        // array_like define the basic parameter type for field classes
        template<typename T, int Dim>
        concept array_like = 
            std::is_same_v<std::remove_cvref_t<T>, std::array<double, Dim>> ||
            std::is_same_v<std::remove_cvref_t<T>, Eigen::Vector<double, Dim>> ||
            std::is_same_v<std::remove_cvref_t<T>, Eigen::RowVector<double, Dim>>;
    
        // those ceoncepts are used to match interp-parameter type
        // a interp-parameter is a pair of std::array<size_t, N> and std::array<double, N>
        // a interp-diff-parameter is a pair of std::array<size_t, N> and std::array<interp-parameter, Dim>
        template<typename Scalar, typename T>
        struct is_std_array : std::false_type {};

        template<typename Scalar, std::size_t N>
        struct is_std_array<Scalar, std::array<Scalar, N>> : std::true_type {};

        template<typename T>
        concept std_array_of_double = is_std_array<double, std::remove_cvref_t<T>>::value;

        template<typename T>
        concept std_array_of_size_t = is_std_array<size_t, std::remove_cvref_t<T>>::value;

        template <typename T>
        concept interp_param = requires(T t) {
            {t.first} -> std_array_of_size_t;
            {t.second} -> std_array_of_double;
        };

        template<typename T>
        concept std_array_of_interp_param = requires {
            typename std::remove_cvref_t<T>::value_type;
            requires std_array_of_double<typename std::remove_cvref_t<T>::value_type>;
            requires is_std_array<
                typename std::remove_cvref_t<T>::value_type, 
                std::remove_cvref_t<T>
            >::value;
        };

        template<typename T>
        concept interp_diff_param = requires(T t) {
            {t.first} -> std_array_of_size_t;
            {t.second} -> std_array_of_interp_param;
        };
    }

}

}

#endif