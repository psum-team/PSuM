#ifndef PSUM_HIGHER_ORDER_INTERP_HPP
#define PSUM_HIGHER_ORDER_INTERP_HPP

#include "simple_grid.hpp"
#include "field_interp.hpp"
#include "device_field.hpp"
#include "host_field.hpp"

namespace psum {

namespace field {

    namespace si = simple_interpolation;

    // Provide interp, interp-diff, and add-back functions based on 2nd/3rd/4th order B-splines
    // for host_field and device_field (accessors) alike.

    namespace detail {
        template <typename T>
        concept device_node_acc = std::is_same_v<
            typename device_field<T::Dimension, var_loc::nodeCentered, typename T::Scalar, T::QuantitySize>::acc_type,
            std::remove_cvref_t<T>>;

        template <typename T>
        concept host_node_field = std::is_same_v<
            host_field<T::Dimension, var_loc::nodeCentered, typename T::Scalar, T::QuantitySize>,
            std::remove_cvref_t<T>>;

        template <typename T>
        concept node_field_acc = device_node_acc<T> || host_node_field<T>;

        template <typename Position, typename FieldAcc>
        concept interp_args = foundation::array_like<Position, FieldAcc::Dimension> &&
             node_field_acc<FieldAcc>;

        // Order routers: one constexpr closure per interpolation kind, closing
        // gen_interp / gen_interp_diff's remaining VecDim template parameter.
        //     2 → <3, &quadratic_bspline_1D>,  3 → <4, &cubic_bspline_1D>, 4 → <5, &quartic_bspline_1D>
        template <int Order>
        constexpr auto interp_router = [](const auto& g, const auto& p) {
            if constexpr (Order == 2)
                return si::gen_interp<3, std::decay_t<decltype(g)>::Dimension, &si::quadratic_bspline_1D>(g, p);
            else if constexpr (Order == 3)
                return si::gen_interp<4, std::decay_t<decltype(g)>::Dimension, &si::cubic_bspline_1D>(g, p);
            else if constexpr (Order == 4)
                return si::gen_interp<5, std::decay_t<decltype(g)>::Dimension, &si::quartic_bspline_1D>(g, p);
        };

        template <int Order>
        constexpr auto diff_interp_router = [](const auto& g, const auto& p) {
            if constexpr (Order == 2)
                return si::gen_interp_diff<3, std::decay_t<decltype(g)>::Dimension, &si::quadratic_bspline_1D>(g, p);
            else if constexpr (Order == 3)
                return si::gen_interp_diff<4, std::decay_t<decltype(g)>::Dimension, &si::cubic_bspline_1D>(g, p);
            else if constexpr (Order == 4)
                return si::gen_interp_diff<5, std::decay_t<decltype(g)>::Dimension, &si::quartic_bspline_1D>(g, p);
        };

        template <int Order, typename Position, typename FieldAcc>
        decltype(auto) interp_impl(const Position& pos, const FieldAcc& f) {
            return interp_tools::generate_interp<Position, FieldAcc, interp_router<Order>>(pos, f);
        }

        template <int Order, typename Position, typename FieldAcc>
        decltype(auto) interp_diff_impl(const Position& pos, const FieldAcc& f) {
            return interp_tools::generate_interp_diff<Position, FieldAcc, diff_interp_router<Order>>(pos, f);
        }

        template <int Order, typename Position, typename FieldAcc>
        decltype(auto) add_back_impl(const Position& pos, const typename std::remove_cvref_t<FieldAcc>::Value& w, FieldAcc&& f) {
            using Acc = std::remove_cvref_t<FieldAcc>;
            if constexpr (device_node_acc<Acc>)
                return interp_tools::generate_add_back<Position, Acc, interp_router<Order>>(pos, w, f);
            else
                return interp_tools::generate_add_back_mutable<Position, Acc, interp_router<Order>>(pos, w, f);
        }
    }

    // ---------------- quadratic interp ----------------
    template <typename Position, typename FieldAcc> requires detail::interp_args<Position, FieldAcc>
    decltype(auto) interp_2nd(const Position& pos, const FieldAcc& f) {
        return detail::interp_impl<2>(pos, f);
    }

    template <typename FieldAcc> requires detail::node_field_acc<FieldAcc>
    decltype(auto) interp_2nd(const double (&pos)[FieldAcc::Dimension], const FieldAcc& f) {
        return interp_2nd(std::to_array(pos), f);
    }

    template <typename Position, typename FieldAcc> requires detail::interp_args<Position, FieldAcc>
    decltype(auto) interp_diff_2nd(const Position& pos, const FieldAcc& f) {
        return detail::interp_diff_impl<2>(pos, f);
    }

    template <typename FieldAcc> requires detail::node_field_acc<FieldAcc>
    decltype(auto) interp_diff_2nd(const double (&pos)[FieldAcc::Dimension], const FieldAcc& f) {
        return interp_diff_2nd(std::to_array(pos), f);
    }

    template <typename Position, typename Acc> requires detail::interp_args<Position, Acc>
    decltype(auto) add_back_2nd(const Position& pos, const typename Acc::Value& w, const Acc& acc) {
        return detail::add_back_impl<2>(pos, w, acc);
    }

    template <typename Acc> requires detail::device_node_acc<Acc>
    decltype(auto) add_back_2nd(const double (&pos)[Acc::Dimension], const typename Acc::Value& w, const Acc& acc) {
        return add_back_2nd(std::to_array(pos), w, acc);
    }

    template <typename Position, typename Field> requires detail::interp_args<Position, Field>
    decltype(auto) add_back_2nd(const Position& pos, const typename Field::Value& w, Field& f) {
        return detail::add_back_impl<2>(pos, w, f);
    }

    template <typename Field> requires detail::host_node_field<Field>
    decltype(auto) add_back_2nd(const double (&pos)[Field::Dimension], const typename Field::Value& w, Field& f) {
        return add_back_2nd(std::to_array(pos), w, f);
    }

    // ---------------- cubic interp ----------------
    template <typename Position, typename FieldAcc> requires detail::interp_args<Position, FieldAcc>
    decltype(auto) interp_3rd(const Position& pos, const FieldAcc& f) {
        return detail::interp_impl<3>(pos, f);
    }

    template <typename FieldAcc> requires detail::node_field_acc<FieldAcc>
    decltype(auto) interp_3rd(const double (&pos)[FieldAcc::Dimension], const FieldAcc& f) {
        return interp_3rd(std::to_array(pos), f);
    }

    template <typename Position, typename FieldAcc> requires detail::interp_args<Position, FieldAcc>
    decltype(auto) interp_diff_3rd(const Position& pos, const FieldAcc& f) {
        return detail::interp_diff_impl<3>(pos, f);
    }

    template <typename FieldAcc> requires detail::node_field_acc<FieldAcc>
    decltype(auto) interp_diff_3rd(const double (&pos)[FieldAcc::Dimension], const FieldAcc& f) {
        return interp_diff_3rd(std::to_array(pos), f);
    }

    template <typename Position, typename Acc> requires detail::interp_args<Position, Acc>
    decltype(auto) add_back_3rd(const Position& pos, const typename Acc::Value& w, const Acc& acc) {
        return detail::add_back_impl<3>(pos, w, acc);
    }

    template <typename Acc> requires detail::device_node_acc<Acc>
    decltype(auto) add_back_3rd(const double (&pos)[Acc::Dimension], const typename Acc::Value& w, const Acc& acc) {
        return add_back_3rd(std::to_array(pos), w, acc);
    }

    template <typename Position, typename Field> requires detail::interp_args<Position, Field>
    decltype(auto) add_back_3rd(const Position& pos, const typename Field::Value& w, Field& f) {
        return detail::add_back_impl<3>(pos, w, f);
    }

    template <typename Field> requires detail::host_node_field<Field>
    decltype(auto) add_back_3rd(const double (&pos)[Field::Dimension], const typename Field::Value& w, Field& f) {
        return add_back_3rd(std::to_array(pos), w, f);
    }


    // ---------------- quartic interp ----------------
    template <typename Position, typename FieldAcc>
    requires detail::interp_args<Position, FieldAcc>
    decltype(auto) interp_4th(const Position& pos, const FieldAcc& f) {
        return detail::interp_impl<4>(pos, f);
    }

    template <typename FieldAcc>
    requires detail::node_field_acc<FieldAcc>
    decltype(auto) interp_4th(const double (&pos)[FieldAcc::Dimension], const FieldAcc& f) {
        return interp_4th(std::to_array(pos), f);
    }

    template <typename Position, typename FieldAcc>
    requires detail::interp_args<Position, FieldAcc>
    decltype(auto) interp_diff_4th(const Position& pos, const FieldAcc& f) {
        return detail::interp_diff_impl<4>(pos, f);
    }

    template <typename FieldAcc>
    requires detail::node_field_acc<FieldAcc>
    decltype(auto) interp_diff_4th(const double (&pos)[FieldAcc::Dimension], const FieldAcc& f) {
        return interp_diff_4th(std::to_array(pos), f);
    }

    template <typename Position, typename Acc> requires detail::interp_args<Position, Acc>
    decltype(auto) add_back_4th(const Position& pos, const typename Acc::Value& w, const Acc& acc) {
        return detail::add_back_impl<4>(pos, w, acc);
    }

    template <typename Acc> requires detail::device_node_acc<Acc>
    decltype(auto) add_back_4th(const double (&pos)[Acc::Dimension], const typename Acc::Value& w, const Acc& acc) {
        return add_back_4th(std::to_array(pos), w, acc);
    }

    template <typename Position, typename Field> requires detail::interp_args<Position, Field>
    decltype(auto) add_back_4th(const Position& pos, const typename Field::Value& w, Field& f) {
        return detail::add_back_impl<4>(pos, w, f);
    }

    template <typename Field> requires detail::host_node_field<Field>
    decltype(auto) add_back_4th(const double (&pos)[Field::Dimension], const typename Field::Value& w, Field& f) {
        return add_back_4th(std::to_array(pos), w, f);
    }

    // ---------------- B-spline order facade ----------------
    // Unconstrained pure forwarders whose template parameter is the B-spline order:
    //     1 → the existing linear implementations (device_field.hpp / host_field.hpp)
    //     2 / 3 / 4 → the higher-order implementations above
    template <int Order, typename Position, typename FieldAcc>
    decltype(auto) interp_B_spline(const Position& pos, const FieldAcc& f) {
        if constexpr (Order == 1) return interp(pos, f);
        else if constexpr (Order == 2) return interp_2nd(pos, f);
        else if constexpr (Order == 3) return interp_3rd(pos, f);
        else if constexpr (Order == 4) return interp_4th(pos, f);
    }

    template <int Order, typename FieldAcc>
    decltype(auto) interp_B_spline(const double (&pos)[FieldAcc::Dimension], const FieldAcc& f) {
        return interp_B_spline<Order>(std::to_array(pos), f);
    }

    template <int Order, typename Position, typename FieldAcc>
    decltype(auto) interp_diff_B_spline(const Position& pos, const FieldAcc& f) {
        if constexpr (Order == 1) return interp_diff(pos, f);
        else if constexpr (Order == 2) return interp_diff_2nd(pos, f);
        else if constexpr (Order == 3) return interp_diff_3rd(pos, f);
        else if constexpr (Order == 4) return interp_diff_4th(pos, f);
    }

    template <int Order, typename FieldAcc>
    decltype(auto) interp_diff_B_spline(const double (&pos)[FieldAcc::Dimension], const FieldAcc& f) {
        return interp_diff_B_spline<Order>(std::to_array(pos), f);
    }

    template <int Order, typename Position, typename FieldAcc>
    decltype(auto) add_back_B_spline(const Position& pos, const typename std::remove_cvref_t<FieldAcc>::Value& w, FieldAcc&& f) {
        if constexpr (Order == 1) return add_back(pos, w, std::forward<FieldAcc>(f));
        else if constexpr (Order == 2) return add_back_2nd(pos, w, std::forward<FieldAcc>(f));
        else if constexpr (Order == 3) return add_back_3rd(pos, w, std::forward<FieldAcc>(f));
        else if constexpr (Order == 4) return add_back_4th(pos, w, std::forward<FieldAcc>(f));
    }

    template <int Order, typename FieldAcc>
    decltype(auto) add_back_B_spline(const double (&pos)[std::remove_cvref_t<FieldAcc>::Dimension], const typename std::remove_cvref_t<FieldAcc>::Value& w, FieldAcc&& f) {
        return add_back_B_spline<Order>(std::to_array(pos), w, std::forward<FieldAcc>(f));
    }
}
}

#endif
