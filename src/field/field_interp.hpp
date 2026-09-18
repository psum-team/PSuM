#ifndef PSUM_FIELD_FIELD_INTERP_HPP
#define PSUM_FIELD_FIELD_INTERP_HPP

#include "foundation.hpp"
#include "atomic_add.hpp"

namespace psum {

namespace field {

namespace interp_tools {

    template <typename FieldAcc>
    concept basic_field_acc = requires(FieldAcc acc, size_t i) {
        { FieldAcc::Dimension } -> std::convertible_to<long long int>;
        { FieldAcc::Location } -> std::convertible_to<var_loc>;
        { FieldAcc::QuantitySize } -> std::convertible_to<long long int>;
        typename FieldAcc::Scalar;
        typename FieldAcc::Value;
        { acc.getGrid() };
        { acc(i) } -> std::convertible_to<typename FieldAcc::Value>;
    };

    template <typename Position, typename FieldAcc>
    requires basic_field_acc<FieldAcc> && foundation::array_like<Position, FieldAcc::Dimension>
    void add_back_nearest(const Position& pos, const typename FieldAcc::Value& _w, const FieldAcc& field_acc){
        typename FieldAcc::Value w = _w;
        constexpr auto Gele = map_vloc_to_gele(FieldAcc::Location);
        size_t idx = field_acc.getGrid().template nearest<Gele>(pos);
        _atomic_add_<typename FieldAcc::Scalar, FieldAcc::QuantitySize>(field_acc(idx), w);
    }

    template <typename FieldAcc>
    void add_back_nearest(const double (&pos)[FieldAcc::Dimension], const typename FieldAcc::Value& _w, const FieldAcc& field_acc){
        add_back_nearest(std::to_array(pos), _w, field_acc);
    }

    template <typename Position, typename FieldAcc>
    requires basic_field_acc<FieldAcc> && foundation::array_like<Position, FieldAcc::Dimension>
    decltype(auto) interp_nearest(const Position& pos, const FieldAcc& field_acc){
        constexpr auto Gele = map_vloc_to_gele(FieldAcc::Location);
        size_t idx = field_acc.getGrid().template nearest<Gele>(pos);
        return field_acc(idx);
    }

    template <typename FieldAcc>
    decltype(auto) interp_nearest(const double (&pos)[FieldAcc::Dimension], const FieldAcc& field_acc){
        return interp_nearest(std::to_array(pos), field_acc);
    }

    template <typename Position, typename FieldAcc, auto Interpolator>
    concept field_can_interp = requires(Position pos, FieldAcc acc, sycl::handler &h) {
        { Interpolator(acc.getGrid(), pos) }
            -> foundation::interp_param;
    };

    template <typename Position, typename FieldAcc, auto Interpolator>
    concept field_can_interp_diff = requires(Position pos, FieldAcc acc, sycl::handler &h) {
        { Interpolator(acc.getGrid(), pos) }
            -> foundation::interp_diff_param;
    };

    template <typename Position, typename FieldAcc, auto Interpolator>
    requires field_can_interp<Position, FieldAcc, Interpolator> &&
             foundation::array_like<Position, FieldAcc::Dimension>
    void generate_add_back(const Position& pos, const typename FieldAcc::Value& _w, const FieldAcc& field_acc, double tol = 1e-10) {
        typename FieldAcc::Value w = _w;
        auto interp_param = Interpolator(field_acc.getGrid(), pos);
        for (int i = 0; i < interp_param.first.size(); ++i) {
            if (std::abs(interp_param.second[i]) > tol)
            _atomic_add_<typename FieldAcc::Scalar, FieldAcc::QuantitySize>(
                field_acc(interp_param.first[i]), w, interp_param.second[i]
            );
        }
    }

    template <typename Position, typename FieldAcc, auto Interpolator>
    requires field_can_interp<Position, FieldAcc, Interpolator> &&
             foundation::array_like<Position, FieldAcc::Dimension>
    decltype(auto) generate_interp(const Position& pos, const FieldAcc& field_acc, double tol = 1e-10){
        auto interp_param = Interpolator(field_acc.getGrid(), pos);
        typename FieldAcc::Value res{};
        if constexpr (FieldAcc::QuantitySize > 1) {
            res = FieldAcc::Value::Zero();
        }
        for (int i = 0; i < interp_param.first.size(); ++i) {
            if (std::abs(interp_param.second[i]) > tol)
                res += field_acc(interp_param.first[i]) * interp_param.second[i];
        }
        return res;
    }

    template <typename Position, typename FieldAcc, auto InterpolatorDiff>
    requires field_can_interp_diff<Position, FieldAcc, InterpolatorDiff> &&
             foundation::array_like<Position, FieldAcc::Dimension>
    decltype(auto) generate_interp_diff(const Position& pos, const FieldAcc& field_acc, double tol = 1e-10){
        auto interp_param = InterpolatorDiff(field_acc.getGrid(), pos);
        std::array<typename FieldAcc::Value, FieldAcc::Dimension> res_all{};
        for (int d = 0; d < FieldAcc::Dimension; d++) {
            auto& res = res_all[d];
            auto& param = interp_param.second[d];
            if constexpr (FieldAcc::QuantitySize > 1) {
                res = FieldAcc::Value::Zero();
            } else {
                res = 0;
            }
            for (int i = 0; i < interp_param.first.size(); ++i) {
                if (std::abs(param[i]) > tol)
                    res += field_acc(interp_param.first[i]) * param[i];
            }
        }
        return res_all;
    }
}

using interp_tools::add_back_nearest;
using interp_tools::interp_nearest;

}

}

#endif