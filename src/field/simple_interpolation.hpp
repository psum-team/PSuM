#ifndef PSUM_FIELD_SIMPLE_INTERPOLATION_HPP
#define PSUM_FIELD_SIMPLE_INTERPOLATION_HPP

#include "foundation.hpp"

namespace psum {

namespace field {

namespace simple_interpolation {

    template<int SupportSize>
    using interp_coeffs = std::array<double, SupportSize>;

    template<auto FuncPtr>
    struct diff_map {}; 

    // match a simple function like linear_interp_1D, with input as a double and output as interp_coeffs
    template <auto Func1D, int SupportSize1D>
    concept FuncWithSupportSize1D = requires(double x) {
        { Func1D(x) } -> std::convertible_to<interp_coeffs<SupportSize1D>>;
    };

    // compute Base^Exp in compile time
    template <int Base, int Exp>
    constexpr int int_pow() {
        int r = 1;
        for (int i = 0; i < Exp; ++i) r *= Base;
        return r;
    }

    // extend 1D interpolation function to N-D
    template <int SupportSize1D, int Dim, auto Func1D, typename VecDim>
    requires (FuncWithSupportSize1D<Func1D, SupportSize1D> && foundation::array_like<VecDim, Dim>)
    interp_coeffs<int_pow<SupportSize1D, Dim>()> interp_tensor(const VecDim& x) {
        constexpr int SupportSize = int_pow<SupportSize1D, Dim>();
        interp_coeffs<SupportSize> res;
        std::array<interp_coeffs<SupportSize1D>, Dim> coefs_on_each_dim;
        for (int i = 0; i < Dim; ++i)
            coefs_on_each_dim[i] = Func1D(x[i]);
        for (int i = 0; i < SupportSize; ++i) {
            int idx = i;
            double coef = 1;
            for (int j = 0; j < Dim; ++j) {
                // 'idx % SupportSize1D' will transform i in base-SupportSize1D
                // such as: i = 6, SupportSize1D = 2, 'idx % SupportSize1D' = [0, 1, 1] ('110' in base-2 is 6)
                coef *= coefs_on_each_dim[Dim - 1 - j][idx % SupportSize1D];
                idx /= SupportSize1D;
            }
            res[i] = coef;
        }
        return res;
    }

    // compute differential of N-D interpolation function
    // c_dx is the factor to multiply the differential of each dimension
    template <int SupportSize1D, int Dim, auto Func1D, typename VecDim>
    requires (FuncWithSupportSize1D<Func1D, SupportSize1D> && foundation::array_like<VecDim, Dim> &&
              FuncWithSupportSize1D<diff_map<Func1D>::value, SupportSize1D>)
    std::array<interp_coeffs<int_pow<SupportSize1D, Dim>()>, Dim> interp_tensor_diff(const VecDim& x, const VecDim& c_dx) {
        constexpr int SupportSize = int_pow<SupportSize1D, Dim>();
        std::array<interp_coeffs<SupportSize>, Dim> res;
        std::array<interp_coeffs<SupportSize1D>, Dim> coefs_on_each_dim;
        std::array<interp_coeffs<SupportSize1D>, Dim> diffs_on_each_dim;
        for (int i = 0; i < Dim; ++i)
            coefs_on_each_dim[i] = Func1D(x[i]);
        for (int i = 0; i < Dim; ++i) {
            diffs_on_each_dim[i] = diff_map<Func1D>::value(x[i]);
            for (int j = 0; j < SupportSize1D; ++j) {
                diffs_on_each_dim[i][j] *= c_dx[i];
            }
        }
        for (int d = 0; d < Dim; ++d) {
            for (int i = 0; i < SupportSize; ++i) {
                int idx = i;
                double coef = 1;
                for (int j = 0; j < Dim; ++j) {
                    if (Dim - 1 - j == d) {
                        coef *= diffs_on_each_dim[Dim - 1 - j][idx % SupportSize1D];
                    } else {
                        coef *= coefs_on_each_dim[Dim - 1 - j][idx % SupportSize1D];
                    }
                    idx /= SupportSize1D;
                }
                res[d][i] = coef;
            }
        }
        return res;
    }

    inline interp_coeffs<2> linear_interp_1D(double x) {
        return interp_coeffs<2>{1 - x, x};
    }

    inline interp_coeffs<2> diff_linear_interp_1D(double x) {
        return interp_coeffs<2>{-1, 1};
    }

    template<>
    struct diff_map<&linear_interp_1D> {
        static constexpr auto value = &diff_linear_interp_1D;
    };

}

}

}

#endif