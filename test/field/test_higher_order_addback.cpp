#include <psum/field.hpp>
#include <iostream>
#include <string>
#include <cmath>
#include <vector>

using namespace psum;
using namespace psum::field;

// add_back with an explicit interpolation kernel
template <typename Field, typename Pos, auto Interpolator>
void add_back_with(const Pos& p, double w, Field& f) {
    interp_tools::generate_add_back_mutable<Pos, Field, Interpolator>(p, w, f);
}

// one (high-order single deposit) vs (k^2 low-order sub-grid deposits) case
template <typename Field, typename Pos, auto Hi, auto Lo>
void run_case(const grid2D& grid, const Pos& p, double w, double delta,
              const std::vector<int>& ks,
              const std::string& prefix, const std::string& stem_hi, const std::string& stem_lo) {
    // the high-order deposit does not depend on k: compute and plot once
    Field f_hi(grid);
    add_back_with<Field, Pos, Hi>(p, w, f_hi);
    f_hi.plot(prefix + "_" + stem_hi + ".plt");

    for (int k : ks) {
        Field f_lo(grid);
        grid2D sub({p[0] - 0.5 * delta, p[1] - 0.5 * delta},
                   {p[0] + 0.5 * delta, p[1] + 0.5 * delta}, {k, k});
        for (size_t c = 0; c < sub.contentSize(var_loc::cellCentered); c++)
            add_back_with<Field, Pos, Lo>(sub.cellCenter(sub.i2c(c)), w / double(k * k), f_lo);

        Field fdiff(grid);
        double l2 = 0, linf = 0, sum_hi = 0, sum_lo = 0;
        for (size_t idx = 0; idx < fdiff.size(); idx++) {
            fdiff(idx) = f_hi(idx) - f_lo(idx);
            l2 += fdiff(idx) * fdiff(idx);
            linf = std::max(linf, std::abs(fdiff(idx)));
            sum_hi += f_hi(idx);
            sum_lo += f_lo(idx);
        }
        l2 = std::sqrt(l2) * delta;

        const std::string ks = std::to_string(k);
        f_lo.plot(prefix + "_" + stem_lo + "_k" + ks + ".plt");
        fdiff.plot(prefix + "_" + "diff" + "_k" + ks + ".plt");

        std::cout << prefix << " k=" << k
                  << "  |diff|: L2=" << l2 << "  Linf=" << linf
                  << "  (charge: " << stem_hi << "=" << sum_hi << ", " << stem_lo << "=" << sum_lo << ")"
                  << std::endl;
    }
}

int main() {
    constexpr int N = 32;
    grid2D grid({0.0, 0.0}, {1.0, 1.0}, {N, N});
    using F = host_node_field2D<double>;
    using Pos = Eigen::RowVector2d;
    const double delta = grid.get_deltas()[0];
    const Pos p(0.50625, 0.50625);   // u = 0.2 within its cell
    const double w = 1.0;
    const std::vector<int> ks = {1, 8, 32};

    // 2nd-order at p vs k^2 1st-order deposits (addback_*)
    run_case<F, Pos,
        simple_interpolation::quadratic_interp<2, Pos>,
        simple_interpolation::linear_interp<2, Pos>
    >(grid, p, w, delta, ks, "addback", "2nd", "1st");

    // 3rd-order at p vs k^2 2nd-order deposits (addback3_*)
    run_case<F, Pos,
        simple_interpolation::cubic_interp<2, Pos>,
        simple_interpolation::quadratic_interp<2, Pos>
    >(grid, p, w, delta, ks, "addback3", "3rd", "2nd");

    std::cout << "test_higher_order_addback: done" << std::endl;
    return 0;
}
