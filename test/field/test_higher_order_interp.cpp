#include <psum/field.hpp>
#include "../../src/random/rander.hpp"
#include "../../src/random/rand_functions.hpp"

#include <vector>
#include <array>
#include <cmath>

using namespace psum::field;
using namespace psum::field::simple_interpolation;
using namespace psum::random;

// ---------------------------------------------------------------------------
// Integral-regression (convolution) criterion for the higher-order (2nd/3rd)
// interpolations.
//
// The p-th order interpolation of a nodal field equals the average of the
// (p-1)-th order interpolation taken over one cell centered at the particle
// (the B-spline recurrence, integrated with a k^d midpoint rule):
//
//      sum_j b_p(u - j) f_j  ==  lim_{k->inf} (1/k^d) * sum_{cells c} sum_j b_{p-1}(x_c - j) f_j
// ---------------------------------------------------------------------------

template<int Dim, int SupportSizeP, int SupportSizeM, auto FuncP, auto FuncM>
bool test_integral_regression(int rounds, int m = 100) {
    std::cout << "Integral regression (order " << SupportSizeP - 1 << " vs " << SupportSizeP - 2 << "):";
    rander R;

    for (int round = 0; round < rounds; round++) {
        Eigen::Vector<double, Dim> lower, upper;
        std::vector<int> size(Dim);
        for (int i = 0; i < Dim; i++) {
            lower[i] = R() - 0.5;
            size[i] = int(20 * R()) + 6 + i;
            upper[i] = lower[i] + (1.0 + 0.7 * i + 0.5 * R()) * size[i] * 0.1;
        }
        simple_grid<Dim> g1(lower, upper, size);

        // f1: white noise on nodes
        std::vector<double> f1(g1.contentSize(var_loc::nodeCentered));
        for (size_t i = 0; i < f1.size(); i++) f1[i] = R();

        // f2: f1 extended by two ghost cell layers per dimension (periodic
        // mapping, consistent with the wrap-around stencil): f2 node i_d reads
        // f1 node (i_d - 2) mod N_d
        std::vector<int> size2(Dim);
        Eigen::Vector<double, Dim> lower2, upper2;
        for (int i = 0; i < Dim; i++) {
            size2[i] = size[i] + 4;
            lower2[i] = lower[i] - 2.0 * g1.get_deltas()[i];
            upper2[i] = upper[i] + 2.0 * g1.get_deltas()[i];
        }
        simple_grid<Dim> g2(lower2, upper2, size2);
        std::vector<double> f2(g2.contentSize(var_loc::nodeCentered));
        for (size_t i = 0; i < f2.size(); i++) {
            auto n = g2.i2n(i);
            for (int d = 0; d < Dim; d++) {
                // periodic mapping consistent with the stencil wrap: the node
                // array of width N+1 is one ring (-2 -> N-1, -1 -> N, N+1 -> 0)
                int t = (n.indices[d] - 2) % (size[d] + 1);
                if (t < 0) t += size[d] + 1;
                n.indices[d] = t;
            }
            f2[i] = f1[g1.n2i(typename simple_grid<Dim>::node_index{n})];
        }

        // convergence ladder: the k^d midpoint rule refines as O(1/k^2),
        // so quadrupling k must reduce the mean error by at least ~16x;
        // we require a conservative 4x per step
        std::vector<double> means, worsts;
        for (int k : {4, 16, 64}) {
            double worst = 0, sum = 0;
            for (int t = 0; t < m; t++) {
                Eigen::Vector<double, Dim> u;
                for (int i = 0; i < Dim; i++)
                    u[i] = lower[i] + g1.get_spans()[i] * double(R());

                // p-th order interpolation at u on f1
                auto [ip, wp] = gen_interp<SupportSizeP, Dim, FuncP>(g1, u);
                double vp = 0;
                for (int j = 0; j < ip.size(); j++) vp += f1[ip[j]] * wp[j];

                // (p-1)-th order interpolation at the k^d sub-cell centers of
                // the one-cell window around u, on f2, then averaged
                double vm = 0;
                int total = 1;
                for (int i = 0; i < Dim; i++) total *= k;
                for (int c = 0; c < total; c++) {
                    Eigen::Vector<double, Dim> x;
                    int r = c;
                    for (int i = 0; i < Dim; i++) {
                        int ic = r % k; r /= k;
                        x[Dim - 1 - i] = u[Dim - 1 - i] - 0.5 * g1.get_deltas()[Dim - 1 - i]
                                         + (ic + 0.5) / k * g1.get_deltas()[Dim - 1 - i];
                    }
                    auto [im, wm] = gen_interp<SupportSizeM, Dim, FuncM>(g2, x);
                    for (int j = 0; j < im.size(); j++) vm += f2[im[j]] * wm[j];
                }
                vm /= total;

                double err = std::abs(vp - vm);
                worst = std::max(worst, err);
                sum += err;
            }
            means.push_back(sum / m);
            worsts.push_back(worst);
        }

        for (size_t i = 1; i < means.size(); i++)
            if (!(means[i] < 0.25 * means[i - 1])) {
                std::cout << "❌ insufficient convergence with k (mean " << means[i - 1] << " -> " << means[i] << ")" << std::endl;
                return false;
            }
        if (means.back() > 1e-4 || worsts.back() > 5e-4) {
            std::cout << "❌ final error too large (mean " << means.back() << ", worst " << worsts.back() << ")" << std::endl;
            return false;
        }
    }
    std::cout << "✅ Integral regression check passed" << std::endl;
    return true;
}

int main() {
    int failures = 0;
    for (int i = 0; i < 3; i++) {
        failures += test_integral_regression<1, 3, 2, &quadratic_bspline_1D, &linear_interp_1D>(2) ? 0 : 1;
        failures += test_integral_regression<2, 3, 2, &quadratic_bspline_1D, &linear_interp_1D>(2) ? 0 : 1;
        failures += test_integral_regression<3, 3, 2, &quadratic_bspline_1D, &linear_interp_1D>(1, 40) ? 0 : 1;
        failures += test_integral_regression<1, 4, 3, &cubic_bspline_1D, &quadratic_bspline_1D>(2) ? 0 : 1;
        failures += test_integral_regression<2, 4, 3, &cubic_bspline_1D, &quadratic_bspline_1D>(2) ? 0 : 1;
        failures += test_integral_regression<3, 4, 3, &cubic_bspline_1D, &quadratic_bspline_1D>(1, 40) ? 0 : 1;
    }
    return failures ? 1 : 0;
}
