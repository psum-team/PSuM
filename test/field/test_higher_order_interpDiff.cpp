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
// Differential interpolation of the higher-order (2nd/3rd) interpolations:
// the direct derivative weights are checked against a central finite
// difference of the value interpolation,
//
//      sum_j dW_j(u) f_j   vs   [I(u + h e_d) - I(u - h e_d)] / (2h)
// ---------------------------------------------------------------------------

template<int Dim, int SupportSize, auto Func>
bool test_diff_vs_difference(int rounds) {
    std::cout << "Diff interp vs finite difference (order " << SupportSize - 1 << "):";
    rander R;
    const int m = 200;
    const double h_rel = 1e-4;      // h as a fraction of the local cell size

    for (int round = 0; round < rounds; round++) {
        // adversarial grid: different cell count and extent per dimension
        Eigen::Vector<double, Dim> lower, upper;
        std::vector<int> size(Dim);
        for (int i = 0; i < Dim; i++) {
            lower[i] = R() - 0.5;
            size[i] = int(20 * R()) + 6 + i;
            upper[i] = lower[i] + (1.0 + 0.7 * i + 0.5 * R()) * size[i] * 0.1;
        }
        simple_grid<Dim> g(lower, upper, size);

        // white noise on nodes (maximally non-smooth)
        std::vector<double> f(g.contentSize(var_loc::nodeCentered));
        for (size_t i = 0; i < f.size(); i++) f[i] = R();

        double worst = 0, sum = 0;
        for (int t = 0; t < m; t++) {
            Eigen::Vector<double, Dim> u;
            // keep every dimension at least 4h away from a node plane
            for (int i = 0; i < Dim; i++) {
                double frac;
                do {
                    u[i] = lower[i] + g.get_spans()[i] * double(R());
                    double v = (u[i] - lower[i]) * g.get_deltas_r()[i];
                    frac = v - std::floor(v);
                } while (frac < 4.0 * h_rel || frac > 1.0 - 4.0 * h_rel);
            }

            auto [idx, wd] = gen_interp_diff<SupportSize, Dim, Func>(g, u);
            for (int d = 0; d < Dim; d++) {
                double direct = 0;
                for (int j = 0; j < idx.size(); j++) direct += f[idx[j]] * wd[d][j];

                Eigen::Vector<double, Dim> pp = u, pm = u;
                pp[d] += h_rel * g.get_deltas()[d];
                auto [ip, wp] = gen_interp<SupportSize, Dim, Func>(g, pp);
                double vp = 0;
                for (int j = 0; j < ip.size(); j++) vp += f[ip[j]] * wp[j];
                pm[d] -= h_rel * g.get_deltas()[d];
                auto [im, wm] = gen_interp<SupportSize, Dim, Func>(g, pm);
                double vm = 0;
                for (int j = 0; j < im.size(); j++) vm += f[im[j]] * wm[j];
                double fd = (vp - vm) / (2.0 * h_rel * g.get_deltas()[d]);

                double err = std::abs(direct - fd);
                worst = std::max(worst, err);
                sum += err;
            }
        }
        double mean = sum / (m * Dim);
        if (mean > 1e-6 || worst > 1e-4) {
            std::cout << "❌ diff mismatch (mean " << mean << ", worst " << worst << ")" << std::endl;
            return false;
        }
    }
    std::cout << "✅ Diff interp vs finite difference check passed" << std::endl;
    return true;
}

int main() {
    int failures = 0;
    for (int i = 0; i < 3; i++) {
        failures += test_diff_vs_difference<1, 3, &quadratic_bspline_1D>(2) ? 0 : 1;
        failures += test_diff_vs_difference<2, 3, &quadratic_bspline_1D>(2) ? 0 : 1;
        failures += test_diff_vs_difference<3, 3, &quadratic_bspline_1D>(1) ? 0 : 1;
        failures += test_diff_vs_difference<1, 4, &cubic_bspline_1D>(2) ? 0 : 1;
        failures += test_diff_vs_difference<2, 4, &cubic_bspline_1D>(2) ? 0 : 1;
        failures += test_diff_vs_difference<3, 4, &cubic_bspline_1D>(1) ? 0 : 1;
    }
    return failures ? 1 : 0;
}
