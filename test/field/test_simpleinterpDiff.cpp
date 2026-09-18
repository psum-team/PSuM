#include <psum/field.hpp>
#include "../../src/random/rander.hpp"
#include "../../src/random/rand_functions.hpp"

using namespace psum::field;
using namespace psum::field::simple_interpolation;
using namespace psum::random;

// check linear property
template<int Dim>
bool test_interp_diff() {
    std::cout << "Differential interp check:";
    rander R;
    Eigen::Vector<double, Dim> lower;
    Eigen::Vector<double, Dim> upper;
    std::vector<int> size(Dim);
    for (int i = 0; i < Dim; i++) {
        lower[i] = R() - 0.5;
        upper[i] = exp(R() - 0.5) + lower[i];
        size[i] = int(60 * R() + 1);
    }
    simple_grid<Dim> g(lower, upper, size);

    auto convert_to_EigenVector = [](auto& vec) {
        Eigen::VectorXd res;
        res.resize(vec.size());
        for (int i = 0; i < vec.size(); i++)
            res[i] = vec[i];
        return res;
    };

    bool check_result = true;
    size_t random_points = std::min<size_t>(g.contentSize(var_loc::cellCentered) * 10, 1e6);
    for (int i = 0; i < random_points; i++) {
        Eigen::Vector<double, Dim> p;
        for (int j = 0; j < Dim; j++) {
            p[j] = g.get_spans()[j] * (R() - 0.5) * 1.0 + g.get_middles()[j];
        }
        auto [indices, diff_coefs_array] = linear_interp_diff(g, p);
        for (int d = 0; d < Dim; d++) {
            double ofs = (R() - 0.5) * g.get_deltas()[d];
            Eigen::Vector<double, Dim> q = p;
            q[d] += ofs;
            // find another point in the same cell
            while (g.c2i(g.nC(q)) != g.c2i(g.nC(p))) {
                ofs = (R() - 0.5) * g.get_deltas()[d];
                q[d] = p[d] + ofs;
            }
            auto [indices_p, coefs_p] = linear_interp(g, p);
            auto [indices_q, coefs_q] = linear_interp(g, q);

            Eigen::VectorXd coefs_p_vec = convert_to_EigenVector(coefs_p);
            Eigen::VectorXd coefs_q_vec = convert_to_EigenVector(coefs_q);

            Eigen::VectorXd diff_coefs_right = (coefs_q_vec - coefs_p_vec) / ofs;
            Eigen::VectorXd diff_coefs = convert_to_EigenVector(diff_coefs_array[d]);

            if ((diff_coefs - diff_coefs_right).norm() > 1e-7 * (diff_coefs+ diff_coefs_right).norm()) {
                check_result = false;
                break;
            }
        }
    }
    if (check_result)
        std::cout << "✅ Linear property check passed" << std::endl;
    else
        std::cout << "❌ Linear property check failed" << std::endl;
    return check_result;
}

int main() {
    int failures = 0;
    for (int i = 0; i < 10; i++) {
        failures += test_interp_diff<1>() ? 0 : 1;
        failures += test_interp_diff<2>() ? 0 : 1;
        failures += test_interp_diff<3>() ? 0 : 1;
        failures += test_interp_diff<4>() ? 0 : 1;
    }
    return failures ? 1 : 0;
}