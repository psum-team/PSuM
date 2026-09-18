#include <psum/field.hpp>
#include "../../src/random/rander.hpp"
#include "../../src/random/rand_functions.hpp"

using namespace psum::field;
using namespace psum::field::simple_interpolation;
using namespace psum::random;

// check interpolation coefficients and indices:
//      sum of coefficients should be 1
//      indices should be within the range of node centered content size
//      coefficients should be within the range of [0, 1]
template<int Dim>
bool test_interp_1() {
    std::cout << "Basic property check:";
    rander R;
    Eigen::Vector<double, Dim> lower;
    Eigen::Vector<double, Dim> upper;
    std::vector<int> size(Dim);
    for (int i = 0; i < Dim; i++) {
        lower[i] = R() - 0.5;
        upper[i] = exp(R() - 0.5) + lower[i];
        size[i] = int(100 * R() + 1);
    }
    simple_grid<Dim> g(lower, upper, size);

    bool check_result = true;
    size_t random_points = std::min<size_t>(g.contentSize(var_loc::cellCentered) * 100, 1e7);
    for (int i = 0; i < random_points; i++) {
        Eigen::Vector<double, Dim> p;
        for (int j = 0; j < Dim; j++) {
            p[j] = g.get_spans()[j] * (R() - 0.5) * 1.2 + g.get_middles()[j];
        }
        bool in_grid = g.inGrid(p);

        if (in_grid) {
            bool is_valid = true;
            auto [indices, coefs] = linear_interp(g, p);
            double coeff_sum = 0;
            for (int j = 0; j < indices.size(); j++) {
                if (indices[j] < 0 || indices[j] >= g.contentSize(var_loc::nodeCentered)) {
                    is_valid = false;
                    break;
                }
                if (coefs[j] < 0 || coefs[j] > 1) {
                    is_valid = false;
                    break;
                }
                coeff_sum += coefs[j];
            }
            if (coeff_sum < 1-1e-12 || coeff_sum > 1+1e-12)
                is_valid = false;
            if (!is_valid) {
                std::cout << "❌ invalid interpolation coefficients" << std::endl;
                check_result = false;
                break;
            }
        }
    }
    if (check_result)
        std::cout << "✅ Basic property check passed" << std::endl;
    else
        std::cout << "❌ Basic property check failed" << std::endl;
    return check_result;
}

// check interpolation uniformity
template<int Dim>
bool test_interp_2(int points_per_cell, double tolerance) {
    std::cout << "Uniformity check:";
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

    Eigen::VectorXd volume;
    volume.resize(g.contentSize(var_loc::nodeCentered));
    volume.setConstant(1);
    for (size_t i = 0; i < g.contentSize(var_loc::nodeCentered); i++) {
        auto node_idx = g.i2n(i);
        for (int j = 0; j < Dim; j++) {
            // i corresponds to a node on boundary
            if (node_idx.indices[j] == 0 || node_idx.indices[j] == g.get_cell_num()[j])
                volume[i] *= 0.5;
        }
    }

    Eigen::VectorXd content;
    content.resize(g.contentSize(var_loc::nodeCentered));
    content.setZero();

    bool check_result = true;
    size_t random_points = g.contentSize(var_loc::cellCentered) * points_per_cell;
    for (int i = 0; i < random_points; i++) {
        Eigen::Vector<double, Dim> p;
        for (int j = 0; j < Dim; j++) {
            p[j] = g.get_spans()[j] * (R() - 0.5) * 1.0 + g.get_middles()[j];
        }

        auto [indices, coefs] = linear_interp(g, p);
        for (int j = 0; j < indices.size(); j++) {
            if (coefs[j] > 1e-14)
                content[indices[j]] += coefs[j] * 1.0 / points_per_cell;
        }
    }
    content << content.array() / volume.array();
    double mean_tol = 5.0 / std::sqrt(static_cast<double>(random_points));
    bool sum_ok = std::abs(content.mean() - 1.0) < mean_tol;
    if (sum_ok)
        std::cout << "✅ sum of coefficients is 1 (approximately)" << std::endl;
    else
        std::cout << "❌ sum of coefficients is not 1 (approximately)" << std::endl;
    double variance = (content.array() - content.mean()).square().mean();
    std::cout << "mean variance: " << variance << std::endl;
    bool uniform_ok = variance < tolerance;
    if (uniform_ok)
        std::cout << "✅ Interpolation uniformity check passed" << std::endl;
    else
        std::cout << "❌ Interpolation uniformity check failed" << std::endl;
    return sum_ok && uniform_ok;
}

// check one-hot property
template<int Dim>
bool test_interp_3() {
    std::cout << "One-hot property check:";
    rander R;
    Eigen::Vector<double, Dim> lower;
    Eigen::Vector<double, Dim> upper;
    std::vector<int> size(Dim);
    for (int i = 0; i < Dim; i++) {
        lower[i] = R() - 0.5;
        upper[i] = exp(R() - 0.5) + lower[i];
        size[i] = int(100 * R() + 1);
    }
    simple_grid<Dim> g(lower, upper, size);

    bool check_result = true;

    for (int i = 0; i < g.contentSize(var_loc::nodeCentered); i++) {
        auto node_pos = g.nodePosition(g.i2n(i));
        auto [indices, coefs] = linear_interp(g, node_pos);
        for (int j = 0; j < indices.size(); j++) {
            if (indices[j] == i) {
                if (coefs[j] < 1.0 - 1e-12 || coefs[j] > 1.0 + 1e-12) {
                    check_result = false;
                    break;
                }
            }
        }
    }
    if (check_result)
        std::cout << "✅ One-hot property check passed" << std::endl;
    else
        std::cout << "❌ One-hot property check failed" << std::endl;
    return check_result;
}

// check linear property
template<int Dim>
bool test_interp_4() {
    std::cout << "Linear property check:";
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
        for (int d = 0; d < Dim; d++) {
            double ofs = (R() - 0.5) * g.get_deltas()[d];
            Eigen::Vector<double, Dim> q = p;
            q[d] += ofs;
            // find another point in the same cell
            while (g.c2i(g.nC(q)) != g.c2i(g.nC(p))) {
                q[d] = p[d] + (R() - 0.5) * g.get_deltas()[d];
            }
            auto [indices_p, coefs_p] = linear_interp(g, p);
            auto [indices_q, coefs_q] = linear_interp(g, q);
            if (indices_p != indices_q) {
                check_result = false;
                break;
            }
            Eigen::VectorXd coefs_p_vec = convert_to_EigenVector(coefs_p);
            Eigen::VectorXd coefs_q_vec = convert_to_EigenVector(coefs_q);
            int inner_point = 10;
            for (int k = 0; k < inner_point; k++) {
                double r = R();
                Eigen::Vector<double, Dim> inner_p = p * (1 - r) + q * r;
                auto [indices_r, coefs_r] = linear_interp(g, inner_p);
                Eigen::VectorXd coefs_r_vec = convert_to_EigenVector(coefs_r);
                Eigen::VectorXd expected_coefs = (coefs_p_vec * (1 - r) + coefs_q_vec * r);
                if ((coefs_r_vec - expected_coefs).norm() > 1e-12) {
                    check_result = false;
                    break;
                }
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
        failures += test_interp_1<1>() ? 0 : 1;
        failures += test_interp_1<2>() ? 0 : 1;
        failures += test_interp_1<3>() ? 0 : 1;
        failures += test_interp_1<4>() ? 0 : 1;
    }
    failures += test_interp_2<1>(40, 0.05) ? 0 : 1;
    failures += test_interp_2<1>(400, 0.01) ? 0 : 1;
    failures += test_interp_2<1>(40000, 0.001) ? 0 : 1;
    failures += test_interp_2<2>(400, 0.01) ? 0 : 1;
    failures += test_interp_2<2>(4000, 0.001) ? 0 : 1;
    failures += test_interp_2<3>(4000, 0.001) ? 0 : 1;
    for (int i = 0; i < 10; i++) {
        failures += test_interp_3<1>() ? 0 : 1;
        failures += test_interp_3<2>() ? 0 : 1;
        failures += test_interp_3<3>() ? 0 : 1;
        failures += test_interp_3<4>() ? 0 : 1;
    }
    for (int i = 0; i < 10; i++) {
        failures += test_interp_4<1>() ? 0 : 1;
        failures += test_interp_4<2>() ? 0 : 1;
        failures += test_interp_4<3>() ? 0 : 1;
        failures += test_interp_4<4>() ? 0 : 1;
    }
    return failures ? 1 : 0;
}