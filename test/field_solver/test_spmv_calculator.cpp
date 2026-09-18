#include "../../src/field_solver/spmv/eigen_spmv_calculator.hpp"
#include "../../src/field_solver/spmv/sycl_spmv_calculator.hpp"
#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <sycl/sycl.hpp>
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

using namespace std;
using namespace psum::field_solver::spmv;

Eigen::SparseMatrix<double> make_fem_like_mass_matrix(int nx, int ny)
{
    int n_nodes = (nx + 1) * (ny + 1);
    vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(nx * ny * 18);

    auto node_idx = [nx](int i, int j) {
        return j * (nx + 1) + i;
    };

    auto add_triangle_mass = [&](int n0, int n1, int n2, double area) {
        int nodes[3] = {n0, n1, n2};
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                double val = (i == j) ? area / 6.0 : area / 12.0;
                triplets.emplace_back(nodes[i], nodes[j], val);
            }
        }
    };

    double hx = 1.0 / nx;
    double hy = 1.0 / ny;
    double tri_area = hx * hy * 0.5;
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            int n00 = node_idx(i, j);
            int n10 = node_idx(i + 1, j);
            int n01 = node_idx(i, j + 1);
            int n11 = node_idx(i + 1, j + 1);
            add_triangle_mass(n00, n10, n11, tri_area);
            add_triangle_mass(n00, n11, n01, tri_area);
        }
    }

    Eigen::SparseMatrix<double> matrix(n_nodes, n_nodes);
    matrix.setFromTriplets(triplets.begin(), triplets.end());
    matrix.makeCompressed();
    return matrix;
}

void check_close(const Eigen::VectorXd& actual, const Eigen::VectorXd& expected, const string& label)
{
    double diff = (actual - expected).norm();
    double max_diff = (actual - expected).cwiseAbs().maxCoeff();
    cout << label << " L2 norm difference: " << diff << endl;
    cout << label << " max absolute difference: " << max_diff << endl;
    assert(diff < 1e-12);
    assert(max_diff < 1e-12);
}

int main()
{
    cout << "\n=== Test SpMV calculators with FEM-like mass matrix ===" << endl;

    Eigen::SparseMatrix<double> matrix = make_fem_like_mass_matrix(8, 6);
    Eigen::VectorXd source(matrix.cols());
    for (int i = 0; i < source.size(); i++) {
        source[i] = std::sin(0.17 * i) + 0.5 * std::cos(0.31 * i);
    }

    Eigen::VectorXd expected = matrix * source;
    cout << "Matrix rows: " << matrix.rows() << endl;
    cout << "Matrix cols: " << matrix.cols() << endl;
    cout << "Matrix nnz: " << matrix.nonZeros() << endl;
    cout << "Eigen baseline norm: " << expected.norm() << endl;

    eigen_spmv_calculator eigen_calc(matrix);
    Eigen::VectorXd eigen_result(matrix.rows());
    eigen_calc.apply(source.data(), eigen_result.data());
    check_close(eigen_result, expected, "Eigen calculator apply");

    Eigen::VectorXd eigen_inplace = source;
    eigen_calc.apply_inplace(eigen_inplace.data());
    check_close(eigen_inplace, expected, "Eigen calculator apply_inplace");

    sycl::queue q{sycl::default_selector_v};
    double* source_dev = sycl::malloc_device<double>(source.size(), q);
    double* result_dev = sycl::malloc_device<double>(expected.size(), q);
    if (source_dev == nullptr || result_dev == nullptr) {
        if (source_dev) sycl::free(source_dev, q);
        if (result_dev) sycl::free(result_dev, q);
        throw runtime_error("failed to allocate SYCL device arrays");
    }

    q.memcpy(source_dev, source.data(), sizeof(double) * source.size()).wait();

    sycl_spmv_calculator sycl_calc(matrix, q);
    sycl_calc.apply(source_dev, result_dev);

    Eigen::VectorXd sycl_result(expected.size());
    q.memcpy(sycl_result.data(), result_dev, sizeof(double) * sycl_result.size()).wait();
    check_close(sycl_result, expected, "SYCL calculator apply");

    q.memcpy(source_dev, source.data(), sizeof(double) * source.size()).wait();
    sycl_calc.apply_inplace(source_dev);

    Eigen::VectorXd sycl_inplace(expected.size());
    q.memcpy(sycl_inplace.data(), source_dev, sizeof(double) * sycl_inplace.size()).wait();
    check_close(sycl_inplace, expected, "SYCL calculator apply_inplace");

    sycl::free(source_dev, q);
    sycl::free(result_dev, q);

    cout << "PASS: SpMV calculator results match Eigen baseline" << endl;
    return 0;
}
