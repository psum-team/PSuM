#include "../../src/field_solver/implements/multigrid/convSparseMatrix.hpp"
#include "../../src/field_solver/implements/multigrid/sparse_linear_trait.hpp"
#include <iostream>
#include <cassert>
#include <random>
#include <cmath>

using namespace std;
using namespace psum;
using namespace psum::field_solver::implements::multigrid;

void build_7point_poisson_matrix(int I, int J, int K,
    std::vector<unsigned long long>& rows,
    std::vector<unsigned long long>& cols,
    std::vector<double>& vals)
{
    for (int k = 0; k < K; ++k)
    {
        for (int j = 0; j < J; ++j)
        {
            for (int i = 0; i < I; ++i)
            {
                int idx = i * J * K + j * K + k;

                bool is_boundary = (i == 0 || i == I - 1 ||
                                   j == 0 || j == J - 1 ||
                                   k == 0 || k == K - 1);

                if (is_boundary)
                {
                    rows.push_back(idx);
                    cols.push_back(idx);
                    vals.push_back(1.0);
                }
                else
                {
                    rows.push_back(idx);
                    cols.push_back(idx);
                    vals.push_back(6.0);

                    if (i > 0)
                    {
                        rows.push_back(idx);
                        cols.push_back((i - 1) * J * K + j * K + k);
                        vals.push_back(-1.0);
                    }
                    if (i < I - 1)
                    {
                        rows.push_back(idx);
                        cols.push_back((i + 1) * J * K + j * K + k);
                        vals.push_back(-1.0);
                    }
                    if (j > 0)
                    {
                        rows.push_back(idx);
                        cols.push_back(i * J * K + (j - 1) * K + k);
                        vals.push_back(-1.0);
                    }
                    if (j < J - 1)
                    {
                        rows.push_back(idx);
                        cols.push_back(i * J * K + (j + 1) * K + k);
                        vals.push_back(-1.0);
                    }
                    if (k > 0)
                    {
                        rows.push_back(idx);
                        cols.push_back(i * J * K + j * K + (k - 1));
                        vals.push_back(-1.0);
                    }
                    if (k < K - 1)
                    {
                        rows.push_back(idx);
                        cols.push_back(i * J * K + j * K + (k + 1));
                        vals.push_back(-1.0);
                    }
                }
            }
        }
    }
}

void test_poisson_matrix_ones_vector()
{
    cout << "\n=== Test Poisson matrix with ones vector ===" << endl;

    int I = 16, J = 16, K = 16;

    std::vector<unsigned long long> rows, cols;
    std::vector<double> vals;
    build_7point_poisson_matrix(I, J, K, rows, cols, vals);

    Eigen::SparseMatrix<double> A(I * J * K, I * J * K);
    std::vector<Eigen::Triplet<double>> triplets;
    for (size_t i = 0; i < rows.size(); ++i)
    {
        triplets.emplace_back(rows[i], cols[i], vals[i]);
    }
    A.setFromTriplets(triplets.begin(), triplets.end());

    ConvSparseMatrix<double> convMat;
    convMat.set(A, 5);

    Eigen::VectorXd x = Eigen::VectorXd::Ones(I * J * K);

    Eigen::VectorXd result_eigen;
    matrix_traits<Eigen::SparseMatrix<double>>::multiply(A, x, result_eigen);
    Eigen::VectorXd result_conv;
    matrix_traits<ConvSparseMatrix<double>>::multiply(convMat, x, result_conv);

    double diff = (result_eigen - result_conv).norm();
    double max_diff = (result_eigen - result_conv).cwiseAbs().maxCoeff();

    cout << "Problem size: " << I << "x" << J << "x" << K << " = " << I * J * K << " unknowns" << endl;
    cout << "L2 norm difference: " << diff << endl;
    cout << "Max absolute difference: " << max_diff << endl;
    cout << "Eigen result norm: " << result_eigen.norm() << endl;
    cout << "Conv result norm: " << result_conv.norm() << endl;

    assert(diff < 1e-10);

    cout << "PASS: Poisson matrix with ones vector" << endl;
}

void test_poisson_matrix_random_vector()
{
    cout << "\n=== Test Poisson matrix with random vector ===" << endl;

    int I = 16, J = 16, K = 16;

    std::vector<unsigned long long> rows, cols;
    std::vector<double> vals;
    build_7point_poisson_matrix(I, J, K, rows, cols, vals);

    Eigen::SparseMatrix<double> A(I * J * K, I * J * K);
    std::vector<Eigen::Triplet<double>> triplets;
    for (size_t i = 0; i < rows.size(); ++i)
    {
        triplets.emplace_back(rows[i], cols[i], vals[i]);
    }
    A.setFromTriplets(triplets.begin(), triplets.end());

    ConvSparseMatrix<double> convMat;
    convMat.set(A, 5);

    std::random_device rd;
    std::mt19937 gen(42);
    std::uniform_real_distribution<> dis(-1.0, 1.0);

    Eigen::VectorXd x(I * J * K);
    for (int i = 0; i < I * J * K; ++i)
    {
        x[i] = dis(gen);
    }

    Eigen::VectorXd result_eigen;
    matrix_traits<Eigen::SparseMatrix<double>>::multiply(A, x, result_eigen);
    Eigen::VectorXd result_conv;
    matrix_traits<ConvSparseMatrix<double>>::multiply(convMat, x, result_conv);

    double diff = (result_eigen - result_conv).norm();
    double max_diff = (result_eigen - result_conv).cwiseAbs().maxCoeff();

    cout << "Problem size: " << I << "x" << J << "x" << K << " = " << I * J * K << " unknowns" << endl;
    cout << "L2 norm difference: " << diff << endl;
    cout << "Max absolute difference: " << max_diff << endl;

    assert(diff < 1e-10);

    cout << "PASS: Poisson matrix with random vector" << endl;
}

void test_random_coefficient_poisson()
{
    cout << "\n=== Test random coefficient Poisson matrix ===" << endl;

    int I = 8, J = 8, K = 8;

    std::random_device rd;
    std::mt19937 gen(42);
    std::uniform_real_distribution<> dis(0.5, 1.5);

    std::vector<unsigned long long> rows, cols;
    std::vector<double> vals;

    for (int k = 0; k < K; ++k)
    {
        for (int j = 0; j < J; ++j)
        {
            for (int i = 0; i < I; ++i)
            {
                int idx = i * J * K + j * K + k;

                bool is_boundary = (i == 0 || i == I - 1 ||
                                   j == 0 || j == J - 1 ||
                                   k == 0 || k == K - 1);

                if (is_boundary)
                {
                    rows.push_back(idx);
                    cols.push_back(idx);
                    vals.push_back(1.0);
                }
                else
                {
                    rows.push_back(idx);
                    cols.push_back(idx);
                    vals.push_back(6.0);

                    if (i > 0)
                    {
                        rows.push_back(idx);
                        cols.push_back((i - 1) * J * K + j * K + k);
                        vals.push_back(-dis(gen));
                    }
                    if (i < I - 1)
                    {
                        rows.push_back(idx);
                        cols.push_back((i + 1) * J * K + j * K + k);
                        vals.push_back(-dis(gen));
                    }
                    if (j > 0)
                    {
                        rows.push_back(idx);
                        cols.push_back(i * J * K + (j - 1) * K + k);
                        vals.push_back(-dis(gen));
                    }
                    if (j < J - 1)
                    {
                        rows.push_back(idx);
                        cols.push_back(i * J * K + (j + 1) * K + k);
                        vals.push_back(-dis(gen));
                    }
                    if (k > 0)
                    {
                        rows.push_back(idx);
                        cols.push_back(i * J * K + j * K + (k - 1));
                        vals.push_back(-dis(gen));
                    }
                    if (k < K - 1)
                    {
                        rows.push_back(idx);
                        cols.push_back(i * J * K + j * K + (k + 1));
                        vals.push_back(-dis(gen));
                    }
                }
            }
        }
    }

    Eigen::SparseMatrix<double> A(I * J * K, I * J * K);
    std::vector<Eigen::Triplet<double>> triplets;
    for (size_t i = 0; i < rows.size(); ++i)
    {
        triplets.emplace_back(rows[i], cols[i], vals[i]);
    }
    A.setFromTriplets(triplets.begin(), triplets.end());

    ConvSparseMatrix<double> convMat;
    convMat.set(A, 5);

    std::mt19937 gen2(42);
    std::uniform_real_distribution<> dis2(-1.0, 1.0);

    Eigen::VectorXd x(I * J * K);
    for (int i = 0; i < I * J * K; ++i)
    {
        x[i] = dis2(gen2);
    }

    Eigen::VectorXd result_eigen;
    matrix_traits<Eigen::SparseMatrix<double>>::multiply(A, x, result_eigen);
    Eigen::VectorXd result_conv;
    matrix_traits<ConvSparseMatrix<double>>::multiply(convMat, x, result_conv);

    double diff = (result_eigen - result_conv).norm();
    double max_diff = (result_eigen - result_conv).cwiseAbs().maxCoeff();

    cout << "Problem size: " << I << "x" << J << "x" << K << " = " << I * J * K << " unknowns" << endl;
    cout << "L2 norm difference: " << diff << endl;
    cout << "Max absolute difference: " << max_diff << endl;

    assert(diff < 1e-10);

    cout << "PASS: Random coefficient Poisson matrix" << endl;
}

void test_2d_poisson_matrix()
{
    cout << "\n=== Test 2D Poisson matrix ===" << endl;

    int I = 32, J = 32;

    std::vector<unsigned long long> rows, cols;
    std::vector<double> vals;

    for (int j = 0; j < J; ++j)
    {
        for (int i = 0; i < I; ++i)
        {
            int idx = i * J + j;

            rows.push_back(idx);
            cols.push_back(idx);
            vals.push_back(4.0);

            if (i > 0)
            {
                rows.push_back(idx);
                cols.push_back((i - 1) * J + j);
                vals.push_back(-1.0);
            }
            if (i < I - 1)
            {
                rows.push_back(idx);
                cols.push_back((i + 1) * J + j);
                vals.push_back(-1.0);
            }
            if (j > 0)
            {
                rows.push_back(idx);
                cols.push_back(i * J + (j - 1));
                vals.push_back(-1.0);
            }
            if (j < J - 1)
            {
                rows.push_back(idx);
                cols.push_back(i * J + (j + 1));
                vals.push_back(-1.0);
            }
        }
    }

    Eigen::SparseMatrix<double> A(I * J, I * J);
    std::vector<Eigen::Triplet<double>> triplets;
    for (size_t i = 0; i < rows.size(); ++i)
    {
        triplets.emplace_back(rows[i], cols[i], vals[i]);
    }
    A.setFromTriplets(triplets.begin(), triplets.end());

    ConvSparseMatrix<double> convMat;
    convMat.set(A, 5);

    Eigen::VectorXd x = Eigen::VectorXd::Ones(I * J);

    Eigen::VectorXd result_eigen;
    matrix_traits<Eigen::SparseMatrix<double>>::multiply(A, x, result_eigen);
    Eigen::VectorXd result_conv;
    matrix_traits<ConvSparseMatrix<double>>::multiply(convMat, x, result_conv);

    double diff = (result_eigen - result_conv).norm();
    double max_diff = (result_eigen - result_conv).cwiseAbs().maxCoeff();

    cout << "Problem size: " << I << "x" << J << " = " << I * J << " unknowns" << endl;
    cout << "L2 norm difference: " << diff << endl;
    cout << "Max absolute difference: " << max_diff << endl;

    assert(diff < 1e-10);

    cout << "PASS: 2D Poisson matrix" << endl;
}

void test_multiply_add_consistency()
{
    cout << "\n=== Test multiply_add consistency ===" << endl;

    int I = 8, J = 8, K = 8;

    std::vector<unsigned long long> rows, cols;
    std::vector<double> vals;
    build_7point_poisson_matrix(I, J, K, rows, cols, vals);

    Eigen::SparseMatrix<double> A(I * J * K, I * J * K);
    std::vector<Eigen::Triplet<double>> triplets;
    for (size_t i = 0; i < rows.size(); ++i)
    {
        triplets.emplace_back(rows[i], cols[i], vals[i]);
    }
    A.setFromTriplets(triplets.begin(), triplets.end());

    ConvSparseMatrix<double> convMat;
    convMat.set(A, 5);

    Eigen::VectorXd x = Eigen::VectorXd::Ones(I * J * K);
    Eigen::VectorXd b = Eigen::VectorXd::Ones(I * J * K);

    double b_coeff = 2.5;
    double ans_coeff = 3.0;

    Eigen::VectorXd result_eigen;
    matrix_traits<Eigen::SparseMatrix<double>>::multiply_add(A, x, b, result_eigen, b_coeff, ans_coeff);
    Eigen::VectorXd result_conv;
    matrix_traits<ConvSparseMatrix<double>>::multiply_add(convMat, x, b, result_conv, b_coeff, ans_coeff);

    double diff = (result_eigen - result_conv).norm();
    double max_diff = (result_eigen - result_conv).cwiseAbs().maxCoeff();

    cout << "Problem size: " << I << "x" << J << "x" << K << " = " << I * J * K << " unknowns" << endl;
    cout << "b_coeff = " << b_coeff << ", ans_coeff = " << ans_coeff << endl;
    cout << "L2 norm difference: " << diff << endl;
    cout << "Max absolute difference: " << max_diff << endl;

    assert(diff < 1e-10);

    cout << "PASS: multiply_add consistency" << endl;
}

void test_conv_info()
{
    cout << "\n=== Test ConvSparseMatrix info ===" << endl;

    int I = 16, J = 16, K = 16;

    std::vector<unsigned long long> rows, cols;
    std::vector<double> vals;
    build_7point_poisson_matrix(I, J, K, rows, cols, vals);

    Eigen::SparseMatrix<double> A(I * J * K, I * J * K);
    std::vector<Eigen::Triplet<double>> triplets;
    for (size_t i = 0; i < rows.size(); ++i)
    {
        triplets.emplace_back(rows[i], cols[i], vals[i]);
    }
    A.setFromTriplets(triplets.begin(), triplets.end());

    ConvSparseMatrix<double> convMat;
    convMat.set(A, 5);

    cout << "Matrix size: " << convMat.rows() << "x" << convMat.cols() << endl;
    cout << "Number of convolution patterns: " << convMat.convCmpt.size() << endl;

    int total_conv_rows = 0;
    for (const auto& conv : convMat.convCmpt)
    {
        total_conv_rows += conv.second.size();
    }
    cout << "Rows handled by convolution: " << total_conv_rows << endl;
    cout << "Rows handled by sparse matrix: " << convMat.non_all_zero_rows.size() << endl;
    cout << "Total rows: " << total_conv_rows + convMat.non_all_zero_rows.size() << endl;

    assert(total_conv_rows + convMat.non_all_zero_rows.size() == I * J * K);

    cout << "PASS: ConvSparseMatrix info" << endl;
}

void test_set_reentrancy()
{
    cout << "\n=== Test ConvSparseMatrix::set reentrancy ===" << endl;

    ConvSparseMatrix<double> convMat;

    Eigen::SparseMatrix<double> A1(64, 64);
    std::vector<Eigen::Triplet<double>> triplets1;
    for (int i = 0; i < 64; ++i)
    {
        triplets1.emplace_back(i, i, 2.0);
        if (i > 0) triplets1.emplace_back(i, i - 1, -1.0);
        if (i < 63) triplets1.emplace_back(i, i + 1, -1.0);
    }
    A1.setFromTriplets(triplets1.begin(), triplets1.end());

    convMat.set(A1);
    Eigen::VectorXd x1 = Eigen::VectorXd::Ones(64);
    Eigen::VectorXd result1;
    matrix_traits<ConvSparseMatrix<double>>::multiply(convMat, x1, result1);
    Eigen::VectorXd expected1;
    matrix_traits<Eigen::SparseMatrix<double>>::multiply(A1, x1, expected1);
    double diff1 = (result1 - expected1).norm();

    cout << "First set: 64x64 matrix" << endl;
    cout << "L2 norm difference: " << diff1 << endl;
    assert(diff1 < 1e-10);

    std::vector<unsigned long long> rows, cols;
    std::vector<double> vals;
    build_7point_poisson_matrix(8, 8, 8, rows, cols, vals);
    Eigen::SparseMatrix<double> A2(512, 512);
    std::vector<Eigen::Triplet<double>> triplets2;
    for (size_t i = 0; i < rows.size(); ++i)
    {
        triplets2.emplace_back(rows[i], cols[i], vals[i]);
    }
    A2.setFromTriplets(triplets2.begin(), triplets2.end());

    convMat.set(A2);
    Eigen::VectorXd x2 = Eigen::VectorXd::Ones(512);
    Eigen::VectorXd result2;
    matrix_traits<ConvSparseMatrix<double>>::multiply(convMat, x2, result2);
    Eigen::VectorXd expected2;
    matrix_traits<Eigen::SparseMatrix<double>>::multiply(A2, x2, expected2);
    double diff2 = (result2 - expected2).norm();

    cout << "Second set: 8x8x8 Poisson matrix" << endl;
    cout << "L2 norm difference: " << diff2 << endl;
    assert(diff2 < 1e-10);

    convMat.set(A1, 3);
    Eigen::VectorXd result3;
    matrix_traits<ConvSparseMatrix<double>>::multiply(convMat, x1, result3);
    Eigen::VectorXd expected3;
    matrix_traits<Eigen::SparseMatrix<double>>::multiply(A1, x1, expected3);
    double diff3 = (result3 - expected3).norm();

    cout << "Third set: 64x64 matrix with acceleration (num_conv_mask=3)" << endl;
    cout << "L2 norm difference: " << diff3 << endl;
    assert(diff3 < 1e-10);

    convMat.set(A1, 7);
    Eigen::VectorXd result4;
    matrix_traits<ConvSparseMatrix<double>>::multiply(convMat, x1, result4);
    Eigen::VectorXd expected4;
    matrix_traits<Eigen::SparseMatrix<double>>::multiply(A1, x1, expected4);
    double diff4 = (result4 - expected4).norm();

    cout << "Fourth set: same matrix, different num_conv_mask=7" << endl;
    cout << "L2 norm difference: " << diff4 << endl;
    assert(diff4 < 1e-10);

    convMat.set(A1);
    Eigen::VectorXd result5;
    matrix_traits<ConvSparseMatrix<double>>::multiply(convMat, x1, result5);
    Eigen::VectorXd expected5;
    matrix_traits<Eigen::SparseMatrix<double>>::multiply(A1, x1, expected5);
    double diff5 = (result5 - expected5).norm();

    cout << "Fifth set: back to single parameter (no acceleration)" << endl;
    cout << "L2 norm difference: " << diff5 << endl;
    assert(diff5 < 1e-10);

    convMat.set(A2, 5);
    Eigen::VectorXd result6;
    matrix_traits<ConvSparseMatrix<double>>::multiply(convMat, x2, result6);
    Eigen::VectorXd expected6;
    matrix_traits<Eigen::SparseMatrix<double>>::multiply(A2, x2, expected6);
    double diff6 = (result6 - expected6).norm();

    cout << "Sixth set: back to 8x8x8 Poisson with acceleration (num_conv_mask=5)" << endl;
    cout << "L2 norm difference: " << diff6 << endl;
    assert(diff6 < 1e-10);

    cout << "PASS: ConvSparseMatrix::set reentrancy" << endl;
}

int main()
{
    try
    {
        test_poisson_matrix_ones_vector();
        test_poisson_matrix_random_vector();
        test_random_coefficient_poisson();
        test_2d_poisson_matrix();
        test_multiply_add_consistency();
        test_conv_info();
        test_set_reentrancy();

        cout << "\n=== All ConvSparseMatrix mat-vec tests passed! ===" << endl;
        return 0;
    }
    catch (const std::exception& e)
    {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }
}
