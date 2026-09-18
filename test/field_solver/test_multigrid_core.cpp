#include "../../src/field_solver/implements/multigrid/multigrid_utilities.hpp"
#include "../../src/field_solver/implements/multigrid/multigrid_core.hpp"
#include <iostream>
#include <cassert>

using namespace std;
using namespace psum;
using namespace psum::field_solver::implements::multigrid;

void build_5point_poisson_matrix_2d(int I, int J,
    std::vector<unsigned long long>& rows,
    std::vector<unsigned long long>& cols,
    std::vector<double>& vals)
{
    for (int j = 0; j < J; ++j)
    {
        for (int i = 0; i < I; ++i)
        {
            int idx = i * J + j;

            bool is_boundary = (i == 0 || i == I - 1 || j == 0 || j == J - 1);

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
    }
}

void build_7point_poisson_matrix_3d(int I, int J, int K,
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

double compute_residual_norm(const Eigen::SparseMatrix<double>& A, const Eigen::VectorXd& x, const Eigen::VectorXd& b)
{
    Eigen::VectorXd Ax = A * x;
    Eigen::VectorXd r = b - Ax;
    return r.norm();
}

void test_multigrid_core_residual_reduction_2d()
{
    cout << "\n=== Test multigrid_core residual reduction (2D) ===" << endl;

    int I = 256, J = 256;
    int depth = 4;
    int num_cycles = 10;

    std::vector<unsigned long long> rows, cols;
    std::vector<double> vals;
    build_5point_poisson_matrix_2d(I, J, rows, cols, vals);

    Eigen::SparseMatrix<double> A(I * J, I * J);
    std::vector<Eigen::Triplet<double>> triplets;
    for (size_t i = 0; i < rows.size(); ++i)
    {
        triplets.emplace_back(rows[i], cols[i], vals[i]);
    }
    A.setFromTriplets(triplets.begin(), triplets.end());

    auto RMats = get_restriction_matrices_2d(I, J, depth);

    using MatrixType = Eigen::SparseMatrix<double>;
    using InnerSolverType = Eigen::SparseLU<Eigen::SparseMatrix<double>>;
    multigrid_core<MatrixType, InnerSolverType> core;
    core.setup(A, RMats, 0.67);
    core.set_relax_counts(2, 1);

    Eigen::VectorXd b = Eigen::VectorXd::Ones(I * J);
    Eigen::VectorXd x = Eigen::VectorXd::Zero(I * J);

    double r0 = compute_residual_norm(A, x, b);
    double r_prev = r0;

    cout << "Problem size: " << I << "x" << J << " = " << I * J << endl;
    cout << "Pre-smoothing: 2, Post-smoothing: 1" << endl;
    cout << "Initial residual norm: " << r0 << endl;
    cout << "Running " << num_cycles << " V-cycles..." << endl;

    for (int cycle = 1; cycle <= num_cycles; ++cycle)
    {
        core.v_cycle(b, x);
        double r = compute_residual_norm(A, x, b);

        double reduction = r / r_prev;
        cout << "Cycle " << cycle << ": residual = " << r << ", reduction factor = " << reduction;

        if (cycle > 1)
        {
            assert(r < r_prev * 1.1 && "Residual should decrease after first cycle");
        }
        else
        {
            cout << " [first cycle]";
        }
        cout << endl;

        r_prev = r;
    }

    double final_reduction = r_prev / r0;
    cout << "Initial residual: " << r0 << endl;
    cout << "Final residual: " << r_prev << endl;
    cout << "Total reduction: " << final_reduction << endl;

    assert(final_reduction < 1e-5 && "Residual should be reduced by factor of at least 1e-5");
    cout << "PASS: Residual reduced by factor of " << final_reduction << endl;
}

void test_multigrid_core_residual_reduction_3d()
{
    cout << "\n=== Test multigrid_core residual reduction (3D) ===" << endl;

    int I = 96, J = 96, K = 96;
    int depth = 3;
    int num_cycles = 25;

    std::vector<unsigned long long> rows, cols;
    std::vector<double> vals;
    build_7point_poisson_matrix_3d(I, J, K, rows, cols, vals);

    Eigen::SparseMatrix<double> A(I * J * K, I * J * K);
    std::vector<Eigen::Triplet<double>> triplets;
    for (size_t i = 0; i < rows.size(); ++i)
    {
        triplets.emplace_back(rows[i], cols[i], vals[i]);
    }
    A.setFromTriplets(triplets.begin(), triplets.end());

    auto RMats = get_restriction_matrices_3d(I, J, K, depth);

    using MatrixType = Eigen::SparseMatrix<double>;
    using InnerSolverType = Eigen::SparseLU<Eigen::SparseMatrix<double>>;
    multigrid_core<MatrixType, InnerSolverType> core;
    core.setup(A, RMats, 0.67);
    core.set_relax_counts(2, 1);

    Eigen::VectorXd b = Eigen::VectorXd::Ones(I * J * K);
    Eigen::VectorXd x = Eigen::VectorXd::Zero(I * J * K);

    double r0 = compute_residual_norm(A, x, b);
    double r_prev = r0;

    cout << "Problem size: " << I << "x" << J << "x" << K << " = " << I * J * K << endl;
    cout << "Pre-smoothing: 2, Post-smoothing: 1" << endl;
    cout << "Initial residual norm: " << r0 << endl;
    cout << "Running " << num_cycles << " V-cycles..." << endl;

    for (int cycle = 1; cycle <= num_cycles; ++cycle)
    {
        core.v_cycle(b, x);
        double r = compute_residual_norm(A, x, b);

        double reduction = r / r_prev;
        cout << "Cycle " << cycle << ": residual = " << r << ", reduction factor = " << reduction;

        if (cycle > 1)
        {
            assert(r < r_prev * 1.1 && "Residual should decrease after first cycle");
        }
        else
        {
            cout << " [first cycle]";
        }
        cout << endl;

        r_prev = r;
    }

    double final_reduction = r_prev / r0;
    cout << "Initial residual: " << r0 << endl;
    cout << "Final residual: " << r_prev << endl;
    cout << "Total reduction: " << final_reduction << endl;

    assert(final_reduction < 1e-5 && "Residual should be reduced by factor of at least 1e-5");
    cout << "PASS: Residual reduced by factor of " << final_reduction << endl;
}

int main()
{
    try
    {
        test_multigrid_core_residual_reduction_2d();
        test_multigrid_core_residual_reduction_3d();

        cout << "\n=== All multigrid_core tests passed! ===" << endl;
        return 0;
    }
    catch (const std::exception& e)
    {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }
}
