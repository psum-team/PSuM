#ifndef PSUM_FIELD_SOLVER_IMPLEMENTS_MULTIGRID_SOLVER_CONV_HPP
#define PSUM_FIELD_SOLVER_IMPLEMENTS_MULTIGRID_SOLVER_CONV_HPP

#include "multigrid_core.hpp"
#include "multigrid_utilities.hpp"
#include "convSparseMatrix.hpp"
#include <Eigen/SparseLU>
#include <Eigen/Sparse>
#include <stdexcept>
#include <iostream>

namespace psum {

namespace field_solver {

namespace implements {

namespace multigrid {

struct MultiGridSolverConv {
    using MatrixType = ConvSparseMatrix<double>;
    using InnerSolverType = Eigen::SparseLU<Eigen::SparseMatrix<double>>;

    multigrid_core<MatrixType, InnerSolverType> core_;
    int max_cycles_ = 10;
    int pre_relax_ = 1;
    int post_relax_ = 1;
    double c_coeff_ = 2.0;
    int num_conv_mask_ = 5;
    int num_conv_mask_relax_ = 5;
    int num_conv_mask_restrict_ = 5;
    int num_conv_mask_interpolate_ = 5;
    int num_conv_mask_a_ = 5;
    bool initialized_ = false;

    inline void set_num_conv_mask(int num_conv_mask)
    {
        num_conv_mask_ = num_conv_mask;
        num_conv_mask_relax_ = num_conv_mask;
        num_conv_mask_restrict_ = num_conv_mask;
        num_conv_mask_interpolate_ = num_conv_mask;
        num_conv_mask_a_ = num_conv_mask;
    }

    inline void set_conv_masks(int conv_relax, int conv_restrict, int conv_interpolate, int conv_a)
    {
        num_conv_mask_relax_ = conv_relax;
        num_conv_mask_restrict_ = conv_restrict;
        num_conv_mask_interpolate_ = conv_interpolate;
        num_conv_mask_a_ = conv_a;
    }

    inline void set_matrix_coo(unsigned long long n_row, unsigned long long n_col,
                               unsigned long long nnz, unsigned long long* rows,
                               unsigned long long* cols, double* vals)
    {
        Eigen::SparseMatrix<double> A(n_row, n_col);
        A.setZero();

        std::vector<Eigen::Triplet<double>> triplets;
        for (unsigned long long i = 0; i < nnz; ++i) {
            triplets.emplace_back(rows[i], cols[i], vals[i]);
        }
        A.setFromTriplets(triplets.begin(), triplets.end());
        matrix_ = A;
    }

    inline void setup_structured_grid(int I, int J, int K, int depth)
    {
        if (!matrix_.rows())
            throw std::runtime_error("Matrix not set");

        std::vector<Eigen::SparseMatrix<double>> RMats;
        if (K <= 1) {
            RMats = get_restriction_matrices_2d(I, J, depth);
        } else {
            RMats = get_restriction_matrices_3d(I, J, K, depth);
        }
        core_.setup(matrix_, RMats, c_coeff_);
        core_.relax_mats_[0].set(core_.relax_mats_[0].original_mat, num_conv_mask_relax_);
        for (size_t i = 0; i < core_.restrict_mats_.size(); i++)
        {
            core_.restrict_mats_[i].set(core_.restrict_mats_[i].original_mat, num_conv_mask_restrict_);
            core_.interpolate_mats_[i].set(core_.interpolate_mats_[i].original_mat, num_conv_mask_interpolate_);
        }
        for (size_t i = 0; i < core_.a_mats_.size(); i++)
        {
            core_.a_mats_[i].set(core_.a_mats_[i].original_mat, num_conv_mask_a_);
        }

        initialized_ = true;
    }

    inline double get_residual_norm(const Eigen::VectorXd& b, const Eigen::VectorXd& x) const
    {
        return (b - matrix_ * x).norm();
    }

    inline void solve(const Eigen::VectorXd& b, Eigen::VectorXd& x)
    {
        core_.set_relax_counts(pre_relax_, post_relax_);
        if (!initialized_)
            throw std::runtime_error("Solver not initialized");

        for (int i = 0; i < max_cycles_; ++i)
        {
            core_.v_cycle(b, x);
        }
    }

private:
    Eigen::SparseMatrix<double> matrix_;
};

}

}

}

}

#endif
