#ifndef PSUM_BATCH_PIC_BATCH_POISSON_SOLVER_1D_HPP
#define PSUM_BATCH_PIC_BATCH_POISSON_SOLVER_1D_HPP

#include "../field_solver/Poisson_solver_1d.hpp"
#include <Eigen/SparseLU>
#include <string>
#include <vector>
#include <stdexcept>
#include <functional>

namespace psum {

namespace batch_pic {

    // Batch version of field_solver::Poisson_solver_1d.
    // solve() receives row-major matrices and solves all cases at once
    // use init(case_id, ...) to initialize each case and init_batch() before solve().
    class batch_poisson_solver_1d {
    public:
        using psolver = field_solver::Poisson_solver_1d;
        using Grid = psum::field::simple_grid<1>;

        explicit batch_poisson_solver_1d(size_t case_num) : solvers_(case_num) {}

        void init(size_t case_id, const Grid& grid, const field_solver::mixed_boundary_1d_list& boundaries, const field_solver::fixed_boundary_1d_list& fbl, double _epsilon = 8.854e-12) {
            init_impl(case_id, grid, &boundaries, fbl, [_epsilon](double) { return _epsilon; });
        }

        void init(size_t case_id, const Grid& grid, const field_solver::mixed_boundary_1d_list& boundaries, const field_solver::fixed_boundary_1d_list& fbl, const std::function<double(double)>& _epsilon) {
            init_impl(case_id, grid, &boundaries, fbl, _epsilon);
        }

        void init(size_t case_id, const Grid& grid, const field_solver::fixed_boundary_1d_list& fbl, double _epsilon = 8.854e-12) {
            init_impl(case_id, grid, nullptr, fbl, [_epsilon](double) { return _epsilon; });
        }

        void init(size_t case_id, const Grid& grid, const field_solver::fixed_boundary_1d_list& fbl, const std::function<double(double)>& _epsilon) {
            init_impl(case_id, grid, nullptr, fbl, _epsilon);
        }

        // checks that all held solvers built the exact same coefficient matrix and factors it once
        void init_batch() {
            if (solvers_.empty())
                throw std::runtime_error("batch_poisson_solver_1d::init_batch: no cases");
            const Eigen::SparseMatrix<double>& A0 = solvers_[0].coeff_matrix();
            for (size_t c = 1; c < solvers_.size(); c++) {
                const Eigen::SparseMatrix<double>& A = solvers_[c].coeff_matrix();
                if (A.rows() != A0.rows() || A.cols() != A0.cols() || A.nonZeros() != A0.nonZeros())
                    throw std::runtime_error("batch_poisson_solver_1d::init_batch: coefficient matrices differ across cases");

                Eigen::SparseMatrix<double> diff(A - A0);
                diff.prune(0.0);
                if (diff.nonZeros() != 0)
                    throw std::runtime_error("batch_poisson_solver_1d::init_batch: coefficient matrices differ across cases");
            }

            batch_size_ = solvers_.size();
            solver_.compute(A0);
            if (solver_.info() != Eigen::Success)
                throw std::runtime_error("batch_poisson_solver_1d::init_batch: factorization failed");
        }

        // row-major input: element (row, case) of phi/source lives at row * batch_size + case.
        void solve(double* phi, double* source) {
            if (batch_size_ == 0)
                throw std::runtime_error("batch_poisson_solver_1d::solve: init_batch() not called");
            const Eigen::Index n = solvers_[0].coeff_matrix().rows();
            const Eigen::Index C = batch_size_;

            // boundary terms are applied to source in place, like every other backend
            Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> B(source, n, C);
            for (size_t c = 0; c < batch_size_; c++) {
                solvers_[c].evaluate_boundary_elements();
                auto& rep = solvers_[c].replace_elements();
                auto& add = solvers_[c].addback_elements();
                for (size_t i = 0; i < add.first.size(); i++)
                    B(add.first[i], c) += add.second[i];
                for (size_t i = 0; i < rep.first.size(); i++)
                    B(rep.first[i], c) = rep.second[i];
            }

            // plain SparseLU on a col-major copy of the rhs
            W_.resize(n, C);
            W_ = B;
            W_ = solver_.solve(W_);
            Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> X(phi, n, C);
            X = W_;
        }

        size_t batch_size() const { return solvers_.size(); }

    private:
        void init_impl(size_t case_id, const Grid& grid, const field_solver::mixed_boundary_1d_list* mbl, const field_solver::fixed_boundary_1d_list& fbl, const std::function<double(double)>& _epsilon) {
            if (case_id >= solvers_.size())
                throw std::runtime_error("batch_poisson_solver_1d::init: case_id out of range");
            if (mbl)
                solvers_[case_id].init(grid, *mbl, fbl, _epsilon);
            else
                solvers_[case_id].init(grid, fbl, _epsilon);
        }

        std::vector<psolver> solvers_;
        Eigen::SparseLU<Eigen::SparseMatrix<double>> solver_;
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> W_;
        size_t batch_size_ = 0;
    };

}

}

#endif
