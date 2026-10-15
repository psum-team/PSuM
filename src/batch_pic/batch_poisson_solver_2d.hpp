#ifndef PSUM_BATCH_PIC_BATCH_POISSON_SOLVER_2D_HPP
#define PSUM_BATCH_PIC_BATCH_POISSON_SOLVER_2D_HPP

#include "../field_solver/Poisson_solver_2d.hpp"
#include "batch_field_solver/mrhs_backend.hpp"
#include <string>
#include <vector>
#include <stdexcept>

namespace psum {

namespace batch_pic {

    // Batch version of field_solver::Poisson_solver_2d.
    // solve() receives row-major matrices and solves all cases at once
    // use init(case_id, ...) to initialize each case and init_batch() before solve().
    class batch_poisson_solver_2d {
    public:
        using psolver = field_solver::Poisson_solver_2d;
        using psolverDomainKind = psolver::psolverDomainKind;
        static constexpr psolverDomainKind Cylindrical = psolverDomainKind::Cylindrical;
        static constexpr psolverDomainKind Cartesian = psolverDomainKind::Cartesian;
        using Grid = psum::field::simple_grid<2>;

        explicit batch_poisson_solver_2d(size_t case_num) : solvers_(case_num) {}

        void init(size_t case_id, const std::string& mrhs_backend_name, const Grid& grid, psolverDomainKind domainType, const field_solver::mixed_boundary_2d_list& boundaries, const field_solver::fixed_boundary_2d_list& fbl, double _epsilon = 8.854e-12) {
            init_impl(case_id, mrhs_backend_name, grid, domainType, &boundaries, fbl, [_epsilon](double, double) { return _epsilon; });
        }

        void init(size_t case_id, const std::string& mrhs_backend_name, const Grid& grid, psolverDomainKind domainType, const field_solver::mixed_boundary_2d_list& boundaries, const field_solver::fixed_boundary_2d_list& fbl, const std::function<double(double, double)>& _epsilon) {
            init_impl(case_id, mrhs_backend_name, grid, domainType, &boundaries, fbl, _epsilon);
        }

        void init(size_t case_id, const std::string& mrhs_backend_name, const Grid& grid, psolverDomainKind domainType, const field_solver::fixed_boundary_2d_list& fbl, double _epsilon = 8.854e-12) {
            init_impl(case_id, mrhs_backend_name, grid, domainType, nullptr, fbl, [_epsilon](double, double) { return _epsilon; });
        }

        void init(size_t case_id, const std::string& mrhs_backend_name, const Grid& grid, psolverDomainKind domainType, const field_solver::fixed_boundary_2d_list& fbl, const std::function<double(double, double)>& _epsilon) {
            init_impl(case_id, mrhs_backend_name, grid, domainType, nullptr, fbl, _epsilon);
        }

        // checks that all held solvers built the exact same coefficient matrix and constructs the MRHS backend
        void init_batch() {
            if (solvers_.empty())
                throw std::runtime_error("batch_poisson_solver_2d::init_batch: no cases");
            const Eigen::SparseMatrix<double>& A0 = solvers_[0].coeff_matrix();
            for (size_t c = 1; c < solvers_.size(); c++) {
                const Eigen::SparseMatrix<double>& A = solvers_[c].coeff_matrix();
                if (A.rows() != A0.rows() || A.cols() != A0.cols() || A.nonZeros() != A0.nonZeros())
                    throw std::runtime_error("batch_poisson_solver_2d::init_batch: coefficient matrices differ across cases");

                Eigen::SparseMatrix<double> diff(A - A0);
                diff.prune(0.0);
                if (diff.nonZeros() != 0)
                    throw std::runtime_error("batch_poisson_solver_2d::init_batch: coefficient matrices differ across cases");
            }

            batch_size_ = solvers_.size();

            // hand the shared coefficient matrix to the MRHS backend (COO,
            // extracted in column order just like the single-solver path)
            const int n_rows = A0.rows();
            const int n_cols = A0.cols();
            const int nnz = A0.nonZeros();
            const int* outerPtr = A0.outerIndexPtr();
            const int* innerIdx = A0.innerIndexPtr();
            const double* values = A0.valuePtr();
            std::vector<unsigned long long> rows, cols;
            std::vector<double> vals;
            rows.reserve(nnz);
            cols.reserve(nnz);
            vals.reserve(nnz);
            for (int k = 0; k < n_cols; ++k) {
                for (int i = outerPtr[k]; i < outerPtr[k+1]; ++i) {
                    rows.push_back(static_cast<unsigned long long>(innerIdx[i]));
                    cols.push_back(static_cast<unsigned long long>(k));
                    vals.push_back(values[i]);
                }
            }
            mrhs_ = batch_field_solver::mrhs_solver_backend(mrhs_backend_name_.c_str());
            mrhs_.set_options(((options_.empty() ? std::string("batch_size=") : options_ + ";batch_size=") +
                               std::to_string(batch_size_)).c_str());
            mrhs_.set_matrix(static_cast<unsigned long long>(n_rows),
                             static_cast<unsigned long long>(n_cols),
                             static_cast<unsigned long long>(nnz),
                             rows.data(), cols.data(), vals.data());
        }

        // row-major input: element (row, case) of phi/source lives at row * batch_size + case. 
        void solve(double* phi, double* source) {
            if (batch_size_ == 0)
                throw std::runtime_error("batch_poisson_solver_2d::solve: init_batch() not called");
            for (size_t c = 0; c < batch_size_; c++) {
                solvers_[c].evaluate_boundary_elements();
                auto& rep = solvers_[c].replace_elements();
                auto& add = solvers_[c].addback_elements();
                mrhs_.set_source_replace(c, rep.first.size(), rep.first.data(), rep.second.data());
                mrhs_.set_source_addback(c, add.first.size(), add.first.data(), add.second.data());
            }
            mrhs_.solve(source, phi);
        }

        void set_options(const std::string& options) {
            if (batch_size_ > 0)
                mrhs_.set_options((options + ";batch_size=" + std::to_string(batch_size_)).c_str());
            else
                options_ = options;
        }

        size_t batch_size() const { return solvers_.size(); }

    private:
        void init_impl(size_t case_id, const std::string& mrhs_backend_name, const Grid& grid, psolverDomainKind domainType, const field_solver::mixed_boundary_2d_list* mbl, const field_solver::fixed_boundary_2d_list& fbl, const std::function<double(double, double)>& _epsilon) {
            if (case_id >= solvers_.size())
                throw std::runtime_error("batch_poisson_solver_2d::init: case_id out of range");
            mrhs_backend_name_ = mrhs_backend_name;
            if (mbl)
                solvers_[case_id].init("dummy", grid, domainType, *mbl, fbl, _epsilon);
            else
                solvers_[case_id].init("dummy", grid, domainType, fbl, _epsilon);
        }

        std::vector<psolver> solvers_;
        batch_field_solver::mrhs_solver_backend mrhs_;
        std::string mrhs_backend_name_;
        std::string options_;
        size_t batch_size_ = 0;
    };

}

}

#endif
