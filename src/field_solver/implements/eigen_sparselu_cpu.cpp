#include "../register/interface.h"
#include <Eigen/Sparse>
#include <Eigen/SparseLU>
#include <iostream>

namespace psum {

namespace field_solver {

namespace implements {

namespace eigen_sparselu_cpu {

    struct EigenLUSolver {
        Eigen::SparseMatrix<double> matrix;
        Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
        bool initialized = false;
        std::vector<unsigned long long> source_replace_idxs;
        std::vector<double> source_replace_values;
        std::vector<unsigned long long> source_addback_idxs;
        std::vector<double> source_addback_values;
    };

    psum_field_solver_handle init() {
        return new EigenLUSolver();
    }

    void set_matrix(psum_field_solver_handle h, unsigned long long n_row, unsigned long long n_col, unsigned long long nnz, unsigned long long* rows, unsigned long long* cols, double* vals) {
        EigenLUSolver* solver = static_cast<EigenLUSolver*>(h);

        solver->matrix.resize(n_row, n_col);
        solver->matrix.setZero();

        std::vector<Eigen::Triplet<double>> triplets;
        for (unsigned long long i = 0; i < nnz; ++i) {
            triplets.emplace_back(rows[i], cols[i], vals[i]);
        }
        solver->matrix.setFromTriplets(triplets.begin(), triplets.end());
        solver->solver.analyzePattern(solver->matrix);
        solver->solver.factorize(solver->matrix);
        solver->initialized = true;
    }

    void solve(psum_field_solver_handle h, double* b, double* x) {
        EigenLUSolver* solver = static_cast<EigenLUSolver*>(h);
        if (!solver->initialized) {
            throw std::runtime_error("Solver not initialized");
        }
        
        for (size_t i = 0; i < solver->source_addback_idxs.size(); ++i) {
            b[solver->source_addback_idxs[i]] += solver->source_addback_values[i];
        }
        
        for (size_t i = 0; i < solver->source_replace_idxs.size(); ++i) {
            b[solver->source_replace_idxs[i]] = solver->source_replace_values[i];
        }
        
        Eigen::Map<Eigen::VectorXd> rhs(b, solver->matrix.rows());
        Eigen::Map<Eigen::VectorXd> result(x, solver->matrix.rows());
        result = solver->solver.solve(rhs);
    }

    void set_options(psum_field_solver_handle, const char*) {
    }

    void set_source_replace(psum_field_solver_handle h, unsigned long long size, unsigned long long* idxs, double* new_values) {
        EigenLUSolver* solver = static_cast<EigenLUSolver*>(h);
        solver->source_replace_idxs.clear();
        solver->source_replace_values.clear();
        solver->source_replace_idxs.reserve(size);
        solver->source_replace_values.reserve(size);
        for (unsigned long long i = 0; i < size; ++i) {
            solver->source_replace_idxs.push_back(idxs[i]);
            solver->source_replace_values.push_back(new_values[i]);
        }
    }

    void set_source_addback(psum_field_solver_handle h, unsigned long long size, unsigned long long* idxs, double* add_values) {
        EigenLUSolver* solver = static_cast<EigenLUSolver*>(h);
        solver->source_addback_idxs.clear();
        solver->source_addback_values.clear();
        solver->source_addback_idxs.reserve(size);
        solver->source_addback_values.reserve(size);
        for (unsigned long long i = 0; i < size; ++i) {
            solver->source_addback_idxs.push_back(idxs[i]);
            solver->source_addback_values.push_back(add_values[i]);
        }
    }

    void solver_free(psum_field_solver_handle h) {
        delete static_cast<EigenLUSolver*>(h);
    }

    static int _ = []() {
        func_table table;
        table.init = init;
        table.set_matrix = set_matrix;
        table.solve = solve;
        table.set_options = set_options;
        table.set_source_replace = set_source_replace;
        table.set_source_addback = set_source_addback;
        table.free = solver_free;
        register_solver("eigen_sparselu_cpu", table);
        return 0;
    }();

}

}

}

}
