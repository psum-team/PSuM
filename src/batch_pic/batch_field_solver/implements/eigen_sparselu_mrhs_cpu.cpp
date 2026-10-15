#include "../../../field_solver/implements/umfpack_lu_decompose.hpp"
#include "../register/interface.h"
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <stdexcept>
#include <string>
#include <vector>

namespace psum {

namespace batch_pic {

namespace implements {

namespace eigen_sparselu_mrhs_cpu {

    struct EigenLUSolverMRHS {
        int N = 0;
        int batch_size = 0;
        bool initialized = false;
        Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic, int> Pmat, Qmat;   // P^T and Q^T of the factorization
        Eigen::SparseMatrix<double> L, U;
        std::vector<std::vector<unsigned long long>> source_replace_idxs;
        std::vector<std::vector<double>> source_replace_values;
        std::vector<std::vector<unsigned long long>> source_addback_idxs;
        std::vector<std::vector<double>> source_addback_values;
    };

    psum_batch_field_solver_handle init() {
        return new EigenLUSolverMRHS();
    }

    void set_matrix(psum_batch_field_solver_handle h, unsigned long long n_row, unsigned long long n_col, unsigned long long nnz, unsigned long long* rows, unsigned long long* cols, double* vals) {
        EigenLUSolverMRHS* solver = static_cast<EigenLUSolverMRHS*>(h);
        solver->N = (int)n_row;

        Eigen::SparseMatrix<double> matrix;
        matrix.resize(n_row, n_col);
        matrix.setZero();

        std::vector<Eigen::Triplet<double>> triplets;
        triplets.reserve(nnz);
        for (unsigned long long i = 0; i < nnz; ++i) {
            triplets.emplace_back(rows[i], cols[i], vals[i]);
        }
        matrix.setFromTriplets(triplets.begin(), triplets.end());

        auto decomp = psum::field_solver::implements::umfpack_lu_decompose(matrix);
        solver->Pmat = decomp.P;
        solver->Qmat = decomp.Q;
        solver->L = std::move(decomp.L);
        solver->U = std::move(decomp.U);
        solver->initialized = true;
    }

    void solve(psum_batch_field_solver_handle h, double* b, double* x) {
        EigenLUSolverMRHS* solver = static_cast<EigenLUSolverMRHS*>(h);
        if (!solver->initialized || solver->batch_size <= 0) {
            throw std::runtime_error("Solver not initialized");
        }

        const Eigen::Index n = solver->N;
        const Eigen::Index C = solver->batch_size;

        Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> B(b, n, C);
        for (Eigen::Index c = 0; c < C; ++c) {
            for (size_t i = 0; i < solver->source_addback_idxs[c].size(); ++i) {
                B(solver->source_addback_idxs[c][i], c) += solver->source_addback_values[c][i];
            }
            for (size_t i = 0; i < solver->source_replace_idxs[c].size(); ++i) {
                B(solver->source_replace_idxs[c][i], c) = solver->source_replace_values[c][i];
            }
        }

        Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> X(x, n, C);
        X = solver->Pmat * B;
        solver->L.triangularView<Eigen::UnitLower>().solveInPlace(X);
        solver->U.triangularView<Eigen::Upper>().solveInPlace(X);
        X = solver->Qmat * X;
    }

    void set_options(psum_batch_field_solver_handle h, const char* options) {
        EigenLUSolverMRHS* solver = static_cast<EigenLUSolverMRHS*>(h);
        if (options == nullptr) return;
        std::string opt(options);
        size_t pos = 0;
        while (pos < opt.size()) {
            size_t sep = opt.find_first_of(";,", pos);
            std::string pair = opt.substr(pos, sep == std::string::npos ? std::string::npos : sep - pos);
            pos = (sep == std::string::npos) ? opt.size() : sep + 1;
            size_t eq = pair.find('=');
            if (eq == std::string::npos) continue;
            std::string key = pair.substr(0, eq);
            std::string value = pair.substr(eq + 1);
            auto trim = [](std::string& v) {
                while (!v.empty() && (v.front() == ' ' || v.front() == '\t')) v.erase(v.begin());
                while (!v.empty() && (v.back() == ' ' || v.back() == '\t')) v.pop_back();
            };
            trim(key);
            trim(value);
            if (key == "batch_size") {
                int b = std::max(1, std::stoi(value));
                solver->batch_size = b;
                solver->source_replace_idxs.resize(b);
                solver->source_replace_values.resize(b);
                solver->source_addback_idxs.resize(b);
                solver->source_addback_values.resize(b);
            }
        }
    }

    void set_source_replace(psum_batch_field_solver_handle h, unsigned long long case_id, unsigned long long size, unsigned long long* idxs, double* new_values) {
        EigenLUSolverMRHS* solver = static_cast<EigenLUSolverMRHS*>(h);
        if ((size_t)case_id >= solver->source_replace_idxs.size())
            throw std::runtime_error("case_id out of range, set batch_size first");
        solver->source_replace_idxs[case_id].assign(idxs, idxs + size);
        solver->source_replace_values[case_id].assign(new_values, new_values + size);
    }

    void set_source_addback(psum_batch_field_solver_handle h, unsigned long long case_id, unsigned long long size, unsigned long long* idxs, double* add_values) {
        EigenLUSolverMRHS* solver = static_cast<EigenLUSolverMRHS*>(h);
        if ((size_t)case_id >= solver->source_addback_idxs.size())
            throw std::runtime_error("case_id out of range, set batch_size first");
        solver->source_addback_idxs[case_id].assign(idxs, idxs + size);
        solver->source_addback_values[case_id].assign(add_values, add_values + size);
    }

    void solver_free(psum_batch_field_solver_handle h) {
        delete static_cast<EigenLUSolverMRHS*>(h);
    }

    static int _ = []() {
        batch_func_table table;
        table.init = init;
        table.set_matrix = set_matrix;
        table.solve = solve;
        table.set_options = set_options;
        table.set_source_replace = set_source_replace;
        table.set_source_addback = set_source_addback;
        table.free = solver_free;
        register_batch_solver("eigen_sparselu_mrhs_cpu", table);
        return 0;
    }();

}

}

}

}
