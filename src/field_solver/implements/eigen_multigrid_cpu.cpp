#include "../register/interface.h"
#include "multigrid/multigrid_solver_conv.hpp"
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <cstring>

namespace psum {

namespace field_solver {

namespace implements {

namespace eigen_multigrid_cpu {

    struct SolverContext {
        multigrid::MultiGridSolverConv solver;
        unsigned long long n_row = 0;
        int I = 0, J = 0, K = 0;
        int depth = 3;
        int num_threads = 0;
        std::vector<unsigned long long> source_replace_idxs;
        std::vector<double> source_replace_values;
        std::vector<unsigned long long> source_addback_idxs;
        std::vector<double> source_addback_values;
    };

    psum_field_solver_handle init() {
        return new SolverContext();
    }

    void set_matrix(psum_field_solver_handle h, unsigned long long n_row, unsigned long long n_col, unsigned long long nnz, unsigned long long* rows, unsigned long long* cols, double* vals) {
        SolverContext* ctx = static_cast<SolverContext*>(h);

        ctx->n_row = n_row;

        ctx->solver.set_matrix_coo(n_row, n_col, nnz, rows, cols, vals);
        ctx->solver.setup_structured_grid(ctx->I, ctx->J, ctx->K, ctx->depth);

        if (ctx->num_threads > 0) {
            ctx->solver.core_.relax_mats_[0].set_thread_num(ctx->num_threads);
            for (size_t i = 0; i < ctx->solver.core_.restrict_mats_.size(); i++) {
                ctx->solver.core_.restrict_mats_[i].set_thread_num(ctx->num_threads);
                ctx->solver.core_.interpolate_mats_[i].set_thread_num(ctx->num_threads);
            }
            for (size_t i = 0; i < ctx->solver.core_.a_mats_.size(); i++) {
                ctx->solver.core_.a_mats_[i].set_thread_num(ctx->num_threads);
            }
        }
    }

    void solve(psum_field_solver_handle h, double* b, double* x) {
        SolverContext* ctx = static_cast<SolverContext*>(h);

        Eigen::Map<Eigen::VectorXd> rhs(b, ctx->n_row);
        Eigen::VectorXd result(ctx->n_row);

        for (size_t i = 0; i < ctx->source_addback_idxs.size(); ++i) {
            rhs[ctx->source_addback_idxs[i]] += ctx->source_addback_values[i];
        }
        
        for (size_t i = 0; i < ctx->source_replace_idxs.size(); ++i) {
            rhs[ctx->source_replace_idxs[i]] = ctx->source_replace_values[i];
        }

        ctx->solver.solve(rhs, result);

        std::memcpy(x, result.data(), sizeof(double) * ctx->n_row);
    }

    void set_options(psum_field_solver_handle h, const char* options) {
        SolverContext* ctx = static_cast<SolverContext*>(h);

        std::istringstream iss(options);
        std::string key;
        bool k_provided = false;
        while (iss >> key) {
            if (key == "I") {
                iss >> ctx->I;
            } else if (key == "J") {
                iss >> ctx->J;
            } else if (key == "K") {
                iss >> ctx->K;
                k_provided = true;
            } else if (key == "depth") {
                iss >> ctx->depth;
            } else if (key == "pre_relax") {
                int pre; iss >> pre; ctx->solver.pre_relax_ = pre;
            } else if (key == "post_relax") {
                int post; iss >> post; ctx->solver.post_relax_ = post;
            } else if (key == "max_cycles") {
                int max; iss >> max; ctx->solver.max_cycles_ = max;
            } else if (key == "num_conv_mask") {
                int num; iss >> num; ctx->solver.set_num_conv_mask(num);
            } else if (key == "num_threads") {
                int num; iss >> num; ctx->num_threads = num;
            } else if (key == "num_conv_relax") {
                int num; iss >> num; ctx->solver.num_conv_mask_relax_ = num;
            } else if (key == "num_conv_restrict") {
                int num; iss >> num; ctx->solver.num_conv_mask_restrict_ = num;
            } else if (key == "num_conv_interpolate") {
                int num; iss >> num; ctx->solver.num_conv_mask_interpolate_ = num;
            } else if (key == "num_conv_a") {
                int num; iss >> num; ctx->solver.num_conv_mask_a_ = num;
            }
        }

        if (!k_provided) {
            ctx->K = 1;
        }
    }

    void set_source_replace(psum_field_solver_handle h, unsigned long long size, unsigned long long* idxs, double* new_values) {
        SolverContext* ctx = static_cast<SolverContext*>(h);
        ctx->source_replace_idxs.clear();
        ctx->source_replace_values.clear();
        ctx->source_replace_idxs.reserve(size);
        ctx->source_replace_values.reserve(size);
        for (unsigned long long i = 0; i < size; ++i) {
            ctx->source_replace_idxs.push_back(idxs[i]);
            ctx->source_replace_values.push_back(new_values[i]);
        }
    }

    void set_source_addback(psum_field_solver_handle h, unsigned long long size, unsigned long long* idxs, double* add_values) {
        SolverContext* ctx = static_cast<SolverContext*>(h);
        ctx->source_addback_idxs.clear();
        ctx->source_addback_values.clear();
        ctx->source_addback_idxs.reserve(size);
        ctx->source_addback_values.reserve(size);
        for (unsigned long long i = 0; i < size; ++i) {
            ctx->source_addback_idxs.push_back(idxs[i]);
            ctx->source_addback_values.push_back(add_values[i]);
        }
    }

    void solver_free(psum_field_solver_handle h) {
        delete static_cast<SolverContext*>(h);
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
        register_solver("eigen_multigrid_cpu", table);
        return 0;
    }();

}

}

}

}
