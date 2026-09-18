#include "../register/interface.h"
#include "multigrid/multigrid_utilities.hpp"
#include "multigrid/multigrid_core.hpp"
#include "multigrid/convSparseMatrix.hpp"
#include "multigrid/sycl_vector.hpp"
#include "multigrid/conv_sycl_sparse_matrix.hpp"
#include "multigrid/sycl_inner_solver.hpp"
#include <sycl/sycl.hpp>
#include <Eigen/SparseCore>
#include <iostream>
#include <sstream>
#include <cstring>

namespace psum {

namespace field_solver {

namespace implements {

namespace sycl_multigrid_gpu {

    struct SolverContext {
        sycl::queue queue;
        multigrid::multigrid_core<multigrid::ConvSyclSparseMatrix<double>,
                                  multigrid::sycl_inner_solver<double>> core;
        unsigned long long n_row = 0;
        int I = 0, J = 0, K = 0;
        int depth = 3;
        int pre_relax = 2;
        int post_relax = 1;
        int max_cycles = 10;
        int num_conv_mask = 5;
        int num_conv_mask_relax = 5;
        int num_conv_mask_restrict = 5;
        int num_conv_mask_interpolate = 5;
        int num_conv_mask_a = 5;
        Eigen::SparseMatrix<double> matrix;
        std::vector<unsigned long long> source_replace_idxs;
        std::vector<double> source_replace_values;
        unsigned long long* d_source_replace_idxs = nullptr;
        double* d_source_replace_values = nullptr;
        unsigned long long source_replace_size = 0;
        std::vector<unsigned long long> source_addback_idxs;
        std::vector<double> source_addback_values;
        unsigned long long* d_source_addback_idxs = nullptr;
        double* d_source_addback_values = nullptr;
        unsigned long long source_addback_size = 0;
    };

    psum_field_solver_handle init() {
        SolverContext* ctx = new SolverContext();
        sycl::device dev = sycl::device{sycl::gpu_selector_v};
        if (!dev.is_gpu()) {
            delete ctx;
            throw std::runtime_error("sycl_multigrid_gpu: no GPU device available");
        }
        ctx->queue = sycl::queue{dev, {sycl::property::queue::in_order{}}};
        multigrid::sycl_vector<double>::default_queue_ = &ctx->queue;
        return ctx;
    }

    void set_matrix(psum_field_solver_handle h, unsigned long long n_row,
                    unsigned long long n_col, unsigned long long nnz,
                    unsigned long long* rows, unsigned long long* cols, double* vals) {
        SolverContext* ctx = static_cast<SolverContext*>(h);

        ctx->n_row = n_row;

        Eigen::SparseMatrix<double> A(n_row, n_col);
        std::vector<Eigen::Triplet<double>> triplets;
        triplets.reserve(nnz);
        for (unsigned long long i = 0; i < nnz; ++i)
            triplets.emplace_back(rows[i], cols[i], vals[i]);
        A.setFromTriplets(triplets.begin(), triplets.end());
        ctx->matrix = A;

        multigrid::sycl_vector<double>::default_queue_ = &ctx->queue;

        std::vector<Eigen::SparseMatrix<double>> RMats;
        if (ctx->K <= 1)
            RMats = multigrid::get_restriction_matrices_2d(ctx->I, ctx->J, ctx->depth);
        else
            RMats = multigrid::get_restriction_matrices_3d(ctx->I, ctx->J, ctx->K, ctx->depth);

        double c_coeff = ctx->K <= 1 ? 2.0 : 8.0;
        ctx->core.setup(A, RMats, c_coeff);
        ctx->core.set_relax_counts(ctx->pre_relax, ctx->post_relax);

        for (size_t i = 0; i < ctx->core.relax_mats_.size(); i++)
            ctx->core.relax_mats_[i].set(ctx->core.relax_mats_[i].original_mat_, ctx->num_conv_mask_relax);
        for (size_t i = 0; i < ctx->core.restrict_mats_.size(); i++) {
            ctx->core.restrict_mats_[i].set(ctx->core.restrict_mats_[i].original_mat_, ctx->num_conv_mask_restrict);
            ctx->core.interpolate_mats_[i].set(ctx->core.interpolate_mats_[i].original_mat_, ctx->num_conv_mask_interpolate);
        }
        for (size_t i = 0; i < ctx->core.a_mats_.size(); i++)
            ctx->core.a_mats_[i].set(ctx->core.a_mats_[i].original_mat_, ctx->num_conv_mask_a);
    }

    void solve(psum_field_solver_handle h, double* b, double* x) {
        SolverContext* ctx = static_cast<SolverContext*>(h);

        multigrid::sycl_vector<double>::default_queue_ = &ctx->queue;

        if (ctx->source_addback_size > 0) {
            ctx->queue.submit([&](sycl::handler& cgh) {
                auto size = ctx->source_addback_size;
                auto idxs = ctx->d_source_addback_idxs;
                auto vals = ctx->d_source_addback_values;
                cgh.parallel_for(sycl::range<1>(size), [=](sycl::id<1> i) {
                    b[idxs[i]] += vals[i];
                });
            });
        }

        if (ctx->source_replace_size > 0) {
            ctx->queue.submit([&](sycl::handler& cgh) {
                auto size = ctx->source_replace_size;
                auto idxs = ctx->d_source_replace_idxs;
                auto vals = ctx->d_source_replace_values;
                cgh.parallel_for(sycl::range<1>(size), [=](sycl::id<1> i) {
                    b[idxs[i]] = vals[i];
                });
            });
        }

        multigrid::sycl_vector<double> d_b, d_x;
        d_b.data_ = b;
        d_b.size_ = static_cast<int>(ctx->n_row);
        d_x.data_ = x;
        d_x.size_ = static_cast<int>(ctx->n_row);

        ctx->core.set_relax_counts(ctx->pre_relax, ctx->post_relax);
        ctx->core.v_cycle_multi(d_b, d_x, ctx->max_cycles);
        ctx->queue.wait();

        d_b.data_ = nullptr;
        d_x.data_ = nullptr;
    }

    void set_options(psum_field_solver_handle h, const char* options) {
        SolverContext* ctx = static_cast<SolverContext*>(h);

        std::istringstream iss(options);
        std::string key;
        bool k_provided = false;
        while (iss >> key) {
            if (key == "I") iss >> ctx->I;
            else if (key == "J") iss >> ctx->J;
            else if (key == "K") { iss >> ctx->K; k_provided = true; }
            else if (key == "depth") iss >> ctx->depth;
            else if (key == "pre_relax") iss >> ctx->pre_relax;
            else if (key == "post_relax") iss >> ctx->post_relax;
            else if (key == "max_cycles") iss >> ctx->max_cycles;
            else if (key == "num_conv_mask") {
                iss >> ctx->num_conv_mask;
                ctx->num_conv_mask_relax = ctx->num_conv_mask;
                ctx->num_conv_mask_restrict = ctx->num_conv_mask;
                ctx->num_conv_mask_interpolate = ctx->num_conv_mask;
                ctx->num_conv_mask_a = ctx->num_conv_mask;
            }
            else if (key == "num_conv_mask_relax") iss >> ctx->num_conv_mask_relax;
            else if (key == "num_conv_mask_restrict") iss >> ctx->num_conv_mask_restrict;
            else if (key == "num_conv_mask_interpolate") iss >> ctx->num_conv_mask_interpolate;
            else if (key == "num_conv_mask_a") iss >> ctx->num_conv_mask_a;
        }
        if (!k_provided)
            ctx->K = 1;
    }

    void upload_source_to_device_(SolverContext* ctx) {
        if (ctx->d_source_replace_idxs) sycl::free(ctx->d_source_replace_idxs, ctx->queue);
        if (ctx->d_source_replace_values) sycl::free(ctx->d_source_replace_values, ctx->queue);
        if (ctx->d_source_addback_idxs) sycl::free(ctx->d_source_addback_idxs, ctx->queue);
        if (ctx->d_source_addback_values) sycl::free(ctx->d_source_addback_values, ctx->queue);

        ctx->source_replace_size = ctx->source_replace_idxs.size();
        ctx->source_addback_size = ctx->source_addback_idxs.size();

        if (ctx->source_replace_size > 0) {
            ctx->d_source_replace_idxs = sycl::malloc_device<unsigned long long>(ctx->source_replace_size, ctx->queue);
            ctx->d_source_replace_values = sycl::malloc_device<double>(ctx->source_replace_size, ctx->queue);
            ctx->queue.memcpy(ctx->d_source_replace_idxs, ctx->source_replace_idxs.data(),
                              sizeof(unsigned long long) * ctx->source_replace_size);
            ctx->queue.memcpy(ctx->d_source_replace_values, ctx->source_replace_values.data(),
                              sizeof(double) * ctx->source_replace_size);
        }

        if (ctx->source_addback_size > 0) {
            ctx->d_source_addback_idxs = sycl::malloc_device<unsigned long long>(ctx->source_addback_size, ctx->queue);
            ctx->d_source_addback_values = sycl::malloc_device<double>(ctx->source_addback_size, ctx->queue);
            ctx->queue.memcpy(ctx->d_source_addback_idxs, ctx->source_addback_idxs.data(),
                              sizeof(unsigned long long) * ctx->source_addback_size);
            ctx->queue.memcpy(ctx->d_source_addback_values, ctx->source_addback_values.data(),
                              sizeof(double) * ctx->source_addback_size);
        }

        ctx->queue.wait();
    }

    void set_source_replace(psum_field_solver_handle h, unsigned long long size,
                            unsigned long long* idxs, double* new_values) {
        SolverContext* ctx = static_cast<SolverContext*>(h);
        ctx->source_replace_idxs.assign(idxs, idxs + size);
        ctx->source_replace_values.assign(new_values, new_values + size);
        upload_source_to_device_(ctx);
    }

    void set_source_addback(psum_field_solver_handle h, unsigned long long size,
                            unsigned long long* idxs, double* add_values) {
        SolverContext* ctx = static_cast<SolverContext*>(h);
        ctx->source_addback_idxs.assign(idxs, idxs + size);
        ctx->source_addback_values.assign(add_values, add_values + size);
        upload_source_to_device_(ctx);
    }

    void solver_free(psum_field_solver_handle h) {
        SolverContext* ctx = static_cast<SolverContext*>(h);
        if (ctx->d_source_replace_idxs) sycl::free(ctx->d_source_replace_idxs, ctx->queue);
        if (ctx->d_source_replace_values) sycl::free(ctx->d_source_replace_values, ctx->queue);
        if (ctx->d_source_addback_idxs) sycl::free(ctx->d_source_addback_idxs, ctx->queue);
        if (ctx->d_source_addback_values) sycl::free(ctx->d_source_addback_values, ctx->queue);
        delete ctx;
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
        register_solver("sycl_multigrid_gpu", table);
        return 0;
    }();

}

}

}

}
