#include "../register/interface.h"
#include "schur_complement/block_sparse_reorder.hpp"
#include "schur_complement/schur_complement_core.hpp"
#include <Eigen/Sparse>
#include <Eigen/SparseLU>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <deque>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace psum {

namespace field_solver {

namespace implements {

namespace eigen_schurcomplement_cpu {

struct Block {
    std::vector<int> core_indices;
    Eigen::SparseMatrix<double> A_ii;
    Eigen::SparseLU<Eigen::SparseMatrix<double>> lu;
    bool has_interior = false;
};

struct SolverContext {
    unsigned long long n = 0;
    int I = 0;
    int J = 0;
    int blocks_x = 1;
    int blocks_y = 1;
    int num_threads = 0;
    bool initialized = false;

    Eigen::SparseMatrix<double> A;
    schur_complement::schur_complement_core core;
    std::deque<Block> blocks;
    std::vector<int> interface_nodes;

    std::vector<unsigned long long> source_replace_idxs;
    std::vector<double> source_replace_values;
    std::vector<unsigned long long> source_addback_idxs;
    std::vector<double> source_addback_values;
};

psum_field_solver_handle init() {
    return new SolverContext();
}

template<typename T>
void write_binary(std::ostream& os, const T& value) {
    os.write(reinterpret_cast<const char*>(&value), sizeof(T));
    if (!os) {
        throw std::runtime_error("eigen_schurcomplement_cpu: failed writing state file");
    }
}

template<typename T>
void read_binary(std::istream& is, T& value) {
    is.read(reinterpret_cast<char*>(&value), sizeof(T));
    if (!is) {
        throw std::runtime_error("eigen_schurcomplement_cpu: failed reading state file");
    }
}

void write_int_vector(std::ostream& os, const std::vector<int>& values) {
    const std::uint64_t size = values.size();
    write_binary(os, size);
    if (size > 0) {
        os.write(reinterpret_cast<const char*>(values.data()), sizeof(int) * size);
        if (!os) {
            throw std::runtime_error("eigen_schurcomplement_cpu: failed writing state vector");
        }
    }
}

std::vector<int> read_int_vector(std::istream& is) {
    std::uint64_t size = 0;
    read_binary(is, size);
    std::vector<int> values(size);
    if (size > 0) {
        is.read(reinterpret_cast<char*>(values.data()), sizeof(int) * size);
        if (!is) {
            throw std::runtime_error("eigen_schurcomplement_cpu: failed reading state vector");
        }
    }
    return values;
}

void write_pair_vector(std::ostream& os, const std::vector<std::pair<int, int>>& values) {
    const std::uint64_t size = values.size();
    write_binary(os, size);
    for (const auto& [first, second] : values) {
        write_binary(os, first);
        write_binary(os, second);
    }
}

std::vector<std::pair<int, int>> read_pair_vector(std::istream& is) {
    std::uint64_t size = 0;
    read_binary(is, size);
    std::vector<std::pair<int, int>> values(size);
    for (auto& [first, second] : values) {
        read_binary(is, first);
        read_binary(is, second);
    }
    return values;
}

void write_sparse_matrix(std::ostream& os, const Eigen::SparseMatrix<double>& matrix) {
    Eigen::SparseMatrix<double> compressed(matrix);
    compressed.makeCompressed();
    const int rows = static_cast<int>(compressed.rows());
    const int cols = static_cast<int>(compressed.cols());
    const int nnz = static_cast<int>(compressed.nonZeros());
    write_binary(os, rows);
    write_binary(os, cols);
    write_binary(os, nnz);
    os.write(reinterpret_cast<const char*>(compressed.outerIndexPtr()), sizeof(int) * (cols + 1));
    os.write(reinterpret_cast<const char*>(compressed.innerIndexPtr()), sizeof(int) * nnz);
    os.write(reinterpret_cast<const char*>(compressed.valuePtr()), sizeof(double) * nnz);
    if (!os) {
        throw std::runtime_error("eigen_schurcomplement_cpu: failed writing sparse matrix state");
    }
}

Eigen::SparseMatrix<double> read_sparse_matrix(std::istream& is) {
    int rows = 0;
    int cols = 0;
    int nnz = 0;
    read_binary(is, rows);
    read_binary(is, cols);
    read_binary(is, nnz);
    std::vector<int> outer(cols + 1);
    std::vector<int> inner(nnz);
    std::vector<double> values(nnz);
    is.read(reinterpret_cast<char*>(outer.data()), sizeof(int) * (cols + 1));
    is.read(reinterpret_cast<char*>(inner.data()), sizeof(int) * nnz);
    is.read(reinterpret_cast<char*>(values.data()), sizeof(double) * nnz);
    if (!is) {
        throw std::runtime_error("eigen_schurcomplement_cpu: failed reading sparse matrix state");
    }

    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(nnz);
    for (int col = 0; col < cols; ++col) {
        for (int k = outer[col]; k < outer[col + 1]; ++k) {
            triplets.emplace_back(inner[k], col, values[k]);
        }
    }
    Eigen::SparseMatrix<double> matrix(rows, cols);
    matrix.setFromTriplets(triplets.begin(), triplets.end());
    matrix.makeCompressed();
    return matrix;
}

void write_dense_matrix(std::ostream& os, const Eigen::MatrixXd& matrix) {
    const int rows = static_cast<int>(matrix.rows());
    const int cols = static_cast<int>(matrix.cols());
    write_binary(os, rows);
    write_binary(os, cols);
    const std::uint64_t size = static_cast<std::uint64_t>(rows) * static_cast<std::uint64_t>(cols);
    if (size > 0) {
        os.write(reinterpret_cast<const char*>(matrix.data()), sizeof(double) * size);
        if (!os) {
            throw std::runtime_error("eigen_schurcomplement_cpu: failed writing dense matrix state");
        }
    }
}

Eigen::MatrixXd read_dense_matrix(std::istream& is) {
    int rows = 0;
    int cols = 0;
    read_binary(is, rows);
    read_binary(is, cols);
    Eigen::MatrixXd matrix(rows, cols);
    const std::uint64_t size = static_cast<std::uint64_t>(rows) * static_cast<std::uint64_t>(cols);
    if (size > 0) {
        is.read(reinterpret_cast<char*>(matrix.data()), sizeof(double) * size);
        if (!is) {
            throw std::runtime_error("eigen_schurcomplement_cpu: failed reading dense matrix state");
        }
    }
    return matrix;
}

int infer_square_size(unsigned long long n) {
    const auto root = static_cast<unsigned long long>(std::llround(std::sqrt(static_cast<double>(n))));
    if (root * root != n) {
        throw std::runtime_error("eigen_schurcomplement_cpu: I and J must be provided for non-square systems");
    }
    return static_cast<int>(root);
}

void ensure_grid_shape(SolverContext* ctx) {
    if (ctx->I <= 0 && ctx->J <= 0) {
        ctx->I = infer_square_size(ctx->n);
        ctx->J = ctx->I;
    } else if (ctx->I <= 0) {
        if (ctx->n % static_cast<unsigned long long>(ctx->J) != 0) {
            throw std::runtime_error("eigen_schurcomplement_cpu: invalid J option for matrix size");
        }
        ctx->I = static_cast<int>(ctx->n / static_cast<unsigned long long>(ctx->J));
    } else if (ctx->J <= 0) {
        if (ctx->n % static_cast<unsigned long long>(ctx->I) != 0) {
            throw std::runtime_error("eigen_schurcomplement_cpu: invalid I option for matrix size");
        }
        ctx->J = static_cast<int>(ctx->n / static_cast<unsigned long long>(ctx->I));
    }

    if (static_cast<unsigned long long>(ctx->I) * static_cast<unsigned long long>(ctx->J) != ctx->n) {
        throw std::runtime_error("eigen_schurcomplement_cpu: I * J must equal matrix dimension");
    }

    ctx->blocks_x = std::max(1, std::min(ctx->blocks_x, ctx->I));
    ctx->blocks_y = std::max(1, std::min(ctx->blocks_y, ctx->J));
}

int block_of_node(const SolverContext* ctx, int global) {
    const int i = global / ctx->J;
    const int j = global % ctx->J;
    const int bx = std::min(ctx->blocks_x - 1, (i * ctx->blocks_x) / ctx->I);
    const int by = std::min(ctx->blocks_y - 1, (j * ctx->blocks_y) / ctx->J);
    return bx * ctx->blocks_y + by;
}

void discover_interface(SolverContext* ctx) {
    const int n = static_cast<int>(ctx->n);

    std::vector<int> node_block(n);
    for (int g = 0; g < n; ++g) {
        node_block[g] = block_of_node(ctx, g);
    }

    std::vector<char> is_interface(n, 0);
    for (int col = 0; col < ctx->A.outerSize(); ++col) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(ctx->A, col); it; ++it) {
            const int row = it.row();
            if (row == col) {
                continue;
            }
            if (node_block[row] != node_block[col]) {
                const int interface_node = node_block[row] > node_block[col] ? row : col;
                is_interface[interface_node] = 1;
            }
        }
    }

    std::vector<int> interface_globals;
    for (int g = 0; g < n; ++g) {
        if (is_interface[g]) {
            interface_globals.push_back(g);
        }
    }

    ctx->interface_nodes = std::move(interface_globals);
}

std::vector<int> make_inner_node_blocks(const SolverContext* ctx) {
    std::vector<int> node_block(ctx->core.n_interior);
    for (int i = 0; i < ctx->core.n_interior; ++i) {
        node_block[i] = block_of_node(ctx, ctx->core.interior_map[i].first);
    }
    return node_block;
}

void build_block_lus(SolverContext* ctx) {
    const auto node_block = make_inner_node_blocks(ctx);
    auto reorder = schur_complement::reorder_sparse_by_block(ctx->core.mat_A, node_block, true, true);

    ctx->blocks.clear();
    ctx->blocks.resize(reorder.A_blocks.size());
    std::vector<int> failed(reorder.A_blocks.size(), 0);

#pragma omp parallel for num_threads(ctx->num_threads > 0 ? ctx->num_threads : 1)
    for (int bidx = 0; bidx < static_cast<int>(reorder.A_blocks.size()); ++bidx) {
        auto& block = ctx->blocks[bidx];
        const int begin = reorder.block_offsets[bidx];
        const int end = reorder.block_offsets[bidx + 1];
        const int ni = end - begin;
        block.has_interior = ni > 0;
        block.core_indices.resize(ni);
        for (int i = 0; i < ni; ++i) {
            block.core_indices[i] = reorder.new_to_old[begin + i];
        }
        block.A_ii = reorder.A_blocks[bidx];

        block.lu.analyzePattern(block.A_ii);
        block.lu.factorize(block.A_ii);
        if (block.lu.info() != Eigen::Success) {
            failed[bidx] = 1;
        }
    }

    for (int bidx = 0; bidx < static_cast<int>(failed.size()); ++bidx) {
        if (failed[bidx]) {
            throw std::runtime_error("eigen_schurcomplement_cpu: block factorization failed");
        }
    }
}

void save_state_file(const SolverContext* ctx, const std::string& path) {
    if (!ctx->initialized) {
        throw std::runtime_error("eigen_schurcomplement_cpu: cannot save state before setup");
    }

    std::ofstream os(path, std::ios::binary);
    if (!os) {
        throw std::runtime_error("eigen_schurcomplement_cpu: cannot open state file for write");
    }

    const char magic[16] = {'P','S','U','M','S','C','H','U','R','S','T','A','T','E','1','\0'};
    os.write(magic, sizeof(magic));
    const std::uint32_t version = 1;
    write_binary(os, version);
    write_binary(os, ctx->n);
    write_binary(os, ctx->I);
    write_binary(os, ctx->J);
    write_binary(os, ctx->blocks_x);
    write_binary(os, ctx->blocks_y);
    write_binary(os, ctx->core.n);
    write_binary(os, ctx->core.n_interface);
    write_binary(os, ctx->core.n_interior);
    write_int_vector(os, ctx->interface_nodes);
    write_pair_vector(os, ctx->core.interface_map);
    write_pair_vector(os, ctx->core.interior_map);
    write_sparse_matrix(os, ctx->core.mat_A);
    write_sparse_matrix(os, ctx->core.mat_B);
    write_sparse_matrix(os, ctx->core.mat_C);
    write_dense_matrix(os, ctx->core.S);
    write_dense_matrix(os, ctx->core.S_inv);
}

void load_state_file(SolverContext* ctx, const std::string& path) {
    std::ifstream is(path, std::ios::binary);
    if (!is) {
        throw std::runtime_error("eigen_schurcomplement_cpu: cannot open state file for read");
    }

    const char expected_magic[16] = {'P','S','U','M','S','C','H','U','R','S','T','A','T','E','1','\0'};
    char magic[16] = {};
    is.read(magic, sizeof(magic));
    if (!is || std::memcmp(magic, expected_magic, sizeof(magic)) != 0) {
        throw std::runtime_error("eigen_schurcomplement_cpu: invalid state file");
    }

    std::uint32_t version = 0;
    read_binary(is, version);
    if (version != 1) {
        throw std::runtime_error("eigen_schurcomplement_cpu: unsupported state file version");
    }

    read_binary(is, ctx->n);
    read_binary(is, ctx->I);
    read_binary(is, ctx->J);
    read_binary(is, ctx->blocks_x);
    read_binary(is, ctx->blocks_y);
    read_binary(is, ctx->core.n);
    read_binary(is, ctx->core.n_interface);
    read_binary(is, ctx->core.n_interior);
    ctx->interface_nodes = read_int_vector(is);
    ctx->core.interface_map = read_pair_vector(is);
    ctx->core.interior_map = read_pair_vector(is);
    ctx->core.mat_A = read_sparse_matrix(is);
    ctx->core.mat_B = read_sparse_matrix(is);
    ctx->core.mat_C = read_sparse_matrix(is);
    ctx->core.S = read_dense_matrix(is);
    ctx->core.S_inv = read_dense_matrix(is);
    if (!is) {
        throw std::runtime_error("eigen_schurcomplement_cpu: failed reading state file");
    }
}

void rebuild_runtime_state(SolverContext* ctx) {
    build_block_lus(ctx);
    ctx->initialized = true;
}

void set_matrix(psum_field_solver_handle h, unsigned long long n_row, unsigned long long n_col, unsigned long long nnz, unsigned long long* rows, unsigned long long* cols, double* vals) {
    if (n_row != n_col) {
        throw std::runtime_error("eigen_schurcomplement_cpu: matrix must be square");
    }

    SolverContext* ctx = static_cast<SolverContext*>(h);
    ctx->n = n_row;
    ensure_grid_shape(ctx);

    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(nnz);
    for (unsigned long long i = 0; i < nnz; ++i) {
        triplets.emplace_back(static_cast<int>(rows[i]), static_cast<int>(cols[i]), vals[i]);
    }

    ctx->A.resize(static_cast<int>(n_row), static_cast<int>(n_col));
    ctx->A.setFromTriplets(triplets.begin(), triplets.end());
    ctx->A.makeCompressed();

    discover_interface(ctx);
    ctx->core.setup(ctx->A, ctx->interface_nodes, ctx->num_threads);
    rebuild_runtime_state(ctx);
}

void solve(psum_field_solver_handle h, double* b, double* x) {
    SolverContext* ctx = static_cast<SolverContext*>(h);
    if (!ctx->initialized) {
        throw std::runtime_error("eigen_schurcomplement_cpu: solver not initialized");
    }

    Eigen::Map<Eigen::VectorXd> rhs(b, static_cast<int>(ctx->n));

    for (size_t i = 0; i < ctx->source_addback_idxs.size(); ++i) {
        rhs[ctx->source_addback_idxs[i]] += ctx->source_addback_values[i];
    }
    for (size_t i = 0; i < ctx->source_replace_idxs.size(); ++i) {
        rhs[ctx->source_replace_idxs[i]] = ctx->source_replace_values[i];
    }

    const int interface_count = ctx->core.n_interface;
    Eigen::VectorXd x_interface(interface_count);
    x_interface.setZero();

    Eigen::VectorXd f_I(ctx->core.n_interior);
    for (const auto& [g, l] : ctx->core.interior_map) f_I[l] = rhs[g];

    Eigen::VectorXd y_full = Eigen::VectorXd::Zero(ctx->core.n_interior);

#pragma omp parallel for num_threads(ctx->num_threads > 0 ? ctx->num_threads : 1)
    for (int bidx = 0; bidx < static_cast<int>(ctx->blocks.size()); ++bidx) {
        auto& block = ctx->blocks[bidx];
        const int ni = static_cast<int>(block.core_indices.size());
        if (!block.has_interior) {
            continue;
        }

        Eigen::VectorXd b_i(ni);
        for (int i = 0; i < ni; ++i) {
            b_i[i] = f_I[block.core_indices[i]];
        }
        Eigen::VectorXd y = block.lu.solve(b_i);
        for (int i = 0; i < ni; ++i) {
            y_full[block.core_indices[i]] = y[i];
        }
    }

    if (interface_count > 0) {
        Eigen::VectorXd rhs_interface(interface_count);
        for (int i = 0; i < interface_count; ++i) {
            rhs_interface[i] = rhs[ctx->core.interface_map[i].first];
        }

        rhs_interface -= ctx->core.mat_C * y_full;
        x_interface = ctx->core.S_inv * rhs_interface;
    }

    Eigen::VectorXd b_mod = f_I - ctx->core.mat_B * x_interface;

#pragma omp parallel for num_threads(ctx->num_threads > 0 ? ctx->num_threads : 1)
    for (int bidx = 0; bidx < static_cast<int>(ctx->blocks.size()); ++bidx) {
        auto& block = ctx->blocks[bidx];
        const int ni = static_cast<int>(block.core_indices.size());
        if (!block.has_interior) {
            continue;
        }

        Eigen::VectorXd b_i(ni);
        for (int i = 0; i < ni; ++i) {
            b_i[i] = b_mod[block.core_indices[i]];
        }

        Eigen::VectorXd x_i = block.lu.solve(b_i);
        for (int i = 0; i < ni; ++i) {
            x[ctx->core.interior_map[block.core_indices[i]].first] = x_i[i];
        }
    }

    for (int i = 0; i < interface_count; ++i) {
        x[ctx->core.interface_map[i].first] = x_interface[i];
    }
}

void set_options(psum_field_solver_handle h, const char* options) {
    SolverContext* ctx = static_cast<SolverContext*>(h);
    std::istringstream iss(options);
    std::string key;
    while (iss >> key) {
        if (key == "I") {
            int value;
            iss >> value;
            if (ctx->initialized) {
                throw std::runtime_error("eigen_schurcomplement_cpu: I must be set before set_matrix");
            }
            ctx->I = value;
        } else if (key == "J") {
            int value;
            iss >> value;
            if (ctx->initialized) {
                throw std::runtime_error("eigen_schurcomplement_cpu: J must be set before set_matrix");
            }
            ctx->J = value;
        } else if (key == "K") {
            int ignored;
            iss >> ignored;
        } else if (key == "blocks_x") {
            int value;
            iss >> value;
            if (ctx->initialized) {
                throw std::runtime_error("eigen_schurcomplement_cpu: blocks_x must be set before set_matrix");
            }
            ctx->blocks_x = value;
        } else if (key == "blocks_y") {
            int value;
            iss >> value;
            if (ctx->initialized) {
                throw std::runtime_error("eigen_schurcomplement_cpu: blocks_y must be set before set_matrix");
            }
            ctx->blocks_y = value;
        } else if (key == "num_threads") {
            iss >> ctx->num_threads;
        } else if (key == "save_state") {
            std::string path;
            iss >> path;
            if (path.empty()) {
                throw std::runtime_error("eigen_schurcomplement_cpu: save_state requires a path");
            }
            save_state_file(ctx, path);
        } else if (key == "load_state") {
            std::string path;
            iss >> path;
            if (path.empty()) {
                throw std::runtime_error("eigen_schurcomplement_cpu: load_state requires a path");
            }
            load_state_file(ctx, path);
            rebuild_runtime_state(ctx);
        }
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
    register_solver("eigen_schurcomplement_cpu", table);
    return 0;
}();

}

}

}

}
