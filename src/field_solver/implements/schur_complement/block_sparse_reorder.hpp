#ifndef PSUM_FIELD_SOLVER_IMPLEMENTS_SCHUR_COMPLEMENT_BLOCK_SPARSE_REORDER_HPP
#define PSUM_FIELD_SOLVER_IMPLEMENTS_SCHUR_COMPLEMENT_BLOCK_SPARSE_REORDER_HPP

#include <Eigen/Sparse>
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace psum {

namespace field_solver {

namespace implements {

namespace schur_complement {

struct block_sparse_reorder_result {
    int n = 0;
    std::vector<int> block_ids;
    std::vector<int> block_offsets;
    std::vector<int> old_to_new;
    std::vector<int> new_to_old;
    Eigen::SparseMatrix<double> A_reordered;
    std::vector<Eigen::SparseMatrix<double>> A_blocks;
};

inline block_sparse_reorder_result reorder_sparse_by_block(const Eigen::SparseMatrix<double>& A,
                                                           const std::vector<int>& node_block,
                                                           bool build_block_matrices = true,
                                                           bool require_block_diagonal = true) {
    const int n = static_cast<int>(A.rows());
    if (static_cast<int>(A.cols()) != n) {
        throw std::runtime_error("block_sparse_reorder: A must be square");
    }
    if (static_cast<int>(node_block.size()) != n) {
        throw std::runtime_error("block_sparse_reorder: node_block size mismatch");
    }

    block_sparse_reorder_result result;
    result.n = n;
    result.old_to_new.assign(n, -1);
    result.new_to_old.resize(n);

    std::vector<int> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int lhs, int rhs) {
        if (node_block[lhs] != node_block[rhs]) {
            return node_block[lhs] < node_block[rhs];
        }
        return lhs < rhs;
    });

    result.block_offsets.clear();
    result.block_offsets.push_back(0);
    for (int new_idx = 0; new_idx < n; ++new_idx) {
        const int old_idx = order[new_idx];
        if (node_block[old_idx] < 0) {
            throw std::runtime_error("block_sparse_reorder: node_block entries must be non-negative");
        }
        if (new_idx == 0 || node_block[old_idx] != result.block_ids.back()) {
            if (new_idx != 0) {
                result.block_offsets.push_back(new_idx);
            }
            result.block_ids.push_back(node_block[old_idx]);
        }
        result.old_to_new[old_idx] = new_idx;
        result.new_to_old[new_idx] = old_idx;
    }
    result.block_offsets.push_back(n);

    std::vector<int> new_to_block(n, -1);
    for (int b = 0; b < static_cast<int>(result.block_ids.size()); ++b) {
        for (int i = result.block_offsets[b]; i < result.block_offsets[b + 1]; ++i) {
            new_to_block[i] = b;
        }
    }

    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(A.nonZeros());

    std::vector<std::vector<Eigen::Triplet<double>>> block_triplets;
    if (build_block_matrices) {
        block_triplets.resize(result.block_ids.size());
    }

    for (int col = 0; col < A.outerSize(); ++col) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(A, col); it; ++it) {
            const int new_row = result.old_to_new[it.row()];
            const int new_col = result.old_to_new[it.col()];
            if (new_row < 0 || new_col < 0) {
                throw std::runtime_error("block_sparse_reorder: invalid permutation");
            }
            const int row_block = new_to_block[new_row];
            const int col_block = new_to_block[new_col];
            if (require_block_diagonal && row_block != col_block) {
                throw std::runtime_error("block_sparse_reorder: A contains cross-block nonzeros");
            }
            triplets.emplace_back(new_row, new_col, it.value());

            if (build_block_matrices && row_block == col_block) {
                const int begin = result.block_offsets[row_block];
                block_triplets[row_block].emplace_back(new_row - begin, new_col - begin, it.value());
            }
        }
    }

    result.A_reordered.resize(n, n);
    result.A_reordered.setFromTriplets(triplets.begin(), triplets.end());
    result.A_reordered.makeCompressed();

    if (build_block_matrices) {
        result.A_blocks.resize(result.block_ids.size());
        for (int b = 0; b < static_cast<int>(result.block_ids.size()); ++b) {
            const int size = result.block_offsets[b + 1] - result.block_offsets[b];
            result.A_blocks[b].resize(size, size);
            result.A_blocks[b].setFromTriplets(block_triplets[b].begin(), block_triplets[b].end());
            result.A_blocks[b].makeCompressed();
        }
    }

    return result;
}

}

}

}

}

#endif
