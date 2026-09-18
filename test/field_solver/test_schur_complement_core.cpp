#include "../../src/field_solver/implements/schur_complement/schur_complement_core.hpp"
#include <Eigen/Sparse>
#include <Eigen/SparseLU>
#include <cmath>
#include <iostream>
#include <vector>

using namespace psum::field_solver::implements::schur_complement;

void build_five_point_stencil(int nx, int ny,
                              std::vector<unsigned long long>& rows,
                              std::vector<unsigned long long>& cols,
                              std::vector<double>& vals,
                              std::vector<double>& b) {
    const int n = nx * ny;
    rows.clear();
    cols.clear();
    vals.clear();
    b.assign(n, 1.0);

    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny; ++j) {
            const int idx = i * ny + j;
            rows.push_back(idx);
            cols.push_back(idx);
            vals.push_back(4.0);

            if (i > 0) {
                rows.push_back(idx);
                cols.push_back((i - 1) * ny + j);
                vals.push_back(-1.0);
            }
            if (i + 1 < nx) {
                rows.push_back(idx);
                cols.push_back((i + 1) * ny + j);
                vals.push_back(-1.0);
            }
            if (j > 0) {
                rows.push_back(idx);
                cols.push_back(i * ny + (j - 1));
                vals.push_back(-1.0);
            }
            if (j + 1 < ny) {
                rows.push_back(idx);
                cols.push_back(i * ny + (j + 1));
                vals.push_back(-1.0);
            }
        }
    }
}

Eigen::SparseMatrix<double> coo_to_sparse(int n,
                                          const std::vector<unsigned long long>& rows,
                                          const std::vector<unsigned long long>& cols,
                                          const std::vector<double>& vals) {
    Eigen::SparseMatrix<double> A(n, n);
    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(vals.size());
    for (size_t k = 0; k < vals.size(); ++k) {
        triplets.emplace_back(static_cast<int>(rows[k]), static_cast<int>(cols[k]), vals[k]);
    }
    A.setFromTriplets(triplets.begin(), triplets.end());
    return A;
}

std::vector<int> pick_interface_nodes_2d(int nx, int ny, int bx, int by) {
    const int n = nx * ny;
    std::vector<int> node_block(n);
    for (int g = 0; g < n; ++g) {
        const int i = g / ny;
        const int j = g % ny;
        const int bxi = std::min(bx - 1, i * bx / nx);
        const int byj = std::min(by - 1, j * by / ny);
        node_block[g] = bxi * by + byj;
    }

    std::vector<char> is_interface(n, 0);
    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny; ++j) {
            const int idx = i * ny + j;
            if (i > 0 && node_block[idx] != node_block[(i - 1) * ny + j]) {
                const int other = (i - 1) * ny + j;
                const int interface_node = node_block[idx] > node_block[other] ? idx : other;
                is_interface[interface_node] = 1;
            }
            if (j > 0 && node_block[idx] != node_block[i * ny + (j - 1)]) {
                const int other = i * ny + (j - 1);
                const int interface_node = node_block[idx] > node_block[other] ? idx : other;
                is_interface[interface_node] = 1;
            }
        }
    }

    std::vector<int> selected;
    for (int g = 0; g < n; ++g) {
        if (is_interface[g]) selected.push_back(g);
    }
    return selected;
}

void solve_with_core(const schur_complement_core& core,
                     const double* b_global, double* x_global) {
    Eigen::VectorXd f_I(core.n_interior);
    Eigen::VectorXd f_Q(core.n_interface);

    for (const auto& [g, l] : core.interior_map) {
        f_I[l] = b_global[g];
    }
    for (const auto& [g, l] : core.interface_map) {
        f_Q[l] = b_global[g];
    }

    Eigen::SparseLU<Eigen::SparseMatrix<double>> lu;
    lu.analyzePattern(core.mat_A);
    lu.factorize(core.mat_A);

    Eigen::VectorXd y = lu.solve(f_I);
    Eigen::VectorXd rhs_Q = f_Q - core.mat_C * y;
    Eigen::VectorXd x_Q = core.S_inv * rhs_Q;
    Eigen::VectorXd x_I = lu.solve(f_I - core.mat_B * x_Q);

    for (const auto& [g, l] : core.interior_map) {
        x_global[g] = x_I[l];
    }
    for (const auto& [g, l] : core.interface_map) {
        x_global[g] = x_Q[l];
    }
}

int test_case(int nx, int ny, int bx, int by) {
    const int n = nx * ny;
    std::cout << "=== " << nx << "x" << ny << " grid, " << bx << "x" << by
              << " blocks ===" << std::endl;

    std::vector<unsigned long long> rows, cols;
    std::vector<double> vals, b_data;
    build_five_point_stencil(nx, ny, rows, cols, vals, b_data);

    Eigen::SparseMatrix<double> A = coo_to_sparse(n, rows, cols, vals);

    std::vector<int> selected = pick_interface_nodes_2d(nx, ny, bx, by);
    std::cout << "  interface nodes: " << selected.size()
              << " / " << n << std::endl;

    if (selected.empty()) {
        std::cout << "  SKIP: no interface nodes (single block)" << std::endl;
        return 0;
    }

    schur_complement_core core;
    core.setup(A, selected, 4);

    if (core.n_interface != static_cast<int>(selected.size())) {
        std::cerr << "  FAIL: n_interface mismatch" << std::endl;
        return 1;
    }
    if (core.n_interior + core.n_interface != n) {
        std::cerr << "  FAIL: dimension mismatch" << std::endl;
        return 1;
    }

    std::vector<double> x_schur(n, 0.0);
    solve_with_core(core, b_data.data(), x_schur.data());

    Eigen::Map<const Eigen::VectorXd> b_vec(b_data.data(), n);
    Eigen::Map<const Eigen::VectorXd> x_schur_vec(x_schur.data(), n);
    double residual = (A * x_schur_vec - b_vec).norm() / b_vec.norm();
    std::cout << "  schur residual: " << residual << std::endl;

    Eigen::VectorXd x_direct = Eigen::SparseLU<Eigen::SparseMatrix<double>>(A).solve(b_vec);
    double diff = (x_schur_vec - x_direct).norm() / x_direct.norm();
    std::cout << "  diff vs direct: " << diff << std::endl;

    if (residual > 1e-8) {
        std::cerr << "  FAIL: residual too large" << std::endl;
        return 1;
    }
    if (diff > 1e-8) {
        std::cerr << "  FAIL: diff too large" << std::endl;
        return 1;
    }

    std::cout << "  PASS" << std::endl;
    return 0;
}

int block_of_node_2d(int global, int nx, int ny, int blocks_x, int blocks_y) {
    const int i = global / ny;
    const int j = global % ny;
    const int bx = std::min(blocks_x - 1, i * blocks_x / nx);
    const int by = std::min(blocks_y - 1, j * blocks_y / ny);
    return bx * blocks_y + by;
}

struct InteriorBlock {
    std::vector<int> core_indices;
    Eigen::SparseMatrix<double> A_ii;
    Eigen::SparseLU<Eigen::SparseMatrix<double>> lu;
};

int test_block_split(int nx, int ny, int blocks_x, int blocks_y) {
    const int n = nx * ny;
    const int num_blocks = blocks_x * blocks_y;
    std::cout << "=== block split: " << nx << "x" << ny << " grid, "
              << blocks_x << "x" << blocks_y << " blocks ===" << std::endl;

    std::vector<unsigned long long> rows, cols;
    std::vector<double> vals, b_data;
    build_five_point_stencil(nx, ny, rows, cols, vals, b_data);
    Eigen::SparseMatrix<double> A = coo_to_sparse(n, rows, cols, vals);

    std::vector<int> selected = pick_interface_nodes_2d(nx, ny, blocks_x, blocks_y);
    if (selected.empty()) {
        std::cout << "  SKIP: no interface nodes" << std::endl;
        return 0;
    }

    schur_complement_core core;
    core.setup(A, selected, 4);

    std::vector<int> block_of_interior(core.n_interior);
    std::vector<int> core_to_block_local(core.n_interior);
    std::vector<InteriorBlock> blocks(num_blocks);

    for (int c = 0; c < core.n_interior; ++c) {
        int global = core.interior_map[c].first;
        int b = block_of_node_2d(global, nx, ny, blocks_x, blocks_y);
        block_of_interior[c] = b;
        core_to_block_local[c] = static_cast<int>(blocks[b].core_indices.size());
        blocks[b].core_indices.push_back(c);
    }

    for (int b = 0; b < num_blocks; ++b) {
        const int nb = static_cast<int>(blocks[b].core_indices.size());
        if (nb == 0) continue;

        std::vector<Eigen::Triplet<double>> triplets;
        for (int local_col = 0; local_col < nb; ++local_col) {
            int core_col = blocks[b].core_indices[local_col];
            for (Eigen::SparseMatrix<double>::InnerIterator it(core.mat_A, core_col); it; ++it) {
                int core_row = it.row();
                if (block_of_interior[core_row] != b) {
                    std::cerr << "  FAIL: cross-block nonzero at core ("
                              << core_row << "," << core_col << ")" << std::endl;
                    return 1;
                }
                triplets.emplace_back(core_to_block_local[core_row], local_col, it.value());
            }
        }

        blocks[b].A_ii.resize(nb, nb);
        blocks[b].A_ii.setFromTriplets(triplets.begin(), triplets.end());
        blocks[b].A_ii.makeCompressed();
        blocks[b].lu.analyzePattern(blocks[b].A_ii);
        blocks[b].lu.factorize(blocks[b].A_ii);
        if (blocks[b].lu.info() != Eigen::Success) {
            std::cerr << "  FAIL: block " << b << " factorization" << std::endl;
            return 1;
        }
    }

    {
        std::vector<Eigen::Triplet<double>> triplets;
        for (int b = 0; b < num_blocks; ++b) {
            const auto& blk = blocks[b];
            for (int lc = 0; lc < static_cast<int>(blk.core_indices.size()); ++lc) {
                int cc = blk.core_indices[lc];
                for (Eigen::SparseMatrix<double>::InnerIterator it(blk.A_ii, lc); it; ++it) {
                    triplets.emplace_back(blk.core_indices[it.row()], cc, it.value());
                }
            }
        }
        Eigen::SparseMatrix<double> recon(core.n_interior, core.n_interior);
        recon.setFromTriplets(triplets.begin(), triplets.end());
        double diff = (Eigen::MatrixXd(core.mat_A - recon)).norm();
        std::cout << "  A reconstruction diff: " << diff << std::endl;
        if (diff > 1e-12) {
            std::cerr << "  FAIL: A reconstruction" << std::endl;
            return 1;
        }
    }

    {
        Eigen::VectorXd f_I(core.n_interior), f_Q(core.n_interface);
        for (const auto& [g, l] : core.interior_map) f_I[l] = b_data[g];
        for (const auto& [g, l] : core.interface_map) f_Q[l] = b_data[g];

        Eigen::VectorXd y_full = Eigen::VectorXd::Zero(core.n_interior);
        for (int b = 0; b < num_blocks; ++b) {
            const auto& blk = blocks[b];
            const int nb = static_cast<int>(blk.core_indices.size());
            if (nb == 0) continue;
            Eigen::VectorXd f_b(nb);
            for (int i = 0; i < nb; ++i) f_b[i] = f_I[blk.core_indices[i]];
            Eigen::VectorXd y_b = blk.lu.solve(f_b);
            for (int i = 0; i < nb; ++i) y_full[blk.core_indices[i]] = y_b[i];
        }

        Eigen::VectorXd x_Q = core.S_inv * (f_Q - core.mat_C * y_full);
        Eigen::VectorXd b_mod = f_I - core.mat_B * x_Q;

        Eigen::VectorXd x_I(core.n_interior);
        for (int b = 0; b < num_blocks; ++b) {
            const auto& blk = blocks[b];
            const int nb = static_cast<int>(blk.core_indices.size());
            if (nb == 0) continue;
            Eigen::VectorXd bm_b(nb);
            for (int i = 0; i < nb; ++i) bm_b[i] = b_mod[blk.core_indices[i]];
            Eigen::VectorXd x_b = blk.lu.solve(bm_b);
            for (int i = 0; i < nb; ++i) x_I[blk.core_indices[i]] = x_b[i];
        }

        std::vector<double> x_block(n, 0.0);
        for (const auto& [g, l] : core.interior_map) x_block[g] = x_I[l];
        for (const auto& [g, l] : core.interface_map) x_block[g] = x_Q[l];

        Eigen::Map<const Eigen::VectorXd> b_vec(b_data.data(), n);
        Eigen::Map<const Eigen::VectorXd> x_vec(x_block.data(), n);
        double residual = (A * x_vec - b_vec).norm() / b_vec.norm();
        std::cout << "  block-solve residual: " << residual << std::endl;

        std::vector<double> x_seq(n, 0.0);
        solve_with_core(core, b_data.data(), x_seq.data());
        Eigen::Map<const Eigen::VectorXd> x_seq_vec(x_seq.data(), n);
        double diff_seq = (x_vec - x_seq_vec).norm();
        std::cout << "  diff vs sequential core: " << diff_seq << std::endl;

        if (residual > 1e-8 || diff_seq > 1e-12) {
            std::cerr << "  FAIL" << std::endl;
            return 1;
        }
    }

    std::cout << "  PASS" << std::endl;
    return 0;
}

int main() {
    int fail = 0;

    fail += test_case(12, 10, 3, 2);
    fail += test_case(20, 20, 4, 4);
    fail += test_case(30, 25, 5, 5);
    fail += test_case(15, 15, 2, 2);

    {
        std::cout << "=== identity matrix, all-interface ===" << std::endl;
        const int n = 10;
        Eigen::SparseMatrix<double> I_mat(n, n);
        I_mat.setIdentity();
        std::vector<int> sel = {0, 2, 4, 6, 8};

        schur_complement_core core;
        core.setup(I_mat, sel);

        std::vector<double> b_data(n, 1.0);
        for (int i = 0; i < n; ++i) b_data[i] = static_cast<double>(i + 1);

        std::vector<double> x_schur(n, 0.0);
        solve_with_core(core, b_data.data(), x_schur.data());

        Eigen::Map<const Eigen::VectorXd> b_vec(b_data.data(), n);
        Eigen::Map<const Eigen::VectorXd> x_vec(x_schur.data(), n);
        double residual = (I_mat * x_vec - b_vec).norm();
        std::cout << "  residual: " << residual << std::endl;
        if (residual > 1e-12) {
            std::cerr << "  FAIL" << std::endl;
            fail++;
        } else {
            std::cout << "  PASS" << std::endl;
        }
    }

    {
        std::cout << "=== zero interface (all interior) ===" << std::endl;
        const int n = 10;
        Eigen::SparseMatrix<double> A(n, n);
        A.setIdentity();

        schur_complement_core core;
        try {
            core.setup(A, {});
            std::cerr << "  FAIL: expected setup error" << std::endl;
            fail++;
        } catch (const std::runtime_error&) {
            std::cout << "  PASS" << std::endl;
        }
    }

    fail += test_block_split(12, 10, 3, 2);
    fail += test_block_split(20, 20, 4, 4);
    fail += test_block_split(30, 25, 5, 5);
    fail += test_block_split(15, 15, 2, 2);
    fail += test_block_split(8, 8, 1, 1);

    std::cout << std::endl;
    if (fail == 0) {
        std::cout << "ALL TESTS PASSED" << std::endl;
    } else {
        std::cout << fail << " TEST(S) FAILED" << std::endl;
    }
    return fail;
}
