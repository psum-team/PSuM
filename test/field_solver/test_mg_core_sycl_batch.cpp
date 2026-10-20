#include <sycl/sycl.hpp>
#include <Eigen/Sparse>
#include <iostream>
#include <vector>
#include <cmath>
#include <string>

#include "field_solver/implements/multigrid/conv_sycl_sparse_matrix.hpp"
#include "field_solver/implements/multigrid/sycl_inner_solver.hpp"
#include "field_solver/implements/multigrid/multigrid_core.hpp"
#include "field_solver/implements/multigrid/multigrid_utilities.hpp"

namespace mg = psum::field_solver::implements::multigrid;

static int g_fail = 0;

static void check(bool ok, const std::string& what) {
    std::cout << (ok ? "   PASS " : "   FAIL ") << what << std::endl;
    if (!ok) g_fail++;
}

// 3d 7-point Laplacian on a nodes^3 grid (healthy diagonal, SPD)
static Eigen::SparseMatrix<double> build_poisson_3d(int nodes) {
    std::vector<Eigen::Triplet<double>> t;
    t.reserve((size_t)nodes * nodes * nodes * 7);
    auto id = [&](int x, int y, int z) {
        return (x * nodes + y) * nodes + z;
    };
    for (int x = 0; x < nodes; x++)
        for (int y = 0; y < nodes; y++)
            for (int z = 0; z < nodes; z++) {
                t.emplace_back(id(x, y, z), id(x, y, z), 8.0);
                if (x > 0) t.emplace_back(id(x, y, z), id(x - 1, y, z), -1.0);
                if (x + 1 < nodes) t.emplace_back(id(x, y, z), id(x + 1, y, z), -1.0);
                if (y > 0) t.emplace_back(id(x, y, z), id(x, y - 1, z), -1.0);
                if (y + 1 < nodes) t.emplace_back(id(x, y, z), id(x, y + 1, z), -1.0);
                if (z > 0) t.emplace_back(id(x, y, z), id(x, y, z - 1), -1.0);
                if (z + 1 < nodes) t.emplace_back(id(x, y, z), id(x, y, z + 1), -1.0);
            }
    Eigen::SparseMatrix<double> A(nodes * nodes * nodes, nodes * nodes * nodes);
    A.setFromTriplets(t.begin(), t.end());
    return A;
}

int main() {
    try {
        sycl::queue q{sycl::default_selector_v, sycl::property::queue::in_order{}};
        mg::sycl_vector<double>::default_queue_ = &q;

        using Core = mg::multigrid_core<mg::ConvSyclSparseMatrix<double>,
                                        mg::sycl_inner_solver<double>>;

        // 8^3 cells -> 9^3 = 729 nodes, two restriction levels (generator
        // takes NODE counts): 9 -> 5 -> 3, coarse inner solver 27x27
        const int cells = 8, C = 4, cycles = 2;
        const int n = 729;
        Eigen::SparseMatrix<double> A = build_poisson_3d(9);
        auto RMats = mg::get_restriction_matrices_3d(9, 9, 9, 2);

        // per-case right-hand sides, interleaved upload (row * C + case)
        std::vector<double> hb((size_t)n * C);
        for (int r = 0; r < n; r++)
            for (int c = 0; c < C; c++)
                hb[(size_t)r * C + c] = ((r * 7 + c * 13) % 17 - 8) * 0.25 + 0.5 * c;

        // --- batch call: one core, batched vectors ---
        Core batch_core;
        batch_core.setup(C, A, RMats, 8.0);
        mg::sycl_vector<double> Bb, Xb;
        Bb.resize(n, C);
        q.memcpy(Bb.data(), hb.data(), sizeof(double) * hb.size()).wait();
        Xb.resize(n, C);
        Xb.setZero();
        batch_core.v_cycle_multi(Bb, Xb, cycles);
        std::vector<double> xb((size_t)n * C);
        q.memcpy(xb.data(), Xb.data(), sizeof(double) * xb.size()).wait();

        // --- reference: one FRESH core per case (core carries state, so a
        //     single core called repeatedly would NOT be equivalent) ---
        double tol_norm = 0.0;
        std::vector<std::vector<double>> xref(C, std::vector<double>(n));
        for (int c = 0; c < C; c++) {
            Core core_c;
            core_c.setup(A, RMats, 8.0);
            std::vector<double> b1(n);
            for (int r = 0; r < n; r++) b1[r] = hb[(size_t)r * C + c];
            mg::sycl_vector<double> B1, X1;
            B1.resize(n, 1);
            q.memcpy(B1.data(), b1.data(), sizeof(double) * n).wait();
            X1.resize(n, 1);
            X1.setZero();
            core_c.v_cycle_multi(B1, X1, cycles);
            q.memcpy(xref[c].data(), X1.data(), sizeof(double) * n).wait();
            for (int r = 0; r < n; r++)
                tol_norm = std::max(tol_norm, std::abs(xref[c][r]));
        }

        // batched solution vs per-case fresh-core solutions
        for (int c = 0; c < C; c++) {
            double max_diff = 0.0;
            for (int r = 0; r < n; r++)
                max_diff = std::max(max_diff,
                    std::abs(xb[(size_t)r * C + c] - xref[c][r]));
            check(max_diff <= 1e-7 * tol_norm,
                  "mg-core batch == fresh-core-per-case (case " + std::to_string(c) +
                  ", max_diff " + std::to_string(max_diff) + ")");
        }

        std::cout << (g_fail == 0 ? "ALL PASS" : "FAILED") << std::endl;
        return g_fail == 0 ? 0 : 1;
    } catch (const std::exception& e) {
        std::cout << "exception: " << e.what() << std::endl;
        return 2;
    }
}
