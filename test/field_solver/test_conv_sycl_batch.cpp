#include <sycl/sycl.hpp>
#include <Eigen/Sparse>
#include <iostream>
#include <vector>
#include <cmath>
#include <string>

#include "field_solver/implements/multigrid/conv_sycl_sparse_matrix.hpp"

namespace mg = psum::field_solver::implements::multigrid;

static int g_fail = 0;

static void check(bool ok, const std::string& what) {
    std::cout << (ok ? "   PASS " : "   FAIL ") << what << std::endl;
    if (!ok) g_fail++;
}

// 1d Laplacian-style matrix: stencil structure (conv groups active under
// the analyzer) plus a few random extra entries (residual rows)
static Eigen::SparseMatrix<double> build_matrix(int n) {
    std::vector<Eigen::Triplet<double>> t;
    t.reserve((size_t)n * 3 + 16);
    for (int i = 0; i < n; i++) {
        t.emplace_back(i, i, 2.0 - 0.01 * (i % 3));
        if (i > 0) t.emplace_back(i, i - 1, -1.0);
        if (i + 1 < n) t.emplace_back(i, i + 1, -1.0);
    }
    for (int k = 0; k < 8; k++) {
        int i = 37 * k + 11, j = (17 * k + 5) % n;
        t.emplace_back(i, j, 0.125);
    }
    Eigen::SparseMatrix<double> A(n, n);
    A.setFromTriplets(t.begin(), t.end());
    return A;
}

int main() {
    try {
        sycl::queue q{sycl::default_selector_v, sycl::property::queue::in_order{}};
        mg::sycl_vector<double>::default_queue_ = &q;

        // sizes near real usage: n = 9^3 coarse-level rows, C = 4 cases
        const int n = 729, C = 4;
        Eigen::SparseMatrix<double> A = build_matrix(n);

        std::vector<double> hx((size_t)n * C), hb((size_t)n * C), hx1(n), hb1(n);
        for (int r = 0; r < n; r++) {
            hx1[r] = 0.5 + 0.01 * r;
            hb1[r] = 1.0 - 0.02 * r;
            for (int c = 0; c < C; c++) {
                hx[(size_t)r * C + c] = 0.5 + 0.01 * r + 0.25 * c;
                hb[(size_t)r * C + c] = 1.0 - 0.02 * r + 0.125 * c;
            }
        }
        Eigen::VectorXd ex1(n), eb1(n);
        for (int r = 0; r < n; r++) { ex1[r] = hx1[r]; eb1[r] = hb1[r]; }

        auto upload = [&](mg::sycl_vector<double>& v, const std::vector<double>& h, int batch) {
            v.resize((int)h.size() / batch, batch);
            q.memcpy(v.data(), h.data(), sizeof(double) * h.size()).wait();
        };
        auto download = [&](auto&& v) {
            std::vector<double> h((size_t)v.rows() * v.batch_size());
            q.memcpy(h.data(), v.data(), sizeof(double) * h.size()).wait();
            return h;
        };
        auto close = [](double a, double b) {
            return std::abs(a - b) <= 1e-9 * (1.0 + std::abs(b));
        };

        // --- batch=1 via set(mat): multiply vs Eigen product ---
        {
            mg::ConvSyclSparseMatrix<double> M;
            M.set(A);
            mg::sycl_vector<double> X, Out;
            upload(X, hx1, 1);
            Out.resize(n, 1);
            mg::matrix_traits<mg::ConvSyclSparseMatrix<double>>::multiply(M, X, Out);
            auto oh = download(Out);
            Eigen::VectorXd ref = A * ex1;
            bool ok = true;
            for (int r = 0; r < n && ok; r++)
                if (!close(oh[r], ref[r])) ok = false;
            check(ok, "batch=1 multiply == A*x");
        }

        // --- batch=4 via set(batch, mat) (residual-dominated analysis) ---
        //     and set(batch, mat, num_conv_mask) (conv groups active):
        //     both must reproduce A*x per case on the interleaved vector
        for (int mode = 0; mode < 2; mode++) {
            mg::ConvSyclSparseMatrix<double> M;
            if (mode == 0) M.set(C, A);
            else M.set(C, A, 5);

            mg::sycl_vector<double> X, Out;
            upload(X, hx, C);
            Out.resize(n, C);
            mg::matrix_traits<mg::ConvSyclSparseMatrix<double>>::multiply(M, X, Out);
            auto oh = download(Out);
            bool ok = true;
            for (int c = 0; c < C && ok; c++) {
                Eigen::VectorXd xc(n);
                for (int r = 0; r < n; r++) xc[r] = hx[(size_t)r * C + c];
                Eigen::VectorXd ref = A * xc;
                for (int r = 0; r < n; r++)
                    if (!close(oh[(size_t)r * C + c], ref[r])) { ok = false; break; }
            }
            check(ok, std::string("batch=4 multiply == per-case A*x (") +
                          (mode == 0 ? "set(C, mat)" : "set(C, mat, mask)") + ")");
        }

        // --- multiply_add: out = A*x*ans_coeff + b*b_coeff*ans_coeff ---
        {
            mg::ConvSyclSparseMatrix<double> M;
            M.set(C, A, 5);
            mg::sycl_vector<double> X, B, Out;
            upload(X, hx, C);
            upload(B, hb, C);
            Out.resize(n, C);
            const double b_coeff = 0.75, ans_coeff = 1.5;
            mg::matrix_traits<mg::ConvSyclSparseMatrix<double>>::multiply_add(
                M, X, B, Out, b_coeff, ans_coeff);
            auto oh = download(Out);
            bool ok = true;
            for (int c = 0; c < C && ok; c++) {
                Eigen::VectorXd xc(n), bc(n);
                for (int r = 0; r < n; r++) {
                    xc[r] = hx[(size_t)r * C + c];
                    bc[r] = hb[(size_t)r * C + c];
                }
                Eigen::VectorXd ref = (A * xc) * ans_coeff + bc * (b_coeff * ans_coeff);
                for (int r = 0; r < n; r++)
                    if (!close(oh[(size_t)r * C + c], ref[r])) { ok = false; break; }
            }
            check(ok, "batch=4 multiply_add == per-case A*x*ans + b*b*ans");
        }

        std::cout << (g_fail == 0 ? "ALL PASS" : "FAILED") << std::endl;
        return g_fail == 0 ? 0 : 1;
    } catch (const std::exception& e) {
        std::cout << "exception: " << e.what() << std::endl;
        return 2;
    }
}
