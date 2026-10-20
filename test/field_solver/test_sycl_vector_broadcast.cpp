#include <sycl/sycl.hpp>
#include <Eigen/Core>
#include <iostream>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <string>

#include "field_solver/implements/multigrid/sycl_vector.hpp"
#include "field_solver/implements/multigrid/sycl_inner_solver.hpp"
#include <Eigen/SparseCore>

namespace mg = psum::field_solver::implements::multigrid;

static int g_fail = 0;

static void check(bool ok, const std::string& what) {
    std::cout << (ok ? "   PASS " : "   FAIL ") << what << std::endl;
    if (!ok) g_fail++;
}

// element (row, case) of a batched vector lives at row * batch + case
static double at(const std::vector<double>& h, int r, int batch, int c) {
    return h[(size_t)r * batch + c];
}

int main() {
    sycl::queue q{sycl::default_selector_v, sycl::property::queue::in_order{}};
    mg::sycl_vector<double>::default_queue_ = &q;

    const int n = 100000, C = 8;

    std::vector<double> ha((size_t)n * C), ha2((size_t)n * C), hb(n);
    std::vector<std::vector<double>> hac(C, std::vector<double>(n)),
        hac2(C, std::vector<double>(n));
    for (int r = 0; r < n; r++) {
        hb[r] = 2.0 - 0.125 * r;
        for (int c = 0; c < C; c++) {
            ha[(size_t)r * C + c] = 0.5 + r + 0.25 * c;
            ha2[(size_t)r * C + c] = -1.0 + 0.5 * r - 0.125 * c;
            hac[c][r] = at(ha, r, C, c);
            hac2[c][r] = at(ha2, r, C, c);
        }
    }

    auto upload = [&](mg::sycl_vector<double>& v, const std::vector<double>& h) {
        v.resize((int)h.size(), 1);
        q.memcpy(v.data(), h.data(), sizeof(double) * h.size()).wait();
    };
    auto download = [&](mg::sycl_vector<double>& v) {
        std::vector<double> h((size_t)v.rows() * v.batch_size());
        q.memcpy(h.data(), v.data(), sizeof(double) * h.size()).wait();
        return h;
    };

    mg::sycl_vector<double> A, A2, B;
    A.resize(n, C); q.memcpy(A.data(), ha.data(), sizeof(double) * ha.size()).wait();
    A2.resize(n, C); q.memcpy(A2.data(), ha2.data(), sizeof(double) * ha2.size()).wait();
    B.resize(n, 1); q.memcpy(B.data(), hb.data(), sizeof(double) * hb.size()).wait();

    // per-case one-case vectors of A / A2
    std::vector<mg::sycl_vector<double>> Ac(C), Ac2(C);
    for (int c = 0; c < C; c++) { upload(Ac[c], hac[c]); upload(Ac2[c], hac2[c]); }

    // --- binary ops: broadcast(A,B) and broadcast(B,A) vs per-case ---
    {
        auto R = mg::vector_traits<mg::sycl_vector<double>>::add(A, B);
        auto Rh = download(R);
        for (int c = 0; c < C; c++) {
            auto Rc = mg::vector_traits<mg::sycl_vector<double>>::add(Ac[c], B);
            auto Rch = download(Rc);
            bool ok = true;
                for (int r = 0; r < n; r++)
                    if (!(Rh[(size_t)r * C + c] == Rch[r])) { ok = false; break; }
                check(ok, "add(A,B) row  case " + std::to_string(c));
        }
    }
    {
        auto R = mg::vector_traits<mg::sycl_vector<double>>::add(B, A);
        auto Rh = download(R);
        for (int c = 0; c < C; c++) {
            auto Rc = mg::vector_traits<mg::sycl_vector<double>>::add(B, Ac[c]);
            auto Rch = download(Rc);
            bool ok = true;
                for (int r = 0; r < n; r++)
                    if (!(Rh[(size_t)r * C + c] == Rch[r])) { ok = false; break; }
                check(ok, "add(B,A) row  case " + std::to_string(c));
        }
    }
    {
        auto R = mg::vector_traits<mg::sycl_vector<double>>::subtract(A, B);
        auto Rh = download(R);
        for (int c = 0; c < C; c++) {
            auto Rc = mg::vector_traits<mg::sycl_vector<double>>::subtract(Ac[c], B);
            auto Rch = download(Rc);
            bool ok = true;
                for (int r = 0; r < n; r++)
                    if (!(Rh[(size_t)r * C + c] == Rch[r])) { ok = false; break; }
                check(ok, "subtract(A,B) row  case " + std::to_string(c));
        }
    }
    {
        auto R = mg::vector_traits<mg::sycl_vector<double>>::subtract(B, A);
        auto Rh = download(R);
        for (int c = 0; c < C; c++) {
            auto Rc = mg::vector_traits<mg::sycl_vector<double>>::subtract(B, Ac[c]);
            auto Rch = download(Rc);
            bool ok = true;
                for (int r = 0; r < n; r++)
                    if (!(Rh[(size_t)r * C + c] == Rch[r])) { ok = false; break; }
                check(ok, "subtract(B,A) row  case " + std::to_string(c));
        }
    }
    {
        auto R = mg::vector_traits<mg::sycl_vector<double>>::multiply_with_array(A, B);
        auto Rh = download(R);
        for (int c = 0; c < C; c++) {
            auto Rc = mg::vector_traits<mg::sycl_vector<double>>::multiply_with_array(Ac[c], B);
            auto Rch = download(Rc);
            bool ok = true;
                for (int r = 0; r < n; r++)
                    if (!(Rh[(size_t)r * C + c] == Rch[r])) { ok = false; break; }
                check(ok, "multiply(A,B) row  case " + std::to_string(c));
        }
    }
    {
        auto R = mg::vector_traits<mg::sycl_vector<double>>::multiply_with_array(B, A);
        auto Rh = download(R);
        for (int c = 0; c < C; c++) {
            auto Rc = mg::vector_traits<mg::sycl_vector<double>>::multiply_with_array(B, Ac[c]);
            auto Rch = download(Rc);
            bool ok = true;
                for (int r = 0; r < n; r++)
                    if (!(Rh[(size_t)r * C + c] == Rch[r])) { ok = false; break; }
                check(ok, "multiply(B,A) row  case " + std::to_string(c));
        }
    }

    // --- equal-batch sanity: batched op vs per-case ops ---
    {
        auto R = mg::vector_traits<mg::sycl_vector<double>>::add(A, A2);
        auto Rh = download(R);
        for (int c = 0; c < C; c++) {
            auto Rc = mg::vector_traits<mg::sycl_vector<double>>::add(Ac[c], Ac2[c]);
            auto Rch = download(Rc);
            bool ok = true;
                for (int r = 0; r < n; r++)
                    if (!(Rh[(size_t)r * C + c] == Rch[r])) { ok = false; break; }
                check(ok, "add(A,A2) equal-batch row  case " + std::to_string(c));
        }
    }

    // --- hadamard_inplace: batched *= one-case vs per-case *= one-case ---
    {
        mg::sycl_vector<double> X; X.resize(n, C);
        q.memcpy(X.data(), ha.data(), sizeof(double) * ha.size()).wait();
        mg::vector_traits<mg::sycl_vector<double>>::hadamard_inplace(X, B);
        auto Xh = download(X);
        for (int c = 0; c < C; c++) {
            mg::sycl_vector<double> Tc; upload(Tc, hac[c]);
            mg::vector_traits<mg::sycl_vector<double>>::hadamard_inplace(Tc, B);
            auto Tch = download(Tc);
            bool ok = true;
                for (int r = 0; r < n; r++)
                    if (!(Xh[(size_t)r * C + c] == Tch[r])) { ok = false; break; }
                check(ok, "hadamard_inplace row  case " + std::to_string(c));
        }
    }

    // --- hadamard_to: both operand orders vs per-case ---
    {
        mg::sycl_vector<double> out; out.resize(n, C);
        mg::vector_traits<mg::sycl_vector<double>>::hadamard_to(A, B, out);
        auto oh = download(out);
        for (int c = 0; c < C; c++) {
            mg::sycl_vector<double> oc; oc.resize(n, 1);
            mg::vector_traits<mg::sycl_vector<double>>::hadamard_to(Ac[c], B, oc);
            auto och = download(oc);
            bool ok = true;
                for (int r = 0; r < n; r++)
                    if (!(oh[(size_t)r * C + c] == och[r])) { ok = false; break; }
                check(ok, "hadamard_to(A,B) row  case " + std::to_string(c));
        }
    }
    {
        mg::sycl_vector<double> out; out.resize(n, C);
        mg::vector_traits<mg::sycl_vector<double>>::hadamard_to(B, A, out);
        auto oh = download(out);
        for (int c = 0; c < C; c++) {
            mg::sycl_vector<double> oc; oc.resize(n, 1);
            mg::vector_traits<mg::sycl_vector<double>>::hadamard_to(B, Ac[c], oc);
            auto och = download(oc);
            bool ok = true;
                for (int r = 0; r < n; r++)
                    if (!(oh[(size_t)r * C + c] == och[r])) { ok = false; break; }
                check(ok, "hadamard_to(B,A) row  case " + std::to_string(c));
        }
    }

    // --- copy_data broadcast: one-case source fans into every case ---
    {
        mg::sycl_vector<double> D; D.resize(n, C);
        mg::vector_traits<mg::sycl_vector<double>>::copy_data(D, B);
        auto Dh = download(D);
        bool ok = true;
        for (int r = 0; r < n && ok; r++)
            for (int c = 0; c < C; c++)
                if (at(Dh, r, C, c) != hb[r]) ok = false;
        check(ok, "copy_data broadcast == one-case source replicated");
    }

    // --- sycl_inner_solver: batch=1 vs Eigen reference; batch=C vs per-case one-case solves ---
    {
        const int m = 6;
        Eigen::MatrixXd dense(m, m);
        for (int i = 0; i < m; i++)
            for (int j = 0; j < m; j++)
                dense(i, j) = 0.5 + 0.125 * i - 0.25 * j;
        dense = dense * dense.transpose() + (double)m * Eigen::MatrixXd::Identity(m, m);
        Eigen::SparseMatrix<double> As = dense.sparseView();
        Eigen::VectorXd b1_ref(m);
        for (int r = 0; r < m; r++) b1_ref[r] = 1.0 - 0.125 * r;
        Eigen::VectorXd inv_diag_ref = dense.inverse() * b1_ref;

        mg::sycl_inner_solver<double> solver;
        solver.compute(As);

        // batch = 1: GEMV against the uploaded inverse, vs Eigen reference
        std::vector<double> b1(m);
        for (int r = 0; r < m; r++) b1[r] = 1.0 - 0.125 * r;
        mg::sycl_vector<double> B1; upload(B1, b1);
        mg::sycl_vector<double> X1;
        solver.solve_to(B1, X1);
        auto x1 = download(X1);
        bool ok = true;
        for (int r = 0; r < m && ok; r++)
            if (std::abs(x1[r] - inv_diag_ref[r]) > 1e-10) ok = false;
        check(ok, "inner_solver batch=1 vs Eigen reference");

        // batch = 2: interleaved result vs per-case one-case solves
        const int CI = 2;
        std::vector<double> hb2((size_t)m * CI);
        for (int r = 0; r < m; r++)
            for (int c = 0; c < CI; c++)
                hb2[(size_t)r * CI + c] = 1.0 - 0.125 * r + 0.5 * c;
        mg::sycl_vector<double> B2; B2.resize(m, CI);
        q.memcpy(B2.data(), hb2.data(), sizeof(double) * hb2.size()).wait();
        mg::sycl_vector<double> X2;
        solver.solve_to(B2, X2);
        auto x2 = download(X2);
        for (int c = 0; c < CI; c++) {
            std::vector<double> bc(m);
            for (int r = 0; r < m; r++) bc[r] = hb2[(size_t)r * CI + c];
            mg::sycl_vector<double> Bc; upload(Bc, bc);
            mg::sycl_vector<double> Xc;
            solver.solve_to(Bc, Xc);
            auto xch = download(Xc);
            bool ok = true;
            for (int r = 0; r < m; r++)
                if (x2[(size_t)r * CI + c] != xch[r]) { ok = false; break; }
            check(ok, "inner_solver batch broadcast == per-case (case " + std::to_string(c) + ")");
        }
    }

    // --- incompatible batches must throw ---
    {
        bool threw = false;
        try {
            mg::sycl_vector<double> P, Q;
            P.resize(n, 2); Q.resize(n, 3);
            mg::vector_traits<mg::sycl_vector<double>>::add(P, Q);
        } catch (const std::runtime_error&) { threw = true; }
        check(threw, "incompatible batch sizes throw");
    }

    std::cout << (g_fail == 0 ? "ALL PASS" : "FAILED") << std::endl;
    return g_fail == 0 ? 0 : 1;
}
