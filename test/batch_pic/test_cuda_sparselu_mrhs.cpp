#include "../../src/batch_pic/batch_field_solver/register/interface.h"
#include "../../src/field_solver/register/interface.h"
#include "../../src/random/rander.hpp"
#include "../../src/timer.hpp"

using namespace psum::random;
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <vector>

static void grid_laplacian(int nx, int ny, std::vector<unsigned long long>& rows, std::vector<unsigned long long>& cols, std::vector<double>& vals) {
    auto add = [&](int r, int c, double v) {
        rows.push_back(r);
        cols.push_back(c);
        vals.push_back(v);
    };
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            int r = j * nx + i;
            add(r, r, 4.0);
            if (i > 0) add(r, r - 1, -1.0);
            if (i < nx - 1) add(r, r + 1, -1.0);
            if (j > 0) add(r, r - nx, -1.0);
            if (j < ny - 1) add(r, r + nx, -1.0);
        }
    }
}

int run(bool record_time = false) {
    const int nx = 256, ny = 512;
    const int N = nx * ny;     // 2D Poisson problem size
    const int BatchSize = 16;  // 16 random right-hand vectors

    batch_func_table mrhs = get_batch_solver_funcs("cuda_sparselu_mrhs_gpu");
    func_table single = get_solver_funcs("cuda_sparselu_gpu");
    if (mrhs.init == nullptr || single.init == nullptr) {
        printf("❌ backends not registered\n");
        return 1;
    }

    std::vector<unsigned long long> rows, cols;
    std::vector<double> vals;
    grid_laplacian(nx, ny, rows, cols, vals);

    // right-hand side: 16 random vectors (col-major matrix)
    rander R;
    std::vector<double> pool((size_t)N * BatchSize);
    for (auto& v : pool) v = 2.0 * R() - 1.0;

    psum_field_solver_handle hs = single.init();
    single.set_matrix(hs, N, N, vals.size(), rows.data(), cols.data(), vals.data());
    psum_batch_field_solver_handle hm = mrhs.init();
    mrhs.set_matrix(hm, N, N, vals.size(), rows.data(), cols.data(), vals.data());
    mrhs.set_options(hm, "batch_size=16");

    double *d_b, *d_x;
    cudaMalloc(&d_b, (size_t)N * BatchSize * sizeof(double));
    cudaMalloc(&d_x, (size_t)N * BatchSize * sizeof(double));

    std::vector<double> x_single((size_t)N * BatchSize);
    for (int c = 0; c < BatchSize; c++) {
        if(record_time) Tic("single_solve");
        cudaMemcpy(d_b, pool.data() + (size_t)c * N, (size_t)N * sizeof(double), cudaMemcpyHostToDevice);
        single.solve(hs, d_b, d_x + (size_t)c * N);
        if(record_time) Toc
    }
    cudaMemcpy(x_single.data(), d_x, x_single.size() * sizeof(double), cudaMemcpyDeviceToHost);

    std::vector<double> b_reshape((size_t)N * BatchSize);
    std::vector<double> x_mrhs((size_t)N * BatchSize);
    if(record_time) Tic("rhs_reshape");
    for (int r = 0; r < N; r++)
        for (int c = 0; c < BatchSize; c++)
            b_reshape[(size_t)r * BatchSize + c] = pool[(size_t)c * N + r];
    if(record_time) TocTic("mrhs_solve");
    cudaMemcpy(d_b, b_reshape.data(), b_reshape.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemset(d_x, 0, b_reshape.size() * sizeof(double));
    mrhs.solve(hm, d_b, d_x);
    cudaMemcpy(x_mrhs.data(), d_x, x_mrhs.size() * sizeof(double), cudaMemcpyDeviceToHost);
    if(record_time) Toc
    single.free(hs);
    mrhs.free(hm);
    cudaFree(d_b);
    cudaFree(d_x);

    // x_single is case-major, x_mrhs is interleaved: compare (r, c) to (r, c)
    double x_norm = 0.0, max_abs_diff = 0.0;
    for (int c = 0; c < BatchSize; c++) {
        for (int r = 0; r < N; r++) {
            double a = x_single[(size_t)c * N + r];
            double b = x_mrhs[(size_t)r * BatchSize + c];
            x_norm = std::max(x_norm, std::abs(a));
            max_abs_diff = std::max(max_abs_diff, std::abs(a - b));
        }
    }
    double rel = max_abs_diff / x_norm;
    bool ok = rel <= 1e-12;
    printf("%s batch solve == repeated single solves (max rel diff %.2e)\n", ok ? "✅" : "❌", rel);
    printf(ok ? "ALL PASS\n" : "FAILED\n");
    return ok ? 0 : 1;
}

int main() {
    for (int i = 0; i < 10; i++) {
        if (run() != 0) {
            return 1;
        }
    }
    for (int i = 0; i < 10; i++) {
        if (run(true) != 0) {
            return 1;
        }
    }
    PrintTimer
}