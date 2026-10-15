#include <psum/batch_pic.hpp>
#include "../../src/field_solver/boundary_creator/Dirichlet.hpp"
#include "../../src/field_solver/boundary_creator/Neumann.hpp"
#include "../../src/field_solver/boundary_creator/Robin.hpp"
#include "../../src/random/rander.hpp"
#include "../../src/timer.hpp"
#include <psum/field.hpp>
#include <cmath>
#include <iostream>
#include <memory>
#include <vector>

using namespace std;
using namespace psum;
using namespace psum::field;
using namespace psum::field_solver;
using namespace psum::random;
using namespace psum::batch_pic;
using namespace boundary_creator;
using PSolver = field_solver::Poisson_solver_2d;

static bool is_gpu(const string& backend) {
    return backend.size() >= 3 && backend.compare(backend.size() - 3, 3, "gpu") == 0;
}

int main(int argc, char** argv) {
    const string backend = argc > 1 ? argv[1] : "native";
    const string mrhs_backend = argc > 2 ? argv[2] : "eigen_sparselu_mrhs_cpu";
    const int gx = argc > 3 ? atoi(argv[3]) : 128;
    const int gy = argc > 4 ? atoi(argv[4]) : 128;
    const bool gpu_world = is_gpu(mrhs_backend);
    if (is_gpu(mrhs_backend) != is_gpu(backend)) {
        std::cout << "backend pair must share one device world (both _cpu or both _gpu)" << std::endl;
        return 1;
    }
    std::cout << "backend " << backend << " | mrhs " << mrhs_backend << " | grid " << gx << "x" << gy << std::endl;

    bool ok = true;
    auto check = [&](bool cond, const std::string& what) {
        if (!cond) {
            ok = false;
            std::cout << "❌ " << what << std::endl;
        } else {
            std::cout << "✅ " << what << std::endl;
        }
    };

    const size_t C = 4;
    grid2D grid({0.0, 0.0}, {1.0, 1.0}, {gx, gy});
    const int N = (gx + 1) * (gy + 1);
    auto make_queue = [](auto selector) {
        return sycl::queue{selector, sycl::property::queue::in_order{}};
    };
    sycl::queue q = gpu_world ? make_queue(sycl::gpu_selector_v) : make_queue(sycl::cpu_selector_v);

    std::vector<std::unique_ptr<PSolver>> ref;
    for (size_t c = 0; c < C; c++) {
        ref.push_back(std::make_unique<PSolver>());
        ref.back()->init(
            backend,
            grid,
            PSolver::Cartesian,
            {
                Neumann_line(grid, boundary_direction_2d::E) = 0.0,
                Neumann_line(grid, boundary_direction_2d::W) = 10.0 + c,
                Robin_Line(grid, boundary_direction_2d::N, {1, 1}) = 100.0 * c,
            },
            {
                Dirichlet_line(grid, boundary_direction_2d::S) = 0.5 + c,
            });
    }

    batch_poisson_solver_2d batch(C);
    for (size_t c = 0; c < C; c++)
        batch.init(
            c,
            mrhs_backend,
            grid,
            PSolver::Cartesian,
            {
                Neumann_line(grid, boundary_direction_2d::E) = 0.0,
                Neumann_line(grid, boundary_direction_2d::W) = 10.0 + c,
                Robin_Line(grid, boundary_direction_2d::N, {1, 1}) = 100.0 * c,
            },
            {
                Dirichlet_line(grid, boundary_direction_2d::S) = 0.5 + c,
            });
    batch.init_batch();

    rander R;

    batch_host_node_field2D<double> Qb_h(grid, C);
    batch_node_field2D<double> Qb(q, grid, C);
    batch_node_field2D<double> PhiB(q, grid, C);
    device_array<double> source_packing(q, (size_t)N * C);
    device_array<double> phi_packing(q, (size_t)N * C);

    for (size_t c = 0; c < C; c++) {
        double* s = reinterpret_cast<double*>(Qb_h.at(c).data());
        for (int r = 0; r < N; r++) s[r] = (2.0 * R() - 1.0) * 1e-6;
    }
    Qb.copy(Qb_h.getContent());
    packing(Qb, source_packing);
    batch.solve(phi_packing.data(), source_packing.data());
    unpacking(phi_packing, PhiB);

    bool all = true;
    double x_norm = 0.0, max_abs_diff = 0.0;
    std::vector<double> case_rel(C, 0.0);
    batch_node_field2D<double> RefB(q, grid, C);
    auto PhiB_h = PhiB.getContent().to_host();
    for (size_t c = 0; c < C; c++) {
        ref[c]->solve(RefB.at(c).data(), Qb.at(c).data());
        std::vector<double> phi_case = RefB.at(c).getContent().to_host();
        double case_norm = 0.0, case_diff = 0.0;
        for (int r = 0; r < N; r++) {
            double a = phi_case[r];
            double b2 = PhiB_h[c][r];
            x_norm = std::max(x_norm, std::abs(a));
            max_abs_diff = std::max(max_abs_diff, std::abs(a - b2));
            case_norm = std::max(case_norm, std::abs(a));
            case_diff = std::max(case_diff, std::abs(a - b2));
        }
        case_rel[c] = case_diff / case_norm;
    }
    for (size_t c = 0; c < C; c++)
        std::cout << "   case " << c << " rel diff " << case_rel[c] << std::endl;
    double rel = max_abs_diff / x_norm;

    std::cout << "   batch-vs-ref max rel diff " << rel << std::endl;
    all = rel <= 1e-12;
    check(all, "batch solve == per-case solves (rel <= 1e-12)");

    PhiB.plot("batch_phi_case%d.plt", "phi");
    RefB.plot("ref_phi_case%d.plt", "phi");

    PrintTimer

    std::cout << (ok ? "ALL PASS" : "FAILED") << std::endl;
    return ok ? 0 : 1;
}
