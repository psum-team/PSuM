#include <psum/batch_pic.hpp>
#include "../../src/field_solver/boundary_creator/Dirichlet.hpp"
#include "../../src/field_solver/boundary_creator/Neumann.hpp"
#include "../../src/field_solver/boundary_creator/Robin.hpp"
#include "../../src/random/rander.hpp"
#include <cmath>
#include <memory>
#include <iostream>
#include <vector>

using namespace std;
using namespace psum;
using namespace psum::field;
using namespace psum::field_solver;
using namespace psum::random;
using namespace psum::batch_pic;
using namespace boundary_creator;

int main() {
    const size_t C = 4;
    const int I = 256;
    grid1D grid({0.0}, {1.0}, {I});
    const int N = grid.contentSize(var_loc::nodeCentered);

    bool ok = true;
    auto check = [&](bool cond, const std::string& what) {
        if (!cond) {
            ok = false;
            std::cout << "❌ " << what << std::endl;
        } else {
            std::cout << "✅ " << what << std::endl;
        }
    };

    auto make_queue = [](auto selector) {
        return sycl::queue{selector, sycl::property::queue::in_order{}};
    };
    sycl::queue q = make_queue(sycl::cpu_selector_v);

    std::vector<std::unique_ptr<Poisson_solver_1d>> ref;
    for (size_t c = 0; c < C; c++) {
        ref.push_back(std::make_unique<Poisson_solver_1d>());
        ref.back()->init(grid,
            {
                Neumann_point(grid, boundary_direction_1d::R) = 1.0 + 0.1 * c,
                Robin_point(grid, boundary_direction_1d::L, std::make_pair(1.0, 1.0)) = 100.0 * c,
            },
            {
                Dirichlet_point(grid, boundary_direction_1d::L) = 0.5 + c,
            });
    }

    batch_poisson_solver_1d batch(C);
    for (size_t c = 0; c < C; c++)
        batch.init(c, grid,
            {
                Neumann_point(grid, boundary_direction_1d::R) = 1.0 + 0.1 * c,
                Robin_point(grid, boundary_direction_1d::L, std::make_pair(1.0, 1.0)) = 100.0 * c,
            },
            {
                Dirichlet_point(grid, boundary_direction_1d::L) = 0.5 + c,
            });
    batch.init_batch();

    rander R;

    batch_host_node_field1D<double> Qb_h(grid, C);
    batch_node_field1D<double> Qb(q, grid, C);
    batch_node_field1D<double> PhiB(q, grid, C);
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
    batch_node_field1D<double> RefB(q, grid, C);
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

    PhiB.plot("batch_phi1d_case%d.plt", "phi");
    RefB.plot("ref_phi1d_case%d.plt", "phi");

    std::cout << (ok ? "ALL PASS" : "FAILED") << std::endl;
    return ok ? 0 : 1;
}
