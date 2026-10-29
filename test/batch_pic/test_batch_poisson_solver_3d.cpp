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
using PSolver = field_solver::Poisson_solver_3d;

int main() {
    const string backend = "sycl_multigrid_gpu";
    const string mrhs_backend = "sycl_multigrid_mrhs_gpu";
    const int gx = 96;
    const size_t C = 8;

    std::cout << "backend " << backend << " | mrhs " << mrhs_backend << " | grid " << gx << "^3 | C " << C << std::endl;

    bool ok = true;
    auto check = [&](bool cond, const std::string& what) {
        if (!cond) {
            ok = false;
            std::cout << "❌ " << what << std::endl;
        } else {
            std::cout << "✅ " << what << std::endl;
        }
    };

    grid3D grid({0.0, 0.0, 0.0}, {1.0, 1.0, 1.0}, {gx, gx, gx});
    const int N = grid.contentSize(var_loc::nodeCentered);
    sycl::queue q = sycl::queue{sycl::default_selector_v};

    std::vector<std::unique_ptr<PSolver>> ref;
    for (size_t c = 0; c < C; c++) {
        ref.push_back(std::make_unique<PSolver>());
        ref.back()->set_options("I " + std::to_string(gx + 1) +
                                " J " + std::to_string(gx + 1) +
                                " K " + std::to_string(gx + 1) +
                                " depth 3 pre_relax 2 post_relax 1 max_cycles 20");
        ref.back()->init(
            backend,
            grid,
            PSolver::Cartesian,
            {
                Neumann_plane(grid, boundary_direction_3d::Zpos) = 1.0 + 0.5 * c,
            },
            {
                Dirichlet_plane(grid, boundary_direction_3d::Xneg) = 0.5 + c,
                Dirichlet_plane(grid, boundary_direction_3d::Xpos) = 12.5 + 0.1 * c,
            });
    }

    batch_poisson_solver_3d batch(C);
    for (size_t c = 0; c < C; c++) {
        batch.init(
            c,
            mrhs_backend,
            grid,
            PSolver::Cartesian,
            {
                Neumann_plane(grid, boundary_direction_3d::Zpos) = 1.0 + 0.5 * c,
            },
            {
                Dirichlet_plane(grid, boundary_direction_3d::Xneg) = 0.5 + c,
                Dirichlet_plane(grid, boundary_direction_3d::Xpos) = 12.5 + 0.1 * c,
            });
    }
    batch.set_options("I " + std::to_string(gx + 1) +
                    " J " + std::to_string(gx + 1) +
                    " K " + std::to_string(gx + 1) +
                    " depth 3 pre_relax 2 post_relax 1 max_cycles 20");
    batch.init_batch();


    rander R;

    batch_host_node_field3D<double> Qb_h(grid, C);
    batch_node_field3D<double> Qb(q, grid, C);
    batch_node_field3D<double> PhiB(q, grid, C);
    device_array<double> source_packing(q, (size_t)N * C);
    device_array<double> phi_packing(q, (size_t)N * C);

    q.memset(phi_packing.data(), 0, sizeof(double) * N * C);

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
    batch_node_field3D<double> RefB(q, grid, C);
    auto PhiB_h = PhiB.getContent().to_host();
    for (size_t c = 0; c < C; c++) {

        q.memset(RefB.at(c).data(), 0, sizeof(double) * N);
        ref[c]->solve(RefB.at(c).data(), Qb.at(c).data());
        auto phi_case = RefB.at(c).getContent().to_host();
        
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

    PhiB.plot("batch_phi3d_case%d.plt", "phi");
    RefB.plot("ref_phi3d_case%d.plt", "phi");


    PrintTimer

    std::cout << (ok ? "ALL PASS" : "FAILED") << std::endl;
    return ok ? 0 : 1;
}
