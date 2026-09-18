#include <psum/field.hpp>
#include "../../src/field_solver/Poisson_solver_3d.hpp"
#include "../../src/field_solver/boundary_creator/Dirichlet.hpp"
#include "../../src/timer.hpp"
#include <cuda_runtime.h>
#include <Eigen/Core>
#include <cmath>
#include <fstream>
#include <iostream>

using namespace std;
using namespace psum;
using namespace psum::field;
using namespace psum::field_solver;
using namespace boundary_creator;

void cuda_check(cudaError_t err) {
    if (err != cudaSuccess)
        throw runtime_error(string("CUDA error: ") + cudaGetErrorString(err));
}

void test_device_mg(const string& backend_name, int SIZE, int N_iter_ap, int num_conv_mask = 8) {
    (void)N_iter_ap;

    int I = SIZE;
    int J = SIZE;
    int K = SIZE;

    grid3D g({0.0, 0.0, 0.0}, {1.0, 1.0, 1.0}, {I, J, K});
    Poisson_solver_3d pSolver;

    char opts[512];
    snprintf(opts, sizeof(opts),
        "I %d J %d K %d depth %d max_cycles %d pre_relax %d post_relax %d num_conv_mask %d",
        I + 1, J + 1, K + 1, 4, 5, 6, 4, num_conv_mask);
    pSolver.set_options(opts);

    pSolver.init(
        backend_name,
        g,
        Poisson_solver_3d::Cartesian,
        {},
        {
            Dirichlet_plane(g, boundary_direction_3d::Xpos) = 0,
            Dirichlet_plane(g, boundary_direction_3d::Xneg) = 0,
            Dirichlet_plane(g, boundary_direction_3d::Ypos) = 0,
            Dirichlet_plane(g, boundary_direction_3d::Yneg) = 0,
            Dirichlet_plane(g, boundary_direction_3d::Zpos) = 0,
            Dirichlet_plane(g, boundary_direction_3d::Zneg) = 0
        },
        1.0
    );

    const size_t n_nodes = g.contentSize(var_loc::nodeCentered);
    vector<double> q_host(n_nodes);
    vector<double> phi_host(n_nodes, 0.0);
    for (size_t i = 0; i < n_nodes; i++) {
        auto pos = g.nodePosition(g.i2n(i));
        q_host[i] = cos((pos.x() + pos.y()) * 12) * sin((pos.x() - pos.y()) * 9) * cos(pos.z() * 6);
    }

    if (backend_name == "cuda_multigrid_gpu") {
        double* d_source = nullptr;
        double* d_phi = nullptr;
        cuda_check(cudaMalloc(&d_source, sizeof(double) * n_nodes));
        cuda_check(cudaMalloc(&d_phi, sizeof(double) * n_nodes));
        cuda_check(cudaMemcpy(d_source, q_host.data(), sizeof(double) * n_nodes, cudaMemcpyHostToDevice));

        double res = 0;
        Tic("loop")
        for (int i = 0; i < 20; i++) {
            Tic("set-zero")
            cuda_check(cudaMemset(d_phi, 0, sizeof(double) * n_nodes));
            cudaDeviceSynchronize();
            Toc
            Tic("solver" + (i >= 10 ? string("") : string("-warmup")))
            pSolver.solve(d_phi, d_source);
            cudaDeviceSynchronize();
            Toc
            if (i == 0) {
                Tic("other")
                cuda_check(cudaMemcpy(phi_host.data(), d_phi, sizeof(double) * n_nodes, cudaMemcpyDeviceToHost));
                Eigen::Map<const Eigen::VectorXd> phi_vec(phi_host.data(), phi_host.size());
                Eigen::VectorXd rho = Eigen::Map<const Eigen::VectorXd>(q_host.data(), q_host.size());
                Eigen::VectorXd b;
                pSolver.make_righthand_item(rho, b);
                res = (pSolver.coeff_matrix() * phi_vec - b).norm() / b.norm();
                Toc
            }
        }
        Toc
        cuda_check(cudaFree(d_source));
        cuda_check(cudaFree(d_phi));

        ofstream fp(CurrentTimeStr + "_1_" + to_string(SIZE) + ".txt");
        gt_print2file(std::filesystem::path(CurrentTimeStr + "_cudamg_" + to_string(SIZE) + ".txt"), timer::print_mode::DetailMode);
        fp << TimeUsed("solver") / 10 << " " << res << endl;
        fp.close();

        cout << backend_name << " " << TimeUsed("solver") / 10 << " " << res << endl;
        return;
    }

    sycl::queue q{sycl::gpu_selector_v};
    node_field3D<double> q_dens(q, g);
    node_field3D<double> phi(q, g);

    q_dens.for_each([&](sycl::handler& h) {
        return [=](size_t i, double& val, const auto& pos) {
            val = cos((pos.x() + pos.y()) * 12) * sin((pos.x() - pos.y()) * 9) * cos(pos.z() * 6);
        };
    });

    double res = 0;
    Tic("loop")
    for (int i = 0; i < 20; i++) {
        Tic("set-zero")
        phi.setZero();
        Toc
        Tic("solver" + (i >= 10 ? string("") : string("-warmup")))
        pSolver.solve(phi.data(), q_dens.data());
        Toc

        if (i == 0) {
            Tic("other")
            auto phi_host = phi.getContent().to_host();
            auto qdens_host = q_dens.getContent().to_host();
            Eigen::Map<const Eigen::VectorXd> phi_vec(phi_host.data(), phi_host.size());
            Eigen::VectorXd rho = Eigen::Map<const Eigen::VectorXd>(qdens_host.data(), qdens_host.size());
            Eigen::VectorXd b;
            pSolver.make_righthand_item(rho, b);
            res = (pSolver.coeff_matrix() * phi_vec - b).norm() / b.norm();
            Toc
        }
    }
    Toc
    string backend_tag;
    if (backend_name == "sycl_multigrid_gpu") backend_tag = "syclmg";
    if (backend_name == "sycl_multigrid32f_gpu") backend_tag = "syclmg32f";
    if (backend_name == "cuda_multigrid_gpu") backend_tag = "cudamg";
    string fname = CurrentTimeStr + "_" + backend_tag + "_" + to_string(SIZE) + ".txt";
    ofstream fp(fname);
    gt_print2file(std::filesystem::path(fname), timer::print_mode::DetailMode);
    fp << TimeUsed("solver") / 10 << " " << res << endl;
    fp.close();

    cout << backend_name << " " << TimeUsed("solver") / 10 << " " << res << endl;
}

int main(int argc, char** argv) {
    if (argc == 1) {
        test_device_mg("cuda_multigrid_gpu", 32, 0);
        return 0;
    }

    if (argc != 4 && argc != 5)
        throw runtime_error("Invalid parameter. Usage: test_device_mg_compare <solver type> <size> <iteration_append> [num_conv_mask]. solver type = 0(sycl) or 1(cuda).");

    int solver_type = stoi(argv[1]);
    int size = stoi(argv[2]);
    int N_app = stoi(argv[3]);
    int num_conv_mask = argc == 5 ? stoi(argv[4]) : 8;

    if (solver_type < 0 || solver_type > 2)
        throw runtime_error("Invalid solver type. solver type = 0(sycl-double) or 1(cuda) or 2(sycl-float32).");

    if (solver_type == 0) {
        test_device_mg("sycl_multigrid_gpu", size, N_app, num_conv_mask);
    } else if (solver_type == 2) {
        test_device_mg("sycl_multigrid32f_gpu", size, N_app, num_conv_mask);
    } else {
        test_device_mg("cuda_multigrid_gpu", size, N_app, num_conv_mask);
    }

    return 0;
}
