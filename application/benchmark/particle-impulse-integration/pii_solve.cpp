// This code is used to solve free-molecular-flow problem using particle-impulse-integration (PII) method.
// It is a axially symmetric problem, with computational domain $D=\{(z, r)\}=[0, L]\times[0, R]$.
// There is disc at z=0 and cold background flow with velocity (0, 0, -u).
// About PII: see https://doi.org/10.1103/8znn-h985

struct problem_parameter {
    double mass, Tw, L, R, lambda, n_0, u;
};

struct solve_config {
    double dt, particle_weight;
    int grid_R, grid_Z;
};

#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <Eigen/Dense>
#include <sycl/sycl.hpp>
#include <psum/psum.hpp>

using namespace psum::prelude;
using namespace psum::tag::property;
using namespace psum::random::RandFunction3D;
using namespace psum::field::simple_interpolation;

using particle = tagged_struct<
    tag_bind<position, Eigen::RowVector3d>,
    tag_bind<velocity, Eigen::RowVector3d>,
    tag_bind<weight, double>,
    tag_bind<random_seed, uint32_t>
>;

using cf = cell_field2D<double>;
using MaterialSet = basic_material_set<particle, pos_x_nan_is_invalid>;
using Router = boundary_router<MaterialSet, particle>;
using material_type = MaterialSet::material_type;

cf once_solve(sycl::queue &q, Router &router, simple_grid<2> &g, problem_parameter param, solve_config config) {
    double dt = config.dt;
    double particle_weight = config.particle_weight;

    particle_group<particle, pos_x_nan_is_invalid> atom(q);

    std::random_device rd;
    psum::random::rander gen(rd());

    double P_ioni = 1 - exp(-param.lambda * dt);

    cf count(q, g);

    double max_z = param.L;
    double radius = param.R;

    std::vector<particle> new_particles;
    double incNum = param.n_0 * M_PI * radius * radius * param.u / particle_weight * dt;
    int num_new_particles = std::floor(incNum) + (gen() < incNum - std::floor(incNum));
    for (int i = 0; i < num_new_particles; ++i) {
        particle p;
        double alpha = 2 * M_PI * gen();
        double randr = radius * sqrt(gen());
        double dt_offset = gen() * dt;
        auto &pos = get<position>(p);
        auto &vel = get<velocity>(p);
        pos = Eigen::RowVector3d(cos(alpha) * randr, sin(alpha) * randr, max_z - dt_offset * param.u);
        vel = Eigen::RowVector3d(0, 0, -param.u);
        get<weight>(p) = particle_weight;
        get<random_seed>(p) = gen() * UINT32_MAX;

        if (gen() > 1 - exp(-param.lambda * dt_offset)) {
            new_particles.push_back(p);
        }
    }
    atom.insert(new_particles);

    int step = 0;
    while (true) {
        atom.for_each([&](sycl::handler& h) {
            auto router_acc = router.get_access(h);
            auto count_acc = count.get_access(h);
            return [=](particle& p) {
                auto& pos = get<position>(p);
                auto& vel = get<velocity>(p);
                double p1x = pos.x(), p1y = pos.y(), p1z = pos.z();
                double r = sqrt(p1x * p1x + p1y * p1y);
                Eigen::Vector2d pos_zr{p1z, r};
                if (count_acc.getGrid().inGrid(pos_zr)) {
                    add_back_nearest(pos_zr, get<weight>(p), count_acc);
                } else {
                    pos_x_nan_is_invalid<particle>::make_invalid(p);
                }

                pos = pos + vel * dt;
                double p2x = pos.x();
                double p2y = pos.y();
                double p2z = pos.z();
                get<weight>(p) *= (1 - P_ioni);
                router_acc.deal(p1x, p1y, p1z, p2x, p2y, p2z, p);
            };
        });

        if (step % 20 == 0) {
            atom.for_each([&](sycl::handler& h) {
                return [=](particle &p) {
                    auto R = psum::random::view_as_rander(get<random_seed>(p));
                    if (R() > get<weight>(p) / particle_weight)
                        pos_x_nan_is_invalid<particle>::make_invalid(p);
                    else
                        get<weight>(p) = particle_weight;
                };
            });
            std::cout << "loop=" << step << "\tsize=" << atom.size() << std::endl;
            double fill_rate = atom.size() / (atom.get_content().size());
            if (fill_rate < 0.8 && atom.size() > 1e5) {
                atom.compress();
            }
        }
        if (atom.size() == 0) break;

        step++;
    }

    count.for_each([&](sycl::handler& h) {
        auto count_acc = count.get_access(h);
        return [=](size_t idx, double& val, const cf::Position& pos) {
            double dz = count_acc.getGrid().del<0>();
            double dr = count_acc.getGrid().del<1>();
            double r_inner = pos.y() - 0.5 * dr;
            double r_outer = pos.y() + 0.5 * dr;
            double vol = dz * M_PI * (r_outer * r_outer - r_inner * r_inner);
            val /= (param.n_0 * vol);
        };
    });

    count.plot("density_pii.plt", "n");

    return count;
}

int main() {
    problem_parameter param;
    param.mass = 2.18e-25;
    param.Tw = 500;
    param.L = 0.15;
    param.R = 0.1;
    param.lambda = 8000;
    param.n_0 = 1e18;
    param.u = 1e3;

    solve_config config;
    config.dt = 1e-6;
    config.particle_weight = 0.01e9;
    config.grid_R = 100;
    config.grid_Z = 150;

    int solve_num = 4;

    sycl::queue q;
    std::cout << "Running on " << q.get_device().get_info<sycl::info::device::name>() << std::endl;

    Router router(q);
    router.get_material_set().set_default_mass(param.mass);
    router.get_material_set().set_default_temperature(param.Tw);

    auto cylinder_wall = mesh_generator::revolve_zr_curve({{0.0, param.R}, {param.L, param.R}}, 64);
    auto z0_plane = mesh_generator::axis_aligned_plane(-param.R, -param.R, 0.0, param.R, param.R, 0.0);
    auto zD_plane = mesh_generator::axis_aligned_plane(-param.R, -param.R, param.L, param.R, param.R, param.L);

    std::vector<triangle_mesh_with_id<material_type>> meshes = {
        {cylinder_wall, material_type::absorb},
        {z0_plane, material_type::maxwell_reflect},
        {zD_plane, material_type::absorb}
    };

    router.set(meshes, 50, 50, 50);

    simple_grid<2> g({0, 0}, {param.L, param.R}, {config.grid_Z, config.grid_R});
    cf dens(q, g);

    std::ifstream fp("reference_results/hp_density_pii.plt");
    if (!fp.is_open()) {
        std::cout << "Reference solution not found, skipping comparison." << std::endl;
        return 0;
    }

    std::cout << "Loading reference solution..." << std::endl;

    auto skip_lines = [](std::ifstream& is, int n) {
        for (int i = 0; i < n; ++i)
            is.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    };
    
    skip_lines(fp, 3);

    simple_grid<2> hp_g({0, 0}, {param.L, param.R}, {300, 200});
    std::vector<double> hp_ans_host_vec(300 * 200);
    for (size_t i = 0; i < hp_ans_host_vec.size(); i++) {
        double x, y, v;
        fp >> x >> y >> v;
        hp_ans_host_vec[i] = v;
    }
    cf hp_ans(device_array<double>(q, hp_ans_host_vec), hp_g);

    simple_grid<2> sample_g({0, 0}, {param.L, param.R}, {30, 20});
    cf hp_sampled(q, sample_g);
    hp_sampled.for_each([&](sycl::handler& h) {
        auto hp_ans_acc = hp_ans.get_access(h);
        return [=](size_t idx, double& val, const cf::Position& pos) {
            val = interp_nearest(pos, hp_ans_acc);
        };
    });
    std::vector<double> hp_ans_sample_host = hp_sampled.getContent().to_host();

    Eigen::VectorXd err_vec(solve_num);
    Eigen::MatrixXd err_mat(hp_sampled.size(), solve_num);

    for (int i = 0; i < solve_num; i++) {
        Tic("solve");
        cf pii_ans = once_solve(q, router, g, param, config);
        Toc;

        cf pii_sampled(q, sample_g);
        pii_sampled.for_each([&](sycl::handler& h) {
            auto pii_ans_acc = pii_ans.get_access(h);
            return [=](size_t idx, double& val, const cf::Position& pos) {
                val = interp_nearest(pos, pii_ans_acc);
            };
        });

        std::vector<double> pii_ans_sample_host = pii_sampled.getContent().to_host();

        for (size_t j = 0; j < hp_ans_sample_host.size(); j++) {
            err_mat(j, i) = hp_ans_sample_host[j] - pii_ans_sample_host[j];
        }
        Eigen::Map<Eigen::VectorXd> hp_map(hp_ans_sample_host.data(), hp_ans_sample_host.size());
        err_vec[i] = err_mat.col(i).norm() / hp_map.norm();
    }

    double mean_err = err_vec.mean();
    double std_dev = std::sqrt((err_vec.array() - mean_err).square().sum() / (err_vec.size() - 1));
    std::cout << mean_err << "|\t" << std_dev << std::endl;

    fp.close();

    PrintTimer;

    return 0;
}
