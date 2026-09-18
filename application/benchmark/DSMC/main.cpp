#include <iostream>
#include <fstream>
#include <psum/psum.hpp>
#include "custom_dpmcc.hpp"
#include "custom_material.hpp"

using namespace psum::prelude;
using property::position;
using property::velocity;
using property::internal_energy;
using property::random_seed;

using Particle = tagged_struct<
    tag_bind<position, Eigen::Vector3d>,
    tag_bind<velocity, Eigen::Vector3d>,
    tag_bind<internal_energy, double>,
    tag_bind<lock_var, int>,
    tag_bind<random_seed, uint32_t>
>;

using ParticleGroup = particle_container::particle_group<Particle, particle_container::pos_x_nan_is_invalid>;

using Field = field::cell_field3D<double>;
using Field2D = field::cell_field2D<double>;

using MaterialSet = simple_material_set<Particle, pos_x_nan_is_invalid>;
using Router = boundary_router<MaterialSet, Particle>;
using material_type = MaterialSet::material_type;

void compute_density_distribution(ParticleGroup& N2, Field& dens, double weight) {
    dens.setZero();
    double cell_v = dens.getGrid().del<0>() * dens.getGrid().del<1>() * dens.getGrid().del<2>();
    N2.for_each([&](sycl::handler &h){
        auto dens_acc = dens.get_access(h);
        return [=](Particle& p) {
            if (dens_acc.getGrid().inGrid(get<position>(p)))
                add_back_nearest(get<position>(p), weight / cell_v, dens_acc);
            else
                ParticleGroup::validator::make_invalid(p);
        };
    });
}

void compute_density_2d(ParticleGroup& N2, Field2D& dens_2d, double weight, double z_span) {
    dens_2d.setZero();
    double cell_area = dens_2d.getGrid().del<0>() * dens_2d.getGrid().del<1>();
    N2.for_each([&](sycl::handler &h){
        auto dens_acc = dens_2d.get_access(h);
        return [=](Particle& p) {
            auto& pos = get<position>(p);
            if (dens_acc.getGrid().inGrid({pos.x(), pos.y()}))
                add_back_nearest({pos.x(), pos.y()}, weight / (cell_area * z_span), dens_acc);
        };
    });
}

void test_stl_injection() {
    sycl::queue q{sycl::default_selector{}};
    std::cout << "Running on " << q.get_device().get_info<sycl::info::device::name>() << std::endl;

    random::rander R;
    ParticleGroup N2(q);

    int I = 350;
	int J = 150;
	double dx = 0.5e-3;
    double dz = 5e-3;
    int max_iterations = 50000;
    double weight = 1e10;
    double dt = 2e-8;

    double truncate_thichness = 0.0025;
    double truncate_pos_x = 0.050;
    double plate_length = 0.1;
    double T_plate = 290.0;

    double pmass = 4.652E-26;
    double boltz = 1.3805e-23;
    double dens = 3.716E20;
	double T = 13.32;
    double v_inf = 1503;

    grid3D g({0, -J * dx, 0}, {I * dx, J * dx, dz}, {I, J * 2, 1});
    Field f_N2(q, g);
    grid2D g2d({0, -J * dx}, {I * dx, J * dx}, {I, J * 2});
    Field2D f_N2_2d(q, g2d);
    Field2D f_N2_2d_mean(q, g2d);
    int f_N2_2d_mean_count = 0;

    auto model = N2_dsmc_model::make(N2, f_N2, g);

    Router router(q);

    auto leave_surf_x_lower = mesh_generator::axis_aligned_plane(
        g.lowerBound<0>(), g.lowerBound<1>(), g.lowerBound<2>(),
        g.lowerBound<0>(), g.upperBound<1>(), g.upperBound<2>()
    );

    auto leave_surf_x_upper = mesh_generator::axis_aligned_plane(
        g.upperBound<0>(), g.lowerBound<1>(), g.lowerBound<2>(),
        g.upperBound<0>(), g.upperBound<1>(), g.upperBound<2>()
    );

    auto leave_surf_y_lower = mesh_generator::axis_aligned_plane(
        g.lowerBound<0>(), g.lowerBound<1>(), g.lowerBound<2>(),
        g.upperBound<0>(), g.lowerBound<1>(), g.upperBound<2>()
    );

    auto leave_surf_y_upper = mesh_generator::axis_aligned_plane(
        g.lowerBound<0>(), g.upperBound<1>(), g.lowerBound<2>(),
        g.upperBound<0>(), g.upperBound<1>(), g.upperBound<2>()
    );

    auto truncate_surf = mesh_generator::axis_aligned_cube(
        truncate_pos_x, -truncate_thichness, -1,
        truncate_pos_x + plate_length, truncate_thichness, 1
    );

    std::vector<triangle_mesh_with_id<material_type>> meshes = {
        {leave_surf_x_lower, material_type::absorb},
        {leave_surf_x_upper, material_type::absorb},
        {leave_surf_y_lower, material_type::specular},
        {leave_surf_y_upper, material_type::specular},
        {truncate_surf, material_type::maxwell_reflect}
    };

    router.set(meshes, 80, 80, 4);
    router.get_material_set().set_default_temperature(T_plate);
    router.get_material_set().set_default_mass(pmass);

    Tic("total")

    for (int iter = 0; iter <= max_iterations; ++iter) {
        Tic("inject")
        int num_inject_per_step = dens * g.span<1>() * g.span<2>() * v_inf * dt / weight;
        std::vector<Particle> new_particles(num_inject_per_step);
        for (int i = 0; i < num_inject_per_step; i++) {
            rander R1 = rander();
            rander R2 = rander();
            get<position>(new_particles[i]) = {R1() * g.del<0>(), R1() * g.span<1>() + g.lowerBound<1>(), R1() * g.span<2>()};
            get<velocity>(new_particles[i]) = random::RandFunction3D::RandV_Maxwell(R2, T, pmass).transpose() + Eigen::RowVector3d{v_inf, 0.0, 0.0};
            get<random_seed>(new_particles[i]) = R1.gen_seed();
            get<internal_energy>(new_particles[i]) = 0.0;
        }
        N2.insert<false>(new_particles);
        if (iter % 20 == 0) {
            TocTic("shuffle")
            N2.shuffle();
        }
        TocTic("density")

        compute_density_distribution(N2, f_N2, weight);
        compute_density_2d(N2, f_N2_2d, weight, dz);
        f_N2_2d_mean.for_each([&](sycl::handler& h) {
            auto dens_acc = f_N2_2d.get_access(h);
            return [=](size_t i, auto& val) {
                val += dens_acc(i);
            };
        });
        f_N2_2d_mean_count++;
        TocTic("collision")

        model(dt);
        TocTic("move")
        
        N2.for_each([&](sycl::handler& h) {
            auto router_acc = router.get_access(h);
            return [=](Particle& p) {
                double p1x = get<position>(p).x();
                double p1y = get<position>(p).y();
                double p1z = get<position>(p).z();

                get<position>(p) += get<velocity>(p) * dt;

                double p2x = get<position>(p).x();
                double p2y = get<position>(p).y();
                double p2z = get<position>(p).z();

                router_acc.deal(p1x, p1y, p1z, p2x, p2y, p2z, p);
                g.wrap_on_dim<2>(get<position>(p));
            };
        });
        Toc

        if (iter % 20 == 0 || iter == max_iterations - 1) {
            size_t active = N2.size();
            std::cout << "Iteration " << iter << ": Active particles = " << active << std::endl;
        }

        if (iter % 100 == 0) {
            Tic("output")
            if (iter % 1000 == 0) {
                PrintTimer
                f_N2.plot("output/"+std::to_string(iter)+"_density_3d.plt", "N2_density");
            }
            f_N2_2d_mean.for_each([&](sycl::handler& h) {
                return [=](size_t i, double& v) {
                    v /= (f_N2_2d_mean_count * dens);
                };
            });
            f_N2_2d_mean.plot("output/"+std::to_string(iter)+"_density_2d.plt", "N2_density");
            f_N2_2d_mean.setZero();
            f_N2_2d_mean_count = 0;
            Toc
        }
    }

    Toc_("total")
    std::cout << "\n========== Performance Summary ==========" << std::endl;
    PrintTimer
}

int main() {
    test_stl_injection();
    return 0;
}
