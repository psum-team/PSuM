#include <iostream>
#include <fstream>
#include <Eigen/Core>
#include <psum/psum.hpp>

using namespace std;
using namespace psum::prelude;

using Particle = tagged_struct<
    tag_bind<property::position, Eigen::RowVector2d>,
    tag_bind<property::velocity, Eigen::RowVector2d>,
    tag_bind<property::random_seed, uint32_t>
>;

using PG = particle_group<Particle, pos_x_nan_is_invalid>;
using Field = node_field2D<double>;
using HField = host_node_field2D<double>;

void move_particles(PG& pg, const auto& grid, double dt) {
    pg.for_each([&](sycl::handler& h) {
        return [=](Particle& p) {
            tag::get<property::position>(p) += tag::get<property::velocity>(p) * dt;
            if (!grid.inGrid(tag::get<property::position>(p)))
                PG::validator::make_invalid(p);
        };
    });
}

int main() {
    sycl::queue q{sycl::default_selector_v};

    auto grid = simple_grid<2>({0.0, 0.0}, {0.1, 0.1}, {200, 200});

    PG electrons(q);

    Field atom_dens(q, grid);
    HField atom_host(grid);

    atom_host.setZero();
    Eigen::RowVector2d center{0.07, 0.05};
    double gas_r = 0.025;
    atom_host.for_each([&](size_t, double& v, const auto& pos) {
        v = 2e22 * exp(-((pos - center).squaredNorm() / gas_r / gas_r));
    });
    atom_host.plot("output/atom_dens.plt", "nAr", 0.0);
    atom_dens.copy(atom_host.getContent());

    ofstream count_file("output/count.plt");
    count_file << "variables=time,ele" << endl;

    double dt = 1e-10;
    int steps = 600, interval = 20;

    for (int step = 0; step <= steps; ++step) {
        move_particles(electrons, grid, dt);

        if (step % interval == 0) {
            count_file << step * dt << "\t" << electrons.size() << endl;
            cout << "step " << step << ": ele=" << electrons.size() << endl;
        }

        Particle p0;
        tag::get<property::position>(p0) = {0.001, 0.05};
        tag::get<property::velocity>(p0) = {6e6, 0.0};
        tag::get<property::random_seed>(p0) = global_random::rand_uint();
        electrons.insert({p0});
    }

    count_file.close();
    cout << "Avalanche stage 2: wrote count.plt and atom_dens.png." << endl;

    return 0;
}
