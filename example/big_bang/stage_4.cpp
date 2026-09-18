#include <iostream>
#include <sycl/sycl.hpp>
#include <Eigen/Core>
#include <psum/psum.hpp>

using namespace std;
using namespace psum::prelude;
using namespace psum::field_solver::boundary_creator;

using Particle = tagged_struct<
    tag_bind<property::position, Eigen::RowVector2d>,
    tag_bind<property::velocity, Eigen::RowVector2d>,
    tag_bind<property::charge, double>,
    tag_bind<property::mass, double>
>;

using ParticleGroup = particle_group<Particle, pos_x_nan_is_invalid>;

void make_particles(size_t count, ParticleGroup& particles) {
    rander R;
    vector<Particle> particle_vec(count);
    for (auto& p : particle_vec) {
        get<property::position>(p) = {R() - 0.5, R() - 0.5};
        auto rand_vec = RandFunction3D::RandV_Maxwell(R, 100000, 2.18e-25);
        get<property::velocity>(p) = {rand_vec.x(), rand_vec.y()};
        get<property::charge>(p) = 1.602e-19 * 1e7;
        get<property::mass>(p) = 2.18e-25 * 1e7;
    }
    particles.insert(particle_vec);
}

void deposit_charge(ParticleGroup& particles, node_field2D<double>& rho) {
    rho.setZero();
    double cell_area = rho.getGrid().del<0>() * rho.getGrid().del<1>();
    particles.for_each([&](sycl::handler& h) {
        auto rho_acc = rho.get_access(h);
        return [=](Particle& p) {
            add_back(
                get<property::position>(p),
                get<property::charge>(p) / cell_area,
                rho_acc
            );
        };
    });
}

void move_particles(ParticleGroup& particles, node_field2D<double>& phi, double dt) {
    particles.for_each([&](sycl::handler& h) {
        auto phi_acc = phi.get_access(h);
        return [=](Particle& p) {
            auto& position = get<property::position>(p);
            auto& velocity = get<property::velocity>(p);

            auto grad_phi = interp_diff(position, phi_acc);
            Eigen::RowVector2d electric_field(-grad_phi[0], -grad_phi[1]);
            velocity += electric_field * get<property::charge>(p) / get<property::mass>(p) * dt;
            position += velocity * dt;

            if (!phi_acc.getGrid().inGrid(position)) {
                ParticleGroup::validator::make_invalid(p);
            }
        };
    });
}

int main() {
    grid2D grid({-1.0, -1.0}, {1.0, 1.0}, {256, 256});

    sycl::queue q{sycl::default_selector_v};
    cout << "device: " << q.get_device().get_info<sycl::info::device::name>() << endl;

    node_field2D<double> phi(q, grid);
    node_field2D<double> rho(q, grid);

    ParticleGroup particles(q);
    make_particles(100000, particles);

    Poisson_solver_2d psolver;
    psolver.init(
        grid,
        Poisson_solver_2d::Cartesian,
        {
            Dirichlet_line(grid, boundary_direction_2d::N) = 0,
            Dirichlet_line(grid, boundary_direction_2d::S) = 0,
            Dirichlet_line(grid, boundary_direction_2d::E) = 0,
            Dirichlet_line(grid, boundary_direction_2d::W) = 0
        }
    );
    host_node_field2D<double> phi_host(grid);
    host_node_field2D<double> rho_host(grid);

    double dt = 2.0e-7;
    int steps = 400;
    int output_interval = 10;

    for (int step = 0; step <= steps; ++step) {
        deposit_charge(particles, rho);
        rho_host.copy(rho.getContent().to_host());
        psolver.solve(phi_host.data(), rho_host.data());
        phi.copy(phi_host.getContent());
        if (step % output_interval == 0) {
            rho.plot("output/rho_step_" + to_string(step) + ".plt", "rho", step * dt);
            phi.plot("output/phi_step_" + to_string(step) + ".plt", "phi", step * dt);
            cout << "step = " << step << ", alive = " << particles.size() << endl;
        }
        if (step == 0) break;
        move_particles(particles, phi, dt);
    }

    cout << "Big Bang step 4: poisson solver finished." << endl;
    return 0;
}
