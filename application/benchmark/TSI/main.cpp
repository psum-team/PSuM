#include <Eigen/Core>
#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <algorithm>

#include <psum/psum.hpp>

using namespace psum::prelude;
using namespace boundary_creator;
using property::position;
using property::velocity;
using namespace std;

using Particle = tagged_struct<
    tag_bind<position, Eigen::Vector<double, 1>>,
    tag_bind<velocity, Eigen::Vector<double, 1>>
>;

using ParticleGroup = particle_group<Particle, pos_x_nan_is_invalid>;

using Nf = node_field1D<double>;
using HNf = host_node_field1D<double>;
using Pf = node_field2D<double>;
using HPf = host_node_field2D<double>;

void init_particles(ParticleGroup& particles, double length, int num, double VT, double VD) {
    std::mt19937 gen(123456789);
    std::uniform_real_distribution<double> R(0, 1);
    double mode = 2;
    std::vector<Particle> buf;
    particles.reserve(num);
    for (int i = 0; i < num; i++) {
        double pos = i * length / num;
        double vel = ((i % 2 == 0) ? VD : -VD) 
                   + VT * cos(2 * M_PI * R(gen)) * sqrt(-2 * log(R(gen)))
                   + 0.1 * VT * sin(2 * M_PI * pos / length * mode);
        buf.emplace_back(Particle());
        get<position>(buf.back()).x() = pos;
        get<velocity>(buf.back()).x() = vel;
    }
    particles.insert(buf);
}

void particle_count(ParticleGroup& particles, Nf& dens, double weight) {
    dens.setZero();
    particles.for_each([&](sycl::handler& h) {
        auto dens_acc = dens.get_access(h);
        return [=](const Particle& p) {
            add_back(get<position>(p), weight, dens_acc);
        };
    });
}

void push_particles(ParticleGroup& particles, Nf& phi, double q_m, double dt) {
    particles.for_each([&](sycl::handler& h) {
        auto phi_acc = phi.get_access(h);
        double length = phi.getGrid().span<0>();
        return [=](Particle& p) {
            auto [dPhi_dx] = interp_diff(get<position>(p), phi_acc);
            double ex = -dPhi_dx;

            get<velocity>(p).x() += ex * q_m * dt;
            get<position>(p).x() += get<velocity>(p).x() * dt;
            
            double& new_pos = get<position>(p).x();
            while (new_pos < 0) new_pos += length;
            while (new_pos >= length) new_pos -= length;
        };
    });
}

int main() {
    double length = 2 * M_PI;
    int cell_num = 64;
    int ppc = 154;
    double omega_p = 1;
    double q_mass_rate = 1;
    double dt = 0.125;
    double VT = 0.01;
    double VD = 0.2;

    double dx = length / cell_num;
    int particle_num = ppc * cell_num;
    double Q = omega_p * omega_p / (q_mass_rate * particle_num / length * 4 * M_PI);
    double rho_background = -Q * particle_num / length;

    sycl::queue q{sycl::default_selector()};

    grid1D grid({0.0}, {length}, {cell_num});
    grid2D phasespace_grid({0.0, -3 * VD}, {length, 3 * VD}, {300, 200});
    
    Nf dens(q, grid), phi(q, grid);
    HNf dens_host(grid), phi_host(grid);
    Pf phasespace_field(q, phasespace_grid);
    HPf phasespace_host(phasespace_grid);

    ParticleGroup electrons(q);
    init_particles(electrons, length, particle_num, VT, VD);

    Poisson_solver_1d solver;
    solver.init(
        grid,
        {},
        {
            Dirichlet_point(grid, boundary_direction_1d::L) = 0.0,
            Dirichlet_point(grid, boundary_direction_1d::R) = 0.0
        },
        0.25 / M_PI
    );

    ofstream fp1("energy.plt");
    fp1 << "variables=time,Venergy,Eenergy,TotalEnergy,relative_E_change" << endl;
    double E0 = 0;

    for (int iter = 0; iter < 1000; iter++) {
        particle_count(electrons, dens, 1.0);
        dens_host.copy(dens.getContent().to_host());
        dens_host.for_each([&](size_t i, double &d) {
            d = d / dx * Q + rho_background;
        });
        solver.solve(phi_host.data(), dens_host.data());
        phi.copy(phi_host.getContent());
        
        push_particles(electrons, phi, q_mass_rate, dt);
        
        double Venergy = 0;
        for (const auto& p : electrons.get_content().to_host()) {
            double vel = get<velocity>(p).x();
            Venergy += 0.5 * abs(Q) / q_mass_rate * vel * vel;
        }
        
        double Eenergy = 0;
        for (size_t i = 0; i < cell_num; i++) {
            Eenergy += pow(phi_host(i) - phi_host(i + 1), 2) * 0.25 / M_PI / 2 / dx;
        }

        if (iter == 0) E0 = Venergy + Eenergy;

        fp1 << iter * dt << "\t" << Venergy << "\t" << Eenergy << "\t"
            << Venergy + Eenergy << "\t" << abs((Venergy + Eenergy) / E0 - 1) << endl;
            
        if (iter % 10 == 0) {
            phasespace_field.setZero();
            electrons.for_each([&](sycl::handler& h) {
                auto phasespace_acc = phasespace_field.get_access(h);
                return [=](const Particle& p) {
                    auto pos = get<position>(p);
                    auto vel = get<velocity>(p);
                    Eigen::Vector<double, 2> pos_v{(pos.x()), (vel.x())};
                    add_back(pos_v, 1.0, phasespace_acc);
                };
            });
            phasespace_host.copy(phasespace_field.getContent().to_host());
            phasespace_host.plot("output/phase" + std::to_string(iter) + ".plt", "phase", iter * dt);
            
            cout << iter << "\t : physical time " << iter * dt << "\t" << Venergy
                 << "\t" << Eenergy << "\t" << Venergy + Eenergy << endl;
        }
    }
    fp1.close();

    return 0;
}