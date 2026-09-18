#include <Eigen/Core>
#include <iostream>
#include <fstream>
#include <cmath>

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

void init_particles(ParticleGroup& ps, double dens, double weight, double dt, double mass,double T, grid1D& c_domain){
    rander R;
    std::vector<Particle> inj_list;
    double Num_to_deal = (dens * c_domain.del<0>() / weight);
    while (true) {
        auto v = psum::random::RandFunction3D::RandV_Maxwell(R, T, mass).x();
        double x_ = c_domain.nodePosition({c_domain.numCells<0>()}).x() + c_domain.del<0>() * R();
        if (v * dt + x_ < c_domain.nodePosition({c_domain.numCells<0>()}).x())
        {
            Particle p;
            get<position>(p).x() = v * dt + x_;
            get<velocity>(p).x() = v;
            inj_list.push_back(p);
        }
        Num_to_deal--;
        if((R() < Num_to_deal)==false) break;
    }
    ps.insert(inj_list);
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

void push_particles(ParticleGroup& particles, Nf& phi, double q_m, double dt, double length) {
    particles.for_each([&](sycl::handler& h) {
        auto phi_acc = phi.get_access(h);
        return [=](Particle& p) {
            auto [dPhi_dx] = interp_diff(get<position>(p), phi_acc);
            double ex = -dPhi_dx;
            get<velocity>(p).x() += ex * q_m * dt;
            get<position>(p).x() += get<velocity>(p).x() * dt;
            double& new_pos = get<position>(p).x();
            if (new_pos < 0 || new_pos >= length) {
                ParticleGroup::validator::make_invalid(p);
            }
        };
    });
}

int main() {
    double dx = 0.04/100;
    double dt = 1.3e-10;
    int I = 100;
    double length = dx * I;

    sycl::queue q{sycl::default_selector()};
    grid1D grid({0.0}, {length}, {I});

    Nf ele_dens(q, grid), ion_dens(q, grid), phi(q, grid);
    HNf ele_dens_host(grid), ion_dens_host(grid), qdens_host(grid), phi_host(grid);

    ParticleGroup ele(q, 512);
    ParticleGroup ion(q, 512);

    Poisson_solver_1d solver;
    solver.init(
        grid, 
        {}, 
        {
            Dirichlet_point(grid, boundary_direction_1d::L) = 0.0, 
            Dirichlet_point(grid, boundary_direction_1d::R) = 12.5
        }
    );

    double mass_ele = 9.1e-31;
    double mass_ar = mass_ele * 39.9 * 1836;
    double Te_ev = 2.2;
    double Ti_ev = 0.5;
    double weight = 6250000;
    
    ofstream fp("numline_xy.plt"); fp.close();

    for (int iter = 0; iter < 200000; iter++) {
        particle_count(ele, ele_dens, weight);
        particle_count(ion, ion_dens, weight);
        ele_dens_host.copy(ele_dens.getContent().to_host());
        ion_dens_host.copy(ion_dens.getContent().to_host());
        double volume = dx;
        for (size_t i = 0; i < qdens_host.size(); i++) {
            ele_dens_host(i) = ele_dens_host(i) / volume;
            ion_dens_host(i) = ion_dens_host(i) / volume;
            qdens_host(i) = (ion_dens_host(i) - ele_dens_host(i)) * 1.602e-19;
        }
        solver.solve(phi_host.data(), qdens_host.data());
        phi.copy(phi_host.getContent());
        push_particles(ele, phi, -1.602e-19 / mass_ele, dt, length);
        push_particles(ion, phi, 1.602e-19 / mass_ar, dt, length);

        init_particles(ion, 5e13, weight, dt, mass_ar, Ti_ev * 11700, grid);
        init_particles(ele, 5e13, weight, dt, mass_ele, Te_ev * 11700, grid);

        if (iter % 50 == 0) {
            if(ele.size() < 0.8 * ele.get_content().size() && ele.size() > 2000)
                ele.compress();
            if(ion.size() < 0.8 * ion.get_content().size() && ion.size() > 2000)
                ion.compress();

            cout << iter << ",\t" << ele.size() << ",\t" << ion.size() << endl;
            ofstream fp("numline_xy.plt", ios::app);
            fp << iter << "," << ele.size() << "," << ion.size() << endl;
            fp.close();
        }
        if (iter % 5000 == 0) {
            phi_host.plot("output/" + to_string(iter) + "Phi_xy.plt", "phi", iter * dt);
            ele_dens_host.plot("output/" + to_string(iter) + "edens_xy.plt", "ne", iter * dt);
            ion_dens_host.plot("output/" + to_string(iter) + "idens_xy.plt", "ni", iter * dt);
        }
    }
    return 0;
}