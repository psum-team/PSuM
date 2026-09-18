#include <Eigen/Core>
#include <iostream>
#include <fstream>
#include <cmath>

#include <psum/psum.hpp>
#include "custom_mccm.hpp"

using namespace psum::prelude;
using namespace boundary_creator;
using property::position;
using property::velocity;
using namespace std;

using Particle = tagged_struct<
    tag_bind<position, Eigen::Vector3d>,
    tag_bind<velocity, Eigen::Vector3d>,
    tag_bind<random_seed, uint32_t>
>;

using ParticleGroup = particle_group<Particle, pos_x_nan_is_invalid>;

using Nf = node_field1D<double>;
using HNf = host_node_field1D<double>;
using Pf = node_field2D<double>;

void push_particles(ParticleGroup& particles, Nf& phi, double q_m, double dt, double length) {
    particles.for_each([&](sycl::handler& h) {
        auto phi_acc = phi.get_access(h);
        return [=](Particle& p) {
            auto [dPhi_dx] = interp_diff({get<position>(p).x()}, phi_acc);
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

void init_particles(ParticleGroup& ps, double dens, double weight, double mass,double T, grid1D& c_domain){
    rander R;
    std::vector<Particle> inj_list;
    double Num = (dens * c_domain.span<0>() / weight);
    int Num_to_deal = static_cast<int>(Num) + (R() < Num - static_cast<int>(Num));
    for (int i = 0; i < Num_to_deal; i++) {
        auto v = psum::random::RandFunction3D::RandV_Maxwell(R, T, mass);
        Particle p;
        get<position>(p) = {R() * c_domain.span<0>() + c_domain.lowerBound<0>(), 0, 0};
        get<velocity>(p) = v;
        get<random_seed>(p) = R.gen_seed();
        inj_list.push_back(p);
    }
    ps.insert(inj_list);
}

void particle_count(ParticleGroup& particles, Nf& dens, double weight) {
    dens.setZero();
    particles.for_each([&](sycl::handler& h) {
        auto dens_acc = dens.get_access(h);
        return [=](const Particle& p) {
            add_back({get<position>(p).x()}, weight, dens_acc);
        };
    });
}

void plot_comparison_with_reference(const std::string &inputFilePath, const std::string &outputFilePath, Nf &electronNumber, Nf &ionNumber)
{
    std::ifstream inputFile(inputFilePath);
    if (!inputFile.is_open()) {
        std::cerr << "Error: reference file cannot be opened." << inputFilePath << std::endl;
        return;
    }

    std::ofstream outputFile(outputFilePath);
    if (!outputFile.is_open()) {
        std::cerr << "Error: output file cannot be opened." << inputFilePath << std::endl;
        inputFile.close();
        return;
    }

    std::string line;
    size_t lineCount = 0;
    size_t dataIndex = 0;

    auto elec_content = electronNumber.getContent().to_host();
    auto ion_content = ionNumber.getContent().to_host();

    while (std::getline(inputFile, line)) {
        if (lineCount == 0) {
            outputFile << line << ",ne_get,ni_get\n";
        } else {
            if (dataIndex >= electronNumber.size()) {
                std::cerr << "Error: different grid size." << std::endl;
                break;
            }
            outputFile << line << "," << elec_content[dataIndex] << "," << ion_content[dataIndex] << "\n";
            dataIndex++;
        }
        lineCount++;
    }
    if (dataIndex != electronNumber.size()) {
        std::cerr << "Error: different grid size." << std::endl;
    }

    inputFile.close();
    outputFile.close();
}

void computeCCP(double nn, double ne0, double Voltage, double Nc0, int GSize, double dt, int FinishStep, int AveStep)
{
    sycl::queue q{sycl::default_selector()};

    // domain size in CCP benchmark.
    double L = 0.067;
    double dx = 0.067 / GSize;

    grid1D grid({0.0}, {L}, {GSize});

    Nf atom_dens(q, grid), ele_dens(q, grid), ion_dens(q, grid), phi(q, grid);
    Nf ele_dens_mean(q, grid), ion_dens_mean(q, grid);
    HNf ele_dens_host(grid), ion_dens_host(grid), qdens_host(grid), phi_host(grid);
    Pf ele_phasespace(q, grid2D({0.0, -1e7}, {L, 1e7}, {400, 200}));
    Pf ion_phasespace(q, grid2D({0.0, -1e5}, {L, 1e5}, {400, 200}));

    double Phi_right;
    Poisson_solver_1d solver;
    solver.init(
        grid, 
        {}, 
        {
            Dirichlet_point(grid, boundary_direction_1d::L) = 0.0, 
            Dirichlet_point(grid, boundary_direction_1d::R) = [&Phi_right](auto){return Phi_right;}
        }
    );

    ParticleGroup ele(q, 512);
    ParticleGroup ion(q, 512);
    ParticleGroup atom(q, 512); // it doesn't matter.

    double mass_He = 6.67e-27;
    double mass_ele = 9.1e-31;
    double weight = ne0 * grid.del<0>() / Nc0;

    init_particles(atom, nn, weight, mass_He, 300, grid);
    init_particles(ion, ne0, weight, mass_He, 300, grid);
    init_particles(ele, ne0, weight, mass_ele, 30000, grid);

    atom_dens.setConstant(nn);
    auto mccm = He_mcc_model::make<ParticleGroup, Nf>(atom, ele, ion, atom_dens, ele_dens, ion_dens);

    for (int iter = 0; iter < FinishStep; iter++) {
        double volume = dx;
        particle_count(ele, ele_dens, weight / volume);
        particle_count(ion, ion_dens, weight / volume);
        ele_dens_host.copy(ele_dens.getContent().to_host());
        ion_dens_host.copy(ion_dens.getContent().to_host());
        ele_dens_host(0) *= 2;
        ele_dens_host(grid.numCells<0>()) *= 2;
        ion_dens_host(0) *= 2;
        ion_dens_host(grid.numCells<0>()) *= 2;
        for (size_t i = 0; i < qdens_host.size(); i++) {
            qdens_host(i) = (ion_dens_host(i) - ele_dens_host(i)) * 1.602e-19;
        }
        Phi_right = Voltage * sin(3.14159 * 2 * 13.56e6 * (iter * dt));
        solver.solve(phi_host.data(), qdens_host.data());
        phi.copy(phi_host.getContent());
        push_particles(ele, phi, -1.602e-19 / mass_ele, dt, L);
        push_particles(ion, phi, 1.602e-19 / mass_He, dt, L);
        
        if (iter > FinishStep - AveStep) {
            ele_dens_mean.for_each([&](sycl::handler& h) {
                auto ele_dens_acc = ele_dens.get_access(h);
                return [=](size_t i, double& v) {
                    v += ele_dens_acc(i);
                };
            });
            ion_dens_mean.for_each([&](sycl::handler& h) {
                auto ion_dens_acc = ion_dens.get_access(h);
                return [=](size_t i, double& v) {
                    v += ion_dens_acc(i);
                };
            });
        }

        mccm(dt);

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
        if (iter % 1000 == 0) {
            phi_host.plot("output/" + to_string(iter) + "Phi_xy.plt", "phi", iter * dt);
            ele_dens_host.plot("output/" + to_string(iter) + "edens_xy.plt", "ne", iter * dt);
            ion_dens_host.plot("output/" + to_string(iter) + "idens_xy.plt", "ni", iter * dt);
            
            ele_phasespace.setZero();
            ele.for_each([&](sycl::handler& h) {
                auto phasespace_acc = ele_phasespace.get_access(h);
                return [=](const Particle& p) {
                    auto pos = get<position>(p);
                    auto vel = get<velocity>(p);
                    Eigen::Vector<double, 2> pos_v{(pos.x()), (vel.x())};
                    if (phasespace_acc.getGrid().inGrid(pos_v))
                        add_back(pos_v, 1.0, phasespace_acc);
                };
            });
            ele_phasespace.plot("output/" + to_string(iter) + "_ele_phase.plt", "ele_phase", iter * dt);
            
            ion_phasespace.setZero();
            ion.for_each([&](sycl::handler& h) {
                auto phasespace_acc = ion_phasespace.get_access(h);
                return [=](const Particle& p) {
                    auto pos = get<position>(p);
                    auto vel = get<velocity>(p);
                    Eigen::Vector<double, 2> pos_v{(pos.x()), (vel.x())};
                    if (phasespace_acc.getGrid().inGrid(pos_v))
                        add_back(pos_v, 1.0, phasespace_acc);
                };
            });
            ion_phasespace.plot("output/" + to_string(iter) + "_ion_phase.plt", "ion_phase", iter * dt);
        }
    }
    ele_dens_mean.for_each([&](sycl::handler& h) {
        return [=](size_t i, double& v) {
            v /= AveStep;
        };
    });
    ion_dens_mean.for_each([&](sycl::handler& h) {
        return [=](size_t i, double& v) {
            v /= AveStep;
        };
    });
    plot_comparison_with_reference("reference_results/literature_case1.plt", "comparison.plt", ele_dens_mean, ion_dens_mean);
}

int main()
{
    double nn = 9.64e20;          // atom density
    double ne0 = 2.56e14;         // initial plasma density
    double Voltage = 450;         // Voltage (amplitude)
    double Nc0 = 512;             // initial PPC (particle per cell)
    int GSize = 128;              // cell number
    double dt = 1.0 / 13.56e6 / 400; // time step
    int AllStep = 256000;         // steps to execute
    int AveStep = 12800;          // steps to average
    computeCCP(nn, ne0, Voltage, Nc0, GSize, dt, AllStep, AveStep);
    return 0;
}