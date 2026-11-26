#include "calc_tau.hpp"
#include <psum/psum.hpp>
#include <sycl/sycl.hpp>
#include <iostream>
#include <fstream>
#include <cmath>
#include "../../../src/timer.hpp"

using namespace psum::prelude;
using Role = psum::particle_collision::coulomb_role;
using property::position;
using property::velocity;
using property::random_seed;

constexpr double Q = 1.602176634e-19;
constexpr double m_e_mass = 9.109e-31;
constexpr double m_p_mass = 1.67e-27;
constexpr double epsilon_0 = 8.854187817e-12;

struct squared_velocity : tag::foundation::abstract_tag {
    inline const static std::string tag_name = "squared_velocity";
};

using Particle = tagged_struct<
    tag_bind<position, Eigen::Vector<double, 1>>,
    tag_bind<velocity, Eigen::Vector<double, 1>>,
    tag_bind<squared_velocity, Eigen::Vector<double, 1>>,
    tag_bind<random_seed, uint32_t>
>;

using ParticleGroup = particle_group<Particle, pos_x_nan_is_invalid>;
using Grid = field::simple_grid<1>;

template<int var_num = 1>
using DCf = field::cell_field1D<double, var_num>;

template<int var_num = 1>
using HCf = field::host_cell_field1D<double, var_num>;

auto to_3d = [](auto& p) -> Eigen::Vector3d {
    psum::random::rander R(psum::tag::get<random_seed>(p));
    double theta = R() * 2.0 * M_PI;
    double vy = std::sqrt(psum::tag::get<squared_velocity>(p).x());
    psum::tag::get<random_seed>(p) = R.get_engine()();
    return {psum::tag::get<velocity>(p).x(), vy * std::cos(theta), vy * std::sin(theta)};
};

auto v2p = [](auto& p, Eigen::Vector3d v_new) {
    psum::tag::get<velocity>(p).x() = v_new.x();
    psum::tag::get<squared_velocity>(p).x() = v_new.y() * v_new.y() + v_new.z() * v_new.z();
};

template<typename Func>
void calc_temperature(ParticleGroup& group, DCf<>& T, double mass, Func _func) {
    DCf<3> moments(T.getQueue(), T.getGrid());
    moments.setZero();

    group.for_each([&](sycl::handler& h) {
        auto mom_acc = moments.get_access(h);
        return [=](Particle& p) {
            double pos = psum::tag::get<position>(p).x();
            Eigen::Vector3d v = _func(p);
            double vx = v.x();
            double v2 = v.x() * v.x() + v.y() * v.y() + v.z() * v.z();
            size_t ci = static_cast<size_t>(mom_acc.getGrid().c2i(
                mom_acc.getGrid().nC(Eigen::Vector<double, 1>{pos})));
            sycl::atomic_ref<double, sycl::memory_order::relaxed,
                            sycl::memory_scope::device,
                            sycl::access::address_space::global_space> num_ref(mom_acc(ci)(0));
            num_ref.fetch_add(1.0);
            sycl::atomic_ref<double, sycl::memory_order::relaxed,
                            sycl::memory_scope::device,
                            sycl::access::address_space::global_space> vx_ref(mom_acc(ci)(1));
            vx_ref.fetch_add(vx);
            sycl::atomic_ref<double, sycl::memory_order::relaxed,
                            sycl::memory_scope::device,
                            sycl::access::address_space::global_space> v2_ref(mom_acc(ci)(2));
            v2_ref.fetch_add(v2);
        };
    });

    T.for_each([&](sycl::handler& h) {
        auto mom_acc = moments.get_access(h);
        return [=](size_t i, double& v) {
            const Eigen::RowVector<double, 3>& m = mom_acc(i);
            double num = m[0];
            if (num == 0) { v = 0; return; }
            double v2 = m[2] / num;
            v = v2 * mass * 0.5 / Q * 2.0 / 3.0;
        };
    });
}

void make_maxwell_particles(std::vector<Particle>& particles, size_t N,
                            double Te_eV, double mass, double pos_range_lo, double pos_range_hi) {
    particles.clear();
    for (size_t i = 0; i < N; i++) {
        psum::random::rander R;
        auto v = psum::random::RandFunction3D::RandV_Maxwell(R, Te_eV * 11600, mass);
        double pos = psum::random::global_random::rand() * (pos_range_hi - pos_range_lo) + pos_range_lo;
        Particle p;
        psum::tag::get<position>(p) = Eigen::Vector<double, 1>{pos};
        psum::tag::get<velocity>(p) = Eigen::Vector<double, 1>{v.x()};
        psum::tag::get<squared_velocity>(p) = Eigen::Vector<double, 1>{v.y() * v.y() + v.z() * v.z()};
        psum::tag::get<random_seed>(p) = psum::random::global_random::rand_uint();
        particles.push_back(p);
    }
}

// =====================================================
// Test 3: Double temperature (two electron populations) with theory
// =====================================================
void test_doubleT() {
    std::cout << "\n========== test_doubleT ==========" << std::endl;
    size_t N = 100000;

    double Te1_init = 300;
    double Te2_init = 150;
    double q_e = -Q;
    double n_dens = 1e19;

    sycl::queue q{sycl::default_selector_v};
    std::cout << "Using device: " << q.get_device().get_info<sycl::info::device::name>() << std::endl;
    Grid grid({-0.1}, {0.9}, {1});

    ParticleGroup ele1(q), ele2(q);
    std::vector<Particle> particles;
    make_maxwell_particles(particles, N, Te1_init, m_e_mass, -0.1, 0.9);
    ele1.insert<false>(particles);
    make_maxwell_particles(particles, N, Te2_init, m_e_mass, -0.1, 0.9);
    ele2.insert<false>(particles);

    auto num2dens = [=](size_t, size_t count) -> double {
        return n_dens;
    };

    DCf<> Te1_f(q, grid), Te2_f(q, grid);

    psum::particle_collision::coulomb_collision_solver<Particle, pos_x_nan_is_invalid, 1>
        solver(q, grid, {{ele1, Role::electron, m_e_mass, q_e, Te1_f},
                         {ele2, Role::electron, m_e_mass, q_e, Te2_f}});

    double dt = 2e-10;
    int total_steps = 200000;
    int output_interval = 200;

    std::ofstream ofs("coulomb_doubleT_validation.plt", std::ios::out);
    ofs << "# step time(s) Te1_sim Te2_sim Te1_theory Te2_theory Te1_err% Te2_err%" << std::endl;

    double Te1_theory = Te1_init;
    double Te2_theory = Te2_init;

    Tic("Double Temperature Relaxation Validation")
    for (int i = 0; i <= total_steps; i++) {
        if (i > 0) {
            double tau_12 = calc_tau(Te1_theory, Te2_theory, m_e_mass, m_e_mass, n_dens, n_dens, q_e, q_e);
            double tau_21 = calc_tau(Te2_theory, Te1_theory, m_e_mass, m_e_mass, n_dens, n_dens, q_e, q_e);

            double dTe1 = (Te2_theory - Te1_theory) / tau_12 * dt;
            double dTe2 = (Te1_theory - Te2_theory) / tau_21 * dt;
            Te1_theory += dTe1;
            Te2_theory += dTe2;

            calc_temperature(ele1, Te1_f, m_e_mass, to_3d);
            calc_temperature(ele2, Te2_f, m_e_mass, to_3d);
            solver.solve(dt, num2dens, to_3d, v2p);
        } else {
            calc_temperature(ele1, Te1_f, m_e_mass, to_3d);
            calc_temperature(ele2, Te2_f, m_e_mass, to_3d);
        }

        if (i % output_interval == 0) {
            HCf<1> Te1_h(grid), Te2_h(grid);
            Te1_h.copy(Te1_f.getContent().to_host());
            Te2_h.copy(Te2_f.getContent().to_host());
            double Te1_sim = Te1_h(0);
            double Te2_sim = Te2_h(0);

            double Te1_err = (Te1_sim - Te1_theory) / Te1_theory * 100.0;
            double Te2_err = (Te2_sim - Te2_theory) / Te2_theory * 100.0;
            double time_s = i * dt;

            ofs << i << " " << time_s << " " << Te1_sim << " " << Te2_sim << " "
                << Te1_theory << " " << Te2_theory << " "
                << Te1_err << " " << Te2_err << std::endl;

            std::cout << "Step " << i << " (t=" << time_s * 1e6 << "us): "
                      << "Te1_sim=" << Te1_sim << ", Te1_th=" << Te1_theory << ", err=" << Te1_err << "%  |  "
                      << "Te2_sim=" << Te2_sim << ", Te2_th=" << Te2_theory << ", err=" << Te2_err << "%"
                      << std::endl;
        }
    }
    Toc

    ofs.close();
    std::cout << "\nValidation complete. Results saved to coulomb_doubleT_validation.plt" << std::endl;
}

// =====================================================
// Test 4: Spitzer relaxation (e-i temperature equilibration)
// =====================================================
void test_spitzer_relaxation() {
    std::cout << "\n========== test_spitzer_relaxation ==========" << std::endl;
    size_t N = 500000;

    double Te_init = 100;
    double Ti_init = 10;
    double q_e = -Q;
    double q_i = Q;
    double n_dens = 1e18;

    sycl::queue q{sycl::default_selector_v};
    std::cout << "Using device: " << q.get_device().get_info<sycl::info::device::name>() << std::endl;
    Grid grid({-0.1}, {0.9}, {10});

    ParticleGroup ele(q), ion(q);
    std::vector<Particle> particles;
    make_maxwell_particles(particles, N, Te_init, m_e_mass, -0.1, 0.9);
    ele.insert<false>(particles);
    make_maxwell_particles(particles, N, Ti_init, m_p_mass, -0.1, 0.9);
    ion.insert<false>(particles);

    auto num2dens = [=](size_t, size_t count) -> double {
        return n_dens;
    };

    DCf<> Te_f(q, grid), Ti_f(q, grid);

    psum::particle_collision::coulomb_collision_solver<Particle, pos_x_nan_is_invalid, 1>
        solver(q, grid, {{ele, Role::electron, m_e_mass, q_e, Te_f},
                         {ion, Role::ion, m_p_mass, q_i, Ti_f}});

    double dt = 1e-7;
    int total_steps = 500000;
    int output_interval = 10000;

    std::ofstream ofs("coulomb_spitzer_validation.plt", std::ios::out);
    ofs << "# step time(s) Te_sim Ti_sim Te_theory Ti_theory Te_err% Ti_err% lambdaD" << std::endl;

    double Te_theory = Te_init;
    double Ti_theory = Ti_init;

    Tic("Spitzer Ion-Electron Relaxation Validation")
    for (int i = 0; i <= total_steps; i++) {
        if (i > 0) {
            double tau_ei = calc_tau(Te_theory, Ti_theory, m_e_mass, m_p_mass, n_dens, n_dens, q_e, q_i);
            double tau_ie = calc_tau(Ti_theory, Te_theory, m_p_mass, m_e_mass, n_dens, n_dens, q_i, q_e);

            double dTe = (Ti_theory - Te_theory) / tau_ei * dt;
            double dTi = (Te_theory - Ti_theory) / tau_ie * dt;
            Te_theory += dTe;
            Ti_theory += dTi;

            calc_temperature(ele, Te_f, m_e_mass, to_3d);
            calc_temperature(ion, Ti_f, m_p_mass, to_3d);
            solver.solve(dt, num2dens, to_3d, v2p);
        } else {
            calc_temperature(ele, Te_f, m_e_mass, to_3d);
            calc_temperature(ion, Ti_f, m_p_mass, to_3d);
        }

        if (i % output_interval == 0) {
            HCf<1> Te_h(grid), Ti_h(grid);
            Te_h.copy(Te_f.getContent().to_host());
            Ti_h.copy(Ti_f.getContent().to_host());
            double Te_sim = Te_h(0);
            double Ti_sim = Ti_h(0);

            double Te_err = (Te_sim - Te_theory) / Te_theory * 100.0;
            double Ti_err = (Ti_sim - Ti_theory) / Ti_theory * 100.0;
            double time_s = i * dt;

            double ld_val = solver.lambdaD().getContent().to_host()[0];
            ofs << i << " " << time_s << " " << Te_sim << " " << Ti_sim << " "
                << Te_theory << " " << Ti_theory << " "
                << Te_err << " " << Ti_err << " " << ld_val << std::endl;

            std::cout << "Step " << i << " (t=" << time_s * 1e6 << "us): "
                      << "Te_sim=" << Te_sim << ", Te_th=" << Te_theory << ", err=" << Te_err << "%  |  "
                      << "Ti_sim=" << Ti_sim << ", Ti_th=" << Ti_theory << ", err=" << Ti_err << "%"
                      << "  lambdaD=" << ld_val << std::endl;
        }
    }
    Toc

    ofs.close();
    std::cout << "\nValidation complete. Results saved to coulomb_spitzer_validation.plt" << std::endl;
}

int main() {
    test_doubleT();
    test_spitzer_relaxation();
    gt_print2screen(psum::timer::MaxkMode);
    return 0;
}
