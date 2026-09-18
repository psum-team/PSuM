#include <psum/psum.hpp>
#include <sycl/sycl.hpp>
#include <iostream>
#include <fstream>
#include <cmath>
#include "../../src/timer.hpp"

using namespace psum::prelude;
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

using Particle3D = tagged_struct<
    tag_bind<position, Eigen::Vector<double, 1>>,
    tag_bind<velocity, Eigen::Vector3d>,
    tag_bind<random_seed, uint32_t>
>;

using ParticleGroup = particle_group<Particle, pos_x_nan_is_invalid>;
using ParticleGroup3D = particle_group<Particle3D, pos_x_nan_is_invalid>;
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

auto v2p_3d = [](auto& p, Eigen::Vector3d v_new) {
    psum::tag::get<velocity>(p) = v_new;
};

auto to_3d_direct = [](auto& p) -> Eigen::Vector3d {
    return psum::tag::get<velocity>(p);
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

template<typename Func>
void calc_temperature_3d(ParticleGroup3D& group, DCf<>& T, double mass, Func _func) {
    DCf<3> moments(T.getQueue(), T.getGrid());
    moments.setZero();

    group.for_each([&](sycl::handler& h) {
        auto mom_acc = moments.get_access(h);
        return [=](const Particle3D& p) {
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

template<typename Func>
void calc_temperature_xy(ParticleGroup3D& group, DCf<>& Tx, DCf<>& Ty, double mass, Func _func) {
    DCf<4> moments(Tx.getQueue(), Tx.getGrid());
    moments.setZero();

    group.for_each([&](sycl::handler& h) {
        auto mom_acc = moments.get_access(h);
        return [=](const Particle3D& p) {
            double pos = psum::tag::get<position>(p).x();
            Eigen::Vector3d v = _func(p);
            size_t ci = static_cast<size_t>(mom_acc.getGrid().c2i(
                mom_acc.getGrid().nC(Eigen::Vector<double, 1>{pos})));
            sycl::atomic_ref<double, sycl::memory_order::relaxed,
                            sycl::memory_scope::device,
                            sycl::access::address_space::global_space> num_ref(mom_acc(ci)(0));
            num_ref.fetch_add(1.0);
            sycl::atomic_ref<double, sycl::memory_order::relaxed,
                            sycl::memory_scope::device,
                            sycl::access::address_space::global_space> vx2_ref(mom_acc(ci)(1));
            vx2_ref.fetch_add(v.x() * v.x());
            sycl::atomic_ref<double, sycl::memory_order::relaxed,
                            sycl::memory_scope::device,
                            sycl::access::address_space::global_space> v2_ref(mom_acc(ci)(2));
            v2_ref.fetch_add(v.x() * v.x() + v.y() * v.y() + v.z() * v.z());
            sycl::atomic_ref<double, sycl::memory_order::relaxed,
                            sycl::memory_scope::device,
                            sycl::access::address_space::global_space> vyp2_ref(mom_acc(ci)(3));
            vyp2_ref.fetch_add(v.y() * v.y() + v.z() * v.z());
        };
    });

    Tx.for_each([&](sycl::handler& h) {
        auto mom_acc = moments.get_access(h);
        return [=](size_t i, double& v) {
            const Eigen::RowVector<double, 4>& m = mom_acc(i);
            double num = m[0];
            if (num == 0) { v = 0; return; }
            double vx2 = m[1] / num;
            v = vx2 * mass * 0.5 / Q * 2.0;
        };
    });

    Ty.for_each([&](sycl::handler& h) {
        auto mom_acc = moments.get_access(h);
        return [=](size_t i, double& v) {
            const Eigen::RowVector<double, 4>& m = mom_acc(i);
            double num = m[0];
            if (num == 0) { v = 0; return; }
            double vyp2 = m[3] / num;
            v = vyp2 * mass * 0.5 / Q;
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

void make_maxwell_particles_3d(std::vector<Particle3D>& particles, size_t N,
                               double Te_eV, double mass, double pos_range_lo, double pos_range_hi) {
    particles.clear();
    for (size_t i = 0; i < N; i++) {
        psum::random::rander R;
        auto v = psum::random::RandFunction3D::RandV_Maxwell(R, Te_eV * 11600, mass);
        double pos = psum::random::global_random::rand() * (pos_range_hi - pos_range_lo) + pos_range_lo;
        Particle3D p;
        psum::tag::get<position>(p) = Eigen::Vector<double, 1>{pos};
        psum::tag::get<velocity>(p) = v;
        psum::tag::get<random_seed>(p) = psum::random::global_random::rand_uint();
        particles.push_back(p);
    }
}

double calc_tau(double T_alpha_eV, double T_beta_eV,
                double m_alpha, double m_beta,
                double n_alpha, double n_beta,
                double q_alpha, double q_beta) {
    double eV_to_J = 1.602176634e-19;
    double eps0_loc = 8.854187817e-12;
    double hbar_loc = 1.054571817e-34;
    double kT_alpha = T_alpha_eV * eV_to_J;
    double kT_beta = T_beta_eV * eV_to_J;
    double v_th = std::sqrt(kT_alpha / m_alpha + kT_beta / m_beta);
    double m12 = m_alpha * m_beta / (m_alpha + m_beta);
    double lambdaD_sq_inv = n_alpha * q_alpha * q_alpha / (eps0_loc * kT_alpha)
                          + n_beta * q_beta * q_beta / (eps0_loc * kT_beta);
    double lambdaD = 1.0 / std::sqrt(lambdaD_sq_inv);
    double b_min = std::max(hbar_loc / (m12 * v_th), std::abs(q_alpha * q_beta) / (v_th * v_th * 4.0 * M_PI * eps0_loc * m12));
    double Lambda = std::max(lambdaD / b_min, 1.01);
    double lnLambda = std::log(Lambda);
    double coeff = 3.0 * std::pow(4.0 * M_PI * eps0_loc, 2) * m_alpha * m_beta;
    double denom = 8.0 * std::sqrt(2.0 * M_PI) * n_beta
                   * q_alpha * q_alpha * q_beta * q_beta * lnLambda;
    double thermal = std::pow(kT_alpha / m_alpha + kT_beta / m_beta, 1.5);
    return coeff / denom * thermal;
}

// =====================================================
// Test 1: Conservation (momentum & energy) - e-i
// =====================================================
void test_conservation() {
    std::cout << "\n========== test_conservation ==========" << std::endl;
    size_t N = 1000000;
    double Te = 300, Ti = 30;

    sycl::queue q{sycl::default_selector_v};
    std::cout << "Using device: " << q.get_device().get_info<sycl::info::device::name>() << std::endl;
    Grid grid({-0.1}, {0.9}, {100});

    ParticleGroup ele(q), ion(q);
    std::vector<Particle> particles;
    make_maxwell_particles(particles, N, Te, m_e_mass, -0.1, 0.9);
    ele.insert<false>(particles);
    make_maxwell_particles(particles, N, Ti, m_p_mass, -0.1, 0.9);
    ion.insert<false>(particles);

    auto num2dens = [=](size_t, size_t count) -> double {
        return count * 1.0e15;
    };

    DCf<> Te_f(q, grid), Ti_f(q, grid);
    calc_temperature(ele, Te_f, m_e_mass, to_3d);
    calc_temperature(ion, Ti_f, m_p_mass, to_3d);

    psum::particle_collision::coulomb_collision_solver<Particle, pos_x_nan_is_invalid, 1>
        solver(q, grid, {&ele, &ion});

    Tic("Coulomb Collision Solve")
    for (int i = 0; i < 20; i++) {
        calc_temperature(ele, Te_f, m_e_mass, to_3d);
        calc_temperature(ion, Ti_f, m_p_mass, to_3d);
        solver.solve(1e-8, num2dens,
                     {Te_f.data(), Ti_f.data()},
                     {m_e_mass, m_p_mass}, {-Q, Q},
                     to_3d, v2p);
        double ele_px = 0, ele_e = 0, ion_px = 0, ion_e = 0;
        {
            std::vector<Particle> host_data = ele.get_content().to_host();
            for (auto& p : host_data) {
                if (std::isnan(psum::tag::get<position>(p).x())) continue;
                ele_px += m_e_mass * psum::tag::get<velocity>(p).x();
                ele_e += 0.5 * m_e_mass * (psum::tag::get<velocity>(p).x() * psum::tag::get<velocity>(p).x()
                                           + psum::tag::get<squared_velocity>(p).x());
            }
        }
        {
            std::vector<Particle> host_data = ion.get_content().to_host();
            for (auto& p : host_data) {
                if (std::isnan(psum::tag::get<position>(p).x())) continue;
                ion_px += m_p_mass * psum::tag::get<velocity>(p).x();
                ion_e += 0.5 * m_p_mass * (psum::tag::get<velocity>(p).x() * psum::tag::get<velocity>(p).x()
                                          + psum::tag::get<squared_velocity>(p).x());
            }
        }
        double total_px = ele_px + ion_px;
        double total_e = ele_e + ion_e;
        std::cout << "Step " << i << ": e: p=" << ele_px << ", e=" << ele_e
                  << " | i: p=" << ion_px << ", e=" << ion_e
                  << " | tot: p=" << total_px << ", e=" << total_e << std::endl;
    }
    Toc
}

// =====================================================
// Test 2: Anisotropic temperature relaxation (Tx != Ty)
// =====================================================
void test_anisotropic() {
    std::cout << "\n========== test_anisotropic ==========" << std::endl;
    size_t N = 100000;
    double Te = 300;
    double m_e = m_e_mass;

    sycl::queue q{sycl::default_selector_v};
    std::cout << "Using device: " << q.get_device().get_info<sycl::info::device::name>() << std::endl;
    Grid grid({-0.1}, {0.9}, {1});

    ParticleGroup3D ele(q);
    std::vector<Particle3D> particles;
    for (size_t i = 0; i < N; i++) {
        psum::random::rander R;
        auto v = psum::random::RandFunction3D::RandV_Maxwell(R, 11600 * Te, m_e);
        auto v2 = psum::random::RandFunction3D::RandV_Maxwell(R, 11600 * Te / 2, m_e);
        Particle3D p;
        psum::tag::get<position>(p) = Eigen::Vector<double, 1>{psum::random::global_random::rand() - 0.1};
        psum::tag::get<velocity>(p) = Eigen::Vector3d{v.x(), v2.y(), v2.z()};
        psum::tag::get<random_seed>(p) = psum::random::global_random::rand_uint();
        particles.push_back(p);
    }
    ele.insert<false>(particles);

    DCf<> Te_f(q, grid), Tex_f(q, grid), Tey_f(q, grid);

    calc_temperature_xy(ele, Tex_f, Tey_f, m_e, to_3d_direct);
    calc_temperature_3d(ele, Te_f, m_e, to_3d_direct);

    psum::particle_collision::coulomb_collision_solver<Particle3D, pos_x_nan_is_invalid, 1>
        solver(q, grid, {&ele});

    auto num2dens = [=](size_t, size_t count) -> double {
        return count * 1.0e14;
    };

    Tic("Coulomb Collision Solve")
    std::ofstream ofs("Texy_evolution.plt", std::ios::out);
    for (int i = 0; i < 20000; i++) {
        calc_temperature_xy(ele, Tex_f, Tey_f, m_e, to_3d_direct);
        calc_temperature_3d(ele, Te_f, m_e, to_3d_direct);
        solver.solve(1e-8, num2dens,
                     {Te_f.data()},
                     {m_e}, {-Q},
                     to_3d_direct, v2p_3d);
        if (i % 100 == 0) {
            HCf<1> Tex_h(grid), Tey_h(grid);
            Tex_h.copy(Tex_f.getContent().to_host());
            Tey_h.copy(Tey_f.getContent().to_host());
            double Tex_mean = Tex_h(0);
            double Tey_mean = Tey_h(0);
            ofs << i << " " << Tex_mean << " " << Tey_mean << " " << Tex_mean - Tey_mean << std::endl;
            std::cout << "Step " << i << ": Tx=" << Tex_mean << ", Ty=" << Tey_mean
                      << ", diff=" << Tex_mean - Tey_mean << std::endl;
        }
    }
    Toc
    ofs.close();
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
        solver(q, grid, {&ele1, &ele2});

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
            solver.solve(dt, num2dens,
                         {Te1_f.data(), Te2_f.data()},
                         {m_e_mass, m_e_mass}, {q_e, q_e},
                         to_3d, v2p);
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
        solver(q, grid, {&ele, &ion});

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
            solver.solve(dt, num2dens,
                         {Te_f.data(), Ti_f.data()},
                         {m_e_mass, m_p_mass}, {q_e, q_i},
                         to_3d, v2p);
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
    // test_conservation();
    // test_anisotropic();
    test_doubleT();
    test_spitzer_relaxation();
    gt_print2screen(psum::timer::MaxkMode);
    return 0;
}