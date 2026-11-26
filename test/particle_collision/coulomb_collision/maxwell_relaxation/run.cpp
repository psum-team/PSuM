#include "../calc_tau.hpp"
#include <psum/psum.hpp>
#include <sycl/sycl.hpp>
#include <iostream>
#include <fstream>
#include <cmath>
#include "src/timer.hpp"

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

void state_file(bool save, const char* name, std::vector<Particle>& ps){
 std::fstream f(name, std::ios::binary | (save?std::ios::out:std::ios::in)); f.exceptions(std::ios::badbit|std::ios::failbit);
 auto io=[&](auto& v){if(save) f.write(reinterpret_cast<const char*>(&v),sizeof(v));else f.read(reinterpret_cast<char*>(&v),sizeof(v));};
 for(auto& p:ps){io(get<position>(p).x());io(get<velocity>(p).x());io(get<squared_velocity>(p).x());io(get<random_seed>(p));}
}
// =====================================================
// Test 3: Double temperature (two electron populations) with theory
// =====================================================
void test_doubleT(unsigned seed) {
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
    particles.resize(N); state_file(false,"group1.bin",particles);
    ele1.insert<false>(particles);
    particles.resize(N); state_file(false,"group2.bin",particles);
    ele2.insert<false>(particles);

    auto num2dens = [=](size_t, size_t count) -> double {
        return n_dens;
    };

    DCf<> Te1_f(q, grid), Te2_f(q, grid);

    psum::particle_collision::coulomb_collision_solver<Particle, pos_x_nan_is_invalid, 1>
        solver(q, grid, {{ele1, Role::electron, m_e_mass, q_e, Te1_f},
                         {ele2, Role::electron, m_e_mass, q_e, Te2_f}},seed);

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

        if ((i<=80000 && i%2000==0) || (i>80000 && i%10000==0)) {
            for (int side=0;side<2;++side) {
                auto v=(side?ele2:ele1).get_content().to_host();
                std::ofstream f("snapshots/"+std::to_string(i)+"-"+std::to_string(side)+".bin",std::ios::binary);
                f.exceptions(std::ios::badbit|std::ios::failbit);
                for(auto &p:v){ const double row[]={get<velocity>(p).x(),get<squared_velocity>(p).x()}; f.write(reinterpret_cast<const char*>(row),sizeof(row)); }
            }
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
int main(int argc,char** argv){
 if(argc==3 && std::string(argv[1])=="--init"){psum::random::global_random::getInstance().engine.seed(std::stoul(argv[2]));std::vector<Particle> p;
 make_maxwell_particles(p,100000,300,m_e_mass,-0.1,0.9);state_file(true,"group1.bin",p);
 make_maxwell_particles(p,100000,150,m_e_mass,-0.1,0.9);state_file(true,"group2.bin",p);return 0;}
 test_doubleT(std::stoul(argv[1]));
}
