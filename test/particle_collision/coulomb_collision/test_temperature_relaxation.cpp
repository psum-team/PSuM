#include "calc_tau.hpp"
#include <psum/psum.hpp>
#include <sycl/sycl.hpp>
#include <iostream>
#include <cmath>
#include <stdexcept>
#include <numeric>


using namespace psum::prelude;
using Role = psum::particle_collision::coulomb_role;
using property::position;
using property::velocity;
using property::random_seed;

constexpr double Q = 1.602176634e-19;
constexpr double m_e_mass = 9.109e-31;
constexpr double m_p_mass = 1.67e-27;


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

void calc_temperature(ParticleGroup& group, DCf<>& temperature, double mass) {
    DCf<2> moments(temperature.getQueue(), temperature.getGrid());
    moments.setZero();
    group.for_each([&](sycl::handler& h) {
        auto m = moments.get_access(h);
        return [=](Particle& p) {
            const auto cell = m.getGrid().c2i(m.getGrid().nC(get<position>(p)));
            using Atomic = sycl::atomic_ref<double, sycl::memory_order::relaxed,
                sycl::memory_scope::device, sycl::access::address_space::global_space>;
            const double vx = get<velocity>(p).x();
            Atomic(m(cell)(0)).fetch_add(1.0);
            Atomic(m(cell)(1)).fetch_add(vx*vx + get<squared_velocity>(p).x());
        };
    });
    temperature.for_each([&](sycl::handler& h) {
        auto m = moments.get_access(h);
        return [=](size_t i, double& t) {
            t = m(i)(0) ? mass*m(i)(1)/(3*Q*m(i)(0)) : 0;
        };
    });
}

void make_maxwell_particles(std::vector<Particle>& particles, size_t N,
                            double Te_eV, double mass, double pos_range_lo, double pos_range_hi) {
    static psum::random::rander R(15343);
    particles.clear();
    for (size_t i = 0; i < N; i++) {
        auto v = psum::random::RandFunction3D::RandV_Maxwell(R, Te_eV * 11600, mass);
        double pos = R() * (pos_range_hi - pos_range_lo) + pos_range_lo;
        Particle p;
        psum::tag::get<position>(p) = Eigen::Vector<double, 1>{pos};
        psum::tag::get<velocity>(p) = Eigen::Vector<double, 1>{v.x()};
        psum::tag::get<squared_velocity>(p) = Eigen::Vector<double, 1>{v.y() * v.y() + v.z() * v.z()};
        psum::tag::get<random_seed>(p) = R.get_engine()();
        particles.push_back(p);
    }
}

// Short runs use the measured initial temperatures to separate sampling error
// in the Maxwellian initialization from the subsequent relaxation error.
void run_relaxation(bool electron_ion) {
    sycl::queue q;
    const size_t count = 32768;
    const int cells = electron_ion ? 10 : 1;
    const double density = electron_ion ? 1e18 : 1e19;
    const double second_mass = electron_ion ? m_p_mass : m_e_mass;
    const double second_charge = electron_ion ? Q : -Q;
    const double dt = electron_ion ? 4e-7 : 4e-9;
    const int steps = 512;
    const int interval = 128;
    double theory1 = electron_ion ? 100 : 300, theory2 = electron_ion ? 10 : 150;

    Grid grid({-0.1}, {0.9}, {cells});
    ParticleGroup first(q), second(q);
    std::vector<Particle> particles;
    make_maxwell_particles(particles, count, theory1, m_e_mass, -0.1, 0.9);
    first.insert<false>(particles);
    make_maxwell_particles(particles, count, theory2, second_mass, -0.1, 0.9);
    second.insert<false>(particles);
    DCf<> t1(q, grid), t2(q, grid);
    psum::particle_collision::coulomb_collision_solver<Particle, pos_x_nan_is_invalid, 1>
        solver(q, grid, {{first, Role::electron, m_e_mass, -Q, t1},
                         {second, electron_ion ? Role::ion : Role::electron, second_mass, second_charge, t2}}, 101);

    // Average the homogeneous cells to reduce temperature sampling noise.
    auto observed_temperature = [](const auto& field) {
        const auto values = field.getContent().to_host();
        return std::accumulate(values.begin(), values.end(), 0.0) / values.size();
    };
    std::cout << (electron_ion ? "Spitzer e-i" : "two electron temperatures")
              << " on " << q.get_device().get_info<sycl::info::device::name>() << '\n';
    for (int step = 0; step <= steps; ++step) {
        calc_temperature(first, t1, m_e_mass);
        calc_temperature(second, t2, second_mass);
        if (step % interval == 0) {
            const double a = observed_temperature(t1);
            const double b = observed_temperature(t2);
            if (step == 0) { theory1 = a; theory2 = b; }
            const double err1 = (a/theory1 - 1)*100, err2 = (b/theory2 - 1)*100;
            std::cout << step << " T_sim=" << a << ',' << b << " T_theory="
                      << theory1 << ',' << theory2 << " errors(%)=" << err1 << ',' << err2 << '\n';
            if (!std::isfinite(err1) || !std::isfinite(err2) ||
                std::max(std::abs(err1), std::abs(err2)) > 2.0)
                throw std::runtime_error("temperature relaxation departed from theory");
        }
        if (step == steps) break;
        const double tau12 = calc_tau(theory1, theory2, m_e_mass, second_mass, density, density, -Q, second_charge);
        const double tau21 = calc_tau(theory2, theory1, second_mass, m_e_mass, density, density, second_charge, -Q);
        const double difference = theory1 - theory2;
        theory1 -= difference * dt / tau12;
        theory2 += difference * dt / tau21;
        solver.solve(dt, [=](size_t, size_t){return density;}, to_3d, v2p);
    }
}

int main() {
    run_relaxation(false);
    run_relaxation(true);
}
