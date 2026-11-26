#include <psum/psum.hpp>
#include <cassert>
#include <cmath>
#include <vector>

using namespace psum::prelude;
using Role = psum::particle_collision::coulomb_role;

using Particle = tagged_struct<
    tag_bind<property::position, Eigen::Vector<double, 1>>,
    tag_bind<property::velocity, Eigen::Vector3d>,
    tag_bind<property::random_seed, uint32_t>
>;
using Group = particle_group<Particle, pos_x_nan_is_invalid>;

constexpr double electron_mass = 9.109e-31;
constexpr double proton_mass = 1.67e-27;
constexpr double elementary_charge = 1.602176634e-19;

struct Moments {
    Eigen::Vector3d momentum = Eigen::Vector3d::Zero();
    double energy = 0;
    double momentum_scale = 0;
};

Moments moments(const std::vector<Particle>& particles, double mass) {
    Moments result;
    for (const auto& p : particles) {
        const auto& v = get<property::velocity>(p);
        assert(v.allFinite());
        result.momentum += mass * v;
        result.energy += 0.5 * mass * v.squaredNorm();
        result.momentum_scale += mass * v.norm();
    }
    return result;
}

void test_scattering(double partner_mass, double partner_charge) {
    sycl::queue q;
    Group electrons(q), partners(q);
    grid1D grid({0.}, {1.}, {1});
    cell_field1D<double> electron_temperature(q, grid), partner_temperature(q, grid);
    electron_temperature.setConstant(100);
    partner_temperature.setConstant(10);

    psum::random::rander random(12345);
    auto make_particles = [&](size_t count, double mass, double temperature) {
        std::vector<Particle> particles(count);
        for (auto& p : particles) {
            get<property::position>(p).x() = 0.5;
            get<property::velocity>(p) =
                psum::random::RandFunction3D::RandV_Maxwell(random, temperature * 11600, mass);
            get<property::random_seed>(p) = random.get_engine()();
        }
        return particles;
    };
    const auto initial_electrons = make_particles(255, electron_mass, 100);
    const auto initial_partners = make_particles(256, partner_mass, 10);
    electrons.insert<false>(initial_electrons);
    partners.insert<false>(initial_partners);
    psum::particle_collision::coulomb_collision_solver<Particle, pos_x_nan_is_invalid, 1>
        solver(q, grid, {{electrons, Role::electron, electron_mass, -elementary_charge, electron_temperature},
                         {partners, partner_charge < 0 ? Role::electron : Role::ion, partner_mass, partner_charge, partner_temperature}});

    const auto before_electrons = moments(initial_electrons, electron_mass);
    const auto before_partners = moments(initial_partners, partner_mass);
    const auto read_velocity = [](const Particle& p) { return get<property::velocity>(p); };
    const auto write_velocity = [](Particle& p, const Eigen::Vector3d& v) { get<property::velocity>(p) = v; };
    solver.solve(1e-8, [](size_t, size_t) { return 1e19; },
                 read_velocity, write_velocity);
    q.wait_and_throw();

    const auto after_electrons = moments(electrons.get_content().to_host(), electron_mass);
    const auto after_partners = moments(partners.get_content().to_host(), partner_mass);
    const double delta_energy = after_electrons.energy - before_electrons.energy;
    const double total_energy = before_electrons.energy + before_partners.energy;
    assert(electrons.size() == initial_electrons.size());
    assert(partners.size() == initial_partners.size());
    assert(std::abs(delta_energy) > 1e-12 * total_energy);
    assert(std::abs(delta_energy + after_partners.energy - before_partners.energy) <= 1e-12 * total_energy);
    const Eigen::Vector3d delta_momentum = after_electrons.momentum + after_partners.momentum
        - before_electrons.momentum - before_partners.momentum;
    assert(delta_momentum.norm() <= 1e-12 * (before_electrons.momentum_scale + before_partners.momentum_scale));
}

using Solver = psum::particle_collision::coulomb_collision_solver<Particle, pos_x_nan_is_invalid, 1>;

Particle make_particle(double x, uint32_t seed) {
    Particle p;
    get<property::position>(p).x() = x;
    get<property::velocity>(p) = Eigen::Vector3d(1e5 * seed, 0, 0);
    get<property::random_seed>(p) = seed;
    return p;
}

constexpr auto read_v = [](const Particle& p) { return get<property::velocity>(p); };
constexpr auto write_v = [](Particle& p, const Eigen::Vector3d& v) { get<property::velocity>(p) = v; };

void test_outside_grid() {
    sycl::queue q;
    grid1D grid({0.}, {1.}, {1});
    Group particles(q);
    cell_field1D<double> temperature(q, grid);
    temperature.setConstant(10);
    const std::vector<Particle> outside{make_particle(-1, 1), make_particle(2, 2)};
    particles.insert<false>(outside);
    Solver solver(q, grid, {{particles, Role::ion, electron_mass, -elementary_charge, temperature}});
    auto advance = [&] {
        solver.solve(1e-8, [](size_t, size_t count) { return count * 1e19; }, read_v, write_v);
        const auto result = particles.get_content().to_host();
        assert(particles.size() == result.size());
        for (size_t i = 0; i < outside.size(); ++i) {
            assert(get<property::velocity>(result[i]) == get<property::velocity>(outside[i]));
            assert(get<property::random_seed>(result[i]) == get<property::random_seed>(outside[i]));
        }
        return result;
    };
    advance();
    assert(std::isnan(solver.lambdaD().getContent().to_host()[0]));
    const std::vector<Particle> inside{make_particle(0.5, 3), make_particle(0.5, 4)};
    particles.insert<false>(inside);
    const auto result = advance();
    assert(get<property::velocity>(result[2]) != get<property::velocity>(inside[0]));
    assert(get<property::velocity>(result[3]) != get<property::velocity>(inside[1]));
}

void test_multispecies_debye_length() {
    sycl::queue q;
    grid1D grid({0.}, {3.}, {3});
    Group electrons(q), ions(q);
    electrons.insert<false>(std::vector<Particle>{make_particle(.5, 1), make_particle(.5, 2),
                                                  make_particle(1.5, 3)});
    ions.insert<false>(std::vector<Particle>{make_particle(.5, 4), make_particle(1.5, 5),
                                             make_particle(1.5, 6), make_particle(1.5, 7)});
    cell_field1D<double> te(q, grid), ti(q, grid);
    ti.setConstant(5);
    Solver solver(q, grid, {{electrons, Role::electron, electron_mass, -elementary_charge, te},
                            {ions, Role::ion, proton_mass, 2 * elementary_charge, ti}});
    constexpr double eps0 = 8.854187817e-12;
    for (double electron_temperature : {20., 40.}) {
        te.setConstant(electron_temperature);
        solver.solve(0., [](size_t, size_t count) { return count * 1e19; }, read_v, write_v);
        const auto lengths = solver.lambdaD().getContent().to_host();
        for (size_t cell = 0; cell < 2; ++cell) {
            const double ne = cell == 0 ? 2e19 : 1e19;
            const double ni = cell == 0 ? 1e19 : 3e19;
            const double expected = std::sqrt(eps0 / (elementary_charge *
                (ne / electron_temperature + 4 * ni / 5.)));
            assert(std::abs(lengths[cell] / expected - 1.) < 1e-12);
        }
        assert(std::isnan(lengths[2]));
    }
}

void test_cross_section_limit() {
    using namespace psum::particle_collision;
    sycl::queue q;
    constexpr double mass = 131.293 * 1.66053906660e-27;
    constexpr double n_eff = 4e19, n_max = 8e19, lambdaD = 5.39e-7, dt = 1e-13;
    double* cosines = sycl::malloc_shared<double>(2, q);
    q.parallel_for(sycl::range<1>(2), [=](sycl::id<1> i) {
        const double speed = i[0] == 0 ? 100.0 : 3000.0;
        Eigen::Vector3d a(speed, 0, 0), b = Eigen::Vector3d::Zero();
        psum::random::rander random(12345);
        scattering_NanBu(a, b, mass, mass, elementary_charge, 2 * elementary_charge,
                         n_eff, coulomb_cross_section_limit(n_max), lambdaD, dt, random);
        cosines[i] = (a.x() - b.x()) / speed;
    }).wait_and_throw();
    for (int i = 0; i < 2; ++i) {
        const double speed = i == 0 ? 100.0 : 3000.0;
        const double b90 = 2 * elementary_charge * elementary_charge /
                           (4 * M_PI * 8.854187817e-12 * (mass / 2) * speed * speed);
        const double sigma = 4 * M_PI * b90 * b90 * std::log(std::max(lambdaD / b90, 1.01));
        const double spacing = std::cbrt(3 / (4 * M_PI * n_max));
        const double limit = 1 / (n_max * spacing);
        assert((sigma > limit) == (i == 0));
        const double strength = std::min(sigma, limit) * n_eff * speed * dt;
        assert(strength < 1e-2);
        psum::random::rander random(12345);
        assert(std::abs(cosines[i] - (1 + strength * std::log(random()))) < 1e-12);
    }
    sycl::free(cosines, q);
}

// An isolated equal-mass pair keeps its relative speed. The Nanbu first
// angular moment therefore gives <gx(k)> / gx(0) = exp(-k*s).
void test_pair_relaxation() {
    constexpr int cells = 16384;
    constexpr double speed = 100., n = 1e19, strength = 0.2;
    constexpr double mass = 131.293 * 1.66053906660e-27;
    const double spacing = std::cbrt(3 / (4 * M_PI * 2 * n));
    const double dt = strength * 2 * spacing / speed; // capped cross section, n_eff=n
    sycl::queue q;
    grid1D grid({0.}, {double(cells)}, {cells});
    Group ions(q);
    std::vector<Particle> input;
    psum::random::rander seeds(8953);
    for (int cell = 0; cell < cells; ++cell) {
        for (double sign : {1., -1.}) {
            Particle p = make_particle(cell + 0.5, seeds.get_engine()());
            get<property::velocity>(p) = Eigen::Vector3d(sign * speed / 2, 0, 0);
            input.push_back(p);
        }
    }
    ions.insert<false>(input);
    cell_field1D<double> temperature(q, grid);
    temperature.setConstant(1);
    Solver solver(q, grid, {{ions, Role::ion, mass, elementary_charge, temperature}}, 193);
    for (int step = 1; step <= 8; ++step) {
        solver.solve(dt, [](size_t, size_t count) { return count * n; }, read_v, write_v);
        if (step != 1 && step != 4 && step != 8) continue;
        const auto particles = ions.get_content().to_host();
        double mean = 0;
        for (int cell = 0; cell < cells; ++cell) {
            mean += (read_v(particles[2*cell]).x() - read_v(particles[2*cell+1]).x()) / speed;
        }
        mean /= cells;
        assert(std::abs(mean - std::exp(-step * strength)) < 0.025);
    }
}

int main() {
    test_pair_relaxation();
    test_cross_section_limit();
    test_outside_grid();
    test_multispecies_debye_length();
    test_scattering(electron_mass, -elementary_charge);
    test_scattering(proton_mass, elementary_charge);
}
