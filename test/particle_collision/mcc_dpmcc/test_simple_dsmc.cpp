#include <psum/psum.hpp>
#include "../../../application/benchmark/DSMC/custom_dpmcc.hpp"
#include <cassert>
#include <cmath>
#include <vector>

using namespace psum::prelude;

using Particle = tagged_struct<
    tag_bind<property::position, Eigen::Vector3d>,
    tag_bind<property::velocity, Eigen::Vector3d>,
    tag_bind<property::internal_energy, double>,
    tag_bind<property::random_seed, uint32_t>
>;

struct Moments {
    Eigen::Vector3d velocity_sum = Eigen::Vector3d::Zero();
    double energy = 0;
    double internal_energy = 0;
};

Moments moments(const std::vector<Particle>& particles) {
    constexpr double mass = 4.652e-26;
    Moments result;
    for (const auto& p : particles) {
        const auto& velocity = get<property::velocity>(p);
        const double internal = get<property::internal_energy>(p);
        assert(velocity.allFinite() && std::isfinite(internal) && internal >= 0);
        result.velocity_sum += velocity;
        result.energy += 0.5 * mass * velocity.squaredNorm() + internal;
        result.internal_energy += internal;
    }
    return result;
}

int main(int argc,char**) {
    if(argc!=1)return 2;
    sycl::queue q;
    particle_group<Particle, pos_x_nan_is_invalid> particles(q);
    cell_field3D<double> density(q, grid3D({0.,0.,0.}, {1.,1.,1.}, {1,1,1}));
    density.setConstant(1e23);

    std::vector<Particle> initial(1024);
    psum::random::rander random(12345);
    for (size_t i = 0; i < initial.size(); ++i) {
        auto& p = initial[i];
        get<property::position>(p).setConstant(0.5);
        get<property::velocity>(p) = Eigen::Vector3d(i % 2 ? 500 : -500, 0, 0);
        get<property::internal_energy>(p) = 0;
        get<property::random_seed>(p) = random.gen_seed();
    }
    particles.insert<false>(initial);
    const auto before = moments(initial);
    auto collide = N2_dsmc_model::make(particles, density);
    for (int step = 0; step < 10; ++step) collide(1e-9);

    const auto after = moments(particles.get_content().to_host());
    assert(particles.size() == initial.size());
    assert(after.internal_energy > 0);
    assert(std::abs(after.energy - before.energy) <= 1e-12 * before.energy);
    assert((after.velocity_sum - before.velocity_sum).norm() <= 1e-12 * initial.size() * 500);
}
