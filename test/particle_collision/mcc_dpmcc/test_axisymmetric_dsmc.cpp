#include <psum/psum.hpp>
#include <array>
#include <cassert>
#include <cstdint>
#include <vector>

using namespace psum::prelude;
using namespace psum::particle_collision;

namespace psum::particle_collision {
template<>
struct position_projection<species_tags::Nitrogen> {
    static std::array<double, 2> project(const Eigen::Vector3d& position) {
        return {position.x(), sycl::hypot(position.y(), position.z())};
    }
};
}

struct collision_hits : tag::foundation::abstract_tag {
    static constexpr auto tag_name = "collision_hits";
};
struct wall_hits : tag::foundation::abstract_tag {
    static constexpr auto tag_name = "wall_hits";
};

using Particle = tagged_struct<
    tag_bind<property::position, Eigen::Vector3d>,
    tag_bind<property::velocity, Eigen::Vector3d>,
    tag_bind<property::random_seed, uint32_t>,
    tag_bind<collision_hits, unsigned>,
    tag_bind<wall_hits, unsigned>
>;
using Group = particle_group<Particle, pos_x_nan_is_invalid>;
using Density = cell_field2D<double>;

struct nitrogen_elastic : deferred_pairing_collision_tag {
    using incident_species_tag = species_tags::Nitrogen;
    using background_species_tag = incident_species_tag;

    // The faster molecule issues the request. The slower one supplies the partner.
    static double request_probability(const Particle& p, const auto&) {
        return get<property::velocity>(p).x() == 2 ? 1.0 : 0.0;
    }

    static double pair_collision_cross_section(const auto&, const auto&) { return 1.0; }

    static void pair_collide(const auto& pair, const auto&) {
        // Reverse the relative velocity of equal-mass molecules.
        const Eigen::Vector3d first_velocity = get<property::velocity>(*pair.first);
        get<property::velocity>(*pair.first) = get<property::velocity>(*pair.second);
        get<property::velocity>(*pair.second) = first_velocity;
        ++get<collision_hits>(*pair.first);
        ++get<collision_hits>(*pair.second);
    }
};

void stream_past_sphere(Group& particles, double dt) {
    particles.for_each([=](sycl::handler&) {
        return [=](Particle& particle) {
            auto& position = get<property::position>(particle);
            auto& velocity = get<property::velocity>(particle);
            // Intersect the swept trajectory with the unit sphere at the origin.
            const double speed_squared = velocity.squaredNorm();
            const double along = position.dot(velocity);
            const double discriminant = along * along
                - speed_squared * (position.squaredNorm() - 1.0);
            double remaining = dt;
            if (along < 0 && discriminant >= 0) {
                const double hit_time = (-along - sycl::sqrt(discriminant)) / speed_squared;
                if (hit_time >= 0 && hit_time <= dt) {
                    position += hit_time * velocity;
                    const Eigen::Vector3d normal = position.normalized();
                    velocity -= 2.0 * velocity.dot(normal) * normal;
                    remaining -= hit_time;
                    ++get<wall_hits>(particle);
                }
            }
            position += remaining * velocity;
        };
    });
}

int main() {
    sycl::queue q;
    Group particles(q);
    const grid2D grid({-3., 0.}, {3., 3.}, {6, 3});
    Density density(q, grid);
    density.for_each([](sycl::handler&) {
        return [](size_t, double& value, const auto& location) {
            value = location[1] < 2.0 ? 16.0 : 0.0;
        };
    });

    std::vector<Particle> initial;
    for (double radius : {0.5, 1.5, 2.5}) {
        // Each pair shares an annulus but differs in azimuth. Neither Cartesian
        // transverse component can stand in for the radius.
        initial.emplace_back(Eigen::Vector3d(-2.75, 0, -radius), Eigen::Vector3d(2, 0, 0), 12345u, 0u, 0u);
        initial.emplace_back(Eigen::Vector3d(-2.75, -radius, 0), Eigen::Vector3d(1, 0, 0), 54321u, 0u, 0u);
    }
    particles.insert<false>(initial);

    using Model = collision_model<nitrogen_elastic>;
    struct Binding { Group* group; Density* density; };
    species_contexts<Model, Group, Density> species{Binding{&particles, &density}};
    collision_channel_contexts<Model, Group, decltype(species)> channels(q, species);
    context_composition context{species, channels};

    constexpr double dt = 0.25;
    stream_past_sphere(particles, dt);
    execute_collision_model<Model>(context, dt);

    // With two molecules per cell, the first two pairs have acceptance exactly one.
    // The outer pair has zero background density.
    const auto counts = species.get<species_tags::Nitrogen>().particle_counts.getContent().to_host();
    std::vector<uint32_t> expected_counts(counts.size());
    for (double radius : {0.5, 1.5, 2.5})
        expected_counts[grid.c2i(grid.nC(std::array{-2.5, radius}))] = 2;
    assert(counts == expected_counts);

    const auto paired = particles.get_content().to_host();
    assert(paired.size() == initial.size());
    for (size_t i = 0; i < paired.size(); ++i) {
        const bool active = i < 4;
        assert(get<collision_hits>(paired[i]) == unsigned(active));
        const double initial_speed = get<property::velocity>(initial[i]).x();
        const double expected_speed = active ? 3.0 - initial_speed : initial_speed;
        assert(get<property::velocity>(paired[i]).x() == expected_speed);
    }

    // Continue the short pass to distinguish trajectories hitting the sphere
    // from those passing outside it.
    for (int step = 0; step < 9; ++step) stream_past_sphere(particles, dt);
    const auto result = particles.get_content().to_host();
    assert(particles.size() == initial.size());
    for (size_t i = 0; i < result.size(); ++i) {
        const bool reflected = i < 2;
        assert(get<wall_hits>(result[i]) == unsigned(reflected));
        assert((get<property::velocity>(result[i]).x() < 0) == reflected);
        assert(get<property::position>(result[i]).norm() >= 1.0);
    }
}
