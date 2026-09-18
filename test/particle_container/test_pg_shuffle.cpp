#include <iostream>
#include <vector>
#include <algorithm>
#include <Eigen/Core>
#include <psum/particle_container.hpp>
#include <psum/tag.hpp>
#include "../../src/random/rander.hpp"

using namespace psum::tag;
using namespace psum::particle_container;
using property::position;
using property::velocity;
using psum::random::rander;

using Particle = tagged_struct<
    tag_bind<position, Eigen::RowVector3f>,
    tag_bind<velocity, Eigen::RowVector3f>>;

using pContainer = particle_group<Particle, pos_x_nan_is_invalid>;

struct ParticleComparator {
    bool operator()(const Particle& a, const Particle& b) const {
        auto pos_a = get<position>(a);
        auto pos_b = get<position>(b);
        if (pos_a.x() != pos_b.x()) return pos_a.x() < pos_b.x();
        if (pos_a.y() != pos_b.y()) return pos_a.y() < pos_b.y();
        if (pos_a.z() != pos_b.z()) return pos_a.z() < pos_b.z();
        
        auto vel_a = get<velocity>(a);
        auto vel_b = get<velocity>(b);
        if (vel_a.x() != vel_b.x()) return vel_a.x() < vel_b.x();
        if (vel_a.y() != vel_b.y()) return vel_a.y() < vel_b.y();
        return vel_a.z() < vel_b.z();
    }
};

void run_shuffle(size_t num_particles) {
    sycl::queue q{ sycl::default_selector_v };
    pContainer container(q);
    rander R;

    std::vector<Particle> p_vec;
    for (size_t i = 0; i < num_particles; ++i) {
        Particle p;
        get<position>(p) = {R(), R(), R()};
        get<velocity>(p) = {R(), R(), R()};
        p_vec.push_back(p);
    }

    container.insert(p_vec);
    
    std::vector<Particle> original_set = p_vec;
    std::sort(original_set.begin(), original_set.end(), ParticleComparator());

    std::cout << "Executing Feistel Shuffle..." << std::endl;
    container.shuffle();

    std::vector<Particle> shuffled_set = container.get_content().to_host();
    
    bool order_is_different = 
        (get<position>(shuffled_set[0]).x() != get<position>(p_vec[0]).x()) || 
        (get<position>(shuffled_set[1]).x() != get<position>(p_vec[1]).x());

    std::sort(shuffled_set.begin(), shuffled_set.end(), ParticleComparator());

    bool is_identical = true;
    if (original_set.size() != shuffled_set.size()) {
        is_identical = false;
    } else {
        for (size_t i = 0; i < original_set.size(); ++i) {
            if (get<position>(original_set[i]) != get<position>(shuffled_set[i]) ||
                get<velocity>(original_set[i]) != get<velocity>(shuffled_set[i])) {
                is_identical = false;
                break;
            }
        }
    }

    std::cout << "Set Bijectivity (Sorted Match): " << (is_identical ? "✅ PASS" : "❌ FAIL") << std::endl;
    std::cout << "Order Randomization: " << (order_is_different ? "✅ PASS" : "❌ FAIL") << std::endl;
}

int main() {
    std::cout << "--- High-Intensity Shuffle Test ---" << std::endl;
    run_shuffle(1e4);
    run_shuffle(1e5);
    run_shuffle(1e6);
    run_shuffle(1e7);
    run_shuffle(2e7);
    run_shuffle(3e7);
}