#include <iostream>
#include <map>
#include <string>
#include <Eigen/Dense>
#include "../../include/psum/field.hpp"
#include "../../src/particle_collision/grid_mapped_device_vector.hpp"
#include "../../src/particle_collision/pairing_engine.hpp"
#include "../../src/utils_sycl.hpp"
#include "../../src/random.hpp"
#include "../../include/psum/tag.hpp"
#include "../../include/psum/particle_container.hpp"

using namespace psum;
using namespace utils_sycl;
using namespace tag;
using namespace tag::property;
using namespace particle_collision;

using particle = tagged_struct<
    tag_bind<position, Eigen::Vector2d>,
    tag_bind<velocity, Eigen::Vector2d>,
    tag_bind<random_seed, uint32_t>
>;

void test_grid_mapped_device_vector() {
    sycl::queue q{sycl::default_selector_v};
    std::cout << "Testing on: " << q.get_device().get_info<sycl::info::device::name>() << std::endl;

    const size_t num_particles = 100000;
    psum::field::grid2D g({0.0, 0.0}, {1.0, 1.0}, {200, 300});

    grid_mapped_device_vector<Eigen::Vector2d, particle*, psum::field::grid2D> mapped_vec(q, g);

    std::vector<particle> host_particles;
    psum::random::rander R;
    for (size_t i = 0; i < num_particles; ++i) {
        particle p;
        get<position>(p) = Eigen::Vector2d(R(), R());
        host_particles.push_back(p);
    }

    psum::particle_container::device_vector<particle> device_particles(q, host_particles);
    psum::particle_container::device_vector<std::pair<Eigen::Vector2d, particle*>> input_data(q);
    input_data.reserve(num_particles);

    device_particles.for_each([&](sycl::handler& h) {
        auto input_data_acc = input_data.get_access(h);
        return [=](particle& p) {
            input_data_acc.push_back({get<position>(p), &p});
        };
    });

    mapped_vec.update(input_data);
    q.wait();

    bool* success = psum::utils_sycl::shared_variable<bool>(q);
    *success = true;

    mapped_vec.content().for_each([&](sycl::handler& h) {
        return [=](auto& pair) {
            if(pair.first != get<position>(*(pair.second))) {
                *success = false;
            }
        };
    });

    auto host_sorted_data = mapped_vec.content().to_host();
    size_t cell_id = g.c2i(g.nC(host_sorted_data[0].first));
    for (size_t i = 1; i < host_sorted_data.size(); ++i) {
        size_t next_id = g.c2i(g.nC(host_sorted_data[i].first));
        if (next_id < cell_id)
            *success = false;
        cell_id = next_id;
    }

    std::cout << "--- Running basic test ---" << std::endl;
    std::cout << "Test " << (*success ? "passed" : "failed") << std::endl;
}

void test_grid_mapped_detailed() {
    sycl::queue q{sycl::default_selector_v};
    
    psum::field::grid2D g({0.0, 0.0}, {1.0, 1.0}, {2, 2});
    grid_mapped_device_vector<Eigen::Vector2d, int, psum::field::grid2D> mapped_vec(q, g);

    std::vector<std::pair<Eigen::Vector2d, int>> host_input = {
        {{0.1, 0.1}, 10}, {{0.2, 0.1}, 11}, {{0.1, 0.2}, 12}, // Cell 0
        {{0.1, 0.6}, 30},                                    // Cell 1
        {{0.6, 0.6}, 40}, {{0.7, 0.7}, 41}                   // Cell 3
    };
    psum::particle_container::device_vector<std::pair<Eigen::Vector2d, int>> input_data(q, host_input);

    std::cout << "--- Running Detailed Partition Test ---" << std::endl;
    mapped_vec.update(input_data);
    q.wait();

    bool* test_passed = shared_variable<bool>(q);
    *test_passed = true;

    q.submit([&](sycl::handler& h) {
        auto acc = mapped_vec.get_access(h);
        h.single_task([=]() {
            if (acc.num_partitions() != 4) *test_passed = false;

            if (acc.partition_size(0) != 3) *test_passed = false;
            if (acc.partition_size(1) != 1) *test_passed = false;
            if (acc.partition_size(2) != 0) *test_passed = false;
            if (acc.partition_size(3) != 2) *test_passed = false;

            if (acc.partition_offset(0) != 0) *test_passed = false;
            if (acc.partition_offset(1) != 3) *test_passed = false;
            if (acc.partition_offset(2) != 4) *test_passed = false;
            if (acc.partition_offset(3) != 4) *test_passed = false;

            if (acc.partition_size(3) > 1) {
                auto p = acc(3, 1);
                if (p.second != 41) *test_passed = false;
            }
        });
    }).wait();

    std::cout << "Detailed Partition & Accessor Test: " << (*test_passed ? "PASSED" : "FAILED") << std::endl;

    std::cout << "--- Running Re-update Stability Test ---" << std::endl;
    
    host_input.clear();
    host_input.push_back({{0.6, 0.1}, 99}); // Cell 2
    input_data = psum::particle_container::device_vector<std::pair<Eigen::Vector2d, int>>(q, host_input);
    
    mapped_vec.update(input_data);
    q.wait();

    q.submit([&](sycl::handler& h) {
        auto acc = mapped_vec.get_access(h);
        h.single_task([=]() {
            if (acc.partition_size(2) != 1 || acc(2, 0).second != 99) *test_passed = false;
            if (acc.partition_size(0) != 0) *test_passed = false;
        });
    }).wait();

    std::cout << "Re-update Test: " << (*test_passed ? "PASSED" : "FAILED") << std::endl;
}

void test_grid_mapped_massive_random() {
    sycl::queue q{sycl::default_selector_v};
    const size_t num_particles = 2000000;
    psum::field::grid2D g({0.0, 0.0}, {1.0, 1.0}, {512, 512}); 
    grid_mapped_device_vector<Eigen::Vector2d, int, psum::field::grid2D> mapped_vec(q, g);

    psum::random::rander R;
    std::vector<std::pair<Eigen::Vector2d, int>> host_input;
    for(size_t i=0; i<num_particles; ++i) {
        host_input.push_back({{R(), R()}, (int)i});
    }
    psum::particle_container::device_vector<std::pair<Eigen::Vector2d, int>> input_data(q, host_input);

    std::cout << "--- Running Massive Random Test ---" << std::endl;
    mapped_vec.update(input_data);
    q.wait();

    size_t* total_count = sycl::malloc_shared<size_t>(1, q);
    *total_count = 0;
    q.submit([&](sycl::handler& h) {
        auto acc = mapped_vec.get_access(h);
        h.parallel_for(sycl::range<1>(acc.num_partitions()), [=](sycl::id<1> i) {
            sycl::atomic_ref<size_t, sycl::memory_order::relaxed, sycl::memory_scope::device> ref(*total_count);
            ref.fetch_add(acc.partition_size(i));
        });
    }).wait();

    std::cout << "Massive Random Test: " << (*total_count == num_particles ? "PASSED" : "FAILED") 
              << " (Counted: " << *total_count << ")" << std::endl;
    sycl::free(total_count, q);
}

void test_pairing_engine() {
    sycl::queue q{sycl::default_selector_v};
    std::cout << "Testing Pairing Engine on: " << q.get_device().get_info<sycl::info::device::name>() << std::endl;

    const size_t num_particles_a = 100000;
    const size_t num_particles_b = 1000;
    
    psum::field::grid2D g({0.0, 0.0}, {1.0, 1.0}, {100, 100});
    
    using QueryData = std::pair<particle*, particle*>;
    using ParticleGroup = psum::particle_container::particle_group<particle, psum::particle_container::pos_x_nan_is_invalid>;
    pairing_engine<ParticleGroup, psum::field::grid2D> engine(q, g);

    psum::random::rander R;
    std::vector<particle> host_a(num_particles_a);
    std::vector<particle> host_b(num_particles_b);

    for (size_t i = 0; i < num_particles_a; ++i) {
        get<position>(host_a[i]) = Eigen::Vector2d(R(), R());
        get<random_seed>(host_a[i]) = R.gen_seed();
        get<velocity>(host_a[i]).x() = 1.0f;
    }
    for (size_t i = 0; i < num_particles_b; ++i) {
        get<position>(host_b[i]) = Eigen::Vector2d(R(), R());
        get<random_seed>(host_b[i]) = R.gen_seed();
        get<velocity>(host_b[i]).x() = 2.0f;
    }

    ParticleGroup group_background(q);
    group_background.insert(host_a);

    psum::particle_container::device_vector<particle> device_incident(q, host_b);
    psum::particle_container::device_vector<std::pair<Eigen::Vector2d, QueryData>> pairing_buffer(q, num_particles_b);
    pairing_buffer.clear();
    
    // bind particle data to input_data
    device_incident.for_each([&](sycl::handler& h) {
        auto buf = pairing_buffer.get_access(h);
        return [=](particle& p) {
            buf.push_back({get<position>(p), {&p, nullptr}});
        };
    });

    std::cout << "Starting deal (Oversampled Atomic Pairing)..." << std::endl;
    
    // Action: swap v.x()
    auto swap_action = [=](QueryData pair_ptr) {
        auto p_a_ptr = pair_ptr.first;
        auto p_b_ptr = pair_ptr.second;
        if (p_a_ptr == nullptr || p_b_ptr == nullptr) return;
        double vx_a = get<velocity>(*p_a_ptr).x();
        double vx_b = get<velocity>(*p_b_ptr).x();
        get<velocity>(*p_a_ptr).x() = vx_b;
        get<velocity>(*p_b_ptr).x() = vx_a;
    };

    engine.deal(group_background, pairing_buffer);
    q.wait();

    pairing_buffer.for_each([&](sycl::handler& h) {
        return [=](auto& pair) {
            swap_action(pair.second);
        };
    });

    auto final_a = group_background.get_content().to_host();
    auto final_b = device_incident.to_host();

    size_t swap_count = 0;
    for(const auto& p : final_a) {
        if(get<velocity>(p).x() == 2.0f) swap_count++;
    }

    std::cout << "--- Pairing Engine Test Report ---" << std::endl;
    std::cout << "Successfully swapped particles: " << swap_count << std::endl;
    
    bool* success = psum::utils_sycl::shared_variable<bool>(q);
    *success = (swap_count > 0); 

    size_t b_changed_count = 0;
    for(const auto& p : final_b) {
        if(get<velocity>(p).x() == 1.0f) b_changed_count++;
    }
    
    if (swap_count != b_changed_count) {
        std::cout << "Logic Error: Swap count mismatch!" << std::endl;
        *success = false;
    }

    std::cout << "Test " << (*success ? "passed" : "failed") << std::endl;
}

int main() {
    test_grid_mapped_device_vector();
    test_grid_mapped_detailed();
    test_grid_mapped_massive_random();
    test_pairing_engine();
    return 0;
}