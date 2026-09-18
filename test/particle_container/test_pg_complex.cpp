#include <Eigen/Core>
#include <vector>
#include <iostream>
#include "../../src/random/rander.hpp"
#include <psum/particle_container.hpp>
#include <psum/tag.hpp>
#include "../../src/timer.hpp"

using namespace psum::particle_container;
using namespace psum::tag;
using property::position;
using property::velocity;
using psum::random::rander;

using Particle =
    tagged_struct<
        tag_bind<position, Eigen::RowVector3f>,
        tag_bind<velocity, Eigen::RowVector3f>>;

// Keep the same validator used in your example
using pContainer = particle_group<Particle, pos_x_nan_is_invalid>;

// Compare container with host reference (detailed)
bool verify_against_host(sycl::queue &q, pContainer &container, const std::vector<Particle> &host_ref) {
    auto content = container.get_content().to_host();
    auto invalid_pos = container.get_invalid_indexes().to_host();
    std::sort(invalid_pos.begin(), invalid_pos.end());
    std::vector<Particle> valid_content;

    std::vector<size_t> invalid_pos_computed;
    for (size_t i = 0; i < content.size(); ++i) {
        if (!pContainer::validator::is_valid(content[i])) {
            invalid_pos_computed.push_back(i);
        } else {
            valid_content.push_back(content[i]);
        }
    }

    if (invalid_pos_computed == invalid_pos)
        std::cout << "Valid: position of invalid particle == invalid_pos✅\n";
    else {
        std::cout << "Invalid: position of invalid particle != invalid_pos❌\n";
        return false;
    }

    std::vector<Particle> sorted_host_ref(host_ref);
    std::erase_if(sorted_host_ref, [=](const Particle& p) {
        return !pContainer::validator::is_valid(p);
    });
    auto particle_cmpr = [](const Particle &a, const Particle &b) {
        auto apos = get<position>(a);
        auto bpos = get<position>(b);
        auto avel = get<velocity>(a);
        auto bvel = get<velocity>(b);
        return std::tie(apos.x(), apos.y(), apos.z(), avel.x(), avel.y(), avel.z()) <
            std::tie(bpos.x(), bpos.y(), bpos.z(), bvel.x(), bvel.y(), bvel.z());
    };
    std::sort(sorted_host_ref.begin(), sorted_host_ref.end(), particle_cmpr);
    std::sort(valid_content.begin(), valid_content.end(), particle_cmpr);
    if (sorted_host_ref == valid_content)
        std::cout << "Valid: content == host_ref✅\n";
    else {
        std::cout << "Invalid: content != host_ref❌\n";
        return false;
    }
    return true;
}

Particle random_particle(rander &R) {
    Particle p;
    get<position>(p) = {R(), R(), R()};
    get<velocity>(p) = {R(), R(), R()};
    return p;
}

void random_erase_insert(sycl::queue & q, int N_0, int N_loop, double delete_ratio, bool verify, const std::string& test_prefix = "") {


    rander R;
    std::vector<Particle> initial_data;
    for (int i = 0; i < N_0; ++i) {
        initial_data.push_back(random_particle(R));
    }

    pContainer container(q);
    container.insert(initial_data);
    std::vector<Particle> host_container(initial_data);

    for (int loop = 0; loop < N_loop; ++loop) {
        double delete_interval_left = R();
        double delete_interval_right = delete_interval_left + delete_ratio;
        std::vector<Particle> new_data;
        int N_new = delete_ratio * N_0;
        for (int i = 0; i < N_new; ++i) {
            new_data.push_back(random_particle(R));
        }
        device_vector<Particle> new_data_device(q, new_data);
        // device operation: remove and insert new
        Tic(test_prefix + "on device(remove)")
        container.for_each([&](sycl::handler&) {
            return [=](Particle& p) {
                auto pos = get<position>(p);
                if (pos.x() > delete_interval_left && pos.x() < delete_interval_right) {
                    pContainer::validator::make_invalid(p);
                }
            };
        });
        TocTic(test_prefix + "on device(insert)")
        container.insert(new_data_device);
        // host operation: do the same on host_container
        TocTic(test_prefix + "on host")
        erase_if(host_container, [=](const Particle& p) {
            auto pos = get<position>(p);
            return pos.x() > delete_interval_left && pos.x() < delete_interval_right;
        });
        for (auto &p : new_data) {
            host_container.push_back(p);
        }
        Toc
        // verify
        if (verify) {
            bool ok = verify_against_host(q, container, host_container);
            if (ok) {
                std::cout << "Valid: loop " << loop << " passed✅\n";
                std::cout << "Capacity: " << container.capacity() << ", size: " << container.size() << "\n";
            }
        }
    }
}

void test_functional_basic(sycl::queue &q) {
    std::cout << "\n=== [Test] Functional Basic ===\n";
    rander R;
    std::vector<Particle> data(100);
    std::generate(data.begin(), data.end(), [&]{ return random_particle(R); });
    for (int i = 0; i < 10; ++i) {
        pos_x_nan_is_invalid<Particle>::make_invalid(data[int(R() * 100)]);
    }

    pContainer container(q);
    container.insert(data);
    auto host_ref = data;

    // empty insert
    container.insert(std::vector<Particle>{});
    if (container.size() != host_ref.size() - 10)
        std::cerr << "❌ Empty insert changed size!\n";
    else
        std::cout << "✅ Empty insert ok\n";

    // full invalidation
    container.for_each([&](sycl::handler&) {
        return [&](Particle &p) { pContainer::validator::make_invalid(p); };
    });
    if (container.size() != 0)
        std::cerr << "❌ container size not updated after invalidation!\n";
    else
        std::cout << "✅ Full invalidation ok\n";
    
    container.insert(std::vector<Particle>{}); // insert no-op

    if (container.size() != 0)
        std::cerr << "❌ container size not updated after invalidation and no-op insert!\n";
    else
        std::cout << "✅ Full invalidation and no-op insert ok\n";

    // reinsert same data
    container.insert(data);
    verify_against_host(q, container, data);
    std::cout << "✅ Basic insert/remove behavior ok\n";
}

void test_roundtrip_consistency(sycl::queue &q) {
    std::cout << "\n=== [Test] Round-trip Consistency ===\n";
    rander R;
    std::vector<Particle> data(1e5);
    std::generate(data.begin(), data.end(), [&]{ return random_particle(R); });

    pContainer container(q);
    container.insert(data);
    auto d2h1 = container.get_content().to_host();

    container.insert(std::vector<Particle>(data.begin(), data.begin() + 1000));
    auto d2h2 = container.get_content().to_host();

    if (d2h1 == d2h2)
        std::cerr << "❌ data unchanged after reinsert!\n";
    else
        std::cout << "✅ round-trip data properly updated\n";
}

void test_capacity_behavior(sycl::queue &q) {
    std::cout << "\n=== [Test] Capacity Behavior ===\n";
    rander R;
    std::vector<Particle> data(1e6);
    std::generate(data.begin(), data.end(), [&]{ return random_particle(R); });

    pContainer container(q);
    container.insert(data);
    std::cout << "Initial capacity: " << container.capacity() << "\n";

    for (int i = 0; i < 20; ++i) {
        container.for_each([&](sycl::handler&) {
            return [=](Particle &p) {
                if (get<position>(p).x() < 0.05f * i)
                    pContainer::validator::make_invalid(p);
            };
        });
        std::vector<Particle> new_data(6e5);
        std::generate(new_data.begin(), new_data.end(), [&]{ return random_particle(R); });
        container.insert(new_data);
        std::cout << "Step " << i
                  << " -> size: " << container.size()
                  << ", capacity: " << container.capacity()
                  << ", invalid_count: " << container.get_invalid_indexes().to_host().size()
                  << "\n";
    }
}

int main() {
    sycl::queue q{ sycl::default_selector_v };
    std::cout << "Running advanced particle_container stress test on device: "
              << q.get_device().get_info<sycl::info::device::name>() << "\n";

    test_functional_basic(q);
    test_roundtrip_consistency(q);
    test_capacity_behavior(q);
    std::cout << "=== [Stress Test] random erase and insert ===\n" << std::endl;
    random_erase_insert(q, 1e8, 5, 0.1, true, "stress dense ");
    random_erase_insert(q, 1e8, 5, 0.001, false, "stress sparse ");
    random_erase_insert(q, 1e7, 20, 0.1, true, "dense ");
    random_erase_insert(q, 1e7, 20, 1e-7, false, "very sparse ");
    random_erase_insert(q, 1e7, 20, 0.001, false, "sparse ");
    random_erase_insert(q, 1e6, 100, 0.1, false, "little dense ");
    random_erase_insert(q, 1e6, 100, 0.001, false, "little sparse ");

    PrintTimer
}
