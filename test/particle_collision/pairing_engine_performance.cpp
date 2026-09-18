#include <iostream>
#include <map>
#include <string>
#include <iomanip>
#include <Eigen/Dense>
#include "../../include/psum/field.hpp"
#include "../../src/particle_collision/pairing_engine.hpp"
#include "../../src/utils_sycl.hpp"
#include "../../src/random.hpp"
#include "../../include/psum/tag.hpp"
#include "../../include/psum/particle_container.hpp"
#include "../../src/timer.hpp"

using namespace psum;
using namespace tag;
using namespace tag::property;
using namespace particle_collision;

using particle = tagged_struct<
    tag_bind<position, Eigen::Vector2d>,
    tag_bind<velocity, Eigen::Vector2d>,
    tag_bind<random_seed, uint32_t>
>;

void run_performance_test(size_t group_size, size_t query_size, int grid_x, int grid_y, int test_id) {
    sycl::queue q{sycl::default_selector_v};

    // Generate fixed group particles
    psum::random::rander R;
    std::vector<particle> host_a(group_size);
    for (size_t i = 0; i < group_size; ++i) {
        get<position>(host_a[i]) = Eigen::Vector2d(R(), R());
        get<random_seed>(host_a[i]) = R.gen_seed();
        get<velocity>(host_a[i]).x() = 1.0f;
    }

    // Create fixed group device data structure
    using ParticleGroup = psum::particle_container::particle_group<particle, psum::particle_container::pos_x_nan_is_invalid>;
    ParticleGroup group_a(q);
    group_a.insert(host_a);

    // Create grid and pairing engine
    psum::field::grid2D g({0.0, 0.0}, {1.0, 1.0}, {grid_x, grid_y});
    pairing_engine<ParticleGroup, psum::field::grid2D> engine(q, g);

    // Generate query particles with completely random positions
    std::vector<particle> host_b(query_size);
    psum::random::rander R_round;
    
    for (size_t i = 0; i < query_size; ++i) {
        double x = R_round();
        double y = R_round();
        get<position>(host_b[i]) = Eigen::Vector2d(x, y);
        get<random_seed>(host_b[i]) = R_round.gen_seed();
        get<velocity>(host_b[i]).x() = 2.0f;
    }

    // Create device data structures for query
    psum::particle_container::device_vector<particle> device_b(q, host_b);
    psum::particle_container::device_vector<std::pair<Eigen::Vector2d, std::pair<particle*, particle*>>> input_data(q, query_size);
    input_data.clear();

    // Bind particle data to input_data
    device_b.for_each([&](sycl::handler& h) {
        auto input_acc = input_data.get_access(h);
        return [=](particle& p) {
            input_acc.push_back({get<position>(p), {&p, nullptr}});
        };
    });

    // Time the pairing operation
    Tic("pairing");
    engine.deal(group_a, input_data);
    
    // Additional synchronization
    q.submit([&](sycl::handler& h) {
        h.single_task([=]() {
            // Empty task to ensure all previous operations complete
        });
    }).wait();

    input_data.for_each([&](sycl::handler& h) {
        return [=](auto& query) {
            auto p_a_ptr = query.second.first;
            auto p_b_ptr = query.second.second;
            if (p_a_ptr == nullptr || p_b_ptr == nullptr) return;
            double vx_a = get<velocity>(*p_a_ptr).x();
            double vx_b = get<velocity>(*p_b_ptr).x();
            get<velocity>(*p_a_ptr).x() = vx_b;
            get<velocity>(*p_b_ptr).x() = vx_a;
        };
    });
    
    double pairing_time = Toc("pairing");

    // Calculate throughput in two different metrics
    double throughput_mps = (group_size / 1000000.0) / (pairing_time);      // Million particles per second
    double throughput_mqs = (query_size / 1000000.0) / (pairing_time);     // Million queries per second
    double query_ratio = static_cast<double>(query_size) / group_size;

    // Validation check and calculate match rate
    auto final_a = group_a.get_content().to_host();
    auto final_b = device_b.to_host();

    size_t swap_count = 0;
    for(const auto& p : final_a) {
        if(get<velocity>(p).x() == 2.0f) swap_count++;
    }

    size_t b_changed_count = 0;
    for(const auto& p : final_b) {
        if(get<velocity>(p).x() == 1.0f) b_changed_count++;
    }

    double match_rate = static_cast<double>(swap_count) / query_size;

    // Print results in table format with all metrics including match rate
    std::cout << test_id << "\t" << group_size/1000 << "K\t" << query_size/1000 << "K\t" 
              << query_ratio << "\t" << grid_x << "x" << grid_y << "\t" 
              << std::fixed << std::setprecision(6) << pairing_time << "\t" 
              << throughput_mps << "\t" << throughput_mqs << "\t"
              << std::setprecision(4) << match_rate << std::endl;

    if (swap_count != b_changed_count || swap_count == 0) {
        std::cout << "WARNING: Validation failed for test " << test_id << std::endl;
    }
}

void comprehensive_performance_test() {
    sycl::queue q{sycl::default_selector_v};
    std::cout << "=== Comprehensive Pairing Engine Performance Test ===" << std::endl;
    std::cout << "Testing on: " << q.get_device().get_info<sycl::info::device::name>() << std::endl;

    // Test configurations
    std::vector<size_t> group_sizes = {300000, 1000000, 3000000, 10000000, 30000000};
    std::vector<double> query_ratios = {0.01, 0.02, 0.04, 0.06, 0.1, 0.15};
    std::vector<int> grid_sizes = {128, 192, 256, 384, 512};
    
    int test_id = 1;

    std::cout << "\nPerformance Test Results:" << std::endl;
    std::cout << "ID\tGroup(K)\tQuery(K)\tRatio\tGrid\t\tTime(s)\t\tMp/s\t\tMq/s\t\tMatchRate" << std::endl;
    std::cout << "--\t-------\t\t-------\t\t-----\t\t----\t\t--------\t\t--------\t\t---------" << std::endl;

    TimerSingleton::getInstance().globalTimer.stop_all();

    // Run tests for different group sizes and query ratios with fixed grid
    int fixed_grid = 256;
    for (size_t group_size : group_sizes) {
        for (double ratio : query_ratios) {
            size_t query_size = static_cast<size_t>(group_size * ratio);
            
            // Skip if query size is too small or too large
            if (query_size < 1000 || query_size > 1000000) continue;
            
            run_performance_test(group_size, query_size, fixed_grid, fixed_grid, test_id++);
        }
    }

    // Run tests for different grid sizes with fixed configuration
    size_t fixed_group = 1000000;
    double fixed_ratio = 0.1;
    size_t fixed_query = static_cast<size_t>(fixed_group * fixed_ratio);
    
    std::cout << "\nGrid Size Sensitivity Test (Group=" << fixed_group/1000 << "K, Query=" << fixed_query/1000 << "K):" << std::endl;
    std::cout << "ID\tGroup(K)\tQuery(K)\tRatio\tGrid\t\tTime(s)\t\tMp/s\t\tMq/s\t\tMatchRate" << std::endl;
    std::cout << "--\t-------\t\t-------\t\t-----\t\t----\t\t--------\t\t--------\t\t---------" << std::endl;
    
    for (int grid_size : grid_sizes) {
        run_performance_test(fixed_group, fixed_query, grid_size, grid_size, test_id++);
    }

    std::cout << "\n=== Comprehensive Performance Test Complete ===" << std::endl;
    
    std::cout << "\nDetailed timing breakdown:" << std::endl;
    TimerSingleton::getInstance().globalTimer.print2screen(psum::timer::print_mode::SumMode, 10);
}

int main() {
    comprehensive_performance_test();
    return 0;
}