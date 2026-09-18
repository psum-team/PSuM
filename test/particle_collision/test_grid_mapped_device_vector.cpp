#include <iostream>
#include <iomanip>
#include <Eigen/Dense>
#include "../../include/psum/field.hpp"
#include "../../src/particle_collision/grid_mapped_device_vector.hpp"
#include "../../src/utils_sycl.hpp"
#include "../../include/psum/particle_container.hpp"
#include "../../src/random.hpp"

using namespace psum;
using namespace particle_collision;

struct exported_elem {
    size_t cell_id;
    int round;
    Eigen::Vector3d pos;
};

int main() {
    sycl::queue q{sycl::default_selector_v};
    std::cout << "Testing on: " << q.get_device().get_info<sycl::info::device::name>() << std::endl;

    psum::field::grid3D g({-1.0, -1.0, -1.0}, {1.0, 1.0, 1.0}, {64, 64, 64});
    grid_mapped_device_vector<Eigen::Vector3d, int, psum::field::grid3D> mapped_vec(q, g);

    const size_t num_queries = 10000;
    psum::random::rander R;

    std::vector<std::pair<Eigen::Vector3d, int>> host_input1;
    for (size_t i = 0; i < num_queries; ++i) {
        double x = R() * 2.0 - 1.0;
        double y = R() * 2.0 - 1.0;
        double z = R() * 2.0 - 1.0;
        host_input1.push_back({{x, y, z}, 1});
    }
    psum::particle_container::device_vector<std::pair<Eigen::Vector3d, int>> input_data1(q, host_input1);

    std::cout << "--- Running first update ---" << std::endl;
    mapped_vec.update(input_data1);
    q.wait();

    auto host_data_round1 = mapped_vec.content().to_host();
    std::vector<exported_elem> exported_round1;
    for (const auto& elem : host_data_round1) {
        size_t cell_id = g.c2i(g.nC(elem.first));
        exported_round1.push_back({cell_id, elem.second, elem.first});
    }

    std::vector<std::pair<Eigen::Vector3d, int>> host_input2;
    for (size_t i = 0; i < num_queries; ++i) {
        double x = R() * 2.0 - 1.0;
        double y = R() * 2.0 - 1.0;
        double z = R() * 2.0 - 1.0;
        host_input2.push_back({{x, y, z}, 2});
    }
    psum::particle_container::device_vector<std::pair<Eigen::Vector3d, int>> input_data2(q, host_input2);

    std::cout << "--- Running second update ---" << std::endl;
    mapped_vec.update(input_data2);
    q.wait();

    auto host_data_round2 = mapped_vec.content().to_host();
    std::vector<exported_elem> exported_round2;
    for (const auto& elem : host_data_round2) {
        size_t cell_id = g.c2i(g.nC(elem.first));
        exported_round2.push_back({cell_id, elem.second, elem.first});
    }

    bool test_passed = true;
    size_t count_round1_in_round2 = 0;
    size_t count_round2_in_round2 = 0;
    for (const auto& elem : exported_round2) {
        if (elem.round == 1) count_round1_in_round2++;
        else if (elem.round == 2) count_round2_in_round2++;
    }

    if (count_round1_in_round2 != 0) {
        std::cout << "Error: round 1 data should be cleared, but found " << count_round1_in_round2 << " elements" << std::endl;
        test_passed = false;
    }
    if (count_round2_in_round2 != num_queries) {
        std::cout << "Error: expected " << num_queries << " round 2 elements, found " << count_round2_in_round2 << std::endl;
        test_passed = false;
    }

    if (!test_passed) {
        size_t num_cells = 8 * 8 * 8;
        
        std::vector<size_t> host_partition_sizes_round1(num_cells, 0);
        for (const auto& elem : exported_round1) {
            host_partition_sizes_round1[elem.cell_id]++;
        }

        std::vector<size_t> host_expected_partition_sizes_round2(num_cells, 0);
        for (const auto& elem : exported_round2) {
            if (elem.round == 2) {
                host_expected_partition_sizes_round2[elem.cell_id]++;
            }
        }

        std::vector<size_t> device_partition_sizes(num_cells, 0);
        size_t* device_sizes_shared = sycl::malloc_shared<size_t>(num_cells, q);
        q.submit([&](sycl::handler& h) {
            auto acc = mapped_vec.get_access(h);
            h.parallel_for(sycl::range<1>(num_cells), [=](sycl::id<1> i) {
                device_sizes_shared[i] = acc.partition_size(i);
            });
        }).wait();
        std::copy(device_sizes_shared, device_sizes_shared + num_cells, device_partition_sizes.begin());
        sycl::free(device_sizes_shared, q);

        std::cout << "\n=== Partition Sizes Comparison ===" << std::endl;
        std::cout << std::setw(10) << "Cell_ID"
                  << std::setw(12) << "R1(Host)"
                  << std::setw(12) << "R2(Host)"
                  << std::setw(12) << "Device"
                  << std::setw(12) << "Diff"
                  << std::endl;
        std::cout << std::string(60, '-') << std::endl;

        size_t mismatch_count = 0;
        for (size_t i = 0; i < num_cells; ++i) {
            if (host_expected_partition_sizes_round2[i] != device_partition_sizes[i]) {
                mismatch_count++;
                if (mismatch_count <= 50) {
                    std::cout << std::setw(10) << i
                              << std::setw(12) << host_partition_sizes_round1[i]
                              << std::setw(12) << host_expected_partition_sizes_round2[i]
                              << std::setw(12) << device_partition_sizes[i]
                              << std::setw(12) << (int)(device_partition_sizes[i] - host_expected_partition_sizes_round2[i])
                              << std::endl;
                }
            }
        }
        if (mismatch_count > 50) {
            std::cout << "... and " << (mismatch_count - 50) << " more mismatches" << std::endl;
        }
        std::cout << "Total mismatches: " << mismatch_count << " / " << num_cells << std::endl;

        std::cout << "\n=== Comparison Table (Round 1 vs Round 2 after second update) ===" << std::endl;
        std::cout << std::setw(8) << "Idx"
                  << std::setw(12) << "R1_cell" << std::setw(8) << "R1_rnd" << std::setw(20) << "R1_position"
                  << " | "
                  << std::setw(12) << "R2_cell" << std::setw(8) << "R2_rnd" << std::setw(20) << "R2_position"
                  << std::endl;
        std::cout << std::string(100, '-') << std::endl;

        size_t max_rows = std::max(exported_round1.size(), exported_round2.size());
        for (size_t i = 0; i < max_rows; ++i) {
            std::cout << std::setw(8) << i;
            if (i < exported_round1.size()) {
                const auto& e1 = exported_round1[i];
                std::cout << std::setw(12) << e1.cell_id
                          << std::setw(8) << e1.round
                          << std::setw(20) << ("(" + std::to_string(e1.pos.x()).substr(0,5) + "," 
                                               + std::to_string(e1.pos.y()).substr(0,5) + ","
                                               + std::to_string(e1.pos.z()).substr(0,5) + ")");
            } else {
                std::cout << std::setw(12) << "-" << std::setw(8) << "-" << std::setw(20) << "-";
            }
            std::cout << " | ";
            if (i < exported_round2.size()) {
                const auto& e2 = exported_round2[i];
                std::cout << std::setw(12) << e2.cell_id
                          << std::setw(8) << e2.round
                          << std::setw(20) << ("(" + std::to_string(e2.pos.x()).substr(0,5) + "," 
                                               + std::to_string(e2.pos.y()).substr(0,5) + ","
                                               + std::to_string(e2.pos.z()).substr(0,5) + ")");
            } else {
                std::cout << std::setw(12) << "-" << std::setw(8) << "-" << std::setw(20) << "-";
            }
            std::cout << std::endl;
        }
        std::cout << "\nRound 1 count (before 2nd update): " << exported_round1.size() << std::endl;
        std::cout << "Round 2 count (after 2nd update): " << exported_round2.size() << std::endl;
        std::cout << "Round 1 elements remaining in round 2: " << count_round1_in_round2 << std::endl;
        std::cout << "Round 2 elements in round 2: " << count_round2_in_round2 << std::endl;
    }

    std::cout << "Test " << (test_passed ? "passed" : "failed") << std::endl;
    return test_passed ? 0 : 1;
}
