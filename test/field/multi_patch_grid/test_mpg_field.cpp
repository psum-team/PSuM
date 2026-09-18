#include <psum/field.hpp>
#include <iostream>
#include <cassert>
#include <cmath>
#include <fstream>
#include <string>

using namespace psum::field;
using namespace multi_patch;

sycl::queue& get_queue() {
    static sycl::queue q;
    return q;
}

void test_device_field() {
    std::cout << "--- Test: Device field set and read ---" << std::endl;

    std::cout << "  Device: " << get_queue().get_device().get_info<sycl::info::device::name>() << std::endl;

    device_multi_patch_field<2, var_loc::cellCentered, double> field(
        multi_patch_grid<2>(
            get_queue(),
            {0.0, 0.0},
            {2.0, 2.0},
            {2, 2},
            std::vector<int>{2, 3, 4, 5}
        )
    );

    assert(field.size() == 1360);

    field.setConstant(1.0);
    auto host_data = field.getContent().to_host();
    for (size_t i = 0; i < host_data.size(); i++) {
        assert(std::abs(host_data[i] - 1.0) < 1e-10);
    }

    field.setConstant(3.14);
    host_data = field.getContent().to_host();
    for (size_t i = 0; i < host_data.size(); i++) {
        assert(std::abs(host_data[i] - 3.14) < 1e-10);
    }

    std::cout << "✅ Device field set and read passed." << std::endl;
}

void test_plot() {
    std::cout << "--- Test: Plot output ---" << std::endl;

    device_multi_patch_field<2, var_loc::cellCentered, double> field(
        multi_patch_grid<2>(
            get_queue(),
            {0.0, 0.0},
            {2.0, 2.0},
            {2, 2},
            std::vector<int>{2, 3, 2, 3}
        )
    );

    field.setConstant(42.0);
    field.plot("test_mpg_output.plt", "density", 1.0);

    std::ifstream fp("test_mpg_output.plt");
    assert(fp.is_open());

    std::string line;
    bool found_header = false;
    int zone_count = 0;
    while (std::getline(fp, line)) {
        if (line.find("variables=") != std::string::npos) {
            assert(line.find("x") != std::string::npos);
            assert(line.find("y") != std::string::npos);
            assert(line.find("density") != std::string::npos);
            found_header = true;
        }
        if (line.find("zone ") != std::string::npos && line.find("F=block") != std::string::npos) {
            zone_count++;
        }
    }
    assert(found_header);
    assert(zone_count == 4);

    fp.close();
    std::cout << "✅ Plot output passed." << std::endl;
}

int main() {
    test_device_field();
    test_plot();
    return 0;
}
