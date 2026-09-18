#include <iostream>
#include <sycl/sycl.hpp>
#include <Eigen/Core>
#include <psum/psum.hpp>

using namespace std;
using namespace psum::prelude;

using Particle = tagged_struct<
    tag_bind<property::position, Eigen::RowVector2d>,
    tag_bind<property::velocity, Eigen::RowVector2d>,
    tag_bind<property::charge, double>,
    tag_bind<property::mass, double>
>;

int main() {
    grid2D grid({-1.0, -1.0}, {1.0, 1.0}, {256, 256});

    sycl::queue q{sycl::default_selector_v};
    cout << "device: " << q.get_device().get_info<sycl::info::device::name>() << endl;

    node_field2D<double> phi(q, grid);
    node_field2D<double> rho(q, grid);

    cout << "Big Bang step 1: grid and fields ready." << endl;

    return 0;
}