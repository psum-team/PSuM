#include <psum/field.hpp>
#include <iostream>
#include <string>
#include <cmath>
#include <vector>

using namespace psum;
using namespace psum::field;

template <int Order, typename Field, typename Pos>
void run_trajectory(const grid2D& grid, const std::vector<Pos>& traj, double w,
                    const std::string& tag) {
    char buf[64];
    for (size_t i = 0; i < traj.size(); i++) {
        Field f(grid);   // clean field for every state of the trajectory
        add_back_B_spline<Order>(traj[i], w, f);
        std::snprintf(buf, sizeof(buf), "%05zu", i);
        f.plot("traj_" + tag + "_f" + buf + ".plt");
    }
}

int main() {
    constexpr int N = 32;
    grid2D grid({0.0, 0.0}, {1.0, 1.0}, {N, N});
    using F = host_node_field2D<double>;
    using Pos = Eigen::RowVector2d;

    // short cycloid arc, dense enough that successive frames stay well below 1 delta
    const double a = 0.15;
    const int n = 64;
    const double t0 = 2.0, t1 = 3.5;
    std::vector<Pos> traj;
    for (int i = 0; i <= n; i++) {
        double t = t0 + (t1 - t0) * i / n;
        traj.push_back(Pos(0.1 + a * (t - std::sin(t)), 0.05 + a * (1.0 - std::cos(t))));
    }

    const double w = 1.0;

    run_trajectory<1, F, Pos>(grid, traj, w, "o1");
    run_trajectory<2, F, Pos>(grid, traj, w, "o2");
    run_trajectory<3, F, Pos>(grid, traj, w, "o3");
    run_trajectory<4, F, Pos>(grid, traj, w, "o4");

    std::cout << "test_addback_trajectory: " << traj.size() << " frames per order written" << std::endl;
    return 0;
}
