#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <psum/psum.hpp>

using namespace std;
using namespace psum::prelude;
using namespace psum::field_solver::boundary_creator;

host_node_field2D<double> read_node_field_plt(const string& filename) {
    ifstream in(filename);

    string line;
    string zone;
    getline(in, line);
    getline(in, zone);

    vector<double> values;
    double xmin = 1e300;
    double xmax = -1e300;
    double ymin = 1e300;
    double ymax = -1e300;
    double x = 0.0;
    double y = 0.0;
    double v = 0.0;
    while (getline(in, line)) {
        istringstream row(line);
        if (!(row >> x >> y >> v)) {
            continue;
        }
        xmin = fmin(xmin, x);
        xmax = fmax(xmax, x);
        ymin = fmin(ymin, y);
        ymax = fmax(ymax, y);
        values.push_back(v);
    }

    int ni = stoi(zone.substr(zone.find('=', zone.find('i')) + 1));
    int nj = stoi(zone.substr(zone.find('=', zone.find('j')) + 1));

    grid2D grid({xmin, ymin}, {xmax, ymax}, {nj - 1, ni - 1});
    return host_node_field2D<double>(std::move(values), grid);
}

int main() {
    const string bitmap_file = "../../docs/psum_tour/psum_bitmap.plt";
    auto mask = read_node_field_plt(bitmap_file);
    mask.plot("output/mask.plt", "mask", 0.0);

    const auto& mask_grid = mask.getGrid();
    grid2D grid(
        {mask_grid.lowerBound<0>(), mask_grid.lowerBound<1>()},
        {mask_grid.upperBound<0>(), mask_grid.upperBound<1>()},
        {200, 200}
    );

    double t = 0.0;
    auto voltage = [&](double time, double x, double y) {
        double high_freq = sin(3000.0 * (x + y) + 100.0 * time) + 1;
        double low_freq = cos(30.0 * (x - y) - 10.0 * time) + 1;
        return (high_freq * 0.2 + low_freq) * cos(60 * y - 3);
    };

    Poisson_solver_2d solver;
    solver.init(
        grid,
        Poisson_solver_2d::Cartesian,
        {
            Dirichlet_line(grid, boundary_direction_2d::N) = 0.0,
            Dirichlet_line(grid, boundary_direction_2d::S) = 0.0,
            Dirichlet_line(grid, boundary_direction_2d::E) = 0.0,
            Dirichlet_line(grid, boundary_direction_2d::W) = 0.0,
            Dirichlet_func(grid, [&](double x, double y) {
                return interp(host_node_field2D<double>::Position{x, y}, mask) > 0.01;
            }) = [&](const auto& pos) {
                return voltage(t, pos[0], pos[1]);
            }
        }
    );

    host_node_field2D<double> phi(grid);
    host_node_field2D<double> source(grid);

    int steps = 80;
    double dt = 0.02;
    for (int step = 0; step <= steps; ++step) {
        t = step * dt;
        source.setZero();
        solver.solve(phi.data(), source.data());
        phi.plot("output/phi_step_" + to_string(step) + ".plt", "phi", t);
        cout << "step = " << step << ", t = " << t << endl;
    }

    cout << "Neon Light stage 4: wrote output/phi_step_*.plt" << endl;
    return 0;
}
