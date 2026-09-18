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
            }) = 1.5
        }
    );

    host_node_field2D<double> phi(grid);
    host_node_field2D<double> source(grid);
    source.setZero();

    solver.solve(phi.data(), source.data());
    phi.plot("output/phi.plt", "phi", 0.0);
    cout << "Neon Light stage 3: wrote output/phi.plt" << endl;
    return 0;
}
