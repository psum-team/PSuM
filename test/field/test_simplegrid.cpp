#include <psum/field.hpp>
#include "../../src/random/rander.hpp"
#include "../../src/random/rand_functions.hpp"

using namespace psum::field;
using namespace psum::random;

// There should be a Vector1D in Eigen, but in reality it doesn't.
// Here we define it manually.
using Vector1D = Eigen::Vector<double, 1>;

template<int Dim>
bool test_grid_set() {
    rander R;
    Eigen::Vector<double, Dim> lower;
    Eigen::Vector<double, Dim> upper;
    std::vector<int> size(Dim);
    for (int i = 0; i < Dim; i++) {
        lower[i] = R() - 0.5;
        upper[i] = exp(R() - 0.5) + lower[i];
        size[i] = int(100 * R() + 1);
    }
    simple_grid<Dim> g(lower, upper, size);

    auto approx = [](double x, double y) {
        return abs(x - y) < (abs(x) + abs(y)) * 1e-10;
    };

    bool check_result = true;

    for (int i = 0; i < Dim; i++) {
        if (g.get_cell_num()[i] != size[i]) {
            check_result = false;
            std::cout << "❌ Cell number is not equal to size in dimension " << i << std::endl;
        }
        
        if (!approx(g.get_lower_bounds()[i] + g.get_deltas()[i] * g.get_cell_num()[i], g.get_upper_bounds()[i])) {
            check_result = false;
            std::cout << "❌ Lower and delta are not right in dimension " << i << std::endl;
        }
        
        if (!approx(g.get_deltas()[i], 1.0 / g.get_deltas_r()[i])) {
            check_result = false;
            std::cout << "❌ Delta and 1/delta_r are not equal in dimension " << i << std::endl;
        }
        
        if (!approx( g.get_middles()[i] + g.get_spans()[i]/2, g.get_upper_bounds()[i]) ||
            !approx( g.get_middles()[i] - g.get_spans()[i]/2, g.get_lower_bounds()[i])) {
            check_result = false;
            std::cout << "❌ Middle and span are not right in dimension " << i << std::endl;
        }
    }

    size_t expected_num_cells = 1;
    size_t expected_num_nodes = 1;
    for (int i = 0; i < Dim; i++) {
        expected_num_cells *= size[i];
        expected_num_nodes *= size[i] + 1;
    }
    if (g.contentSize(var_loc::cellCentered) != expected_num_cells ||
        g.contentSize(var_loc::nodeCentered) != expected_num_nodes) {
        check_result = false;
        std::cout << "❌ Content size is not correct" << std::endl;
    }
    if (check_result)
        std::cout << "✅ Grid set is correct" << std::endl;
    return check_result;
}

template<int Dim>
bool test_grid_inGrid() {
    rander R;
    Eigen::Vector<double, Dim> lower;
    Eigen::Vector<double, Dim> upper;
    std::vector<int> size(Dim);
    for (int i = 0; i < Dim; i++) {
        lower[i] = R() - 0.5;
        upper[i] = exp(R() - 0.5) + lower[i];
        size[i] = int(100 * R() + 1);
    }
    simple_grid<Dim> g(lower, upper, size);
    
    bool check_result = true;
    for (int i = 0; i < 1000; i++) {
        Eigen::Vector<double, Dim> p;
        for (int j = 0; j < Dim; j++) {
            p[j] = g.get_spans()[j] * (R() - 0.5)+ g.get_middles()[j];
        }
        bool in_grid = g.inGrid(p);
        if (in_grid == false) {
            std::cout << "❌ in_Grid is not correct" << std::endl;
            check_result = false;
            break;
        }
    }
    if (check_result)
        std::cout << "✅ All points are in grid" << std::endl;
    return check_result;
}

template<int Dim>
bool test_grid_inGrid_and_nN_nC() {
    rander R;
    Eigen::Vector<double, Dim> lower;
    Eigen::Vector<double, Dim> upper;
    std::vector<int> size(Dim);
    for (int i = 0; i < Dim; i++) {
        lower[i] = R() - 0.5;
        upper[i] = exp(R() - 0.5) + lower[i];
        size[i] = int(100 * R() + 1);
    }
    simple_grid<Dim> g(lower, upper, size);

    bool check_result = true;
    for (int i = 0; i < 1000; i++) {
        Eigen::Vector<double, Dim> p;
        for (int j = 0; j < Dim; j++) {
            p[j] = g.get_spans()[j] * (R() - 0.5) * 1.2 + g.get_middles()[j];
        }
        bool in_grid = g.inGrid(p);

        if (in_grid) {
            bool in_grid_check_nN = true;
            auto node_index = g.nN(p);
            for (int j = 0; j < Dim; j++) {
                if (node_index.indices[j] < 0 || node_index.indices[j] > size[j])
                    in_grid_check_nN = false;
            }
            if (in_grid_check_nN != in_grid) {
                check_result = false;
                std::cout << "❌ inGrid() and nN() are not consistent" << std::endl;
                break;
            }

            bool in_grid_check_nC = true;
            auto cell_index = g.nC(p);
            for (int j = 0; j < Dim; j++) {
                if (cell_index.indices[j] < 0 || cell_index.indices[j] >= size[j])
                    in_grid_check_nC = false;
            }
            if (in_grid_check_nC != in_grid) {
                check_result = false;
                std::cout << "❌ inGrid() and nC() are not consistent" << std::endl;
                break;
            }
        }
    }
    if (check_result)
        std::cout << "✅ inGrid() and nN/nC are consistent" << std::endl;
    return check_result;
}

template<int Dim>
bool test_grid_nearest() {
    rander R;
    Eigen::Vector<double, Dim> lower;
    Eigen::Vector<double, Dim> upper;
    std::vector<int> size(Dim);
    for (int i = 0; i < Dim; i++) {
        lower[i] = R() - 0.5;
        upper[i] = exp(R() - 0.5) + lower[i];
        size[i] = int(100 * R() + 1);
    }
    simple_grid<Dim> g(lower, upper, size);

    bool check_result = true;
    // avoid too much computation
    for (int i = 0; i < 1000/(g.contentSize(var_loc::cellCentered)/10000.0); i++) {
        Eigen::RowVector<double, Dim> p;
        for (int j = 0; j < Dim; j++) {
            p[j] = g.get_spans()[j] * (R() - 0.5)+ g.get_middles()[j];
        }
        // check nearest cell
        auto p_nc = g.cellCenter(g.nC(p));
        double dist = (p - p_nc).norm();
        bool is_nearest_cell = true;
        for (int j = 0; j < g.contentSize(var_loc::cellCentered); j++) {
            double dist_to_cell = (p - g.cellCenter(g.i2c(j))).norm();
            if (dist_to_cell < dist) {
                is_nearest_cell = false;
                break;
            }
        }
        if (is_nearest_cell == false) {
            check_result = false;
            break;
        }
        // check nearest node
        auto p_nn = g.nodePosition(g.nN(p));
        dist = (p - p_nn).norm();
        bool is_nearest_node = true;
        for (int j = 0; j < g.contentSize(var_loc::nodeCentered); j++) {
            double dist_to_node = (p - g.nodePosition(g.i2n(j))).norm();
            if (dist_to_node < dist) {
                is_nearest_node = false;
                break;
            }
        }
        if (is_nearest_node == false) {
            check_result = false;
            break;
        }
    }
    if (check_result)
        std::cout << "✅ Nearest cell/node is correct" << std::endl;
    else
        std::cout << "❌ Nearest cell/node is not correct" << std::endl;
    return check_result;
}

int main() {
    int failures = 0;
    auto expect_throw = [](const char* what, auto&& f) {
        try {
            f();
            std::cout << "❌ " << what << " but no exception caught" << std::endl;
            return false;
        } catch (std::exception& e) {
            std::cout << "✅ " << what << " caught exception." << std::endl;
            return true;
        }
    };
    failures += expect_throw("construction with wrong upper_bound size", [] {
        grid3D g(std::vector<double>{0, 0, 0}, std::vector<double>{0, 1}, {1, 1, 1});
    }) ? 0 : 1;
    failures += expect_throw("construction with wrong lower_bound size", [] {
        grid2D g(std::vector<double>{0, 0, 0}, std::vector<double>{0, 1}, {1, 1, 1});
    }) ? 0 : 1;
    failures += expect_throw("construction with wrong num_cells", [] {
        grid3D g(std::vector<double>{0, 0, 0}, std::vector<double>{0, 1, 2}, {1, 1, 0});
    }) ? 0 : 1;
    failures += expect_throw("construction with wrong num_cells size", [] {
        grid3D g(std::vector<double>{0, 0, 0}, std::vector<double>{0, 1, 2}, {1, 1});
    }) ? 0 : 1;

    for (int i = 0; i < 10; i++) {
        failures += test_grid_set<1>() ? 0 : 1;
        failures += test_grid_set<2>() ? 0 : 1;
        failures += test_grid_set<3>() ? 0 : 1;
        failures += test_grid_set<4>() ? 0 : 1;
    }
    for (int i = 0; i < 10; i++) {
        failures += test_grid_inGrid<1>() ? 0 : 1;
        failures += test_grid_inGrid<2>() ? 0 : 1;
        failures += test_grid_inGrid<3>() ? 0 : 1;
        failures += test_grid_inGrid<4>() ? 0 : 1;
    }
    for (int i = 0; i < 10; i++) {
        failures += test_grid_inGrid_and_nN_nC<1>() ? 0 : 1;
        failures += test_grid_inGrid_and_nN_nC<2>() ? 0 : 1;
        failures += test_grid_inGrid_and_nN_nC<3>() ? 0 : 1;
        failures += test_grid_inGrid_and_nN_nC<4>() ? 0 : 1;
    }
    for (int i = 0; i < 10; i++) {
        failures += test_grid_nearest<1>() ? 0 : 1;
        failures += test_grid_nearest<2>() ? 0 : 1;
        failures += test_grid_nearest<3>() ? 0 : 1;
        failures += test_grid_nearest<4>() ? 0 : 1;
    }

    return failures ? 1 : 0;
}