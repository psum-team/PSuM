#ifndef PSUM_FIELD_SOLVER_BOUNDARY_CREATOR_NEUMANN_HPP
#define PSUM_FIELD_SOLVER_BOUNDARY_CREATOR_NEUMANN_HPP

#include <functional>
#include <optional>
#include <stdexcept>
#include <memory>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include "../mixed_boundary.hpp"
#include "../../field/simple_grid.hpp"

namespace psum {

namespace field_solver {

namespace boundary_creator {

    using boundary_direction_1d = psum::field::boundary_direction_1d;
    using boundary_direction_2d = psum::field::boundary_direction_2d;
    using boundary_direction_3d = psum::field::boundary_direction_3d;

    mixed_boundary_1d Neumann_point(const field::grid1D &g, boundary_direction_1d dir)
    {
        mixed_boundary_1d ans;
        int grid_I = g.numCells<0>();
        std::unordered_map<size_t, std::array<double, 1>> related_elements;
        
        size_t idx;
        if (dir == boundary_direction_1d::L) {
            idx = 0;
        } else if (dir == boundary_direction_1d::R) {
            idx = grid_I;
        }
        auto pos = g.nodePosition(g.i2n(idx));
        related_elements[idx] = {pos.x()};
        
        ans.set_related_elements(related_elements);
        ans.set_coeff_func([](const std::array<double, 1>&) { return std::make_pair(0.0, 1.0); });
        ans.set_direction(dir);
        
        return ans;
    }

    mixed_boundary_2d Neumann_line(const field::grid2D &g, boundary_direction_2d dir, std::pair<double, double> min_max = {nan(""), nan("")})
    {
        mixed_boundary_2d ans;
        size_t start, end, step;
        int grid_I = g.numCells<0>();
        int grid_J = g.numCells<1>();
        std::unordered_map<size_t, std::array<double, 2>> related_elements;

        if (dir == boundary_direction_2d::S) {
            start = 0;
            end = (grid_I + 1) * (grid_J + 1);
            step = grid_J + 1;
        }
        if (dir == boundary_direction_2d::N)
        {
            start = grid_J;
            end = (grid_I + 1) * (grid_J + 1);
            step = grid_J + 1;
        }
        if (dir == boundary_direction_2d::W)
        {
            start = 0;
            end = grid_J + 1;
            step = 1;
        }
        if (dir == boundary_direction_2d::E)
        {
            start = grid_I * (grid_J + 1);
            end = (grid_I + 1) * (grid_J + 1);
            step = 1;
        }
        for (size_t j = start; j < end; j+=step)
        {
            auto pos = g.nodePosition(g.i2n(j));
            double pos_component;
            if (dir == boundary_direction_2d::W || dir == boundary_direction_2d::E) pos_component = pos.y();
            if (dir == boundary_direction_2d::S || dir == boundary_direction_2d::N) pos_component = pos.x();
            if(!((pos_component < min_max.first) || (pos_component > min_max.second))) // it is true when min_max={nan, nan}
            {
                related_elements[j] = {pos.x(), pos.y()};
            }
        }

        ans.set_related_elements(related_elements);
        ans.set_coeff_func([](const std::array<double, 2>&) { return std::make_pair(0, 1); });
        ans.set_direction(dir);

        return ans;
    }

    mixed_boundary_3d Neumann_plane(const field::grid3D &g, boundary_direction_3d dir, std::pair<double, double> min_max = {nan(""), nan("")})
    {
        mixed_boundary_3d ans;
        int grid_I = g.numCells<0>();
        int grid_J = g.numCells<1>();
        int grid_K = g.numCells<2>();
        std::unordered_map<size_t, std::array<double, 3>> related_elements;

        for (int i = 0; i <= grid_I; i++) {
            for (int j = 0; j <= grid_J; j++) {
                for (int k = 0; k <= grid_K; k++) {
                    bool is_boundary = false;
                    if (dir == boundary_direction_3d::Xneg && i == 0) is_boundary = true;
                    else if (dir == boundary_direction_3d::Xpos && i == grid_I) is_boundary = true;
                    else if (dir == boundary_direction_3d::Yneg && j == 0) is_boundary = true;
                    else if (dir == boundary_direction_3d::Ypos && j == grid_J) is_boundary = true;
                    else if (dir == boundary_direction_3d::Zneg && k == 0) is_boundary = true;
                    else if (dir == boundary_direction_3d::Zpos && k == grid_K) is_boundary = true;

                    if (is_boundary) {
                        auto pos = g.nodePosition({i, j, k});
                        double pos_component;
                        if (dir == boundary_direction_3d::Yneg || dir == boundary_direction_3d::Ypos) pos_component = pos.x();
                        else if (dir == boundary_direction_3d::Zneg || dir == boundary_direction_3d::Zpos) pos_component = pos.x();
                        else pos_component = pos.y();
                        if(!((pos_component < min_max.first) || (pos_component > min_max.second))) {
                            related_elements[g.n2i({i, j, k})] = {pos.x(), pos.y(), pos.z()};
                        }
                    }
                }
            }
        }

        ans.set_related_elements(related_elements);
        ans.set_coeff_func([](const std::array<double, 3>&) { return std::make_pair(0, 1); });
        ans.set_direction(dir);

        return ans;
    }

}

}

}

#endif