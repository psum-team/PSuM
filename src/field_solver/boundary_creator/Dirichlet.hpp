#ifndef PSUM_FIELD_SOLVER_BOUNDARY_CREATOR_DIRICHLET_HPP
#define PSUM_FIELD_SOLVER_BOUNDARY_CREATOR_DIRICHLET_HPP

#include <functional>
#include <optional>
#include <stdexcept>
#include <memory>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include "../fixed_boundary.hpp"
#include "../../field/simple_grid.hpp"

namespace psum {

namespace field_solver {

namespace boundary_creator {

    using boundary_direction_1d = psum::field::boundary_direction_1d;
    using boundary_direction_2d = psum::field::boundary_direction_2d;
    using boundary_direction_3d = psum::field::boundary_direction_3d;

    fixed_boundary_1d Dirichlet_point(const field::grid1D &g, boundary_direction_1d dir)
    {
        fixed_boundary_1d ans;
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
        return ans;
    }

    fixed_boundary_1d Dirichlet_range(const field::grid1D& g, double x0, double x1, double geo_tolerance = 1e-10)
    {
        fixed_boundary_1d ans;
        std::unordered_map<size_t, std::array<double, 1>> related_elements;
        double tol = geo_tolerance * g.span<0>();
        for (int i = 0; i <= g.numCells<0>(); i++) {
            auto pos = g.nodePosition({i});
            if (pos.x() >= x0-tol && pos.x() <= x1+tol) {
                related_elements[g.n2i({i})] = {pos.x()};
            }
        }
        ans.set_related_elements(related_elements);
        return ans;
    }

    fixed_boundary_2d Dirichlet_line(const field::grid2D &g, boundary_direction_2d dir, std::pair<double, double> min_max = {nan(""), nan("")})
    {
        fixed_boundary_2d ans;
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
        return ans;
    }

    fixed_boundary_2d Dirichlet_box(const field::grid2D& g, double x0, double y0, double x1, double y1, double geo_tolerance = 1e-10)
    {
        fixed_boundary_2d ans;
        std::unordered_map<size_t, std::array<double, 2>> related_elements;
        double tol = geo_tolerance * sqrt(g.span<0>() * g.span<1>());
        for (int i = 0; i <= g.numCells<0>(); i++) {
            for (int j = 0; j <= g.numCells<1>(); j++) {
                auto pos = g.nodePosition({i, j});
                if (pos.x() >= x0-tol && pos.x() <= x1+tol && pos.y() >= y0-tol && pos.y() <= y1+tol) {
                    related_elements[g.n2i({i, j})] = {pos.x(), pos.y()};
                }
            }
        }
        ans.set_related_elements(related_elements);
        return ans;
    }

    fixed_boundary_2d Dirichlet_func(const field::grid2D& g, const std::function<bool(double x, double y)>& func)
    {
        fixed_boundary_2d ans;
        std::unordered_map<size_t, std::array<double, 2>> related_elements;
        for (int i = 0; i <= g.numCells<0>(); i++) {
            for (int j = 0; j <= g.numCells<1>(); j++) {
                auto pos = g.nodePosition({i, j});
                if (func(pos.x(), pos.y())) {
                    related_elements[g.n2i({i, j})] = {pos.x(), pos.y()};
                }
            }
        }
        ans.set_related_elements(related_elements);
        return ans;
    }

    fixed_boundary_3d Dirichlet_plane(const field::grid3D &g, boundary_direction_3d dir, 
        std::function<bool(double, double)> shape = std::function<bool(double, double)>())
    {
        fixed_boundary_3d ans;
        int grid_I = g.numCells<0>();
        int grid_J = g.numCells<1>();
        int grid_K = g.numCells<2>();
        std::unordered_map<size_t, std::array<double, 3>> related_elements;
        int start, end, step;
        int i_begin = 0;
        int j_begin = 0;
        int k_begin = 0;
        int i_end = grid_I;
        int j_end = grid_J;
        int k_end = grid_K;

        if (dir == boundary_direction_3d::Xpos) i_begin = i_end = grid_I;
        if (dir == boundary_direction_3d::Ypos) j_begin = j_end = grid_J;
        if (dir == boundary_direction_3d::Zpos) k_begin = k_end = grid_K;

        if (dir == boundary_direction_3d::Xneg) i_begin = i_end = 0;
        if (dir == boundary_direction_3d::Yneg) j_begin = j_end = 0;
        if (dir == boundary_direction_3d::Zneg) k_begin = k_end = 0;

        for (int i = i_begin; i <= i_end; i++) {
            for (int j = j_begin; j <= j_end; j++) {
                for (int k = k_begin; k <= k_end; k++) {
                    auto pos = g.nodePosition({i, j, k});
                    bool add_it = true;
                    if (shape) {
                        if (dir == boundary_direction_3d::Xpos || dir == boundary_direction_3d::Xneg)
                            add_it = shape(pos.y(), pos.z());
                        if (dir == boundary_direction_3d::Ypos || dir == boundary_direction_3d::Yneg)
                            add_it = shape(pos.x(), pos.z());
                        if (dir == boundary_direction_3d::Zpos || dir == boundary_direction_3d::Zneg)
                            add_it = shape(pos.x(), pos.y());
                    }
                    if(add_it) {
                        related_elements[g.n2i({i, j, k})] = {pos.x(), pos.y(), pos.z()};
                    }
                }
            }
        }
        ans.set_related_elements(related_elements);
        return ans;
    }

    fixed_boundary_3d Dirichlet_box(const field::grid3D& g, double x0, double y0, double z0, double x1, double y1, double z1, double geo_tolerance = 1e-10)
    {
        fixed_boundary_3d ans;
        std::unordered_map<size_t, std::array<double, 3>> related_elements;
        double tol = geo_tolerance * cbrt(g.span<0>() * g.span<1>() * g.span<2>());
        for (int i = 0; i <= g.numCells<0>(); i++) {
            for (int j = 0; j <= g.numCells<1>(); j++) {
                for (int k = 0; k <= g.numCells<2>(); k++) {
                    auto pos = g.nodePosition({i, j, k});
                    if (pos.x() >= x0-tol && pos.x() <= x1+tol && pos.y() >= y0-tol && pos.y() <= y1+tol && pos.z() >= z0-tol && pos.z() <= z1+tol) {
                        related_elements[g.n2i({i, j, k})] = {pos.x(), pos.y(), pos.z()};
                    }
                }
            }
        }
        ans.set_related_elements(related_elements);
        return ans;
    }


    fixed_boundary_3d Dirichlet_box_rz(const field::grid3D& g, double z0, double r0, double z1, double r1, double geo_tolerance = 1e-10)
    {
        fixed_boundary_3d ans;
        std::unordered_map<size_t, std::array<double, 3>> related_elements;
        double tol = geo_tolerance * pow(g.span<0>() * g.span<1>() * g.span<2>(), 1.0 / 3);
        for (int i = 0; i <= g.numCells<0>(); i++) {
            for (int j = 0; j <= g.numCells<1>(); j++) {
                for (int k = 0; k <= g.numCells<2>(); k++) {
                    auto pos = g.nodePosition({i, j, k});
                    double z = pos.z();
                    double r = sqrt(pos.x() * pos.x() + pos.y() * pos.y());
                    if (z >= z0 - tol && z <= z1 + tol && r >= r0 - tol && r <= r1 + tol) {
                        related_elements[g.n2i({i, j, k})] = {pos.x(), pos.y(), pos.z()};
                    }
                }
            }
        }
        ans.set_related_elements(related_elements);
        return ans;
    }


    fixed_boundary_3d Dirichlet_func(const field::grid3D& g, const std::function<bool(double x, double y, double z)>& func)
    {
        fixed_boundary_3d ans;
        std::unordered_map<size_t, std::array<double, 3>> related_elements;
        for (int i = 0; i <= g.numCells<0>(); i++) {
            for (int j = 0; j <= g.numCells<1>(); j++) {
                for (int k = 0; k <= g.numCells<2>(); k++) {
                    auto pos = g.nodePosition({i, j, k});
                    if (func(pos.x(), pos.y(), pos.z())) {
                        related_elements[g.n2i({i, j, k})] = {pos.x(), pos.y(), pos.z()};
                    }
                }
            }
        }
        ans.set_related_elements(related_elements);
        return ans;
    }

}

}

}

#endif