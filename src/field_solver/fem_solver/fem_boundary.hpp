#ifndef PSUM_FIELD_SOLVER_FEM_FEM_BOUNDARY_HPP
#define PSUM_FIELD_SOLVER_FEM_FEM_BOUNDARY_HPP

#include <functional>
#include <optional>
#include <stdexcept>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include "tri_mesh.hpp"
#include "../fixed_boundary.hpp"

namespace psum {

namespace field_solver {

namespace fem_solver {

struct fem_neumann_boundary {
private:
    std::function<double(double, double)> value_func_;
    std::optional<double> value_const_;
    std::unordered_map<size_t, std::array<double, 2>> related_nodes_;

public:
    fem_neumann_boundary() = default;

    inline void set_related_nodes(const std::unordered_map<size_t, std::array<double, 2>>& nodes) {
        related_nodes_ = nodes;
    }

    inline fem_neumann_boundary& operator=(double val) {
        value_const_ = val;
        value_func_ = nullptr;
        return *this;
    }

    inline fem_neumann_boundary& operator=(const std::function<double(double, double)>& func) {
        value_func_ = func;
        value_const_ = std::nullopt;
        return *this;
    }

    inline double evaluate(double x, double y) const {
        if (value_func_) return value_func_(x, y);
        if (value_const_) return *value_const_;
        throw std::runtime_error("fem_neumann_boundary: no value set");
    }

    inline const auto& get_nodes() const { return related_nodes_; }
    inline bool has_node(size_t idx) const { return related_nodes_.count(idx) > 0; }
};

struct fem_robin_boundary {
private:
    std::function<std::pair<double, double>(double, double)> coeff_func_;
    std::function<double(double, double)> value_func_;
    std::optional<double> value_const_;
    std::unordered_map<size_t, std::array<double, 2>> related_nodes_;

public:
    fem_robin_boundary() = default;

    inline void set_related_nodes(const std::unordered_map<size_t, std::array<double, 2>>& nodes) {
        related_nodes_ = nodes;
    }

    inline void set_coeff_function(const std::function<std::pair<double, double>(double, double)>& func) {
        coeff_func_ = func;
    }

    inline fem_robin_boundary& operator=(double val) {
        value_const_ = val;
        value_func_ = nullptr;
        return *this;
    }

    inline fem_robin_boundary& operator=(const std::function<double(double, double)>& func) {
        value_func_ = func;
        value_const_ = std::nullopt;
        return *this;
    }

    inline auto evaluate_coeffs(double x, double y) const {
        if (!coeff_func_) throw std::runtime_error("fem_robin_boundary: coeff func not set");
        return coeff_func_(x, y);
    }

    inline double evaluate_value(double x, double y) const {
        if (value_func_) return value_func_(x, y);
        if (value_const_) return *value_const_;
        throw std::runtime_error("fem_robin_boundary: no value set");
    }

    inline const auto& get_nodes() const { return related_nodes_; }
    inline bool has_node(size_t idx) const { return related_nodes_.count(idx) > 0; }
};

namespace fem_boundary_creator {

inline field_solver::fixed_boundary_2d Dirichlet_region(
    const tri_mesh& mesh,
    const std::function<bool(double, double)>& region_func)
{
    fixed_boundary_2d ans;
    std::unordered_map<size_t, std::array<double, 2>> related;
    for (size_t i = 0; i < mesh.n_nodes(); i++) {
        auto& p = mesh.node_position(i);
        if (region_func(p.x(), p.y())) {
            related[i] = {p.x(), p.y()};
        }
    }
    ans.set_related_elements(related);
    return ans;
}

inline fem_neumann_boundary Neumann_region(
    const tri_mesh& mesh,
    const std::function<bool(double, double)>& region_func)
{
    fem_neumann_boundary ans;
    std::unordered_map<size_t, std::array<double, 2>> related;
    for (size_t i = 0; i < mesh.n_nodes(); i++) {
        auto& p = mesh.node_position(i);
        if (region_func(p.x(), p.y())) {
            related[i] = {p.x(), p.y()};
        }
    }
    ans.set_related_nodes(related);
    return ans;
}

inline fem_robin_boundary Robin_region(
    const tri_mesh& mesh,
    const std::function<std::pair<double, double>(double, double)>& coeff_func,
    const std::function<bool(double, double)>& region_func)
{
    fem_robin_boundary ans;
    std::unordered_map<size_t, std::array<double, 2>> related;
    for (size_t i = 0; i < mesh.n_nodes(); i++) {
        auto& p = mesh.node_position(i);
        if (region_func(p.x(), p.y())) {
            related[i] = {p.x(), p.y()};
        }
    }
    ans.set_related_nodes(related);
    ans.set_coeff_function(coeff_func);
    return ans;
}

}

}

}

}

#endif
