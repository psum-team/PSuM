#ifndef PSUM_FIELD_SOLVER_FEM_TRI_MESH_HPP
#define PSUM_FIELD_SOLVER_FEM_TRI_MESH_HPP

#include <cmath>
#include <vector>
#include <array>
#include <limits>
#include <unordered_map>
#include <unordered_set>
#include <stdexcept>
#include <Eigen/Core>

namespace psum {

namespace field_solver {

namespace fem_solver {

struct tri_mesh {
private:
    std::vector<Eigen::Vector2d> nodes_;
    std::vector<std::array<size_t, 3>> elements_;
    std::vector<std::array<size_t, 2>> boundary_edges_;
    std::vector<Eigen::Vector2d> boundary_edge_normals_;
    std::unordered_map<size_t, std::vector<size_t>> node_to_elements_;

    void build_topology() {
        auto make_edge_key = [](size_t a, size_t b) -> size_t {
            return a < b ? a * 1000000000ULL + b : b * 1000000000ULL + a;
        };

        std::unordered_map<size_t, int> edge_appear;
        std::unordered_map<size_t, std::pair<size_t, int>> edge_first_elem;

        for (size_t e = 0; e < elements_.size(); e++) {
            auto& tri = elements_[e];
            for (int k = 0; k < 3; k++) {
                size_t a = tri[k];
                size_t b = tri[(k + 1) % 3];
                size_t key = make_edge_key(a, b);
                edge_appear[key]++;
                if (edge_appear[key] == 1)
                    edge_first_elem[key] = {e, k};
            }
            for (int k = 0; k < 3; k++) {
                node_to_elements_[tri[k]].push_back(e);
            }
        }

        boundary_edges_.clear();
        boundary_edge_normals_.clear();
        for (auto& [key, count] : edge_appear) {
            if (count != 1) continue;

            auto [e0, local_edge] = edge_first_elem[key];

            size_t n0 = elements_[e0][local_edge];
            size_t n1 = elements_[e0][(local_edge + 1) % 3];
            size_t n_opp = elements_[e0][(local_edge + 2) % 3];

            Eigen::Vector2d p0 = nodes_[n0];
            Eigen::Vector2d p1 = nodes_[n1];
            Eigen::Vector2d tangent = (p1 - p0).normalized();
            Eigen::Vector2d normal(-tangent.y(), tangent.x());

            Eigen::Vector2d p_opp = nodes_[n_opp];
            if ((p_opp - p0).dot(normal) > 0) {
                normal = -normal;
            }

            boundary_edges_.push_back({n0, n1});
            boundary_edge_normals_.push_back(normal);
        }
    }

public:
    tri_mesh() = default;

    tri_mesh(const std::vector<Eigen::Vector2d>& nodes,
             const std::vector<std::array<size_t, 3>>& elements)
        : nodes_(nodes), elements_(elements) {
        build_topology();
    }

    void set_mesh(const std::vector<Eigen::Vector2d>& nodes,
                  const std::vector<std::array<size_t, 3>>& elements) {
        nodes_ = nodes;
        elements_ = elements;
        node_to_elements_.clear();
        boundary_edges_.clear();
        boundary_edge_normals_.clear();
        build_topology();
    }

    inline size_t n_nodes() const { return nodes_.size(); }
    inline size_t n_elements() const { return elements_.size(); }
    inline size_t n_boundary_edges() const { return boundary_edges_.size(); }

    inline const Eigen::Vector2d& node_position(size_t i) const { return nodes_[i]; }
    inline const std::vector<Eigen::Vector2d>& nodes() const { return nodes_; }
    inline const std::vector<std::array<size_t, 3>>& elements() const { return elements_; }
    inline const std::vector<std::array<size_t, 2>>& boundary_edges() const { return boundary_edges_; }
    inline const std::vector<Eigen::Vector2d>& boundary_edge_normals() const { return boundary_edge_normals_; }
    inline const auto& node_to_elements() const { return node_to_elements_; }

    inline double element_area(size_t e) const {
        auto& tri = elements_[e];
        Eigen::Vector2d p0 = nodes_[tri[0]];
        Eigen::Vector2d p1 = nodes_[tri[1]];
        Eigen::Vector2d p2 = nodes_[tri[2]];
        return 0.5 * std::abs((p1.x() - p0.x()) * (p2.y() - p0.y()) -
                               (p2.x() - p0.x()) * (p1.y() - p0.y()));
    }

    inline Eigen::Vector2d element_centroid(size_t e) const {
        auto& tri = elements_[e];
        return (nodes_[tri[0]] + nodes_[tri[1]] + nodes_[tri[2]]) / 3.0;
    }

    inline double edge_length(size_t edge_idx) const {
        auto& edge = boundary_edges_[edge_idx];
        return (nodes_[edge[1]] - nodes_[edge[0]]).norm();
    }

    inline std::array<Eigen::Vector2d, 3> element_gradients(size_t e) const {
        auto& tri = elements_[e];
        Eigen::Vector2d p0 = nodes_[tri[0]];
        Eigen::Vector2d p1 = nodes_[tri[1]];
        Eigen::Vector2d p2 = nodes_[tri[2]];
        double A2 = (p1.x() - p0.x()) * (p2.y() - p0.y()) -
                     (p2.x() - p0.x()) * (p1.y() - p0.y());
        double inv2A = 1.0 / A2;

        std::array<Eigen::Vector2d, 3> grads;
        grads[0] = Eigen::Vector2d(p1.y() - p2.y(), p2.x() - p1.x()) * inv2A;
        grads[1] = Eigen::Vector2d(p2.y() - p0.y(), p0.x() - p2.x()) * inv2A;
        grads[2] = Eigen::Vector2d(p0.y() - p1.y(), p1.x() - p0.x()) * inv2A;
        return grads;
    }
};

}

}

}

#endif
