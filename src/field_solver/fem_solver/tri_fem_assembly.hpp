#ifndef PSUM_FIELD_SOLVER_FEM_TRI_FEM_ASSEMBLY_HPP
#define PSUM_FIELD_SOLVER_FEM_TRI_FEM_ASSEMBLY_HPP

#include <cmath>
#include <vector>
#include <array>
#include <unordered_set>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include "tri_mesh.hpp"

namespace psum {

namespace field_solver {

namespace fem_solver {

enum class fem_domain_kind {
	Cartesian,
	Cylindrical
};

inline Eigen::SparseMatrix<double> tri_fem_assemble_stiffness(
    const tri_mesh& mesh,
    const std::function<double(double, double)>& epsilon,
    fem_domain_kind domain = fem_domain_kind::Cartesian)
{
    size_t n = mesh.n_nodes();
    size_t ne = mesh.n_elements();

    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(ne * 9);

    for (size_t e = 0; e < ne; e++) {
        auto& tri = mesh.elements()[e];
        auto grads = mesh.element_gradients(e);
        double area = mesh.element_area(e);

        Eigen::Vector2d centroid = mesh.element_centroid(e);
        double eps = epsilon(centroid.x(), centroid.y());

        double w = 1.0;
        if (domain == fem_domain_kind::Cylindrical)
            w = centroid.y();

        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                double val = eps * area * w * grads[i].dot(grads[j]);
                if (std::abs(val) > 1e-30) {
                    triplets.push_back(
                        Eigen::Triplet<double>(tri[i], tri[j], val));
                }
            }
        }
    }

    Eigen::SparseMatrix<double> K(n, n);
    K.setFromTriplets(triplets.begin(), triplets.end());
    return K;
}

inline Eigen::SparseMatrix<double> tri_fem_assemble_mass(
    const tri_mesh& mesh,
    fem_domain_kind domain = fem_domain_kind::Cartesian)
{
    size_t n = mesh.n_nodes();
    size_t ne = mesh.n_elements();

    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(ne * 9);

    for (size_t e = 0; e < ne; e++) {
        auto& tri = mesh.elements()[e];
        double area = mesh.element_area(e);

        double w = 1.0;
        if (domain == fem_domain_kind::Cylindrical) {
            Eigen::Vector2d centroid = mesh.element_centroid(e);
            w = centroid.y();
        }

        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                double val = w * area / 12.0;
                if (i == j) val = w * area / 6.0;
                triplets.push_back(
                    Eigen::Triplet<double>(tri[i], tri[j], val));
            }
        }
    }

    Eigen::SparseMatrix<double> M(n, n);
    M.setFromTriplets(triplets.begin(), triplets.end());
    return M;
}

}

}

}

#endif
