#ifndef PSUM_FIELD_SOLVER_GHOST_ELEMENT_MANAGER_HPP
#define PSUM_FIELD_SOLVER_GHOST_ELEMENT_MANAGER_HPP

#include <cmath>
#include <map>
#include <vector>
#include <Eigen/Sparse>
#include "local_equation.hpp"

namespace psum {

namespace field_solver {

struct ghost_element_manager {
private:
    std::vector<local_equation> ghost_equations;
    std::map<size_t, local_equation> ghost_contribute;
    size_t size_of_main_system;
    size_t number_of_ghost_elements;

public:
    ghost_element_manager(size_t N) : size_of_main_system(N), number_of_ghost_elements(0) {}

    size_t new_ghost_element() {
        number_of_ghost_elements++;
        return number_of_ghost_elements - 1 + size_of_main_system;
    }

    void extract_ghost_contribute(size_t element, const local_equation& eq) {
        if (ghost_contribute.count(element) == 0) {
            std::map<size_t, double> terms;
            for (const auto& term : eq.get_terms()) {
                if (term.first >= size_of_main_system) {
                    terms.insert(term);
                }
            }
            ghost_contribute[element] = local_equation(terms);
        } else {
            throw std::runtime_error("Error: ghost contributes already exists. row=" + std::to_string(element));
        }
    }

    void set_ghost_equation(const local_equation& eq) {
        ghost_equations.push_back(eq);
    }

    auto get_n_ghost_equations() const { return ghost_equations.size(); }

    auto& get_ghost_contribute() const {
        return ghost_contribute;
    }

    // [A, B; C, D][x; x_ghost] = [b; boundary_value]
    // -> A' x = b + M * boundary_value, where A' = A - B * D^-1 * C, M = -B * D^-1
    // get A' and M
    std::pair<Eigen::SparseMatrix<double>, Eigen::SparseMatrix<double>> get_elimination_system(const Eigen::SparseMatrix<double>& A) {
    if (ghost_equations.size() != number_of_ghost_elements)
        throw std::runtime_error("ghost equation count mismatch:" + std::to_string(ghost_equations.size()) + " vs " + std::to_string(number_of_ghost_elements));
        
        Eigen::SparseMatrix<double> B, C, D_inv;

        // compute B: contribution from ghost elements to main system
        {
            std::vector<Eigen::Triplet<double>> triplets;
            B.resize(size_of_main_system, number_of_ghost_elements);
            for (const auto& [idx_in_main_system, eq] : ghost_contribute) {
                for (auto [idx_in_global_system, coeff] : eq.get_terms()) {
                    size_t idx_in_ghost_system = idx_in_global_system - size_of_main_system;
                    triplets.push_back(Eigen::Triplet<double>(
                        idx_in_main_system, 
                        idx_in_ghost_system,
                        coeff
                    ));
                }
            }
            B.setFromTriplets(triplets.begin(), triplets.end());
        }

        // compute C and D: boundary contribution by main system
        {
            C.resize(number_of_ghost_elements, size_of_main_system);
            D_inv.resize(number_of_ghost_elements, number_of_ghost_elements);
            std::vector<Eigen::Triplet<double>> triplets_C;
            std::vector<Eigen::Triplet<double>> triplets_D_inv;
            std::vector<int> nnz_per_row(number_of_ghost_elements, 0);
            for (size_t i = 0; i < ghost_equations.size(); i++) {
                for (auto [idx_in_global_system, coeff] : ghost_equations[i].get_terms()) {
                    if (idx_in_global_system < size_of_main_system) {
                        size_t idx_in_main_system = idx_in_global_system;
                        triplets_C.push_back(Eigen::Triplet<double>(
                            i, 
                            idx_in_main_system,
                            coeff
                        ));
                    } else {
                        size_t idx_in_ghost_system = idx_in_global_system - size_of_main_system;
                        triplets_D_inv.push_back(Eigen::Triplet<double>(
                            idx_in_ghost_system, 
                            i,
                            1.0 / coeff
                        ));
                        nnz_per_row[idx_in_ghost_system]++;
                    }
                }
            }
            C.setFromTriplets(triplets_C.begin(), triplets_C.end());
            D_inv.setFromTriplets(triplets_D_inv.begin(), triplets_D_inv.end());
            // check whether D_inv is inverse of D
            for (auto i : nnz_per_row)
                if (i != 1)
                throw std::runtime_error("Error: unsupported boundary equations. D is not permutation-like matrix.");
        }

        Eigen::SparseMatrix<double> M = -B * D_inv;
        Eigen::SparseMatrix<double> A_elimination = A + M * C;
        return {A_elimination, M};
    }
};

}

}

#endif