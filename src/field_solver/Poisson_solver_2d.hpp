#ifndef PSUM_FIELD_SOLVER_POISSON_SOLVER_2D_HPP
#define PSUM_FIELD_SOLVER_POISSON_SOLVER_2D_HPP

#include <fstream>
#include <iostream>
#include <string>
#include <cmath>
#include <Eigen/Core>
#include <Eigen/SparseLU>
#include "../field/simple_grid.hpp"
#include "fixed_boundary.hpp"
#include "mixed_boundary.hpp"
#include "local_equation.hpp"
#include "ghost_element_manager.hpp"
#include "backend.hpp"

namespace psum {

namespace field_solver {

// 2D Poisson solver based on staggered-grid finite difference.
class Poisson_solver_2d{
    using Grid = psum::field::simple_grid<2>;
    using boundary_direction_2d = psum::field::boundary_direction_2d;
public:
    enum psolverDomainKind
    {
        Cylindrical,
        Cartesian,
    };

    Poisson_solver_2d() : backend_name_("native"), grid_({0.0, 0.0}, {1.0, 1.0}, {1, 1}) {};
    inline void init(const Grid &grid, psolverDomainKind domainType, const mixed_boundary_2d_list &boundaries, const fixed_boundary_2d_list &fbl, double _epsilon = 8.854e-12) {
        _init_impl("native", grid, domainType, boundaries, fbl, [_epsilon](double, double) { return _epsilon; });
    }
    inline void init(const Grid& grid, psolverDomainKind domainType, const mixed_boundary_2d_list &boundaries, const fixed_boundary_2d_list& fbl, const std::function<double(double, double)>& _epsilon){
        _init_impl("native", grid, domainType, boundaries, fbl, _epsilon);
    }
    inline void init(const Grid& grid, psolverDomainKind domainType, const fixed_boundary_2d_list& fbl, double _epsilon = 8.854e-12){
        _init_impl("native", grid, domainType, {}, fbl, [_epsilon](double, double) { return _epsilon; });
    }
    inline void init(const Grid& grid, psolverDomainKind domainType, const fixed_boundary_2d_list& fbl, const std::function<double(double, double)>& _epsilon){
        _init_impl("native", grid, domainType, {}, fbl, _epsilon);
    }
    inline void init(const std::string& backend_name, const Grid &grid, psolverDomainKind domainType, const mixed_boundary_2d_list &boundaries, const fixed_boundary_2d_list &fbl, double _epsilon = 8.854e-12) {
        _init_impl(backend_name, grid, domainType, boundaries, fbl, [_epsilon](double, double) { return _epsilon; });
    }
    inline void init(const std::string& backend_name, const Grid &grid, psolverDomainKind domainType, const mixed_boundary_2d_list &boundaries, const fixed_boundary_2d_list &fbl, const std::function<double(double, double)>& _epsilon) {
        _init_impl(backend_name, grid, domainType, boundaries, fbl, _epsilon);
    }
    inline void init(const std::string& backend_name, const Grid& grid, psolverDomainKind domainType, const fixed_boundary_2d_list& fbl, double _epsilon = 8.854e-12){
        _init_impl(backend_name, grid, domainType, {}, fbl, [_epsilon](double, double) { return _epsilon; });
    }
    inline void init(const std::string& backend_name, const Grid& grid, psolverDomainKind domainType, const fixed_boundary_2d_list& fbl, const std::function<double(double, double)>& _epsilon){
        _init_impl(backend_name, grid, domainType, {}, fbl, _epsilon);
    }

    inline const Eigen::SparseMatrix<double>& coeff_matrix() const {
        return A_;
    }

    inline void solve(Eigen::Ref<Eigen::VectorXd> phi, const Eigen::Ref<const Eigen::VectorXd>& source) {
        if(b_.size()!=A_.rows())
            b_.resize(A_.rows());
        if (source.size() != b_.size())
            std::cerr << "Poisson_solver_2d solve: source and grid differ in size" << std::endl;
        b_ = source;
        if (phi.size() != b_.size())
            phi.resize(A_.cols());
        solve(phi.data(), b_.data());
    }

    inline void make_righthand_item(double* b) {
        fbl_.evaluate(replace_elements_.second);
        mbl_.evaluate(mixed_boundary_values_);

        for (auto& v: addback_elements_.second)
            v = 0.0;
        int count = 0;
        for (int k = 0; k < M_.outerSize(); ++k) {
            bool reach_inner_loop = false;
            for (decltype(M_)::InnerIterator it(M_, k); it; ++it) {
                reach_inner_loop = true;
                if (it.row() != addback_elements_.first[count])
                    throw std::runtime_error("make_righthand_item: addback_elements_ status broken.");
                addback_elements_.second[count] += it.value() * mixed_boundary_values_[it.col()];
            }
            if (reach_inner_loop)
                count++;
        }
        if (backend_name_ == "native") {
            // add back source here
            for (size_t i = 0; i < addback_elements_.first.size(); i++)
                b[addback_elements_.first[i]] += addback_elements_.second[i];
            // replace source here
            for (size_t i = 0; i < replace_elements_.first.size(); i++)
                b[replace_elements_.first[i]] = replace_elements_.second[i];
        } else {
            // use set_source_replace
            backend_.set_source_addback(addback_elements_.first.size(), addback_elements_.first.data(), addback_elements_.second.data());
            backend_.set_source_replace(replace_elements_.first.size(), replace_elements_.first.data(), replace_elements_.second.data());
        }
    }

    inline void solve(double* phi, double* source) {
        make_righthand_item(source);
        if (backend_name_ == "native") {
            try {
                Eigen::Map<Eigen::VectorXd> b(source, A_.rows());
                Eigen::Map<Eigen::VectorXd> x(phi, A_.cols());
                x = solver_.solve(b);
                if (solver_.info() != Eigen::Success) {
                    std::cerr << "SolveError: native solver solving failed" << std::endl;
                }
            } catch(const std::exception& e) {
                std::cerr<<"native solving failed"<<std::endl;
                std::cerr << e.what() << '\n';
            }
        } else {
            try {
                backend_.solve(source, phi);
            } catch(const std::exception& e) {
                std::cerr << "backend solving failed" << std::endl;
                std::cerr << e.what() << '\n';
            }
        }
    }

    // set_options: set backend options string, must be called before init if needed
    inline void set_options(const std::string& options) {
        options_ = options;
        if (backend_name_ != "native") {
            if (!backend_.is_null())
                backend_.set_options(options_.c_str());
        }
    }

protected:
    std::string backend_name_;
    std::string options_;
    Eigen::SparseMatrix<double> A_;
    Eigen::SparseMatrix<double, Eigen::RowMajor> M_;
    Eigen::SparseLU<Eigen::SparseMatrix<double>> solver_;
    Eigen::VectorXd b_;
    fixed_boundary_2d_list fbl_;
    mixed_boundary_2d_list mbl_;
    std::pair<std::vector<unsigned long long>, std::vector<double>> replace_elements_;
    std::pair<std::vector<unsigned long long>, std::vector<double>> addback_elements_;
    std::vector<double> mixed_boundary_values_;
    Grid grid_;
    solver_backend backend_;

    inline void _init_impl(const std::string& backend_name, const Grid& grid, psolverDomainKind domainType, const mixed_boundary_2d_list& mbl, const fixed_boundary_2d_list& fbl, const std::function<double(double, double)>& epsilon){
        
        grid_ = grid;
        int n_nodes = grid_.contentSize(psum::field::var_loc::nodeCentered);
        A_.resize(n_nodes, n_nodes);

        fbl_ = fbl;
        mbl_ = mbl;

        // fixed boundary will cover mixed boundary
        {
            auto& fixed_indexes = fbl_.get_element_indexes();
            std::vector<size_t> indexes_need_remove;
            for (auto& idx : fixed_indexes) {
                if (mbl_.element_exists(idx))
                    indexes_need_remove.push_back(idx);
            }
            mbl_.erase_at(indexes_need_remove);
        }

        std::vector<std::pair<size_t, double>> fixed_array;
        fbl_.evaluate(fixed_array);
        for (auto& [idx, val] : fixed_array) {
            replace_elements_.first.push_back(idx);
            replace_elements_.second.push_back(val);
        }

        ghost_element_manager ghost_mng(n_nodes);
        auto mb_nodes = mbl_.get_boundary_nodes();
        std::vector<local_equation> ghost_equations(mb_nodes.size());
        std::vector<std::vector<size_t>> element_idx_to_mb_node_idx(n_nodes);
        for (size_t i = 0; i < mb_nodes.size(); i++) {
            auto& node = mb_nodes[i];
            element_idx_to_mb_node_idx[node.element].push_back(i);
        }

        std::vector<Eigen::Triplet<double>> tripletlist;

        int I = grid_.numCells<0>();
        int J = grid_.numCells<1>();
        double dx_r = grid_.del_r<0>();
        double dy_r = grid_.del_r<1>();
        auto cyclic_index = [I, J](std::array<int, 2> idx) {
            std::array<int, 2> cyc_idx = idx;
            cyc_idx[0] = (cyc_idx[0] % (I + 1) + I + 1) % (I + 1);
            cyc_idx[1] = (cyc_idx[1] % (J + 1) + J + 1) % (J + 1);
            return Grid::node_index(cyc_idx);
        };
        for (int i = 0; i <= I; i++) {
            for (int j = 0; j <= J; j++) {
                size_t idx = grid_.n2i({i, j});
                local_equation final_eq;
                local_equation local_value({{idx, 1.0}});

                size_t idx_N = grid_.n2i(cyclic_index({i, j + 1}));
                size_t idx_S = grid_.n2i(cyclic_index({i, j - 1}));
                size_t idx_E = grid_.n2i(cyclic_index({i + 1, j}));
                size_t idx_W = grid_.n2i(cyclic_index({i - 1, j}));

                if(fbl_.element_exists(idx)) {
                    // case1: fixed node
                    final_eq = local_value;
                } else {
                    // case2: mixed node case
                    for (auto node_idx: element_idx_to_mb_node_idx[idx]) {
                        boundary_direction_2d dir = mb_nodes[node_idx].direction;
                        switch (dir) {
                        case boundary_direction_2d::N:
                            idx_N = ghost_mng.new_ghost_element();
                            break;
                        case boundary_direction_2d::S:
                            idx_S = ghost_mng.new_ghost_element();
                            break;
                        case boundary_direction_2d::E:
                            idx_E = ghost_mng.new_ghost_element();
                            break;
                        case boundary_direction_2d::W:
                            idx_W = ghost_mng.new_ghost_element();
                            break;
                        default:
                            break;
                        }
                    }
                    local_equation grad_X;
                    local_equation grad_Y;
                    grad_X = local_equation({{idx_W, dx_r * -0.5}, {idx_E, dx_r * 0.5}});
                    grad_Y = local_equation({{idx_S, dy_r * -0.5}, {idx_N, dy_r * 0.5}});

                    for (auto node_idx: element_idx_to_mb_node_idx[idx]) {
                        auto& mixed_boundary_node = mb_nodes[node_idx];
                        boundary_direction_2d dir = mixed_boundary_node.direction;
                        double v_coef = mixed_boundary_node.coeffs.first;
                        double g_coef = mixed_boundary_node.coeffs.second;
                        local_equation grad;
                        local_equation final_eq;
                        switch (dir) {
                        case boundary_direction_2d::N:
                            final_eq += grad_Y * g_coef + local_value * v_coef;
                            break;
                        case boundary_direction_2d::S:
                            final_eq += grad_Y * -g_coef + local_value * v_coef;
                            break;
                        case boundary_direction_2d::E:
                            final_eq += grad_X * g_coef + local_value * v_coef;
                            break;
                        case boundary_direction_2d::W:
                            final_eq += grad_X * -g_coef + local_value * v_coef;
                            break;
                        default:
                            break;
                        }
                        ghost_equations[node_idx] = final_eq;
                    }
                    // case3: trivial node case
                    local_equation grad_N;
                    local_equation grad_S;
                    local_equation grad_E;
                    local_equation grad_W;
                    // compute gradient: center difference
                    grad_N = local_equation({{idx, dy_r * -1}, {idx_N, dy_r}});
                    grad_S = local_equation({{idx_S, dy_r * -1}, {idx, dy_r}});
                    grad_E = local_equation({{idx, dx_r * -1}, {idx_E, dx_r}});
                    grad_W = local_equation({{idx_W, dx_r * -1}, {idx, dx_r}});

                    // compute epsilon
                    double eps_N, eps_S, eps_E, eps_W;
                    Eigen::RowVector2d pos_N = (grid_.nodePosition({i, j}) + grid_.nodePosition({i, j + 1})) / 2;
                    Eigen::RowVector2d pos_S = (grid_.nodePosition({i, j}) + grid_.nodePosition({i, j - 1})) / 2;
                    Eigen::RowVector2d pos_E = (grid_.nodePosition({i, j}) + grid_.nodePosition({i + 1, j})) / 2;
                    Eigen::RowVector2d pos_W = (grid_.nodePosition({i, j}) + grid_.nodePosition({i - 1, j})) / 2;
                    eps_N = epsilon(pos_N.x(), pos_N.y());
                    eps_S = epsilon(pos_S.x(), pos_S.y());
                    eps_E = epsilon(pos_E.x(), pos_E.y());
                    eps_W = epsilon(pos_W.x(), pos_W.y());

                    // compute 'div': gradient * surface / volume
                    double surface_W, surface_E, surface_S, surface_N;
                    double volume;
                    if (domainType == Cylindrical) {
                        double r_upper = (grid_.nodePosition({i, j}).y() + grid_.nodePosition({i, j + 1}).y()) / 2;
                        double r_lower = (grid_.nodePosition({i, j}).y() + grid_.nodePosition({i, j - 1}).y()) / 2;
                        if(r_lower<0) r_lower=0;
                        surface_N = grid_.del<0>() * M_PI * 2 * r_upper;
                        surface_S = grid_.del<0>() * M_PI * 2 * r_lower;
                        surface_E = surface_W = M_PI * (r_upper * r_upper - r_lower * r_lower);
                        volume = grid_.del<0>() * M_PI * (r_upper * r_upper - r_lower * r_lower);
                    } else {
                        surface_N = surface_S = grid_.del<0>();
                        surface_E = surface_W = grid_.del<1>();
                        volume = grid_.del<0>() * grid_.del<1>();
                    }
                    // Poisson equation: s = -div(grad(phi)) ~ sum(flux on cell surface) / -vol
                    final_eq = (grad_E * surface_E * eps_E - grad_W * surface_W * eps_W + grad_N * surface_N * eps_N - grad_S * surface_S * eps_S) * (-1.0 / volume);
                }
                ghost_mng.extract_ghost_contribute(idx, final_eq);
                for (auto [idx_coeff, coeff] : final_eq.get_terms()) {
                    if (idx_coeff < n_nodes)
                        tripletlist.push_back({(int)idx, (int)idx_coeff, coeff});
                }
            }
        }

        for (size_t i = 0; i < mb_nodes.size(); i++) {
            ghost_mng.set_ghost_equation(ghost_equations[i]);
        }

        Eigen::SparseMatrix<double> A_inner;
        A_inner.resize(n_nodes, n_nodes);
        A_inner.setFromTriplets(tripletlist.begin(), tripletlist.end());

        auto [A, M] = ghost_mng.get_elimination_system(A_inner);
        A_ = A;
        A_.pruned();
        A_.makeCompressed();
        
        M_ = M;
        M_.pruned();

        addback_elements_.first.resize(0);
        for (int k = 0; k < M_.outerSize(); ++k) {
            for (decltype(M_)::InnerIterator it(M_, k); it; ++it) {
                addback_elements_.first.push_back(it.row());
                break;
            }
        }
        addback_elements_.second.resize(addback_elements_.first.size());

        backend_name_ = backend_name;

        if (backend_name_ == "native") {
            solver_.compute(A_);
            if (solver_.info() != Eigen::Success)
            {
                std::cerr << "SolveError: native solver buidling failed" << std::endl;
                return;
            }
        } else {
            try {
                solver_backend new_backend(backend_name_.c_str());
                backend_ = std::move(new_backend);

                if (!options_.empty()) {
                    backend_.set_options(options_.c_str());
                }

                int n_rows = A_.rows();
                int n_cols = A_.cols();
                int nnz = A_.nonZeros();
                const int* outerPtr = A_.outerIndexPtr();
                const int* innerIdx = A_.innerIndexPtr();
                const double* values = A_.valuePtr();

                std::vector<unsigned long long> rows, cols;
                rows.reserve(nnz);
                cols.reserve(nnz);
                for (int k = 0; k < n_cols; ++k) {
                    for (int i = outerPtr[k]; i < outerPtr[k+1]; ++i) {
                        rows.push_back(innerIdx[i]);
                        cols.push_back(k);
                    }
                }

                backend_.set_matrix(n_rows, n_cols, nnz, rows.data(), cols.data(), const_cast<double*>(values));
            } catch(const std::exception& e) {
                std::cerr << "Backend initialization failed: " << e.what() << std::endl;
                return;
            }
        }
	}
};

}

}

#endif
