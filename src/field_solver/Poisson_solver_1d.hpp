#ifndef PSUM_FIELD_SOLVER_POISSON_SOLVER_1D_HPP
#define PSUM_FIELD_SOLVER_POISSON_SOLVER_1D_HPP

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
#include "backend.hpp"
#include <functional>

namespace psum {

namespace field_solver {

class Poisson_solver_1d{
    using Grid = psum::field::simple_grid<1>;
    using boundary_direction_1d = psum::field::boundary_direction_1d;
public:
    Poisson_solver_1d() : backend_name_("native"), grid_({0.0}, {1.0}, {1}) {};
    inline void init(const Grid &grid, const mixed_boundary_1d_list &boundaries, const fixed_boundary_1d_list &fbl, double _epsilon = 8.854e-12) {
        _init_impl("native", grid, boundaries, fbl, [_epsilon](double) { return _epsilon; });
    }
    inline void init(const Grid& grid, const mixed_boundary_1d_list& boundaries, const fixed_boundary_1d_list& fbl, const std::function<double(double)>& epsilon){
        _init_impl("native", grid, boundaries, fbl, epsilon);
    }
    inline void init(const Grid& grid, const fixed_boundary_1d_list& fbl, double _epsilon = 8.854e-12){
        _init_impl("native", grid, {}, fbl, [_epsilon](double) { return _epsilon; });
    }
    inline void init(const Grid& grid, const fixed_boundary_1d_list& fbl, const std::function<double(double)>& _epsilon){
        _init_impl("native", grid, {}, fbl, _epsilon);
    }
    inline void init(const std::string& backend_name, const Grid &grid, const mixed_boundary_1d_list &boundaries, const fixed_boundary_1d_list &fbl, double _epsilon = 8.854e-12) {
        _init_impl(backend_name, grid, boundaries, fbl, [_epsilon](double) { return _epsilon; });
    }
    inline void init(const std::string& backend_name, const Grid& grid, const mixed_boundary_1d_list& boundaries, const fixed_boundary_1d_list& fbl, const std::function<double(double)>& epsilon){
        _init_impl(backend_name, grid, boundaries, fbl, epsilon);
    }
    inline void init(const std::string& backend_name, const Grid& grid, const fixed_boundary_1d_list& fbl, double _epsilon = 8.854e-12){
        _init_impl(backend_name, grid, {}, fbl, [_epsilon](double) { return _epsilon; });
    }
    inline void init(const std::string& backend_name, const Grid& grid, const fixed_boundary_1d_list& fbl, const std::function<double(double)>& _epsilon){
        _init_impl(backend_name, grid, {}, fbl, _epsilon);
    }

    inline const Eigen::SparseMatrix<double>& coeff_matrix() const {
        return A_;
    }

    inline void solve(Eigen::Ref<Eigen::VectorXd> phi, const Eigen::Ref<const Eigen::VectorXd>& source) {
        if(b_.size()!=A_.rows())
            b_.resize(A_.rows());
        if (source.size() != b_.size())
            std::cerr << "Poisson_solver_1d solve: source and grid differ in size" << std::endl;
        b_ = source;
        if (phi.size() != b_.size())
            phi.resize(A_.cols());
        solve(phi.data(), b_.data());
    }

    inline void make_righthand_item(double* b) {
        fbl_.evaluate(replace_elements_.second);
        mbl_.evaluate(addback_elements_.second);

        if (backend_name_ == "native") {
            // add back source here
            for (size_t i = 0; i < addback_elements_.first.size(); i++)
                b[addback_elements_.first[i]] += addback_elements_.second[i];
            // replace source here
            for (size_t i = 0; i < replace_elements_.first.size(); i++)
                b[replace_elements_.first[i]] = replace_elements_.second[i];
        } else {
            backend_.set_source_addback(addback_elements_.first.size(), addback_elements_.first.data(), addback_elements_.second.data());
            backend_.set_source_replace(replace_elements_.first.size(), replace_elements_.first.data(), replace_elements_.second.data());
        }
    }
    // assemble the right-hand side from rho without solving (3d-style
    // overload); 1d has no ghost-elimination matrix, so mixed-boundary
    // contributions are applied directly from mbl_ evaluation
    inline void make_righthand_item(const Eigen::VectorXd& rho, Eigen::VectorXd& b) {
        fbl_.evaluate(replace_elements_.second);
        mbl_.evaluate(addback_elements_.second);
        b = rho;
        for (size_t i = 0; i < addback_elements_.first.size(); i++)
            b[addback_elements_.first[i]] += addback_elements_.second[i];
        for (size_t i = 0; i < replace_elements_.first.size(); i++)
            b[replace_elements_.first[i]] = replace_elements_.second[i];
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
                throw;
            }
        } else {
            try {
                backend_.solve(source, phi);
            } catch(const std::exception& e) {
                std::cerr << "backend solving failed" << std::endl;
                std::cerr << e.what() << '\n';
                throw;
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
    Eigen::SparseLU<Eigen::SparseMatrix<double>> solver_;
    Eigen::VectorXd b_;
    fixed_boundary_1d_list fbl_;
    mixed_boundary_1d_list mbl_;
    std::pair<std::vector<unsigned long long>, std::vector<double>> replace_elements_;
    std::pair<std::vector<unsigned long long>, std::vector<double>> addback_elements_;
    std::vector<double> mixed_boundary_values_;
    Grid grid_;
    solver_backend backend_;

    inline void _init_impl(const std::string& backend_name, const Grid& grid, const mixed_boundary_1d_list& mbl, const fixed_boundary_1d_list& fbl, const std::function<double(double)>& epsilon){
        
        grid_ = grid;
        int n_nodes = grid_.contentSize(psum::field::var_loc::nodeCentered);
        A_.resize(n_nodes, n_nodes);

        fbl_ = fbl;
        mbl_ = mbl;

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

        auto nodes = mbl_.get_boundary_nodes();
        for (auto& node : nodes) {
            addback_elements_.first.push_back(node.element);
        }

        std::vector<Eigen::Triplet<double>> tripletlist;

        int I = grid_.numCells<0>();
        double dx_r = grid_.del_r<0>();
        auto cyclic_index = [I](std::array<int, 1> idx) {
            std::array<int, 1> cyc_idx = idx;
            cyc_idx[0] = (cyc_idx[0] % (I + 1) + I + 1) % (I + 1);
            return Grid::node_index(cyc_idx);
        };
        for (int i = 0; i <= I; i++) {
            size_t idx = grid_.n2i({i});
            local_equation final_eq;
            local_equation local_value({{idx, 1.0}});

            size_t idx_R = grid_.n2i(cyclic_index({i + 1}));
            size_t idx_L = grid_.n2i(cyclic_index({i - 1}));

            if(fbl_.element_exists(idx)) {
                final_eq = local_value;
            } else if (mbl_.element_exists(idx)) {
                local_equation grad_X;
                size_t idx_RR = grid_.n2i(cyclic_index({i + 2}));
                size_t idx_LL = grid_.n2i(cyclic_index({i - 2}));

                if (i==I) {
                    grad_X = local_equation({{idx_LL, dx_r * +0.5}, {idx_L, dx_r * -2}, {idx, dx_r * +1.5}});
                } else if (i==0) {
                    grad_X = local_equation({{idx, dx_r * -1.5}, {idx_R, dx_r * +2}, {idx_RR, dx_r * -0.5}});
                } else {
                    grad_X = local_equation({{idx_L, dx_r * -0.5}, {idx_R, dx_r * 0.5}});
                }

                int n_direction_collect = 0;
                for (int i_dir = 0; i_dir < (int)boundary_direction_1d::Count; i_dir++) {
                    boundary_direction_1d dir = (boundary_direction_1d)i_dir;
                    if (mbl_.element_exists(idx, dir)) {
                        auto [v_coef, g_coef] = mbl_.evaluate_coeffs_at(idx, dir);
                        switch (dir) {
                        case boundary_direction_1d::R:
                            final_eq += grad_X * g_coef + local_value * v_coef;
                            n_direction_collect++;
                            break;
                        case boundary_direction_1d::L:
                            final_eq += grad_X * -g_coef + local_value * v_coef;
                            n_direction_collect++;
                            break;
                        default:
                            break;
                        }
                    }
                }
                final_eq *= (1.0 / n_direction_collect);
            } else {
                local_equation grad_R;
                local_equation grad_L;
                grad_R = local_equation({{idx, dx_r * -1}, {idx_R, dx_r}});
                grad_L = local_equation({{idx_L, dx_r * -1}, {idx, dx_r}});

                double eps_R, eps_L;
                double pos_R = (grid_.nodePosition({i})[0] + grid_.nodePosition({i + 1})[0]) / 2;
                double pos_L = (grid_.nodePosition({i})[0] + grid_.nodePosition({i - 1})[0]) / 2;
                eps_R = epsilon(pos_R);
                eps_L = epsilon(pos_L);

                double surface_L = 1.0, surface_R = 1.0;
                double volume = grid_.del<0>();
                final_eq = (grad_R * surface_R * eps_R - grad_L * surface_L * eps_L) * (-1.0 / volume);
            }
            for (auto [idx_coeff, coeff] : final_eq.get_terms()) {
                tripletlist.push_back({static_cast<int>(idx), static_cast<int>(idx_coeff), coeff});
            }
        }

        A_.setFromTriplets(tripletlist.begin(), tripletlist.end());
        A_.makeCompressed();

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
                throw;
            }
        }
    }
};

}

}

#endif
