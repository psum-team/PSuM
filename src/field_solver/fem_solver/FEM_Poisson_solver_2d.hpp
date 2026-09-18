#ifndef PSUM_FIELD_SOLVER_FEM_POISSON_SOLVER_2D_HPP
#define PSUM_FIELD_SOLVER_FEM_POISSON_SOLVER_2D_HPP

#include <fstream>
#include <iostream>
#include <string>
#include <cmath>
#include <unordered_set>
#include <unordered_map>
#include <sycl/sycl.hpp>
#include <Eigen/Core>
#include <Eigen/SparseLU>
#include "../spmv/eigen_spmv_calculator.hpp"
#include "../spmv/spmv_calculator.hpp"
#include "../spmv/sycl_spmv_calculator.hpp"
#include "tri_mesh.hpp"
#include "tri_fem_assembly.hpp"
#include "fem_boundary.hpp"
#include "../fixed_boundary.hpp"
#include "../backend.hpp"

namespace psum {

namespace field_solver {

namespace fem_solver {

// 2D Poisson solver based on finite element method.
class FEM_Poisson_solver_2d{
    using Grid = tri_mesh;
public:
    enum psolverDomainKind
    {
        Cylindrical,
        Cartesian,
    };

    FEM_Poisson_solver_2d() : backend_name_("native"), domain_kind_(fem_domain_kind::Cartesian), grid_({}, {}) {}

    inline ~FEM_Poisson_solver_2d() = default;

    inline void init( const Grid& grid, psolverDomainKind domainType, const field_solver::fixed_boundary_2d_list& fbl, double _epsilon = 8.854e-12) {
        _init_impl("native", grid, domainType, {}, {}, fbl, [_epsilon](double, double) { return _epsilon; });
    }

    inline void init(const Grid &grid, psolverDomainKind domainType, const field_solver::fixed_boundary_2d_list& fbl, const std::function<double(double, double)>& epsilon) {
        _init_impl("native", grid, domainType, {}, {}, fbl, epsilon);
    }

    inline void init(const Grid& grid, psolverDomainKind domainType, const std::vector<fem_neumann_boundary>& neumann_boundaries, const field_solver::fixed_boundary_2d_list& fbl, double _epsilon = 8.854e-12) {
        _init_impl("native", grid, domainType, neumann_boundaries, {}, fbl, [_epsilon](double, double) { return _epsilon; });
    }

    inline void init(const Grid& grid, psolverDomainKind domainType, const std::vector<fem_neumann_boundary>& neumann_boundaries, const std::vector<fem_robin_boundary>& robin_boundaries, const field_solver::fixed_boundary_2d_list& fbl, double _epsilon = 8.854e-12) {
        _init_impl("native", grid, domainType, neumann_boundaries, robin_boundaries, fbl, [_epsilon](double, double) { return _epsilon; });
    }

    inline void init(const Grid& grid, psolverDomainKind domainType, const std::vector<fem_neumann_boundary>& neumann_boundaries, const std::vector<fem_robin_boundary>& robin_boundaries, const field_solver::fixed_boundary_2d_list& fbl, const std::function<double(double, double)>& epsilon) {
        _init_impl("native", grid, domainType, neumann_boundaries, robin_boundaries, fbl, epsilon);
    }

    inline void init(const std::string& backend_name, const Grid& grid, psolverDomainKind domainType, const field_solver::fixed_boundary_2d_list& fbl, double _epsilon = 8.854e-12) {
        _init_impl(backend_name, grid, domainType, {}, {}, fbl, [_epsilon](double, double) { return _epsilon; });
    }

    inline void init(const std::string& backend_name, const Grid& grid, psolverDomainKind domainType, const field_solver::fixed_boundary_2d_list& fbl, const std::function<double(double, double)>& epsilon) {
        _init_impl(backend_name, grid, domainType, {}, {}, fbl, epsilon);
    }

    inline void init(const std::string& backend_name, const Grid& grid, psolverDomainKind domainType, const std::vector<fem_neumann_boundary>& neumann_boundaries, const field_solver::fixed_boundary_2d_list& fbl, double _epsilon = 8.854e-12) {
        _init_impl(backend_name, grid, domainType, neumann_boundaries, {}, fbl, [_epsilon](double, double) { return _epsilon; });
    }

    inline void init(const std::string& backend_name, const Grid& grid, psolverDomainKind domainType, const std::vector<fem_neumann_boundary>& neumann_boundaries, const std::vector<fem_robin_boundary>& robin_boundaries, const field_solver::fixed_boundary_2d_list& fbl, double _epsilon = 8.854e-12) {
        _init_impl(backend_name, grid, domainType, neumann_boundaries, robin_boundaries, fbl, [_epsilon](double, double) { return _epsilon; });
    }

    inline void init(const std::string& backend_name, const Grid& grid, psolverDomainKind domainType, const std::vector<fem_neumann_boundary>& neumann_boundaries, const std::vector<fem_robin_boundary>& robin_boundaries, const field_solver::fixed_boundary_2d_list& fbl, const std::function<double(double, double)>& epsilon) {
        _init_impl(backend_name, grid, domainType, neumann_boundaries, robin_boundaries, fbl, epsilon);
    }

    inline const Eigen::SparseMatrix<double>& coeff_matrix() const {
        return A_;
    }

    template <typename SpMVCalculator, typename... Args>
    inline void set_spmv_calculator(Args... args) {
        static_assert(std::is_base_of_v<field_solver::spmv::spmv_calculator, SpMVCalculator>,
                      "SpMVCalculator must derive from spmv_calculator");
        spmv_calculator_ = std::make_unique<SpMVCalculator>(args...);
        if (mass_matrix_.rows() != 0 || mass_matrix_.cols() != 0) {
            spmv_calculator_->set_matrix(mass_matrix_);
        }
    }

    inline void solve(Eigen::Ref<Eigen::VectorXd> phi, const Eigen::Ref<const Eigen::VectorXd>& source) {
        if (_backend_uses_device_name()) {
            throw std::runtime_error("FEM_Poisson_solver_2d: Eigen solve overload is host-only for non-CPU backends");
        }
        if(b_.size()!=A_.rows())
            b_.resize(A_.rows());
        if (source.size() != b_.size())
            std::cerr << "FEM_Poisson_solver_2d solve: source size mismatch" << std::endl;
        b_ = source;
        if (phi.size() != b_.size())
            phi.resize(A_.cols());
        solve(phi.data(), b_.data());
    }

    inline void make_righthand_item(double* b) {
        if (!spmv_calculator_) {
            throw std::runtime_error("FEM_Poisson_solver_2d: spmv calculator is not initialized");
        }

        spmv_calculator_->apply_inplace(b);

        fbl_.evaluate(replace_elements_.second);
        addback_elements_.first.clear();
        addback_elements_.second.clear();

        auto add_rhs = [&](size_t idx, double value) {
            if (backend_name_ == "native") {
                b[idx] += value;
                return;
            }
            addback_elements_.first.push_back(idx);
            addback_elements_.second.push_back(value);
        };

        for (auto& info : boundary_edge_infos_) {
            double r_w = (domain_kind_ == fem_domain_kind::Cylindrical) ? info.mid_y : 1.0;
            if (info.is_neumann) {
                double g = neumann_boundaries_[info.neumann_idx].evaluate(info.mid_x, info.mid_y);
                double c0 = g * r_w * info.length / 3.0;
                double c1 = g * r_w * info.length / 6.0;
                add_rhs(info.n0, c0 + c1);
                add_rhs(info.n1, c1 + c0);
            }
            if (info.is_robin) {
                double g = robin_boundaries_[info.robin_idx].evaluate_value(info.mid_x, info.mid_y);
                auto [_, beta] = robin_boundaries_[info.robin_idx].evaluate_coeffs(info.mid_x, info.mid_y);
                _validate_robin_beta(beta);
                double rhs = g / beta * r_w * info.length / 3.0;
                add_rhs(info.n0, rhs);
                add_rhs(info.n1, rhs);
            }
        }

        if (backend_name_ == "native") {
            for (size_t i = 0; i < replace_elements_.first.size(); i++)
                b[replace_elements_.first[i]] = replace_elements_.second[i];
        } else {
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
                    std::cerr << "FEM_Poisson_solver_2d: native solver failed" << std::endl;
                }
            } catch (const std::exception& e) {
                std::cerr << "FEM_Poisson_solver_2d native solve failed: " << e.what() << std::endl;
            }
        } else {
            try {
                backend_.solve(source, phi);
            } catch (const std::exception& e) {
                std::cerr << "FEM_Poisson_solver_2d backend solve failed: " << e.what() << std::endl;
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
    fem_domain_kind domain_kind_;
    Eigen::SparseMatrix<double> A_;
    Eigen::SparseLU<Eigen::SparseMatrix<double>> solver_;
    Eigen::VectorXd b_;
    Grid grid_;

    field_solver::fixed_boundary_2d_list fbl_;
    std::vector<fem_neumann_boundary> neumann_boundaries_;
    std::vector<fem_robin_boundary> robin_boundaries_;

    std::pair<std::vector<unsigned long long>, std::vector<double>> replace_elements_;
    std::pair<std::vector<unsigned long long>, std::vector<double>> addback_elements_;
    Eigen::SparseMatrix<double> mass_matrix_;
    std::unique_ptr<field_solver::spmv::spmv_calculator> spmv_calculator_;

    struct _boundary_edge_info {
        size_t n0, n1;
        double length;
        double mid_x, mid_y;
        bool is_neumann;
        bool is_robin;
        int neumann_idx;
        int robin_idx;
    };
    std::vector<_boundary_edge_info> boundary_edge_infos_;
    field_solver::solver_backend backend_;

    sycl::queue sycl_queue_{sycl::default_selector_v};

    inline bool _backend_uses_device_name() const {
        if (backend_name_ == "native") return false;
        if (backend_name_.size() < 3) return false;
        return backend_name_.substr(backend_name_.size() - 3) != "cpu";
    }

    inline void _prepare_spmv_calculator() {
        if (!spmv_calculator_) {
            if (_backend_uses_device_name()) {
                spmv_calculator_ = std::make_unique<field_solver::spmv::sycl_spmv_calculator>(sycl_queue_);
            } else {
                spmv_calculator_ = std::make_unique<field_solver::spmv::eigen_spmv_calculator>();
            }
        }
        spmv_calculator_->set_matrix(mass_matrix_);
    }

    inline void _init_impl(const std::string& backend_name, const Grid& grid, psolverDomainKind domainType, const std::vector<fem_neumann_boundary>& neumann_boundaries, const std::vector<fem_robin_boundary>& robin_boundaries, const field_solver::fixed_boundary_2d_list& fbl, const std::function<double(double, double)>& epsilon) {
        
        grid_ = grid;
        backend_name_ = backend_name;
        domain_kind_ = domainType == Cylindrical ? fem_domain_kind::Cylindrical : fem_domain_kind::Cartesian;
        fbl_ = fbl;
        neumann_boundaries_ = neumann_boundaries;
        robin_boundaries_ = robin_boundaries;

        size_t n = grid_.n_nodes();
        Eigen::SparseMatrix<double> K = tri_fem_assemble_stiffness(grid_, epsilon, domain_kind_);
        mass_matrix_ = tri_fem_assemble_mass(grid_, domain_kind_);
        _prepare_spmv_calculator();

        std::unordered_set<size_t> dirichlet_nodes = fbl_.get_element_indexes();
        std::unordered_map<size_t, std::vector<int>> node_to_neumann;
        std::unordered_map<size_t, std::vector<int>> node_to_robin;

        for (int bi = 0; bi < (int)neumann_boundaries_.size(); bi++) {
            for (auto& [idx, pos] : neumann_boundaries_[bi].get_nodes()) {
                if (dirichlet_nodes.count(idx) == 0)
                    node_to_neumann[idx].push_back(bi);
            }
        }
        for (int bi = 0; bi < (int)robin_boundaries_.size(); bi++) {
            for (auto& [idx, pos] : robin_boundaries_[bi].get_nodes()) {
                if (dirichlet_nodes.count(idx) == 0 && node_to_neumann.count(idx) == 0)
                    node_to_robin[idx].push_back(bi);
            }
        }

        std::vector<Eigen::Triplet<double>> robin_stiffness_triplets;
        boundary_edge_infos_.clear();
        boundary_edge_infos_.reserve(grid_.n_boundary_edges());

        for (size_t be = 0; be < grid_.n_boundary_edges(); be++) {
            auto& edge = grid_.boundary_edges()[be];
            size_t n0 = edge[0], n1 = edge[1];
            Eigen::Vector2d p0 = grid_.node_position(n0);
            Eigen::Vector2d p1 = grid_.node_position(n1);
            double length = (p1 - p0).norm();
            double mid_x = (p0.x() + p1.x()) / 2.0;
            double mid_y = (p0.y() + p1.y()) / 2.0;

            int nbi = _get_common_boundary_idx(n0, n1, node_to_neumann);
            bool is_neumann = nbi >= 0;

            int rbi = -1;
            if (!is_neumann) {
                rbi = _get_common_boundary_idx(n0, n1, node_to_robin);
            }
            bool is_robin = !is_neumann && rbi >= 0;

            _boundary_edge_info info;
            info.n0 = n0;
            info.n1 = n1;
            info.length = length;
            info.mid_x = mid_x;
            info.mid_y = mid_y;
            info.is_neumann = is_neumann;
            info.is_robin = is_robin;
            info.neumann_idx = is_neumann ? nbi : -1;
            info.robin_idx = is_robin ? rbi : -1;
            boundary_edge_infos_.push_back(info);

            if (is_robin) {
                auto [alpha, beta] = robin_boundaries_[rbi].evaluate_coeffs(mid_x, mid_y);
                _validate_robin_beta(beta);
                double ratio = alpha / beta;
                double r_w = (domain_kind_ == fem_domain_kind::Cylindrical) ? mid_y : 1.0;
                double s00 = ratio * r_w * length / 3.0;
                double s01 = ratio * r_w * length / 6.0;
                robin_stiffness_triplets.push_back({(int)n0, (int)n0, s00});
                robin_stiffness_triplets.push_back({(int)n0, (int)n1, s01});
                robin_stiffness_triplets.push_back({(int)n1, (int)n0, s01});
                robin_stiffness_triplets.push_back({(int)n1, (int)n1, s00});
            }
        }

        A_ = K;
        if (!robin_stiffness_triplets.empty()) {
            Eigen::SparseMatrix<double> R(n, n);
            R.setFromTriplets(robin_stiffness_triplets.begin(), robin_stiffness_triplets.end());
            A_ = A_ + R;
        }

        std::vector<std::pair<size_t, double>> fixed_array;
        fbl_.evaluate(fixed_array);
        replace_elements_.first.clear();
        replace_elements_.second.clear();
        for (auto& [idx, val] : fixed_array) {
            replace_elements_.first.push_back(idx);
            replace_elements_.second.push_back(val);
        }

        {
            std::unordered_set<size_t> fixed_set(dirichlet_nodes.begin(), dirichlet_nodes.end());
            std::vector<Eigen::Triplet<double>> triplets;
            triplets.reserve(A_.nonZeros());
            for (int j = 0; j < A_.outerSize(); ++j) {
                for (Eigen::SparseMatrix<double>::InnerIterator it(A_, j); it; ++it) {
                    if (fixed_set.count(it.row()) == 0) {
                        triplets.push_back(Eigen::Triplet<double>(it.row(), it.col(), it.value()));
                    }
                }
            }
            for (auto idx : fixed_set) {
                triplets.push_back(Eigen::Triplet<double>((int)idx, (int)idx, 1.0));
            }
            A_.setZero();
            A_.setFromTriplets(triplets.begin(), triplets.end());
        }

        A_.pruned();
        A_.makeCompressed();

        if (backend_name_ == "native") {
            solver_.compute(A_);
            if (solver_.info() != Eigen::Success) {
                std::cerr << "FEM_Poisson_solver_2d: native solver build failed" << std::endl;
                return;
            }
        } else {
            try {
                field_solver::solver_backend new_backend(backend_name_.c_str());
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
            } catch (const std::exception& e) {
                throw std::runtime_error(std::string("FEM_Poisson_solver_2d backend init failed: ") + e.what());
            }
        }
	}

    static inline int _get_common_boundary_idx(size_t n0, size_t n1, const std::unordered_map<size_t, std::vector<int>>& map) {
        auto it0 = map.find(n0);
        auto it1 = map.find(n1);
        if (it0 == map.end() || it1 == map.end()) return -1;
        for (int idx0 : it0->second) {
            for (int idx1 : it1->second) {
                if (idx0 == idx1) return idx0;
            }
        }
        return -1;
    }

    static inline void _validate_robin_beta(double beta) {
        if (std::abs(beta) < 1e-30) {
            throw std::runtime_error("FEM_Poisson_solver_2d: Robin gradient coefficient beta must be non-zero");
        }
    }
};

}

}

}

#endif
