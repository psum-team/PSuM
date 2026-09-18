#ifndef PSUM_FIELD_SOLVER_MPG_POISSON_SOLVER_2D_HPP
#define PSUM_FIELD_SOLVER_MPG_POISSON_SOLVER_2D_HPP

#include "../../field/multi_patch/multi_patch_grid.hpp"
#include "../../field/multi_patch/duplicate_map.hpp"
#include "FEM_Poisson_solver_2d.hpp"
#include "multi_patch_to_tri_mesh.hpp"
#include "fem_boundary.hpp"
#include "../fixed_boundary.hpp"
#include <vector>
#include <functional>
#include <string>
#include <stdexcept>
#include <Eigen/Core>
#include <memory>
#include <optional>
#include <sycl/sycl.hpp>
#include "../spmv/eigen_spmv_calculator.hpp"
#include "../spmv/spmv_calculator.hpp"
#include "../spmv/sycl_spmv_calculator.hpp"

namespace psum {

namespace field_solver {

namespace fem_solver {

inline field_solver::fixed_boundary_2d Dirichlet_region(
    const field::multi_patch::multi_patch_grid<2>& mpg,
    const std::function<bool(double, double)>& region_func)
{
    auto [mesh, dm] = multi_patch_to_tri_mesh(mpg);
    return fem_boundary_creator::Dirichlet_region(mesh, region_func);
}

inline fem_neumann_boundary Neumann_region(
    const field::multi_patch::multi_patch_grid<2>& mpg,
    const std::function<bool(double, double)>& region_func)
{
    auto [mesh, dm] = multi_patch_to_tri_mesh(mpg);
    return fem_boundary_creator::Neumann_region(mesh, region_func);
}

inline fem_robin_boundary Robin_region(
    const field::multi_patch::multi_patch_grid<2>& mpg,
    const std::function<std::pair<double, double>(double, double)>& coeff_func,
    const std::function<bool(double, double)>& region_func)
{
    auto [mesh, dm] = multi_patch_to_tri_mesh(mpg);
    return fem_boundary_creator::Robin_region(mesh, coeff_func, region_func);
}

// 2D Poisson solver on multi_patch_grid<2> (FEM with local refinement grid).
class MPG_Poisson_solver_2d{
    using MPG = field::multi_patch::multi_patch_grid<2>;
    using DM = field::multi_patch::duplicate_map<2>;
public:
    using psolverDomainKind = FEM_Poisson_solver_2d::psolverDomainKind;
    static constexpr psolverDomainKind Cylindrical = FEM_Poisson_solver_2d::Cylindrical;
    static constexpr psolverDomainKind Cartesian = FEM_Poisson_solver_2d::Cartesian;

    MPG_Poisson_solver_2d() : solver_(FEM_Poisson_solver_2d()) {}

    inline ~MPG_Poisson_solver_2d() {
        _free_sycl_buffers();
    }

    inline void init(const MPG& mpg, psolverDomainKind domainType, const field_solver::fixed_boundary_2d_list& fbl, double _epsilon = 8.854e-12) {
        _do_init("native", mpg, domainType, {}, {}, fbl, [_epsilon](double, double) { return _epsilon; });
    }

    inline void init(const MPG& mpg, psolverDomainKind domainType, const field_solver::fixed_boundary_2d_list& fbl, const std::function<double(double, double)>& epsilon) {
        _do_init("native", mpg, domainType, {}, {}, fbl, epsilon);
    }

    inline void init(const MPG& mpg, psolverDomainKind domainType, const std::vector<fem_neumann_boundary>& neumann_boundaries, const field_solver::fixed_boundary_2d_list& fbl, double _epsilon = 8.854e-12) {
        _do_init("native", mpg, domainType, neumann_boundaries, {}, fbl, [_epsilon](double, double) { return _epsilon; });
    }

    inline void init(const MPG& mpg, psolverDomainKind domainType, const std::vector<fem_neumann_boundary>& neumann_boundaries, const std::vector<fem_robin_boundary>& robin_boundaries, const field_solver::fixed_boundary_2d_list& fbl, double _epsilon = 8.854e-12) {
        _do_init("native", mpg, domainType, neumann_boundaries, robin_boundaries, fbl, [_epsilon](double, double) { return _epsilon; });
    }

    inline void init(const MPG& mpg, psolverDomainKind domainType, const std::vector<fem_neumann_boundary>& neumann_boundaries, const std::vector<fem_robin_boundary>& robin_boundaries, const field_solver::fixed_boundary_2d_list& fbl, const std::function<double(double, double)>& epsilon) {
        _do_init("native", mpg, domainType, neumann_boundaries, robin_boundaries, fbl, epsilon);
    }

    inline void init(const std::string& backend_name, const MPG& mpg, psolverDomainKind domainType, const field_solver::fixed_boundary_2d_list& fbl, double _epsilon = 8.854e-12) {
        _do_init(backend_name, mpg, domainType, {}, {}, fbl, [_epsilon](double, double) { return _epsilon; });
    }

    inline void init(const std::string& backend_name, const MPG& mpg, psolverDomainKind domainType, const field_solver::fixed_boundary_2d_list& fbl, const std::function<double(double, double)>& epsilon) {
        _do_init(backend_name, mpg, domainType, {}, {}, fbl, epsilon);
    }

    inline void init(const std::string& backend_name, const MPG& mpg, psolverDomainKind domainType, const std::vector<fem_neumann_boundary>& neumann_boundaries, const field_solver::fixed_boundary_2d_list& fbl, double _epsilon = 8.854e-12) {
        _do_init(backend_name, mpg, domainType, neumann_boundaries, {}, fbl, [_epsilon](double, double) { return _epsilon; });
    }

    inline void init(const std::string& backend_name, const MPG& mpg, psolverDomainKind domainType, const std::vector<fem_neumann_boundary>& neumann_boundaries, const std::vector<fem_robin_boundary>& robin_boundaries, const field_solver::fixed_boundary_2d_list& fbl, double _epsilon = 8.854e-12) {
        _do_init(backend_name, mpg, domainType, neumann_boundaries, robin_boundaries, fbl, [_epsilon](double, double) { return _epsilon; });
    }

    inline void init(const std::string& backend_name, const MPG& mpg, psolverDomainKind domainType, const std::vector<fem_neumann_boundary>& neumann_boundaries, const std::vector<fem_robin_boundary>& robin_boundaries, const field_solver::fixed_boundary_2d_list& fbl, const std::function<double(double, double)>& epsilon) {
        _do_init(backend_name, mpg, domainType, neumann_boundaries, robin_boundaries, fbl, epsilon);
    }

    inline const Eigen::SparseMatrix<double>& coeff_matrix() const {
        return solver_.coeff_matrix();
    }

    inline void set_queue(sycl::queue queue) {
        _free_sycl_buffers();
        sycl_queue_ = queue;
        if (_backend_uses_device_name()) {
            solver_.set_spmv_calculator<field_solver::spmv::sycl_spmv_calculator>(*sycl_queue_);
            if (dm_ptr_) {
                _prepare_transform_spmv_calculators();
            }
        }
    }

    inline void solve(Eigen::Ref<Eigen::VectorXd> phi, const Eigen::Ref<const Eigen::VectorXd>& source) {
        if (_backend_uses_device_name()) {
            throw std::runtime_error("MPG_Poisson_solver_2d: Eigen solve overload is host-only for non-CPU backends");
        }
        if (!dm_ptr_) {
            throw std::runtime_error("MPG_Poisson_solver_2d::solve called before init");
        }
        if (source.size() != static_cast<Eigen::Index>(dm_ptr_->n_index)) {
            throw std::runtime_error("MPG_Poisson_solver_2d::solve source size mismatch");
        }
        if (phi.size() != static_cast<Eigen::Index>(dm_ptr_->n_index)) {
            throw std::runtime_error("MPG_Poisson_solver_2d::solve phi size mismatch");
        }
        solve(phi.data(), source.data());
    }

    inline void solve(std::vector<double>& phi_mpg, const std::vector<double>& source_mpg) {
        if (_backend_uses_device_name()) {
            throw std::runtime_error("MPG_Poisson_solver_2d: std::vector solve overload is host-only for non-CPU backends");
        }
        if (!dm_ptr_) {
            throw std::runtime_error("MPG_Poisson_solver_2d::solve called before init");
        }
        if (source_mpg.size() != dm_ptr_->n_index) {
            throw std::runtime_error("MPG_Poisson_solver_2d::solve source size mismatch");
        }
        phi_mpg.resize(dm_ptr_->n_index);
        solve(phi_mpg.data(), source_mpg.data());
    }

    inline void solve(double* phi_mpg, const double* source_mpg) {
        if (!dm_ptr_) {
            throw std::runtime_error("MPG_Poisson_solver_2d::solve called before init");
        }
        if (!gather_spmv_calculator_ || !scatter_spmv_calculator_)
            throw std::runtime_error("MPG_Poisson_solver_2d: transform spmv calculators are not initialized");
        if (_backend_uses_device_name()) {
            if (!sycl_queue_) {
                throw std::runtime_error("MPG_Poisson_solver_2d: set_queue must be called before solve with a device backend");
            }
            if (sycl_source_mesh_ == nullptr || sycl_phi_mesh_ == nullptr) {
                _free_sycl_buffers();
                sycl_source_mesh_ = sycl::malloc_device<double>(dm_ptr_->n_dof, *sycl_queue_);
                sycl_phi_mesh_ = sycl::malloc_device<double>(dm_ptr_->n_dof, *sycl_queue_);
                if (sycl_source_mesh_ == nullptr || sycl_phi_mesh_ == nullptr) {
                    _free_sycl_buffers();
                    throw std::runtime_error("MPG_Poisson_solver_2d: failed to allocate device dof buffers");
                }
            }
            if (sycl_source_mesh_ == nullptr || sycl_phi_mesh_ == nullptr)
                throw std::runtime_error("MPG_Poisson_solver_2d: device dof buffers are not initialized");
            gather_spmv_calculator_->apply(source_mpg, sycl_source_mesh_);
            solver_.solve(sycl_phi_mesh_, sycl_source_mesh_);
            scatter_spmv_calculator_->apply(sycl_phi_mesh_, phi_mpg);
        } else {
            gather_spmv_calculator_->apply(source_mpg, source_mesh_.data());
            solver_.solve(phi_mesh_, source_mesh_);
            scatter_spmv_calculator_->apply(phi_mesh_.data(), phi_mpg);
        }
    }

    inline void make_righthand_item(double* b) {
        solver_.make_righthand_item(b);
    }

    inline void set_options(const std::string& options) {
        solver_.set_options(options);
    }

    inline const DM& duplicate_map() const {
        return *dm_ptr_;
    }

private:
    const MPG* mpg_ptr_ = nullptr;
    std::string backend_name_ = "native";
    FEM_Poisson_solver_2d solver_;
    std::unique_ptr<DM> dm_ptr_;
    Eigen::VectorXd source_mesh_;
    Eigen::VectorXd phi_mesh_;
    std::optional<sycl::queue> sycl_queue_;
    std::unique_ptr<field_solver::spmv::spmv_calculator> gather_spmv_calculator_;
    std::unique_ptr<field_solver::spmv::spmv_calculator> scatter_spmv_calculator_;
    double* sycl_source_mesh_ = nullptr;
    double* sycl_phi_mesh_ = nullptr;

    inline bool _backend_uses_device_name() const {
        if (backend_name_.size() < 3) return false;
        return backend_name_.substr(backend_name_.size() - 3) != "cpu" && backend_name_ != "native";
    }

    inline void _free_sycl_buffers() {
        if (sycl_source_mesh_ != nullptr) {
            sycl::free(sycl_source_mesh_, *sycl_queue_);
            sycl_source_mesh_ = nullptr;
        }
        if (sycl_phi_mesh_ != nullptr) {
            sycl::free(sycl_phi_mesh_, *sycl_queue_);
            sycl_phi_mesh_ = nullptr;
        }
    }

    inline void _prepare_transform_spmv_calculators() {
        if (_backend_uses_device_name() && sycl_queue_) {
            gather_spmv_calculator_ = std::make_unique<field_solver::spmv::sycl_spmv_calculator>(*sycl_queue_);
            scatter_spmv_calculator_ = std::make_unique<field_solver::spmv::sycl_spmv_calculator>(*sycl_queue_);
        } else {
            gather_spmv_calculator_ = std::make_unique<field_solver::spmv::eigen_spmv_calculator>();
            scatter_spmv_calculator_ = std::make_unique<field_solver::spmv::eigen_spmv_calculator>();
        }
        if (!gather_spmv_calculator_ || !scatter_spmv_calculator_) {
            throw std::runtime_error("MPG_Poisson_solver_2d: transform spmv calculators are not initialized");
        }
        gather_spmv_calculator_->set_matrix(field::multi_patch::build_gather_matrix(*dm_ptr_));
        scatter_spmv_calculator_->set_matrix(dm_ptr_->P);
    }

    inline void _do_init(
        const std::string& backend_name,
        const MPG& mpg,
        psolverDomainKind domainType,
        const std::vector<fem_neumann_boundary>& neumann_boundaries,
        const std::vector<fem_robin_boundary>& robin_boundaries,
        const field_solver::fixed_boundary_2d_list& fbl,
        const std::function<double(double, double)>& epsilon)
    {
        mpg_ptr_ = &mpg;
        backend_name_ = backend_name;
        _free_sycl_buffers();
        auto [mesh, dm] = multi_patch_to_tri_mesh(mpg);
        dm_ptr_ = std::make_unique<DM>(std::move(dm));
        source_mesh_.resize(dm_ptr_->n_dof);
        phi_mesh_.resize(dm_ptr_->n_dof);
        if (_backend_uses_device_name() && sycl_queue_) {
            solver_.set_spmv_calculator<field_solver::spmv::sycl_spmv_calculator>(*sycl_queue_);
        } else {
            solver_.set_spmv_calculator<field_solver::spmv::eigen_spmv_calculator>();
        }
        solver_.init(backend_name, mesh, domainType, neumann_boundaries, robin_boundaries, fbl, epsilon);
        _prepare_transform_spmv_calculators();
    }
};

}

}

}

#endif
