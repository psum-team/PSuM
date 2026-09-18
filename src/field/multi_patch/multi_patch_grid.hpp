#ifndef PSUM_FIELD_MULTI_PATCH_MULTI_PATCH_GRID_HPP
#define PSUM_FIELD_MULTI_PATCH_MULTI_PATCH_GRID_HPP

#include <vector>
#include <array>
#include <stdexcept>
#include <cmath>
#include <ostream>
#include <algorithm>
#include "../simple_grid.hpp"
#include "../foundation.hpp"
#include "../device_array.hpp"

namespace psum {

namespace field {

namespace multi_patch {

    template <int Dim>
    struct multi_patch_grid_acc {
        simple_grid<Dim> coarse;
        device_array_acc<simple_grid<Dim>> patch_grids;
        device_array_acc<size_t> cell_offsets;
        device_array_acc<size_t> node_offsets;
        size_t num_patches;
        size_t total_cells;
        size_t total_nodes;

        size_t contentSize(var_loc loc) const {
            return (loc == var_loc::cellCentered) ? total_cells : total_nodes;
        }

        template <grid_element Gele, typename VecDim>
        requires (foundation::array_like<VecDim, Dim>)
        size_t nearest(const VecDim& p) const {
            if constexpr (Gele == grid_element::cell) {
                auto ci = coarse.nC(p);
                size_t pid = coarse.c2i(ci);
                auto& pg = patch_grids[pid];
                auto local_ci = pg.nC(p);
                return cell_offsets[pid] + pg.c2i(local_ci);
            } else {
                auto ci = coarse.nC(p);
                size_t pid = coarse.c2i(ci);
                auto& pg = patch_grids[pid];
                auto local_ni = pg.nN(p);
                return node_offsets[pid] + pg.n2i(local_ni);
            }
        }
    };

    template <int Dim, typename VecDim>
    auto linear_interp_mpg(const multi_patch_grid_acc<Dim>& ga, const VecDim& p) {
        auto ci = ga.coarse.nC(p);
        size_t pid = ga.coarse.c2i(ci);
        auto& pg = ga.patch_grids[pid];
        auto result = simple_interpolation::linear_interp<Dim, VecDim>(pg, p);
        size_t offset = ga.node_offsets[pid];
        for (size_t i = 0; i < result.first.size(); i++)
            result.first[i] += offset;
        return result;
    }

    template <int Dim, typename VecDim>
    auto linear_interp_diff_mpg(const multi_patch_grid_acc<Dim>& ga, const VecDim& p) {
        auto ci = ga.coarse.nC(p);
        size_t pid = ga.coarse.c2i(ci);
        auto& pg = ga.patch_grids[pid];
        auto result = simple_interpolation::linear_interp_diff<Dim, VecDim>(pg, p);
        size_t offset = ga.node_offsets[pid];
        for (size_t i = 0; i < result.first.size(); i++)
            result.first[i] += offset;
        return result;
    }

    template <int Dim, int MinResExp = 2, int MaxResExp = 6>
    requires (Dim > 0 && MinResExp > 0 && MaxResExp >= MinResExp)
    class multi_patch_grid {
    public:
        static constexpr int Dimension = Dim;
        static constexpr int MinResolutionExponent = MinResExp;
        static constexpr int MaxResolutionExponent = MaxResExp;
        using acc_type = multi_patch_grid_acc<Dim>;

    private:
        simple_grid<Dim> coarse_grid_;
        std::vector<int> resolution_exponents_;
        std::vector<size_t> patch_cell_offsets_;
        std::vector<size_t> patch_node_offsets_;
        size_t total_cells_;
        size_t total_nodes_;

        mutable sycl::queue q_;
        device_array<simple_grid<Dim>> patch_grids_dev_;
        device_array<size_t> patch_cell_offsets_dev_;
        device_array<size_t> patch_node_offsets_dev_;

        int cells_per_dim(size_t patch_id) const {
            return 1 << resolution_exponents_[patch_id];
        }

        size_t find_patch_cell(size_t linear_idx) const {
            auto it = std::upper_bound(patch_cell_offsets_.begin(), patch_cell_offsets_.end(), linear_idx);
            return (it - patch_cell_offsets_.begin()) - 1;
        }

        size_t find_patch_node(size_t linear_idx) const {
            auto it = std::upper_bound(patch_node_offsets_.begin(), patch_node_offsets_.end(), linear_idx);
            return (it - patch_node_offsets_.begin()) - 1;
        }

        void build_offsets() {
            size_t np = resolution_exponents_.size();
            patch_cell_offsets_.resize(np + 1);
            patch_node_offsets_.resize(np + 1);
            patch_cell_offsets_[0] = 0;
            patch_node_offsets_[0] = 0;
            for (size_t i = 0; i < np; i++) {
                int cpd = cells_per_dim(i);
                size_t cc = 1, nc = 1;
                for (int d = 0; d < Dim; d++) {
                    cc *= cpd;
                    nc *= (cpd + 1);
                }
                patch_cell_offsets_[i + 1] = patch_cell_offsets_[i] + cc;
                patch_node_offsets_[i + 1] = patch_node_offsets_[i] + nc;
            }
            total_cells_ = patch_cell_offsets_.back();
            total_nodes_ = patch_node_offsets_.back();
        }

        void validate() const {
            size_t expected = coarse_grid_.contentSize(var_loc::cellCentered);
            if (resolution_exponents_.size() != expected) {
                throw std::runtime_error("Error: resolution_exponents size must equal number of patches.");
            }
            for (auto exp : resolution_exponents_) {
                if (exp < MinResExp || exp > MaxResExp) {
                    throw std::runtime_error("Error: resolution exponent out of range.");
                }
            }
        }

        std::vector<simple_grid<Dim>> build_patch_grids_vec() const {
            std::vector<simple_grid<Dim>> grids;
            grids.reserve(patches_count());
            for (size_t i = 0; i < patches_count(); i++)
                grids.push_back(patch_grid(i));
            return grids;
        }

    public:
        multi_patch_grid(const sycl::queue& q,
            const std::vector<double>& lower_point,
            const std::vector<double>& upper_point,
            const std::vector<int>& coarse_cell_num,
            const std::vector<int>& resolution_exponents
        ) : coarse_grid_(lower_point, upper_point, coarse_cell_num),
            resolution_exponents_(resolution_exponents),
            q_(q),
            patch_grids_dev_(q, 1),
            patch_cell_offsets_dev_(q, 1),
            patch_node_offsets_dev_(q, 1)
        {
            validate();
            build_offsets();
            patch_grids_dev_ = device_array<simple_grid<Dim>>(q_, build_patch_grids_vec());
            patch_cell_offsets_dev_ = device_array<size_t>(q_, patch_cell_offsets_);
            patch_node_offsets_dev_ = device_array<size_t>(q_, patch_node_offsets_);
        }

        multi_patch_grid(const multi_patch_grid&) = delete;
        multi_patch_grid& operator=(const multi_patch_grid&) = delete;

        multi_patch_grid(multi_patch_grid&&) noexcept = default;
        multi_patch_grid& operator=(multi_patch_grid&&) noexcept = default;

        multi_patch_grid make_copy() const {
            const auto& lower = coarse_grid_.get_lower_bounds();
            const auto& upper = coarse_grid_.get_upper_bounds();
            const auto& cell_num = coarse_grid_.get_cell_num();
            return multi_patch_grid(
                q_,
                std::vector<double>(lower.begin(), lower.end()),
                std::vector<double>(upper.begin(), upper.end()),
                std::vector<int>(cell_num.begin(), cell_num.end()),
                resolution_exponents_
            );
        }

        acc_type get_access(sycl::handler& h) const {
            return acc_type{
                coarse_grid_,
                patch_grids_dev_.get_access(h),
                patch_cell_offsets_dev_.get_access(h),
                patch_node_offsets_dev_.get_access(h),
                patches_count(),
                total_cells_,
                total_nodes_
            };
        }

        sycl::queue getQueue() const { return q_; }

        struct patch_cell_index { size_t patch_id; std::array<int, Dim> indices; };
        struct patch_node_index { size_t patch_id; std::array<int, Dim> indices; };

        /// nearest node
        template<typename VecDim> requires (foundation::array_like<VecDim, Dim>)
        patch_node_index nN(const VecDim& p) const {
            auto coarse_ci = coarse_grid_.nC(p);
            size_t pid = coarse_grid_.c2i(coarse_ci);
            auto pg = patch_grid(pid);
            auto local_ni = pg.nN(p);
            return {pid, local_ni.indices};
        }

        /// corner node
        template<typename VecDim> requires (foundation::array_like<VecDim, Dim>)
        patch_node_index cN(const VecDim& p) const {
            auto coarse_ci = coarse_grid_.nC(p);
            size_t pid = coarse_grid_.c2i(coarse_ci);
            auto pg = patch_grid(pid);
            auto local_ni = pg.cN(p);
            return {pid, local_ni.indices};
        }

        /// nearest cell
        template<typename VecDim> requires (foundation::array_like<VecDim, Dim>)
        patch_cell_index nC(const VecDim& p) const {
            auto coarse_ci = coarse_grid_.nC(p);
            size_t pid = coarse_grid_.c2i(coarse_ci);
            auto pg = patch_grid(pid);
            auto local_ci = pg.nC(p);
            return {pid, local_ci.indices};
        }

        /// node <---> index
        size_t n2i(const patch_node_index& ni) const {
            auto pg = patch_grid(ni.patch_id);
            typename simple_grid<Dim>::node_index local_ni{ni.indices};
            return patch_node_offsets_[ni.patch_id] + pg.n2i(local_ni);
        }
        patch_node_index i2n(size_t i) const {
            size_t pid = find_patch_node(i);
            size_t local_i = i - patch_node_offsets_[pid];
            auto pg = patch_grid(pid);
            auto local_ni = pg.i2n(local_i);
            return {pid, local_ni.indices};
        }

        /// cell <---> index
        size_t c2i(const patch_cell_index& ci) const {
            auto pg = patch_grid(ci.patch_id);
            typename simple_grid<Dim>::cell_index local_ci{ci.indices};
            return patch_cell_offsets_[ci.patch_id] + pg.c2i(local_ci);
        }
        patch_cell_index i2c(size_t i) const {
            size_t pid = find_patch_cell(i);
            size_t local_i = i - patch_cell_offsets_[pid];
            auto pg = patch_grid(pid);
            auto local_ci = pg.i2c(local_i);
            return {pid, local_ci.indices};
        }

        /// grid geometry
        template <int _Dim> requires (_Dim < Dim && _Dim >= 0)
        double patch_span() const { return coarse_grid_.template del<_Dim>(); }

        template <int _Dim> requires (_Dim < Dim && _Dim >= 0)
        double lowerBound() const { return coarse_grid_.template lowerBound<_Dim>(); }

        template <int _Dim> requires (_Dim < Dim && _Dim >= 0)
        double upperBound() const { return coarse_grid_.template upperBound<_Dim>(); }

        template <int _Dim> requires (_Dim < Dim && _Dim >= 0)
        int numPatches() const { return coarse_grid_.template numCells<_Dim>(); }

        template <int _Dim> requires (_Dim < Dim && _Dim >= 0)
        double middle() const { return coarse_grid_.template middle<_Dim>(); }

        template <int _Dim> requires (_Dim < Dim && _Dim >= 0)
        double span() const { return coarse_grid_.template span<_Dim>(); }

        const auto& get_lower_bounds() const { return coarse_grid_.get_lower_bounds(); }
        const auto& get_upper_bounds() const { return coarse_grid_.get_upper_bounds(); }
        const auto& get_patch_num() const { return coarse_grid_.get_cell_num(); }
        const auto& get_patch_spans() const { return coarse_grid_.get_deltas(); }
        const auto& get_middles() const { return coarse_grid_.get_middles(); }
        const auto& get_spans() const { return coarse_grid_.get_spans(); }

        size_t patch_content_size(size_t patch_id, var_loc loc) const {
            int cpd = cells_per_dim(patch_id);
            int n = (loc == var_loc::cellCentered) ? cpd : (cpd + 1);
            size_t count = 1;
            for (int d = 0; d < Dim; d++)
                count *= n;
            return count;
        }

        size_t patch_offset(size_t patch_id) const {
            return patch_cell_offsets_[patch_id];
        }

        simple_grid<Dim> patch_grid(size_t patch_id) const {
            auto cell_idx = coarse_grid_.i2c(patch_id);
            Eigen::Vector<double, Dim> patch_lower;
            Eigen::Vector<double, Dim> patch_upper;
            const auto& deltas = coarse_grid_.get_deltas();
            const auto& lower = coarse_grid_.get_lower_bounds();
            for (int d = 0; d < Dim; d++) {
                patch_lower[d] = lower[d] + cell_idx.indices[d] * deltas[d];
                patch_upper[d] = patch_lower[d] + deltas[d];
            }
            int cpd = cells_per_dim(patch_id);
            std::vector<int> fine_size(Dim, cpd);
            return simple_grid<Dim>(patch_lower, patch_upper, fine_size);
        }

        const simple_grid<Dim>& coarse_grid() const {
            return coarse_grid_;
        }

        const std::vector<int>& get_resolution_exponents() const {
            return resolution_exponents_;
        }

        const std::vector<size_t>& get_patch_cell_offsets() const {
            return patch_cell_offsets_;
        }

        const std::vector<size_t>& get_patch_node_offsets() const {
            return patch_node_offsets_;
        }

        template<typename VecDim> requires (foundation::array_like<VecDim, Dim>)
        bool inGrid(const VecDim& p) const {
            return coarse_grid_.inGrid(p);
        }

        bool inGrid(const double(&p)[Dim]) const {
            return coarse_grid_.inGrid(p);
        }

        template<int ChosenDim, typename VecDim> requires (foundation::array_like<VecDim, Dim>)
        void wrap_on_dim(VecDim& p) const {
            coarse_grid_.template wrap_on_dim<ChosenDim>(p);
        }

        size_t contentSize(var_loc loc) const {
            return (loc == var_loc::cellCentered) ? total_cells_ : total_nodes_;
        }

        size_t patches_count() const {
            return resolution_exponents_.size();
        }

        Eigen::RowVector<double, Dim> nodePosition(const patch_node_index& ni) const {
            auto pg = patch_grid(ni.patch_id);
            return pg.nodePosition(typename simple_grid<Dim>::node_index{ni.indices});
        }
        Eigen::RowVector<double, Dim> cellCenter(const patch_cell_index& ci) const {
            auto pg = patch_grid(ci.patch_id);
            return pg.cellCenter(typename simple_grid<Dim>::cell_index{ci.indices});
        }
        Eigen::RowVector<double, Dim> cellCorner(const patch_cell_index& ci) const {
            auto pg = patch_grid(ci.patch_id);
            return pg.cellCorner(typename simple_grid<Dim>::cell_index{ci.indices});
        }

        template <grid_element Gele>
        requires (Gele == grid_element::cell||Gele == grid_element::node)
        Eigen::RowVector<double, Dim> position(const size_t& idx) const {
            if constexpr (Gele == grid_element::cell) {
                auto pci = i2c(idx);
                auto pg = patch_grid(pci.patch_id);
                return pg.cellCenter(typename simple_grid<Dim>::cell_index{pci.indices});
            } else {
                auto pni = i2n(idx);
                auto pg = patch_grid(pni.patch_id);
                return pg.nodePosition(typename simple_grid<Dim>::node_index{pni.indices});
            }
        }

        template <typename Scalar, int Vnum>
        requires (std::is_floating_point_v<Scalar> && Vnum > 0)
		inline void plot(std::ostream& fp, const Scalar *content, double time, var_loc loc) const {
            for (size_t p = 0; p < patches_count(); p++) {
                auto pg = patch_grid(p);
                size_t offset = (loc == var_loc::cellCentered) ? patch_cell_offsets_[p] : patch_node_offsets_[p];
                pg.template plot<Scalar, Vnum>(fp, content + offset * Vnum, time, loc);
            }
		}

        template <grid_element Gele, typename VecDim> requires (foundation::array_like<VecDim, Dim>)
        size_t nearest(const VecDim& p) const {
            if constexpr (Gele == grid_element::cell) {
                return c2i(nC(p));
            } else if constexpr (Gele == grid_element::node) {
                return n2i(nN(p));
            } else {
                throw std::runtime_error("Error: grid element not supported in simple_grid.");
            }
        }
    };

}

}

}

#endif
