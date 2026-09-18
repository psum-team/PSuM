#ifndef PSUM_FIELD_SIMPLE_GRID_HPP
#define PSUM_FIELD_SIMPLE_GRID_HPP

#include <math.h>
#include <iostream>
#include "foundation.hpp"
#include "simple_interpolation.hpp"
#include "field_interp.hpp"
#include "../serialization/container_sl.hpp"

namespace psum {

namespace field {

    template<int Dim>
    requires (Dim > 0)
    struct simple_grid {
        static constexpr int Dimension = Dim;
    private:
        std::array<double, Dim> lower_bounds;
        std::array<double, Dim> upper_bounds;
        std::array<int, Dim> cell_num;
        std::array<double, Dim> delta;
        std::array<double, Dim> delta_r;
        std::array<double, Dim> midpoint;
        std::array<double, Dim> length;

        void check_dimension(int dim) const {
            if (dim != Dimension) throw std::runtime_error("Error: dimension mismatch in simple_grid.");
        }

        void check_var_loc(var_loc loc) const {
            if (loc != var_loc::cellCentered && loc != var_loc::nodeCentered)
                throw std::runtime_error("Error: variable location not supported in simple_grid.");
        }

        void generate_from_eigen(const Eigen::Vector<double, Dim>& lower_point, const Eigen::Vector<double, Dim>& upper_point, const std::vector<int>& size) {
            check_dimension(size.size());
            for (int i = 0; i < Dim; i++) {
                if (lower_point[i] >= upper_point[i]) throw std::runtime_error("Error: lower bound should be less than upper bound.");
                if (size[i] <= 0) throw std::runtime_error("Error: number of cells in each dimension should be positive.");
                lower_bounds[i] = lower_point[i];
                upper_bounds[i] = upper_point[i];
                cell_num[i] = size[i];
                delta[i] = (upper_bounds[i] - lower_bounds[i]) / size[i];
                delta_r[i] = 1.0 / delta[i];
                midpoint[i] = (upper_bounds[i] + lower_bounds[i]) / 2.0;
                length[i] = upper_bounds[i] - lower_bounds[i];
            }
        }

    public:
        simple_grid(const Eigen::Vector<double, Dim>& lower_point, const Eigen::Vector<double, Dim>& upper_point, const std::vector<int>& size) {
            generate_from_eigen(lower_point, upper_point, size);
        }
        simple_grid(const std::vector<double>& lower_point, const std::vector<double>& upper_point, const std::vector<int>& size) {
            check_dimension(lower_point.size());
            check_dimension(upper_point.size());
            // cast to Eigen::Vector
            Eigen::Vector<double, Dim> lower_point_eigen(lower_point.data());
            Eigen::Vector<double, Dim> upper_point_eigen(upper_point.data());
            generate_from_eigen(lower_point_eigen, upper_point_eigen, size);
        }

        simple_grid(const std::initializer_list<double>& lower_point,
                    const std::initializer_list<double>& upper_point,
                    const std::initializer_list<int>& size)
        : simple_grid(std::vector<double>(lower_point), 
                      std::vector<double>(upper_point), 
                      std::vector<int>(size)) 
        {}

        struct node_index { std::array<int, Dim> indices; };
        struct cell_index { std::array<int, Dim> indices; };

        /// nearest node
        template<typename VecDim> requires (foundation::array_like<VecDim, Dim>)
        node_index nN(const VecDim& p) const {
            node_index result;
            for (int i = 0; i < Dim; i++)
                result.indices[i] = round((p[i] - lower_bounds[i]) * delta_r[i]);
            return result;
        }

        /// corner node
        template<typename VecDim> requires (foundation::array_like<VecDim, Dim>)
        node_index cN(const VecDim& p) const {
            node_index result;
            for (int i = 0; i < Dim; i++)
                result.indices[i] = int((p[i] - lower_bounds[i]) * delta_r[i]);
            return result;
        }

        /// nearest cell
        template<typename VecDim> requires (foundation::array_like<VecDim, Dim>)
        cell_index nC(const VecDim& p) const {
            cell_index result;
            for (int i = 0; i < Dim; i++)
                result.indices[i] = int((p[i] - lower_bounds[i]) * delta_r[i]);
            return result;
        }

        /// node <---> index
        size_t n2i(const node_index& n) const {
            size_t ans = n.indices[0];
            for (int i = 1; i < Dim; i++)
                ans = ans * (cell_num[i] + 1) + n.indices[i];
            return ans;
        }
        node_index i2n(size_t i) const {
            node_index result;
            for (int j = Dim - 1; j >= 0; j--) {
                result.indices[j] = i % (cell_num[j] + 1);
                i /= (cell_num[j] + 1);
            }
            return result;
        }

        /// cell <---> index
        size_t c2i(const cell_index& c) const {
            size_t ans = c.indices[0];
            for (int i = 1; i < Dim; i++)
                ans = ans * cell_num[i] + c.indices[i];
            return ans;
        }
        cell_index i2c(size_t i) const {
            cell_index result;
            for (int j = Dim - 1; j >= 0; j--) {
                result.indices[j] = i % cell_num[j];
                i /= cell_num[j];
            }
            return result;
        }

        /// grid geometry
        template <int _Dim> requires (_Dim < Dim && _Dim >= 0)
        double del() const { return delta[_Dim]; }

        template <int _Dim> requires (_Dim < Dim && _Dim >= 0)
        double del_r() const { return delta_r[_Dim]; }

        template <int _Dim> requires (_Dim < Dim && _Dim >= 0)
        double lowerBound() const { return lower_bounds[_Dim]; }

        template <int _Dim> requires (_Dim < Dim && _Dim >= 0)
        double upperBound() const { return upper_bounds[_Dim]; }

        template <int _Dim> requires (_Dim < Dim && _Dim >= 0)
        int numCells() const { return cell_num[_Dim]; }

        template <int _Dim> requires (_Dim < Dim && _Dim >= 0)
        double middle() const { return midpoint[_Dim]; }

        template <int _Dim> requires (_Dim < Dim && _Dim >= 0)
        double span() const { return length[_Dim]; }

        const auto& get_lower_bounds() const { return lower_bounds; }
        const auto& get_upper_bounds() const { return upper_bounds; }
        const auto& get_cell_num() const { return cell_num; }
        const auto& get_deltas() const { return delta; }
        const auto& get_deltas_r() const { return delta_r; }
        const auto& get_middles() const { return midpoint; }
        const auto& get_spans() const { return length; }

        template<typename VecDim> requires (foundation::array_like<VecDim, Dim>)
        bool inGrid(const VecDim& p) const {
            for (int i = 0; i < Dim; i++) {
                if (p[i] < lower_bounds[i] || p[i] >= upper_bounds[i]) return false;
            }
            return true;
        }

        bool inGrid(const double(&p)[Dim]) const {
            return inGrid(std::to_array(p));
        }

        template<int ChosenDim, typename VecDim> requires (foundation::array_like<VecDim, Dim>)
        void wrap_on_dim(VecDim& p) const {
            if (p[ChosenDim] < lower_bounds[ChosenDim] || p[ChosenDim] >= upper_bounds[ChosenDim]) {
                p[ChosenDim] = p[ChosenDim] - floor((p[ChosenDim] - lower_bounds[ChosenDim]) / length[ChosenDim]) * length[ChosenDim];
            }
        }

        size_t contentSize(var_loc loc) const {
            check_var_loc(loc);
            size_t ans = 1;
            if (loc == var_loc::cellCentered) {
                for (int i = 0; i < Dim; i++)
                    ans *= cell_num[i];
            } else if (loc == var_loc::nodeCentered) {
                for (int i = 0; i < Dim; i++)
                    ans *= (cell_num[i] + 1);
            }
            return ans;
        }

        /// grid element geometry
        Eigen::RowVector<double, Dim> nodePosition(const node_index& n) const {
            Eigen::RowVector<double, Dim> result;
            for (int i = 0; i < Dim; i++)
                result[i] = lower_bounds[i] + n.indices[i] * delta[i];
            return result;
        }
        Eigen::RowVector<double, Dim> cellCenter(const cell_index& c) const {
            Eigen::RowVector<double, Dim> result;
            for (int i = 0; i < Dim; i++)
                result[i] = lower_bounds[i] + (c.indices[i] + 0.5) * delta[i];
            return result;
        }
        Eigen::RowVector<double, Dim> cellCorner(const cell_index& c) const {
            Eigen::RowVector<double, Dim> result;
            for (int i = 0; i < Dim; i++)
                result[i] = lower_bounds[i] + c.indices[i] * delta[i];
            return result;
        }

        template <grid_element Gele>
        requires (Gele == grid_element::cell||Gele == grid_element::node)
        Eigen::RowVector<double, Dim> position(const size_t& index) const {
            if constexpr (Gele == grid_element::cell) {
                return cellCenter(i2c(index));
            } else if constexpr (Gele == grid_element::node) {
                return nodePosition(i2n(index));
            } else {
                throw std::runtime_error("Error: grid element not supported in simple_grid.");
            }
        }

        template <typename Scalar, int Vnum>
        requires (std::is_floating_point_v<Scalar> && Vnum > 0)
		inline void plot(std::ostream& fp, const Scalar *content, double time, var_loc loc) const {
            check_var_loc(loc);

            std::vector<char> ijk_chars;
            if (Dim == 1) {
                ijk_chars = { 'i' };
            } else if (Dim == 2) {
                ijk_chars = { 'i', 'j' };
            } else if (Dim == 3) {
                ijk_chars = { 'i', 'j', 'k' };
            } else {
                throw std::runtime_error("Error: dimension not supported in simple_grid.");
            }
            if (Dim > 1) {
                fp << "\nzone ";
                for (int i = 0; i < (int)ijk_chars.size(); i++) {
                    fp << ijk_chars[i] << "=" << (cell_num[Dim - 1 - i] + 1) << ",";
                }
                if (loc == var_loc::cellCentered) {
                    fp << "F=block";
                    int first_data = Dim + 1;
                    int last_data = Dim + Vnum;
                    if (first_data == last_data)
                        fp << ", VARLOCATION=([" << first_data << "]=CELLCENTERED)";
                    else
                        fp << ", VARLOCATION=([" << first_data << "-" << last_data << "]=CELLCENTERED)";
                    fp << std::endl;
                } else {
                    fp << "F=point" << std::endl;
                }
            }

			if (!std::isnan(time))
				fp << "AUXDATA TimeUnits=\"Seconds\" SolutionTime=" << time;

            if (loc == var_loc::cellCentered) {
                size_t node_count = contentSize(var_loc::nodeCentered);
                for (int d = 0; d < Dim; d++) {
                    for (size_t i = 0; i < node_count; i++) {
                        auto n = i2n(i);
                        fp << std::endl << lower_bounds[d] + n.indices[d] * delta[d];
                    }
                }
                size_t cell_count = contentSize(var_loc::cellCentered);
                for (int v = 0; v < Vnum; v++) {
                    for (size_t i = 0; i < cell_count; i++) {
                        fp << std::endl << content[i * Vnum + v];
                    }
                }
            } else {
                for (size_t i = 0; i < contentSize(loc); i++) {
                    auto p = nodePosition(i2n(i));
                    fp << std::endl << p[0];
                    for (int j = 1; j < Dim; j++) {
                        fp << "\t" << p[j];
                    }
                    for (int j = 0; j < Vnum; j++) {
                        fp << "\t" << content[i * Vnum + j];
                    }
                }
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
    
    using grid1D = simple_grid<1>;
    using grid2D = simple_grid<2>;
    using grid3D = simple_grid<3>;

    namespace simple_interpolation {

        // linear interpolation
        template<int Dim, typename VecDim> requires (foundation::array_like<VecDim, Dim>)
        auto linear_interp(const simple_grid<Dim>& grid, const VecDim& p) {
            auto cornerNode = grid.cN(p);
            auto cornerPos = grid.nodePosition(cornerNode);
            std::array<double, Dim> normalizatedPos;
            auto& delta_r = grid.get_deltas_r();
            for (int i = 0; i < Dim; i++)
                normalizatedPos[i] = (p[i] - cornerPos[i]) * delta_r[i];
            constexpr int SupportSize1D = 2;
            auto coefs = simple_interpolation::interp_tensor<SupportSize1D, Dim, &simple_interpolation::linear_interp_1D>(normalizatedPos);
            constexpr int SupportSize = coefs.size();
            std::array<size_t, SupportSize> indices;
            for (int i = 0; i < SupportSize; i++) {
                auto node = cornerNode;
                int idx = i;
                for (int j = 0; j < Dim; j++) {
                    int idx_in_dim = idx % SupportSize1D;
                    node.indices[Dim - 1 - j] += idx_in_dim;
                    idx /= SupportSize1D;
                }
                indices[i] = grid.n2i(node);
            }
            return std::make_pair(indices, coefs);
        }

        // linear interpolation
        template<int Dim, typename VecDim> requires (foundation::array_like<VecDim, Dim>)
        auto linear_interp_diff(const simple_grid<Dim>& grid, const VecDim& p) {
            auto cornerNode = grid.cN(p);
            auto cornerPos = grid.nodePosition(cornerNode);
            std::array<double, Dim> normalizatedPos;
            auto& delta_r = grid.get_deltas_r();
            for (int i = 0; i < Dim; i++)
                normalizatedPos[i] = (p[i] - cornerPos[i]) * delta_r[i];
            constexpr int SupportSize1D = 2;
            auto coefs = simple_interpolation::interp_tensor_diff<SupportSize1D, Dim, &simple_interpolation::linear_interp_1D>(normalizatedPos, grid.get_deltas_r());
            constexpr int SupportSize = coefs[0].size();
            std::array<size_t, SupportSize> indices;
            for (int i = 0; i < SupportSize; i++) {
                auto node = cornerNode;
                int idx = i;
                for (int j = 0; j < Dim; j++) {
                    int idx_in_dim = idx % SupportSize1D;
                    node.indices[Dim - 1 - j] += idx_in_dim;
                    idx /= SupportSize1D;
                }
                indices[i] = grid.n2i(node);
            }
            return std::make_pair(indices, coefs);
        }
    }
}

}

#endif