#ifndef PSUM_FIELD_HOST_FIELD_HPP
#define PSUM_FIELD_HOST_FIELD_HPP

#include <optional>
#include "simple_grid.hpp"
#include "atomic_add.hpp"

namespace psum {

namespace field {
    template <int _Dimension, var_loc _location, typename _Scalar, int _QuantitySize>
    class host_field;
}

namespace serialization {
    template <int Dim, field::var_loc Loc, typename Scalar, int QS>
    void save(mas_file& fp, const std::string& name, const field::host_field<Dim, Loc, Scalar, QS>& obj);
    template <int Dim, field::var_loc Loc, typename Scalar, int QS>
    void load(mas_file& fp, const std::string& name, field::host_field<Dim, Loc, Scalar, QS>& obj);
}

namespace field {

    template <typename Func, typename Data>
    concept lambda_on_field_value =
        requires(Func f, size_t i, Data& v) {
            { f(i, v) } -> std::same_as<void>;
        };

    template <typename Func, typename Data, typename Position>
    concept lambda_on_field_element =
        requires(Func f, size_t i, Data& v, const Position &p) {
            { f(i, v, p) } -> std::same_as<void>;
        };

    template <int _Dimension, var_loc _location, typename _Scalar, int _QuantitySize = 1>
	class host_field {
    public:
		static constexpr int Dimension = _Dimension;
        static constexpr var_loc Location = _location;
        static constexpr int QuantitySize = _QuantitySize;
        using Scalar = _Scalar;
        using Self = host_field<_Dimension, _location, _Scalar, _QuantitySize>;
        using Value = typename std::conditional<QuantitySize == 1, Scalar, Eigen::RowVector<Scalar, QuantitySize>>::type;
        using Position = Eigen::RowVector<double, _Dimension>;
		using Grid = simple_grid<Dimension>;

    protected:
        std::vector<Value> content;
		std::optional<Grid> grid;
        static constexpr auto Gele = map_vloc_to_gele(Location);

        void require_grid() const {
            if (!grid) throw std::runtime_error("Error: host_field has no grid.");
        }

	public:
        host_field() = default;

        template <int Dim, var_loc Loc, typename Scalar, int QS>
        friend void serialization::save(serialization::mas_file&, const std::string&, const host_field<Dim, Loc, Scalar, QS>&);
        template <int Dim, var_loc Loc, typename Scalar, int QS>
        friend void serialization::load(serialization::mas_file&, const std::string&, host_field<Dim, Loc, Scalar, QS>&);

        host_field(const Grid &_grid) : grid(_grid), content(_grid.contentSize(Location), Value{}) {
            setZero();
        }
        host_field(std::vector<Value> &&_content, const Grid &_grid):
            grid(_grid), content(std::move(_content)) {
            if (content.size() != grid->contentSize(Location))
                throw std::runtime_error("Error: content size is not equal to grid size.");
        }
        host_field(Self&& other) noexcept
            : content(std::move(other.content)), grid(std::move(other.grid)) {
            other.grid.reset();
        }
        Self& operator=(Self&& other) noexcept {
            if (this != &other) {
                content = std::move(other.content);
                grid = std::move(other.grid);
                other.grid.reset();
            }
            return *this;
        }

        bool has_grid() const { return grid.has_value(); }
        size_t size() const { require_grid(); return grid->contentSize(Location); }
        Value* data() { require_grid(); return content.data(); }
        const Value* data() const { require_grid(); return content.data(); }
		const Grid& getGrid() const { require_grid(); return *grid; }
		var_loc getLocation() const { return Location; }
        auto getContent() const { require_grid(); return content; }

        template <typename FuncType>
        requires lambda_on_field_value<FuncType, Value>
        void for_each(FuncType&& func) {
            size_t data_size = size();
            for (size_t i = 0; i < data_size; i++) {
                func(i, content[i]);
            }
        }

        template <typename FuncType>
        requires lambda_on_field_element<FuncType, Value, Position>
        void for_each(FuncType&& func) {
            size_t data_size = size();
            for (size_t i = 0; i < data_size; i++) {
                func(i, content[i], getGrid().template position<Gele>(i));
            }
        }

		void setConstant(const Value& value) {
            require_grid();
            for (auto& v : content)
                v = value;
		}

        void setZero() {
            require_grid();
            for (auto& v : content) {
                if constexpr (QuantitySize == 1) {
                    v = 0;
                } else {
                    for (int i = 0; i < QuantitySize; i++) {
                        v[i] = 0;
                    }
                }
            };
        }

        Self& operator<<(const Self& b) {
            if (size() != b.size())
                throw std::runtime_error("Error: operator<< requires fields of the same size.");
            std::copy(b.data(), b.data() + size(), data());
            return *this;
        }

        void copy(const std::vector<Value>& host_vec) {
            if (host_vec.size() != size()) throw std::runtime_error("Error: host_vec size is not equal to field size.");
            content = host_vec;
        }

        Value& operator()(size_t index) {
            return content[index];
		}

        const Value& operator()(size_t index) const {
            return content[index];
		}

        Scalar& operator()(size_t index_row, size_t index_col) {
			static_assert(QuantitySize > 1, "QuantitySize must be greater than 1 for this operator.");
			return content[index_row][index_col];
		}
        
        const Scalar& operator()(size_t index_row, size_t index_col) const {
			static_assert(QuantitySize > 1, "QuantitySize must be greater than 1 for this operator.");
			return content[index_row][index_col];
		}


		void plot(const std::string &filename, std::string varName = "_not_defined_", double time = std::nan("")) const {
			std::ofstream fp(filename, std::ios::out);
            if (!fp) {
                std::cerr << filename << " cannot be openned!" << std::endl;
                return;
            }
			fp << std::fixed << std::setprecision(4) << std::scientific;
			if (varName == "_not_defined_"){
                varName = "v1";
                for (int i = 1; i < QuantitySize; i++) {
                    varName += std::string(",v") + std::to_string(i+1);
                }
            }
            fp << "variables=" << (Dimension == 1 ? std::string("x,") : (Dimension == 2 ? std::string("x,y,") : std::string("x,y,z,"))) << varName;
            
            std::vector<Scalar> unfolded_content(getGrid().contentSize(Location) * QuantitySize);
            for (size_t i = 0; i < getGrid().contentSize(Location); i++) {
                if constexpr (QuantitySize == 1) {
                    unfolded_content[i] = content[i];
                } else {
                    for (int j = 0; j < QuantitySize; j++) {
                        unfolded_content[i * QuantitySize + j] = content[i][j];
                    }
                }
            }
            getGrid().template plot<Scalar, QuantitySize>(fp, unfolded_content.data(), time, Location);
        }
    };

    template <typename Scalar, int QuantitySize = 1>
	using host_cell_field1D = host_field<1, var_loc::cellCentered, Scalar, QuantitySize>;
	template <typename Scalar, int QuantitySize = 1>
	using host_node_field1D = host_field<1, var_loc::nodeCentered, Scalar, QuantitySize>;
	template <typename Scalar, int QuantitySize = 1>
	using host_cell_field2D = host_field<2, var_loc::cellCentered, Scalar, QuantitySize>;
	template <typename Scalar, int QuantitySize = 1>
	using host_node_field2D = host_field<2, var_loc::nodeCentered, Scalar, QuantitySize>;
	template <typename Scalar, int QuantitySize = 1>
	using host_cell_field3D = host_field<3, var_loc::cellCentered, Scalar, QuantitySize>;
	template <typename Scalar, int QuantitySize = 1>
	using host_node_field3D = host_field<3, var_loc::nodeCentered, Scalar, QuantitySize>;

    template <typename Position, typename Field>
    requires foundation::array_like<Position, Field::Dimension> &&
             std::is_same_v<
                host_field<Field::Dimension, var_loc::nodeCentered, typename Field::Scalar, Field::QuantitySize>,
                std::remove_cvref_t<Field>
             >
    decltype(auto) interp(
        const Position& pos,
        const Field& f
    ) {
        return interp_tools::generate_interp<
            Position,
            Field,
            simple_interpolation::linear_interp<Field::Dimension, Position>
        >(pos, f);
    }

    template <typename Field>
    requires std::is_same_v<
                host_field<Field::Dimension, var_loc::nodeCentered, typename Field::Scalar, Field::QuantitySize>,
                std::remove_cvref_t<Field>
             >
    decltype(auto) interp(
        const double (&pos)[Field::Dimension],
        const Field& f) {
        return interp(std::to_array(pos), f);
    }

    template <typename Position, typename Field>
    requires foundation::array_like<Position, Field::Dimension> &&
             std::is_same_v<
                host_field<Field::Dimension, var_loc::nodeCentered, typename Field::Scalar, Field::QuantitySize>,
                std::remove_cvref_t<Field>
             >
    decltype(auto) interp_diff(
        const Position& pos,
        const Field& f
    ) {
        return interp_tools::generate_interp_diff<
            Position,
            Field,
            simple_interpolation::linear_interp_diff<Field::Dimension, Position>
        >(pos, f);
    }

    template <typename Field>
    requires std::is_same_v<
                host_field<Field::Dimension, var_loc::nodeCentered, typename Field::Scalar, Field::QuantitySize>,
                std::remove_cvref_t<Field>
             >
    decltype(auto) interp_diff(
        const double (&pos)[Field::Dimension],
        const Field& f) {
        return interp_diff(std::to_array(pos), f);
    }

    namespace interp_tools {
        // UGLY patch!
        // just for hots_field
        template <typename Position, typename FieldAcc, auto Interpolator>
        requires field_can_interp<Position, FieldAcc, Interpolator> &&
                 foundation::array_like<Position, FieldAcc::Dimension>
        void generate_add_back_mutable(const Position& pos, const typename FieldAcc::Value& _w, FieldAcc& field_acc, double tol = 1e-10) {
            typename FieldAcc::Value w = _w;
            auto interp_param = Interpolator(field_acc.getGrid(), pos);
            for (int i = 0; i < interp_param.first.size(); ++i) {
                if (std::abs(interp_param.second[i]) > tol)
                _atomic_add_<typename FieldAcc::Scalar, FieldAcc::QuantitySize>(
                    field_acc(interp_param.first[i]), w, interp_param.second[i]
                );
            }
        }
    }

    template <typename Position, typename Field>
    requires foundation::array_like<Position, Field::Dimension> &&
             std::is_same_v<
                host_field<Field::Dimension, var_loc::nodeCentered, typename Field::Scalar, Field::QuantitySize>,
                std::remove_cvref_t<Field>
             >
    decltype(auto) add_back(
        const Position& pos,
        const typename Field::Value& w,
        Field& f
    ) {
        return interp_tools::generate_add_back_mutable<
            Position,
            Field,
            simple_interpolation::linear_interp<Field::Dimension, Position>
        >(pos, w, f);
    }

    template <typename Field>
    requires std::is_same_v<
                host_field<Field::Dimension, var_loc::nodeCentered, typename Field::Scalar, Field::QuantitySize>,
                std::remove_cvref_t<Field>
             >
    decltype(auto) add_back(
        const double (&pos)[Field::Dimension],
        const typename Field::Value& w,
        Field& f) {
        return add_back(std::to_array(pos), w, f);
    }

}

namespace serialization {

    template <int Dim, field::var_loc Loc, typename Scalar, int QS>
    void save(mas_file& fp, const std::string& name, const field::host_field<Dim, Loc, Scalar, QS>& obj) {
        save(fp, name + ".location", (int)obj.getLocation());
        save(fp, name + ".grid", obj.getGrid());
        save(fp, name + ".content", obj.getContent());
    }

    template <int Dim, field::var_loc Loc, typename Scalar, int QS>
    void load(mas_file& fp, const std::string& name, field::host_field<Dim, Loc, Scalar, QS>& obj) {
        int loc = -1;
        load(fp, name + ".location", loc);
        if (loc != (int)obj.getLocation())
            throw std::runtime_error(
                "Error: block '" + name + "' has location " + 
                std::to_string(loc) + " which cannot be loaded into a field of location " + 
                std::to_string((int)obj.getLocation()) + "."
            );
        field::simple_grid<Dim> grid;
        load(fp, name + ".grid", grid);
        std::vector<typename field::host_field<Dim, Loc, Scalar, QS>::Value> content;
        load(fp, name + ".content", content);
        if (content.size() != grid.contentSize(Loc))
            throw std::runtime_error("Error: block '" + name + "' has a content size which does not match its grid.");
        obj.grid = grid;
        obj.content = std::move(content);
    }
}

}

#endif