#ifndef PSUM_FIELD_DEVICE_FIELD_HPP
#define PSUM_FIELD_DEVICE_FIELD_HPP

#include <optional>
#include "simple_grid.hpp"
#include "device_array.hpp"

namespace psum {

namespace field {
    template <int _Dimension, var_loc _location, typename _Scalar, int _QuantitySize>
    class device_field;
}

namespace serialization {
    template <int Dim, field::var_loc Loc, typename Scalar, int QS>
    void save(mas_file& fp, const std::string& name, const field::device_field<Dim, Loc, Scalar, QS>& obj);
    template <int Dim, field::var_loc Loc, typename Scalar, int QS>
    void load(mas_file& fp, const std::string& name, field::device_field<Dim, Loc, Scalar, QS>& obj);
}

namespace field {

    template <typename Func, typename Data>
    concept handler_to_field_value_func =
        requires(Func f, sycl::handler &h, size_t i, Data& v) {
            { f(h)(i, v) } -> std::same_as<void>;
        };

    template <typename Func, typename Data, typename Position>
    concept handler_to_field_element_func =
        requires(Func f, sycl::handler &h, size_t i, Data& v, const Position &p) {
            { f(h)(i, v, p) } -> std::same_as<void>;
        };

    template <int _Dimension, var_loc _location, typename _Scalar, typename _Value, int _QuantitySize = 1>
    struct device_field_acc {
    public:
		static constexpr int Dimension = _Dimension;
        static constexpr var_loc Location = _location;
        static constexpr int QuantitySize = _QuantitySize;
        using Scalar = _Scalar;
        using Grid = simple_grid<Dimension>;
        using Value = _Value;

        device_field_acc(const device_array_acc<Value> &_content_acc, const Grid &_grid)
            : content_acc(_content_acc), grid(_grid) {
            grid_contentSize = grid.contentSize(Location);
        }

        Value& operator()(size_t index) const {
            return content_acc[index];
		}

        Scalar& operator()(size_t index_row, size_t index_col) {
            static_assert(QuantitySize > 1, "QuantitySize must be greater than 1 for this operator.");
			return content_acc[index_row][index_col];
		}
        
        Scalar& operator()(size_t index_row, size_t index_col) const {
			static_assert(QuantitySize > 1, "QuantitySize must be greater than 1 for this operator.");
			return content_acc[index_row][index_col];
		}

        Grid& getGrid() { return grid; }
        const Grid& getGrid() const { return grid; }

    private:
        device_array_acc<Value> content_acc;
        Grid grid;
        size_t grid_contentSize;
    };

    template <int _Dimension, var_loc _location, typename _Scalar, int _QuantitySize = 1>
	class device_field {
    public:
		static constexpr int Dimension = _Dimension;
        static constexpr var_loc Location = _location;
        static constexpr int QuantitySize = _QuantitySize;
        using Scalar = _Scalar;
        using Self = device_field<_Dimension, _location, _Scalar, _QuantitySize>;
        using Value = typename std::conditional<QuantitySize == 1, Scalar, Eigen::RowVector<Scalar, QuantitySize>>::type;
        using Position = Eigen::RowVector<double, _Dimension>;
		using Grid = simple_grid<Dimension>;
        using acc_type = device_field_acc<Dimension, Location, Scalar, Value, QuantitySize>;

    protected:
        device_array<Value> device_content;
		std::optional<Grid> grid;
        static constexpr auto Gele = map_vloc_to_gele(Location);

        void require_grid() const {
            if (!grid) throw std::runtime_error("Error: device_field has no grid.");
        }

	public:
        device_field(const sycl::queue& q) : device_content(q, 1 << Dimension) {}

        template <int Dim, var_loc Loc, typename Scalar, int QS>
        friend void serialization::save(serialization::mas_file&, const std::string&, const device_field<Dim, Loc, Scalar, QS>&);
        template <int Dim, var_loc Loc, typename Scalar, int QS>
        friend void serialization::load(serialization::mas_file&, const std::string&, device_field<Dim, Loc, Scalar, QS>&);

        device_field(const sycl::queue& q, const Grid &_grid) : grid(_grid), device_content(q, _grid.contentSize(Location)) {
            setZero();
        }
        device_field(device_array<Value> &&_content, const Grid &_grid):
            grid(_grid), device_content(std::move(_content)) {
            if (device_content.size() != grid->contentSize(Location))
                throw std::runtime_error("Error: device_content size is not equal to grid size.");
        }
        device_field(Self&& other) noexcept
            : device_content(std::move(other.device_content)), grid(std::move(other.grid)) {
            other.grid.reset();
        }
        Self& operator=(Self&& other) noexcept {
            if (this != &other) {
                device_content = std::move(other.device_content);
                grid = std::move(other.grid);
                other.grid.reset();
            }
            return *this;
        }

        decltype(auto) get_access(sycl::handler& h) {
            return acc_type(device_content.get_access(h), getGrid());
        }

        bool has_grid() const { return grid.has_value(); }
        size_t size() const { require_grid(); return grid->contentSize(Location); }
        Value* data() { require_grid(); return device_content.data(); }
        const Value* data() const { require_grid(); return device_content.data(); }
		const Grid& getGrid() const { require_grid(); return *grid; }
		var_loc getLocation() const { return Location; }
        sycl::queue getQueue() const { return device_content.get_queue(); }
        auto& getContent() const { require_grid(); return device_content; }

        template <typename FuncType>
        requires handler_to_field_value_func<FuncType, Value>
        void for_each(FuncType&& func) {
            size_t data_size = size();
            device_content.get_queue().submit([&](sycl::handler& h) {
                auto acc = get_access(h);
                auto v_func = func(h);
                h.parallel_for(sycl::range<1>(data_size), [=](sycl::id<1> idx) {
                    v_func(idx, acc(idx));
                });
            }).wait();
        }

        template <typename FuncType>
        requires handler_to_field_element_func<FuncType, Value, Position>
        void for_each(FuncType&& func) {
            constexpr auto Gele = map_vloc_to_gele(Location);
            size_t data_size = size();
            device_content.get_queue().submit([&](sycl::handler& h) {
                auto acc = get_access(h);
                auto v_func = func(h);
                h.parallel_for(sycl::range<1>(data_size), [=](sycl::id<1> idx) {
                    v_func(idx, acc(idx), acc.getGrid().template position<Gele>(idx));
                });
            }).wait();
        }

		void setConstant(const Value& value) {
            require_grid();
            device_content.for_each([value](sycl::handler &h) {
                return [value](Value& v) {
                    v = value;
                };
            });
		}

        void setZero() {
            require_grid();
            device_content.for_each([](sycl::handler &h) {
                return [](Value& v) {
                    if constexpr (QuantitySize == 1) {
                        v = 0;
                    } else {
                        for (int i = 0; i < QuantitySize; i++) {
                            v[i] = 0;
                        }
                    }
                };
            });
        }

        Self& operator<<(const Self& b) {
            if (size() != b.size())
                throw std::runtime_error("Error: operator<< requires fields of the same size.");
            if (getQueue() != b.getQueue())
                throw std::runtime_error("Error: operator<< requires fields on the same queue.");
            device_content.get_queue().memcpy(device_content.data(), b.data(), device_content.size() * sizeof(Value)).wait();
            return *this;
        }

        void copy(const std::vector<Value>& host_vec) {
            require_grid();
            device_content.copy(host_vec);
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
            auto copied_content = device_content.to_host();
            std::vector<Scalar> unfolded_content(getGrid().contentSize(Location) * QuantitySize);
            for (size_t i = 0; i < getGrid().contentSize(Location); i++) {
                if constexpr (QuantitySize == 1) {
                    unfolded_content[i] = copied_content[i];
                } else {
                    for (int j = 0; j < QuantitySize; j++) {
                        unfolded_content[i * QuantitySize + j] = copied_content[i][j];
                    }
                }
            }
            getGrid().template plot<Scalar, QuantitySize>(fp, unfolded_content.data(), time, Location);
        }
    };

    template <typename Scalar, int QuantitySize = 1>
	using cell_field1D = device_field<1, var_loc::cellCentered, Scalar, QuantitySize>;
	template <typename Scalar, int QuantitySize = 1>
	using node_field1D = device_field<1, var_loc::nodeCentered, Scalar, QuantitySize>;
	template <typename Scalar, int QuantitySize = 1>
	using cell_field2D = device_field<2, var_loc::cellCentered, Scalar, QuantitySize>;
	template <typename Scalar, int QuantitySize = 1>
	using node_field2D = device_field<2, var_loc::nodeCentered, Scalar, QuantitySize>;
	template <typename Scalar, int QuantitySize = 1>
	using cell_field3D = device_field<3, var_loc::cellCentered, Scalar, QuantitySize>;
	template <typename Scalar, int QuantitySize = 1>
	using node_field3D = device_field<3, var_loc::nodeCentered, Scalar, QuantitySize>;

    template <typename Position, typename Acc>
    requires foundation::array_like<Position, Acc::Dimension> &&
             std::is_same_v<typename device_field<Acc::Dimension, var_loc::nodeCentered, typename Acc::Scalar, Acc::QuantitySize>::acc_type,
                std::remove_cvref_t<Acc>
             >
    decltype(auto) interp(
        const Position& pos,
        const Acc& acc
    ) {
        return interp_tools::generate_interp<
            Position,
            Acc,
            simple_interpolation::linear_interp<Acc::Dimension, Position>
        >(pos, acc);
    }

    template <typename Acc>
    requires std::is_same_v<typename device_field<Acc::Dimension, var_loc::nodeCentered, typename Acc::Scalar, Acc::QuantitySize>::acc_type,
                std::remove_cvref_t<Acc>
             >
    decltype(auto) interp(
        const double (&pos)[Acc::Dimension], const Acc& acc) {
        return interp(std::to_array(pos), acc);
    }

    template <typename Position, typename Acc>
    requires foundation::array_like<Position, Acc::Dimension> &&
             std::is_same_v<typename device_field<Acc::Dimension, var_loc::nodeCentered, typename Acc::Scalar, Acc::QuantitySize>::acc_type,
                std::remove_cvref_t<Acc>
             >
    decltype(auto) interp_diff(
        const Position& pos,
        const Acc& acc
    ) {
        return interp_tools::generate_interp_diff<
            Position,
            Acc,
            simple_interpolation::linear_interp_diff<Acc::Dimension, Position>
        >(pos, acc);
    }

    template <typename Acc>
    requires std::is_same_v<typename device_field<Acc::Dimension, var_loc::nodeCentered, typename Acc::Scalar, Acc::QuantitySize>::acc_type,
                std::remove_cvref_t<Acc>
             >
    decltype(auto) interp_diff(
        const double (&pos)[Acc::Dimension], const Acc& acc) {
        return interp_diff(std::to_array(pos), acc);
    }

    template <typename Position, typename Acc>
    requires foundation::array_like<Position, Acc::Dimension> &&
             std::is_same_v<typename device_field<Acc::Dimension, var_loc::nodeCentered, typename Acc::Scalar, Acc::QuantitySize>::acc_type,
                std::remove_cvref_t<Acc>
             >
    decltype(auto) add_back(
        const Position& pos,
        const typename Acc::Value& w,
        const Acc& acc
    ) {
        return interp_tools::generate_add_back<
            Position,
            Acc,
            simple_interpolation::linear_interp<Acc::Dimension, Position>
        >(pos, w, acc);
    }

    template <typename Acc>
    requires std::is_same_v<typename device_field<Acc::Dimension, var_loc::nodeCentered, typename Acc::Scalar, Acc::QuantitySize>::acc_type,
                std::remove_cvref_t<Acc>
             >
    decltype(auto) add_back(
        const double (&pos)[Acc::Dimension],
        const typename Acc::Value& w,
        const Acc& acc) {
        return add_back(std::to_array(pos), w, acc);
    }
}

namespace serialization {

    template <int Dim, field::var_loc Loc, typename Scalar, int QS>
    void save(mas_file& fp, const std::string& name, const field::device_field<Dim, Loc, Scalar, QS>& obj) {
        save(fp, name + ".location", (int)obj.getLocation());
        save(fp, name + ".grid", obj.getGrid());
        save(fp, name + ".content", obj.getContent());
    }

    template <int Dim, field::var_loc Loc, typename Scalar, int QS>
    void load(mas_file& fp, const std::string& name, field::device_field<Dim, Loc, Scalar, QS>& obj) {
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
        field::device_array<typename field::device_field<Dim, Loc, Scalar, QS>::Value> content(obj.device_content.get_queue(), 1);
        load(fp, name + ".content", content);
        if (content.size() != grid.contentSize(Loc))
            throw std::runtime_error("Error: block '" + name + "' has a content size which does not match its grid.");
        obj.grid = grid;
        obj.device_content = std::move(content);
    }
}

}

#endif