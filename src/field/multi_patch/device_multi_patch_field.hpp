#ifndef PSUM_FIELD_MULTI_PATCH_DEVICE_MULTI_PATCH_FIELD_HPP
#define PSUM_FIELD_MULTI_PATCH_DEVICE_MULTI_PATCH_FIELD_HPP

#include "multi_patch_grid.hpp"
#include "../device_array.hpp"

namespace psum {

namespace field {

namespace multi_patch {

    template <typename Func, typename Data>
    concept handler_to_mpf_value_func =
        requires(Func f, sycl::handler &h, size_t i, Data& v) {
            { f(h)(i, v) } -> std::same_as<void>;
        };

    template <typename Func, typename Data, typename Position>
    concept handler_to_mpf_element_func =
        requires(Func f, sycl::handler &h, size_t i, Data& v, const Position &p) {
            { f(h)(i, v, p) } -> std::same_as<void>;
        };

    template <int _Dimension, var_loc _location, typename _Scalar, typename _Value, int _QuantitySize = 1>
    struct device_multi_patch_field_acc {
    public:
        static constexpr int Dimension = _Dimension;
        static constexpr var_loc Location = _location;
        static constexpr int QuantitySize = _QuantitySize;
        using Scalar = _Scalar;
        using Grid = multi_patch_grid_acc<Dimension>;
        using Value = _Value;
        using Position = Eigen::RowVector<double, _Dimension>;

        device_multi_patch_field_acc(
            const device_array_acc<Value>& _content_acc,
            const device_array_acc<Position>& _positions_acc,
            const Grid& _grid_acc
        ) : content_acc(_content_acc), positions_acc(_positions_acc), grid_acc(_grid_acc) {}

        Value& operator()(size_t index) const {
            return content_acc[index];
        }

        Scalar& operator()(size_t index_row, size_t index_col) {
            return content_acc[index_row][index_col];
        }

        Scalar& operator()(size_t index_row, size_t index_col) const {
            return content_acc[index_row][index_col];
        }

        const Position& position(size_t index) const {
            return positions_acc[index];
        }

        Grid& getGrid() { return grid_acc; }
        const Grid& getGrid() const { return grid_acc; }

    private:
        device_array_acc<Value> content_acc;
        device_array_acc<Position> positions_acc;
        Grid grid_acc;
    };

    template <int _Dimension, var_loc _location, typename _Scalar, int _QuantitySize = 1>
    class device_multi_patch_field {
    public:
        static constexpr int Dimension = _Dimension;
        static constexpr var_loc Location = _location;
        static constexpr int QuantitySize = _QuantitySize;
        using Scalar = _Scalar;
        using Self = device_multi_patch_field<_Dimension, _location, _Scalar, _QuantitySize>;
        using Value = typename std::conditional<QuantitySize == 1, Scalar, Eigen::RowVector<Scalar, QuantitySize>>::type;
        using Position = Eigen::RowVector<double, _Dimension>;
        using Grid = multi_patch_grid<Dimension>;
        using acc_type = device_multi_patch_field_acc<Dimension, Location, Scalar, Value, QuantitySize>;

    protected:
        device_array<Value> device_content;
        Grid grid_;
        device_array<Position> positions_;
        static constexpr auto Gele = map_vloc_to_gele(Location);

        static std::vector<Position> build_positions(const Grid& g) {
            size_t total = g.contentSize(Location);
            std::vector<Position> pos(total);
            size_t idx = 0;
            for (size_t p = 0; p < g.patches_count(); p++) {
                auto pg = g.patch_grid(p);
                size_t patch_size = g.patch_content_size(p, Location);
                for (size_t i = 0; i < patch_size; i++) {
                    if constexpr (Gele == grid_element::cell) {
                        pos[idx] = pg.cellCenter(pg.i2c(i));
                    } else {
                        pos[idx] = pg.nodePosition(pg.i2n(i));
                    }
                    idx++;
                }
            }
            return pos;
        }

        static void check_device(const sycl::queue& q, const Grid& grid) {
            if (q.get_device() != grid.getQueue().get_device())
                throw std::runtime_error("device_multi_patch_field: queue and grid must be on the same device");
        }

    public:
        device_multi_patch_field(Grid&& _grid)
            : device_multi_patch_field(_grid.getQueue(), std::move(_grid)) {}

        device_multi_patch_field(const sycl::queue& q, Grid&& _grid)
            : grid_(std::move(_grid)),
              device_content(q, _grid.contentSize(Location)),
              positions_(q, build_positions(_grid))
        {
            check_device(q, grid_);
            setZero();
        }

        device_multi_patch_field(device_array<Value>&& _content, Grid&& _grid)
            : grid_(std::move(_grid)),
              device_content(std::move(_content)),
              positions_(_grid.getQueue(), build_positions(_grid))
        {
            check_device(device_content.get_queue(), grid_);
            if (device_content.size() != grid_.contentSize(Location))
                throw std::runtime_error("Error: device_content size is not equal to grid size.");
        }

        decltype(auto) get_access(sycl::handler& h) {
            return acc_type(
                device_content.get_access(h),
                positions_.get_access(h),
                grid_.get_access(h)
            );
        }

        size_t size() const { return grid_.contentSize(Location); }
        Value* data() { return device_content.data(); }
        const Value* data() const { return device_content.data(); }
        const Grid& getGrid() const { return grid_; }
        var_loc getLocation() const { return Location; }
        sycl::queue getQueue() const { return device_content.get_queue(); }
        auto& getContent() const { return device_content; }

        template <typename FuncType>
        requires handler_to_mpf_value_func<FuncType, Value>
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
        requires handler_to_mpf_element_func<FuncType, Value, Position>
        void for_each(FuncType&& func) {
            size_t data_size = size();
            device_content.get_queue().submit([&](sycl::handler& h) {
                auto acc = get_access(h);
                auto v_func = func(h);
                h.parallel_for(sycl::range<1>(data_size), [=](sycl::id<1> idx) {
                    v_func(idx, acc(idx), acc.position(idx));
                });
            }).wait();
        }

        void setConstant(const Value& value) {
            device_content.for_each([value](sycl::handler &h) {
                return [value](Value& v) {
                    v = value;
                };
            });
        }

        void setZero() {
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
            if (getQueue() == b.getQueue() && size() == b.size()) {
                device_content.get_queue().memcpy(device_content.data(), b.data(), device_content.size() * sizeof(Value)).wait();
            }
            return *this;
        }

        void copy(const std::vector<Value>& host_vec) {
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
            std::vector<Scalar> unfolded_content(grid_.contentSize(Location) * QuantitySize);
            for (int i = 0; i < grid_.contentSize(Location); i++) {
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
	using cell_mp_field2D = device_multi_patch_field<2, var_loc::cellCentered, Scalar, QuantitySize>;
	template <typename Scalar, int QuantitySize = 1>
	using node_mp_field2D = device_multi_patch_field<2, var_loc::nodeCentered, Scalar, QuantitySize>;
    template <typename Scalar, int QuantitySize = 1>
	using cell_mp_field3D = device_multi_patch_field<3, var_loc::cellCentered, Scalar, QuantitySize>;
	template <typename Scalar, int QuantitySize = 1>
	using node_mp_field3D = device_multi_patch_field<3, var_loc::nodeCentered, Scalar, QuantitySize>;

    template <typename Position, typename Acc>
    requires foundation::array_like<Position, Acc::Dimension> &&
             std::is_same_v<
                typename device_multi_patch_field<Acc::Dimension, var_loc::nodeCentered, typename Acc::Scalar, Acc::QuantitySize>::acc_type,
                std::remove_cvref_t<Acc>
             >
    decltype(auto) interp(
        const Position& pos,
        const Acc& acc
    ) {
        return interp_tools::generate_interp<
            Position,
            Acc,
            linear_interp_mpg<Acc::Dimension, Position>
        >(pos, acc);
    }

    template <typename Acc>
    requires std::is_same_v<
                typename device_multi_patch_field<Acc::Dimension, var_loc::nodeCentered, typename Acc::Scalar, Acc::QuantitySize>::acc_type,
                std::remove_cvref_t<Acc>
             >
    decltype(auto) interp(
        const double (&pos)[Acc::Dimension], const Acc& acc) {
        return interp(std::to_array(pos), acc);
    }

    template <typename Position, typename Acc>
    requires foundation::array_like<Position, Acc::Dimension> &&
             std::is_same_v<
                typename device_multi_patch_field<Acc::Dimension, var_loc::nodeCentered, typename Acc::Scalar, Acc::QuantitySize>::acc_type,
                std::remove_cvref_t<Acc>
             >
    decltype(auto) interp_diff(
        const Position& pos,
        const Acc& acc
    ) {
        return interp_tools::generate_interp_diff<
            Position,
            Acc,
            linear_interp_diff_mpg<Acc::Dimension, Position>
        >(pos, acc);
    }

    template <typename Acc>
    requires std::is_same_v<
                typename device_multi_patch_field<Acc::Dimension, var_loc::nodeCentered, typename Acc::Scalar, Acc::QuantitySize>::acc_type,
                std::remove_cvref_t<Acc>
             >
    decltype(auto) interp_diff(
        const double (&pos)[Acc::Dimension], const Acc& acc) {
        return interp_diff(std::to_array(pos), acc);
    }

    template <typename Position, typename Acc>
    requires foundation::array_like<Position, Acc::Dimension> &&
             std::is_same_v<
                typename device_multi_patch_field<Acc::Dimension, var_loc::nodeCentered, typename Acc::Scalar, Acc::QuantitySize>::acc_type,
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
            linear_interp_mpg<Acc::Dimension, Position>
        >(pos, w, acc);
    }

    template <typename Acc>
    requires std::is_same_v<
                typename device_multi_patch_field<Acc::Dimension, var_loc::nodeCentered, typename Acc::Scalar, Acc::QuantitySize>::acc_type,
                std::remove_cvref_t<Acc>
             >
    decltype(auto) add_back(
        const double (&pos)[Acc::Dimension],
        const typename Acc::Value& w,
        const Acc& acc) {
        return add_back(std::to_array(pos), w, acc);
    }

}

}

}

#endif
