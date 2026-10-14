#ifndef PSUM_BATCH_PIC_HOST_BATCH_FIELD_HPP
#define PSUM_BATCH_PIC_HOST_BATCH_FIELD_HPP

#include <vector>
#include <type_traits>
#include "../field/host_field.hpp"
#include <cstdio>

namespace psum {

namespace batch_pic {

    // A batch field is an array of host fields with same grid and queue.
    template <typename Field>
    class host_batch_field {
    public:
        static constexpr int Dimension = Field::Dimension;
        static constexpr field::var_loc Location = Field::Location;
        static constexpr int QuantitySize = Field::QuantitySize;
        using Scalar = typename Field::Scalar;
        using Value = typename Field::Value;
        using Grid = typename Field::Grid;

        host_batch_field(const Grid& grid, size_t case_num) {
            fields.reserve(case_num);
            for (size_t c = 0; c < case_num; c++) {
                fields.emplace_back(grid);
            }
        }

        size_t batch_size() const { return fields.size(); }

        Field& at(size_t c) { return fields.at(c); }
        const Field& at(size_t c) const { return fields.at(c); }

        const Grid& getGrid() const { return fields.front().getGrid(); }
        field::var_loc getLocation() const { return Location; }

        auto getContent() const {
            std::vector<std::vector<Value>> out;
            out.reserve(fields.size());
            for (auto& f : fields) out.push_back(f.getContent());
            return out;
        }

        // access the whole per-case field array
        auto& getFields() { return fields; }
        auto& getFields() const { return fields; }

        void setConstant(const Value& value) {
            for (auto& f : fields) f.setConstant(value);
        }

        void setZero() {
            for (auto& f : fields) f.setZero();
        }

        void copy(const std::vector<std::vector<Value>>& host_batches) {
            if (host_batches.size() != fields.size())
                throw std::runtime_error("host_batch_field::copy: batch size mismatch");
            for (size_t c = 0; c < fields.size(); c++) {
                fields[c].copy(host_batches[c]);
            }
        }

        // case index substituted for %d
        void plot(const std::string& pattern, std::string varName = "_not_defined_", double time = std::nan("")) const {
            if (pattern.find("%d") == std::string::npos)
                throw std::runtime_error("batch plot: filename pattern has no %d for the case index; every case would overwrite the same file");
            for (size_t c = 0; c < fields.size(); c++) {
                char buf[512];
                int n = snprintf(buf, sizeof(buf), pattern.c_str(), static_cast<int>(c));
                if (n < 0 || static_cast<size_t>(n) >= sizeof(buf))
                    throw std::runtime_error("batch plot: rendered filename too long");
                fields[c].plot(std::string(buf, n), varName, time);
            }
        }

    private:
        std::vector<Field> fields;
    };

    // interp/add_back for batch fields.
    template <typename Position, typename Batch>
    requires field::foundation::array_like<Position, Batch::Dimension> &&
             std::is_same_v<
                host_batch_field<field::host_field<Batch::Dimension, field::var_loc::nodeCentered, typename Batch::Scalar, Batch::QuantitySize>>,
                std::remove_cvref_t<Batch>
             >
    decltype(auto) interp(
        const Position& pos,
        size_t case_id,
        const Batch& batch
    ) {
        return field::interp(pos, batch.getFields()[case_id]);
    }

    template <typename Batch>
    requires std::is_same_v<
                host_batch_field<field::host_field<Batch::Dimension, field::var_loc::nodeCentered, typename Batch::Scalar, Batch::QuantitySize>>,
                std::remove_cvref_t<Batch>
             >
    decltype(auto) interp(
        const double (&pos)[Batch::Dimension],
        size_t case_id,
        const Batch& batch
    ) {
        return field::interp(std::to_array(pos), batch.getFields()[case_id]);
    }

    template <typename Position, typename Batch>
    requires field::foundation::array_like<Position, Batch::Dimension> &&
             std::is_same_v<
                host_batch_field<field::host_field<Batch::Dimension, field::var_loc::nodeCentered, typename Batch::Scalar, Batch::QuantitySize>>,
                std::remove_cvref_t<Batch>
             >
    decltype(auto) interp_diff(
        const Position& pos,
        size_t case_id,
        const Batch& batch
    ) {
        return field::interp_diff(pos, batch.getFields()[case_id]);
    }

    template <typename Batch>
    requires std::is_same_v<
                host_batch_field<field::host_field<Batch::Dimension, field::var_loc::nodeCentered, typename Batch::Scalar, Batch::QuantitySize>>,
                std::remove_cvref_t<Batch>
             >
    decltype(auto) interp_diff(
        const double (&pos)[Batch::Dimension],
        size_t case_id,
        const Batch& batch
    ) {
        return field::interp_diff(std::to_array(pos), batch.getFields()[case_id]);
    }

    template <typename Position, typename Batch>
    requires field::foundation::array_like<Position, Batch::Dimension> &&
             std::is_same_v<
                host_batch_field<field::host_field<Batch::Dimension, field::var_loc::nodeCentered, typename Batch::Scalar, Batch::QuantitySize>>,
                std::remove_cvref_t<Batch>
             >
    decltype(auto) add_back(
        const Position& pos,
        const typename Batch::Value& w,
        size_t case_id,
        Batch& batch
    ) {
        return field::add_back(pos, w, batch.getFields()[case_id]);
    }

    template <typename Batch>
    requires std::is_same_v<
                host_batch_field<field::host_field<Batch::Dimension, field::var_loc::nodeCentered, typename Batch::Scalar, Batch::QuantitySize>>,
                std::remove_cvref_t<Batch>
             >
    decltype(auto) add_back(
        const double (&pos)[Batch::Dimension],
        const typename Batch::Value& w,
        size_t case_id,
        Batch& batch
    ) {
        return field::add_back(std::to_array(pos), w, batch.getFields()[case_id]);
    }

    template <typename Scalar, int QuantitySize = 1>
    using batch_host_cell_field1D = host_batch_field<field::host_cell_field1D<Scalar, QuantitySize>>;
    template <typename Scalar, int QuantitySize = 1>
    using batch_host_node_field1D = host_batch_field<field::host_node_field1D<Scalar, QuantitySize>>;
    template <typename Scalar, int QuantitySize = 1>
    using batch_host_cell_field2D = host_batch_field<field::host_cell_field2D<Scalar, QuantitySize>>;
    template <typename Scalar, int QuantitySize = 1>
    using batch_host_node_field2D = host_batch_field<field::host_node_field2D<Scalar, QuantitySize>>;
    template <typename Scalar, int QuantitySize = 1>
    using batch_host_cell_field3D = host_batch_field<field::host_cell_field3D<Scalar, QuantitySize>>;
    template <typename Scalar, int QuantitySize = 1>
    using batch_host_node_field3D = host_batch_field<field::host_node_field3D<Scalar, QuantitySize>>;
}

}

#endif
