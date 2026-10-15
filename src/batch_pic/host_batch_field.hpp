#ifndef PSUM_BATCH_PIC_HOST_BATCH_FIELD_HPP
#define PSUM_BATCH_PIC_HOST_BATCH_FIELD_HPP

#include <vector>
#include <type_traits>
#include "../field/host_field.hpp"
#include <cstdio>

namespace psum {

namespace batch_pic {

    // A batch field is an array of host fields with same grid and queue.
    template <int Dim_, field::var_loc Loc_, typename ScalarT, int QuantitySizeT = 1>
    class host_batch_field {
    public:
        using Field = field::host_field<Dim_, Loc_, ScalarT, QuantitySizeT>;
        static constexpr int Dimension = Dim_;
        static constexpr field::var_loc Location = Loc_;
        static constexpr int QuantitySize = QuantitySizeT;
        using Scalar = ScalarT;
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

    template <typename T>
    struct is_host_batch_field : std::false_type {};
    template <int D, field::var_loc L, typename S, int Q>
    struct is_host_batch_field<host_batch_field<D, L, S, Q>> : std::true_type {};
    template <typename Batch>
    concept host_batch_field_type = is_host_batch_field<std::remove_cvref_t<Batch>>::value;

    // interp/add_back for batch fields.
    template <typename Position, typename Batch>
    requires field::foundation::array_like<Position, Batch::Dimension> && host_batch_field_type<Batch>
    decltype(auto) interp(
        const Position& pos,
        size_t case_id,
        const Batch& batch
    ) {
        return field::interp(pos, batch.getFields()[case_id]);
    }

    template <typename Batch>
    requires host_batch_field_type<Batch>
    decltype(auto) interp(
        const double (&pos)[Batch::Dimension],
        size_t case_id,
        const Batch& batch
    ) {
        return field::interp(std::to_array(pos), batch.getFields()[case_id]);
    }

    template <typename Position, typename Batch>
    requires field::foundation::array_like<Position, Batch::Dimension> && host_batch_field_type<Batch>
    decltype(auto) interp_diff(
        const Position& pos,
        size_t case_id,
        const Batch& batch
    ) {
        return field::interp_diff(pos, batch.getFields()[case_id]);
    }

    template <typename Batch>
    requires host_batch_field_type<Batch>
    decltype(auto) interp_diff(
        const double (&pos)[Batch::Dimension],
        size_t case_id,
        const Batch& batch
    ) {
        return field::interp_diff(std::to_array(pos), batch.getFields()[case_id]);
    }

    template <typename Position, typename Batch>
    requires field::foundation::array_like<Position, Batch::Dimension> && host_batch_field_type<Batch>
    decltype(auto) add_back(
        const Position& pos,
        const typename Batch::Value& w,
        size_t case_id,
        Batch& batch
    ) {
        return field::add_back(pos, w, batch.getFields()[case_id]);
    }

    template <typename Batch>
    requires host_batch_field_type<Batch>
    decltype(auto) add_back(
        const double (&pos)[Batch::Dimension],
        const typename Batch::Value& w,
        size_t case_id,
        Batch& batch
    ) {
        return field::add_back(std::to_array(pos), w, batch.getFields()[case_id]);
    }

    template <typename Scalar, int QuantitySize = 1>
    using batch_host_cell_field1D = host_batch_field<1, field::var_loc::cellCentered, Scalar, QuantitySize>;
    template <typename Scalar, int QuantitySize = 1>
    using batch_host_node_field1D = host_batch_field<1, field::var_loc::nodeCentered, Scalar, QuantitySize>;
    template <typename Scalar, int QuantitySize = 1>
    using batch_host_cell_field2D = host_batch_field<2, field::var_loc::cellCentered, Scalar, QuantitySize>;
    template <typename Scalar, int QuantitySize = 1>
    using batch_host_node_field2D = host_batch_field<2, field::var_loc::nodeCentered, Scalar, QuantitySize>;
    template <typename Scalar, int QuantitySize = 1>
    using batch_host_cell_field3D = host_batch_field<3, field::var_loc::cellCentered, Scalar, QuantitySize>;
    template <typename Scalar, int QuantitySize = 1>
    using batch_host_node_field3D = host_batch_field<3, field::var_loc::nodeCentered, Scalar, QuantitySize>;
}

}

#endif
