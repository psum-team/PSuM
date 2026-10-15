#ifndef PSUM_BATCH_PIC_DEVICE_BATCH_FIELD_HPP
#define PSUM_BATCH_PIC_DEVICE_BATCH_FIELD_HPP

#include <vector>
#include <stdexcept>
#include <cstdio>
#include "../field/device_field.hpp"

namespace psum {

namespace batch_pic {

    namespace foundation {

        // this class is only used to support the usage of `f.getContent().to_host()` for device_batch_field.
        template <typename Field>
        class device_batch_content {
        public:
            using Value = typename Field::Value;

            explicit device_batch_content(const std::vector<const field::device_array<Value>*>& arrays)
                : arrays_(arrays) {}

            size_t batch_size() const { return arrays_.size(); }

            std::vector<std::vector<Value>> to_host() const {
                std::vector<std::vector<Value>> out;
                out.reserve(batch_size());
                for (auto* a : arrays_)
                    out.push_back(a->to_host());
                return out;
            }

        private:
            std::vector<const field::device_array<Value>*> arrays_;
        };
    }

    // A batch field is an array of device fields with same grid and queue.
    template <int Dim_, field::var_loc Loc_, typename ScalarT, int QuantitySizeT = 1>
    class device_batch_field {
    public:
        using Field = field::device_field<Dim_, Loc_, ScalarT, QuantitySizeT>;
        static constexpr int Dimension = Dim_;
        static constexpr field::var_loc Location = Loc_;
        static constexpr int QuantitySize = QuantitySizeT;
        using Scalar = ScalarT;
        using Value = typename Field::Value;
        using Grid = typename Field::Grid;
        using FieldAcc = typename Field::acc_type;
        using acc_type = field::device_array_acc<FieldAcc>;

        device_batch_field(const sycl::queue& q, const Grid& grid, size_t case_num)
            : fields(), acc_table(q, case_num) {
            fields.reserve(case_num);
            for (size_t c = 0; c < case_num; c++) {
                fields.emplace_back(q, grid);
            }
            for (auto& f : fields) established_ptrs.push_back(f.data());

            std::vector<FieldAcc> host_accs;
            host_accs.reserve(fields.size());
            for (auto& f : fields) {
                host_accs.emplace_back(field::device_array_acc<Value>(f.data(), f.size()), f.getGrid());
            }
            acc_table.copy(host_accs);
        }

        acc_type get_access(sycl::handler& h) {
            // establish the invariant for get_access.
            // this loop does not make any effect (in usual situations)
            for (size_t c = 0; c < fields.size(); c++) {
                if (fields[c].data() != established_ptrs[c]) {
                    throw std::runtime_error(
                        "device_batch_field: per-case buffer address changed after construction; "
                        "device_batch_field requires stable exclusive field buffers");
                }
            }
            return acc_table.get_access(h);
        }

        size_t batch_size() const { return fields.size(); }

        Field& at(size_t c) { return fields.at(c); }
        const Field& at(size_t c) const { return fields.at(c); }

        const Grid& getGrid() const { return fields.front().getGrid(); }
        field::var_loc getLocation() const { return Location; }
        sycl::queue getQueue() const { return fields.front().getQueue(); }

        auto getContent() {
            std::vector<const field::device_array<Value>*> arrays;
            for (auto& f : fields) arrays.push_back(&f.getContent());
            return foundation::device_batch_content<Field>(arrays);
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
                throw std::runtime_error("device_batch_field::copy: batch size mismatch");
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
        // kernel-side table of the established per-case accessors
        field::device_array<FieldAcc> acc_table;
        // host-side record of the established accessor contents (the data
        // pointers); the invariance every get_access must confirm
        std::vector<Value*> established_ptrs;
    };

    template <typename T>
    struct is_device_field_acc : std::false_type {};
    template <int D, field::var_loc L, typename S, typename V, int Q>
    struct is_device_field_acc<field::device_field_acc<D, L, S, V, Q>> : std::true_type {};

    template <typename BatchAcc>
    concept device_batch_acc = requires(BatchAcc acc, size_t c) {
            typename BatchAcc::value_type;
            { acc[c] } -> std::convertible_to<typename BatchAcc::value_type&>;
        } && is_device_field_acc<typename BatchAcc::value_type>::value;

    // interp/add_back for batch fields.
    template <typename Position, typename BatchAcc>
    requires device_batch_acc<BatchAcc> &&
             field::foundation::array_like<Position, BatchAcc::value_type::Dimension>
    decltype(auto) interp(
        const Position& pos,
        size_t case_id,
        const BatchAcc& batch_acc
    ) {
        return field::interp(pos, batch_acc[case_id]);
    }

    template <typename BatchAcc>
    requires device_batch_acc<BatchAcc>
    decltype(auto) interp(
        const double (&pos)[BatchAcc::value_type::Dimension],
        size_t case_id,
        const BatchAcc& batch_acc
    ) {
        return field::interp(std::to_array(pos), batch_acc[case_id]);
    }

    template <typename Position, typename BatchAcc>
    requires device_batch_acc<BatchAcc> &&
             field::foundation::array_like<Position, BatchAcc::value_type::Dimension>
    decltype(auto) interp_diff(
        const Position& pos,
        size_t case_id,
        const BatchAcc& batch_acc
    ) {
        return field::interp_diff(pos, batch_acc[case_id]);
    }

    template <typename BatchAcc>
    requires device_batch_acc<BatchAcc>
    decltype(auto) interp_diff(
        const double (&pos)[BatchAcc::value_type::Dimension],
        size_t case_id,
        const BatchAcc& batch_acc
    ) {
        return field::interp_diff(std::to_array(pos), batch_acc[case_id]);
    }

    template <typename Position, typename BatchAcc>
    requires device_batch_acc<BatchAcc> &&
             field::foundation::array_like<Position, BatchAcc::value_type::Dimension>
    decltype(auto) add_back(
        const Position& pos,
        const typename BatchAcc::value_type::Value& w,
        size_t case_id,
        const BatchAcc& batch_acc
    ) {
        return field::add_back(pos, w, batch_acc[case_id]);
    }

    template <typename BatchAcc>
    requires device_batch_acc<BatchAcc>
    decltype(auto) add_back(
        const double (&pos)[BatchAcc::value_type::Dimension],
        const typename BatchAcc::value_type::Value& w,
        size_t case_id,
        const BatchAcc& batch_acc
    ) {
        return field::add_back(std::to_array(pos), w, batch_acc[case_id]);
    }

    template <typename Scalar, int QuantitySize = 1>
    using batch_cell_field1D = device_batch_field<1, field::var_loc::cellCentered, Scalar, QuantitySize>;
    template <typename Scalar, int QuantitySize = 1>
    using batch_node_field1D = device_batch_field<1, field::var_loc::nodeCentered, Scalar, QuantitySize>;
    template <typename Scalar, int QuantitySize = 1>
    using batch_cell_field2D = device_batch_field<2, field::var_loc::cellCentered, Scalar, QuantitySize>;
    template <typename Scalar, int QuantitySize = 1>
    using batch_node_field2D = device_batch_field<2, field::var_loc::nodeCentered, Scalar, QuantitySize>;
    template <typename Scalar, int QuantitySize = 1>
    using batch_cell_field3D = device_batch_field<3, field::var_loc::cellCentered, Scalar, QuantitySize>;
    template <typename Scalar, int QuantitySize = 1>
    using batch_node_field3D = device_batch_field<3, field::var_loc::nodeCentered, Scalar, QuantitySize>;
}

}

#endif
