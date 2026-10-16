#ifndef PSUM_BATCH_PIC_BATCH_FIELD_COPY_HPP
#define PSUM_BATCH_PIC_BATCH_FIELD_COPY_HPP

#include <vector>
#include <stdexcept>
#include <cstddef>
#include "host_batch_field.hpp"
#include "device_batch_field.hpp"
#include "../field/device_array.hpp"

namespace psum {

namespace batch_pic {


    // ---- host <-> device ----

    template <int Dim_, field::var_loc Loc_, typename Scalar, int QuantitySize>
    requires (QuantitySize == 1)
    void batch_copy(const host_batch_field<Dim_, Loc_, Scalar, QuantitySize>& from, device_batch_field<Dim_, Loc_, Scalar, QuantitySize>& to) {
        if (from.batch_size() != to.batch_size())
            throw std::runtime_error("batch field copy: batch size mismatch");
        auto q = to.getQueue();
        for (size_t c = 0; c < to.batch_size(); c++) {
            if (from.at(c).size() != to.at(c).size())
                throw std::runtime_error("batch field copy: field size mismatch");
            q.memcpy(to.at(c).data(), from.at(c).data(), to.at(c).size() * sizeof(Scalar));
        }
        q.wait();   // queues are in-order, one sync for the whole batch
    }

    template <int Dim_, field::var_loc Loc_, typename Scalar, int QuantitySize>
    requires (QuantitySize == 1)
    void batch_copy(const device_batch_field<Dim_, Loc_, Scalar, QuantitySize>& from, host_batch_field<Dim_, Loc_, Scalar, QuantitySize>& to) {
        if (from.batch_size() != to.batch_size())
            throw std::runtime_error("batch field copy: batch size mismatch");
        auto q = from.getQueue();
        for (size_t c = 0; c < from.batch_size(); c++) {
            if (from.at(c).size() != to.at(c).size())
                throw std::runtime_error("batch field copy: field size mismatch");
            q.memcpy(to.at(c).data(), from.at(c).data(), to.at(c).size() * sizeof(Scalar));
        }
        q.wait();   // queues are in-order, one sync for the whole batch
    }

    // conversions between a batch field (quantity size 1) and the linear arrangement used by the batch solver interfaces (row-major interleaved).
    // packing: batch field -> linear arrangement; [c][i] -> i * batch_size + c.
    // unpacking: linear arrangement -> batch field; i * batch_size + c -> [c][i].

    template <int Dim_, field::var_loc Loc_, typename Scalar, int QuantitySize>
    requires (QuantitySize == 1)
    void packing(const host_batch_field<Dim_, Loc_, Scalar, QuantitySize>& batch, std::vector<Scalar>& out) {
        const size_t C = batch.batch_size();
        const size_t n = C ? batch.at(0).size() : 0;
        if (out.size() != n * C)
            throw std::runtime_error("batch field packing: linear buffer size mismatch");
        Scalar* dst = out.data();
        for (size_t c = 0; c < C; c++) {
            const Scalar* src = reinterpret_cast<const Scalar*>(batch.at(c).data());
            for (size_t i = 0; i < n; i++)
                dst[i * C + c] = src[i];
        }
    }

    template <int Dim_, field::var_loc Loc_, typename Scalar, int QuantitySize>
    requires (QuantitySize == 1)
    void unpacking(const std::vector<Scalar>& in, host_batch_field<Dim_, Loc_, Scalar, QuantitySize>& batch) {
        const size_t C = batch.batch_size();
        const size_t n = C ? batch.at(0).size() : 0;
        if (in.size() != n * C)
            throw std::runtime_error("batch field unpacking: linear buffer size mismatch");
        const Scalar* src = in.data();
        for (size_t c = 0; c < C; c++) {
            Scalar* dst = reinterpret_cast<Scalar*>(batch.at(c).data());
            for (size_t i = 0; i < n; i++)
                dst[i] = src[i * C + c];
        }
    }

    // ---- device ----

    template <int Dim_, field::var_loc Loc_, typename Scalar, int QuantitySize>
    requires (QuantitySize == 1)
    void packing(device_batch_field<Dim_, Loc_, Scalar, QuantitySize>& batch, field::device_array<Scalar>& out) {
        const size_t C = batch.batch_size();
        const size_t n = C ? batch.at(0).size() : 0;
        if (out.size() != n * C)
            throw std::runtime_error("batch field packing: linear buffer size mismatch");
        if (n * C == 0)
            return;
        sycl::queue q = batch.getQueue();
        q.submit([&](sycl::handler& h) {
            auto bacc = batch.get_access(h);
            Scalar* dst = out.data();
            h.parallel_for(sycl::range<2>(n, C), [=](sycl::item<2> it) {
                const size_t i = it.get_id(0);
                const size_t c = it.get_id(1);
                dst[i * C + c] = bacc[c](i);
            });
        });
        q.wait();
    }

    template <int Dim_, field::var_loc Loc_, typename Scalar, int QuantitySize>
    requires (QuantitySize == 1)
    void unpacking(const field::device_array<Scalar>& in, device_batch_field<Dim_, Loc_, Scalar, QuantitySize>& batch) {
        const size_t C = batch.batch_size();
        const size_t n = C ? batch.at(0).size() : 0;
        if (in.size() != n * C)
            throw std::runtime_error("batch field unpacking: linear buffer size mismatch");
        if (n * C == 0)
            return;
        sycl::queue q = batch.getQueue();
        q.submit([&](sycl::handler& h) {
            auto bacc = batch.get_access(h);
            const Scalar* src = in.data();
            h.parallel_for(sycl::range<2>(n, C), [=](sycl::item<2> it) {
                const size_t i = it.get_id(0);
                const size_t c = it.get_id(1);
                bacc[c](i) = src[i * C + c];
            });
        });
        q.wait();
    }

}

}

#endif
