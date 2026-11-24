#ifndef PSUM_RUNTIME_FIELD_OPERATION_HPP
#define PSUM_RUNTIME_FIELD_OPERATION_HPP

// Numerical operations on fields: fill, add, copy, zero, read a node, turn a
// deposited charge into a density.
//
// These are algorithms performed ON field objects, not part of managing them:
// creating, querying and accessing fields is field_manager's job, and that is
// all this file uses of it — every function below reaches a field through the
// manager's dispatch (device window or host window) and touches nothing else.
// It is the field-side counterpart of particle_operation_dispatch.hpp.
//
// Each operation exists in a device and a host form because the two field
// types store their data differently (SYCL device memory vs. a plain vector);
// that difference lives here and nowhere else.

#include <psum/psum.hpp>
#include "field_manager.hpp"
#include <array>
#include <utility>

namespace psum_runtime {

    using namespace psum::prelude;

    template <typename Universe>
    class field_operation {
        field_manager<Universe>& fields_;
        using status = manager_status_of<200>;   // fields: the field segment

    public:
        explicit field_operation(field_manager<Universe>& fields) : fields_(fields) {}

        status fill_field(const char* name, double value) {
            if (!name) return st_bad_arg;
            return fields_.dispatch_field(name, [&](auto& f) {
                using F = std::remove_cvref_t<decltype(f)>;
                if constexpr (F::QuantitySize == 1)
                    f.setConstant(typename F::Scalar(value));
                else
                    f.setConstant(F::Value::Constant(typename F::Scalar(value)));
                return st_ok;
            });
        }

        status field_get_node(const char* name, const int* node_indices, double* value) {
            if (!name || !node_indices || !value) return st_bad_arg;
            return fields_.dispatch_field(name, [&](auto& f) {
                using F = std::remove_cvref_t<decltype(f)>;
                std::array<int, F::Dimension> idx;
                bool in_range = true;
                [&]<int... D>(std::integer_sequence<int, D...>) {
                    ((in_range = in_range && node_indices[D] >= 0 &&
                        node_indices[D] < f.getGrid().template numCells<D>() + 1), ...);
                }(std::make_integer_sequence<int, F::Dimension>{});
                if (!in_range) return st_bad_arg;
                for (int d = 0; d < F::Dimension; d++) idx[d] = node_indices[d];
                auto host = f.getContent().to_host();
                size_t linear = f.getGrid().n2i({idx});
                if constexpr (F::QuantitySize == 1)
                    *value = double(host[linear]);
                else
                    for (int c = 0; c < F::QuantitySize; c++)
                        value[c] = double(host[linear][c]);
                return st_ok;
            });
        }

        status field_shape(const char* name, int* dim, int* node_counts, int* quantity_size, int* scalar_kind) {
            if (!name || !dim || !node_counts || !quantity_size || !scalar_kind) return st_bad_arg;
            return fields_.dispatch_field(name, [&](auto& f) {
                using F = std::remove_cvref_t<decltype(f)>;
                *dim = F::Dimension;
                [&]<int... D>(std::integer_sequence<int, D...>) {
                    ((node_counts[D] = f.getGrid().template numCells<D>() + 1), ...);
                }(std::make_integer_sequence<int, F::Dimension>{});
                *quantity_size = F::QuantitySize;
                *scalar_kind = std::is_same_v<typename F::Scalar, float> ? 1 : 0;
                return st_ok;
            });
        }

        status host_field_fill(const char* name, double value) {
            if (!name) return st_bad_arg;
            return fields_.dispatch_host_field(name, [&](auto& f) {
                using F = std::remove_cvref_t<decltype(f)>;
                if constexpr (F::QuantitySize == 1)
                    f.setConstant(typename F::Scalar(value));
                else
                    f.setConstant(F::Value::Constant(typename F::Scalar(value)));
                return st_ok;
            });
        }

        status host_field_get_node(const char* name, const int* node_indices, double* value) {
            if (!name || !node_indices || !value) return st_bad_arg;
            return fields_.dispatch_host_field(name, [&](auto& f) {
                return host_get_node(f, node_indices, value);
            });
        }

        status host_field_shape(const char* name, int* dim, int* node_counts, int* quantity_size, int* scalar_kind) {
            if (!name || !dim || !node_counts || !quantity_size || !scalar_kind) return st_bad_arg;
            return fields_.dispatch_host_field(name, [&](auto& f) {
                using F = std::remove_cvref_t<decltype(f)>;
                *dim = F::Dimension;
                [&]<int... D>(std::integer_sequence<int, D...>) {
                    ((node_counts[D] = f.getGrid().template numCells<D>() + 1), ...);
                }(std::make_integer_sequence<int, F::Dimension>{});
                *quantity_size = F::QuantitySize;
                *scalar_kind = std::is_same_v<typename F::Scalar, float> ? 1 : 0;
                return st_ok;
            });
        }

        // zeroing (cheaper name for fill(0), kept as its own port)
        status zero_field(const char* name) {
            if (!name) return st_bad_arg;
            return fields_.dispatch_field(name, [&](auto& fld) { fld.setZero(); return st_ok; });
        }

        status host_zero_field(const char* name) {
            if (!name) return st_bad_arg;
            return fields_.dispatch_host_field(name, [&](auto& fld) {
                using F = std::remove_cvref_t<decltype(fld)>;
                if constexpr (F::QuantitySize == 1)
                    fld.setConstant(typename F::Scalar(0));
                else
                    fld.setConstant(F::Value::Constant(typename F::Scalar(0)));
                return st_ok;
            });
        }

        // deposited charge -> density: multiplies by scale / cell measure.
        status deposit_to_density(const char* name, int axisymmetric, double scale) {
            if (!name) return st_bad_arg;
            status r = fields_.dispatch_field(name, [&](auto& fld) {
                using F = std::remove_cvref_t<decltype(fld)>;
                return density_one<F, true>(fld, axisymmetric, scale);
            });
            if (r != st_missing) return r;   // not a device field -> try the host side
            return fields_.dispatch_host_field(name, [&](auto& fld) {
                using F = std::remove_cvref_t<decltype(fld)>;
                return density_one<F, false>(fld, axisymmetric, scale);
            });
        }

        // in-place += value on every node (the canonical use is adding
        // the neutralizing background charge after a density pass)
        status add_field(const char* name, double value) {
            if (!name) return st_bad_arg;
            return fields_.dispatch_field(name, [&](auto& fld) {
                using F = std::remove_cvref_t<decltype(fld)>;
                auto host = fld.getContent().to_host();
                for (auto& v : host) {
                    if constexpr (F::QuantitySize == 1)
                        v = typename F::Scalar(double(v) + value);
                    else
                        for (int c = 0; c < F::QuantitySize; c++)
                            v[c] = typename F::Scalar(double(v[c]) + value);
                }
                fld.copy(host);
                return st_ok;
            });
        }

        status host_add_field(const char* name, double value) {
            if (!name) return st_bad_arg;
            return fields_.dispatch_host_field(name, [&](auto& fld) {
                using F = std::remove_cvref_t<decltype(fld)>;
                auto host = fld.getContent();
                for (auto& v : host) {
                    if constexpr (F::QuantitySize == 1)
                        v = typename F::Scalar(double(v) + value);
                    else
                        for (int c = 0; c < F::QuantitySize; c++)
                            v[c] = typename F::Scalar(double(v[c]) + value);
                }
                fld.copy(host);
                return st_ok;
            });
        }

        // dst += src element-wise (both device fields; dims/QS must match)
        status add_field_from(const char* dst_name, const char* src_name) {
            if (!dst_name || !src_name) return st_bad_arg;
            return fields_.dispatch_field(dst_name, [&](auto& dst) {
                return fields_.dispatch_field(src_name, [&](auto& src) {
                    using D = std::remove_cvref_t<decltype(dst)>;
                    using S = std::remove_cvref_t<decltype(src)>;
                    if constexpr (D::Dimension == S::Dimension &&
                                    D::QuantitySize == S::QuantitySize) {
                        auto hs = src.getContent().to_host();
                        auto hd = dst.getContent().to_host();
                        for (size_t k = 0; k < hd.size(); k++) {
                            if constexpr (D::QuantitySize == 1)
                                hd[k] = typename D::Scalar(double(hd[k]) + double(hs[k]));
                            else
                                for (int c = 0; c < D::QuantitySize; c++)
                                    hd[k][c] = typename D::Scalar(double(hd[k][c]) + double(hs[k][c]));
                        }
                        dst.copy(hd);
                        return st_ok;
                    } else
                        return st_bad_arg;
                });
            });
        }

        // host twin; the source may live in EITHER window (host or
        // device) — the device fallback stages through the host
        status host_add_field_from(const char* dst_name, const char* src_name) {
            if (!dst_name || !src_name) return st_bad_arg;
            return fields_.dispatch_host_field(dst_name, [&](auto& dst) {
                status r = fields_.dispatch_host_field(src_name, [&](auto& src) {
                    using D = std::remove_cvref_t<decltype(dst)>;
                    using S = std::remove_cvref_t<decltype(src)>;
                    if constexpr (D::Dimension == S::Dimension &&
                                    D::QuantitySize == S::QuantitySize) {
                        auto hs = src.getContent();
                        auto hd = dst.getContent();
                        for (size_t k = 0; k < hd.size(); k++) {
                            if constexpr (D::QuantitySize == 1)
                                hd[k] = typename D::Scalar(double(hd[k]) + double(hs[k]));
                            else
                                for (int c = 0; c < D::QuantitySize; c++)
                                    hd[k][c] = typename D::Scalar(double(hd[k][c]) + double(hs[k][c]));
                        }
                        dst.copy(hd);
                        return st_ok;
                    } else
                        return st_bad_arg;
                });
                if (r != st_missing) return r;
                return fields_.dispatch_field(src_name, [&](auto& src) {
                    using D = std::remove_cvref_t<decltype(dst)>;
                    using S = std::remove_cvref_t<decltype(src)>;
                    if constexpr (D::Dimension == S::Dimension &&
                                    D::QuantitySize == S::QuantitySize) {
                        auto hs = src.getContent().to_host();
                        auto hd = dst.getContent();
                        for (size_t k = 0; k < hd.size(); k++) {
                            if constexpr (D::QuantitySize == 1)
                                hd[k] = typename D::Scalar(double(hd[k]) + double(hs[k]));
                            else
                                for (int c = 0; c < D::QuantitySize; c++)
                                    hd[k][c] = typename D::Scalar(double(hd[k][c]) + double(hs[k][c]));
                        }
                        dst.copy(hd);
                        return st_ok;
                    } else
                        return st_bad_arg;
                });
            });
        }

        status field_copy(const char* dst_name, const char* src_name) {
            if (!dst_name || !src_name) return st_bad_arg;
            return fields_.dispatch_field(dst_name, [&](auto& dst) {
                // the source may live in EITHER window (device or host)
                status r = fields_.dispatch_field(src_name, [&](auto& src) {
                    using D = std::remove_cvref_t<decltype(dst)>;
                    using S = std::remove_cvref_t<decltype(src)>;
                    if constexpr (D::Dimension == S::Dimension &&
                                    D::QuantitySize == S::QuantitySize) {
                        auto hs = src.getContent().to_host();
                        std::vector<typename D::Value> hd(hs.size());
                        for (size_t k = 0; k < hs.size(); k++) {
                            if constexpr (D::QuantitySize == 1)
                                hd[k] = typename D::Scalar(hs[k]);
                            else
                                for (int c = 0; c < D::QuantitySize; c++)
                                    hd[k][c] = typename D::Scalar(hs[k][c]);
                        }
                        dst.copy(hd);
                        return st_ok;
                    } else
                        return st_bad_arg;
                });
                if (r != st_missing) return r;
                return fields_.dispatch_host_field(src_name, [&](auto& src) {
                    using D = std::remove_cvref_t<decltype(dst)>;
                    using S = std::remove_cvref_t<decltype(src)>;
                    if constexpr (D::Dimension == S::Dimension &&
                                    D::QuantitySize == S::QuantitySize) {
                        auto hs = src.getContent();
                        std::vector<typename D::Value> hd(hs.size());
                        for (size_t k = 0; k < hs.size(); k++) {
                            if constexpr (D::QuantitySize == 1)
                                hd[k] = typename D::Scalar(hs[k]);
                            else
                                for (int c = 0; c < D::QuantitySize; c++)
                                    hd[k][c] = typename D::Scalar(hs[k][c]);
                        }
                        dst.copy(hd);
                        return st_ok;
                    } else
                        return st_bad_arg;
                });
            });
        }

        status host_field_copy(const char* dst_name, const char* src_name) {
            if (!dst_name || !src_name) return st_bad_arg;
            return fields_.dispatch_host_field(dst_name, [&](auto& dst) {
                // the source may live in EITHER window (host or device)
                status r = fields_.dispatch_host_field(src_name, [&](auto& src) {
                    using D = std::remove_cvref_t<decltype(dst)>;
                    using S = std::remove_cvref_t<decltype(src)>;
                    if constexpr (D::Dimension == S::Dimension &&
                                    D::QuantitySize == S::QuantitySize) {
                        auto hs = src.getContent();
                        std::vector<typename D::Value> hd(hs.size());
                        for (size_t k = 0; k < hs.size(); k++) {
                            if constexpr (D::QuantitySize == 1)
                                hd[k] = typename D::Scalar(hs[k]);
                            else
                                for (int c = 0; c < D::QuantitySize; c++)
                                    hd[k][c] = typename D::Scalar(hs[k][c]);
                        }
                        dst.copy(hd);
                        return st_ok;
                    } else
                        return st_bad_arg;
                });
                if (r != st_missing) return r;
                return fields_.dispatch_field(src_name, [&](auto& src) {
                    using D = std::remove_cvref_t<decltype(dst)>;
                    using S = std::remove_cvref_t<decltype(src)>;
                    if constexpr (D::Dimension == S::Dimension &&
                                    D::QuantitySize == S::QuantitySize) {
                        auto hs = src.getContent().to_host();
                        std::vector<typename D::Value> hd(hs.size());
                        for (size_t k = 0; k < hs.size(); k++) {
                            if constexpr (D::QuantitySize == 1)
                                hd[k] = typename D::Scalar(hs[k]);
                            else
                                for (int c = 0; c < D::QuantitySize; c++)
                                    hd[k][c] = typename D::Scalar(hs[k][c]);
                        }
                        dst.copy(hd);
                        return st_ok;
                    } else
                        return st_bad_arg;
                });
            });
        }

        template <typename FieldT, bool OnDevice>
        static void density_scale_uniform(FieldT& fld, double ff) {
            if constexpr (OnDevice) {
                fld.for_each([ff](sycl::handler& h) {
                    return [ff](size_t, typename FieldT::Value& v) {
                        v = typename FieldT::Scalar(double(v) * ff);
                    };
                });
            } else {
                fld.for_each([ff](size_t, typename FieldT::Value& v) {
                    v = typename FieldT::Scalar(double(v) * ff);
                });
            }
        }

        template <typename FieldT, bool OnDevice>
        static manager_status density_one(FieldT& fld, int axisymmetric, double scale) {
            if constexpr (FieldT::QuantitySize != 1) {
                return st_bad_arg;   // non-scalar fields cannot hold a density
            } else {
                if (axisymmetric && FieldT::Dimension != 2) return st_bad_arg;   // the axisymmetric density needs a 2d field
                auto& g = fld.getGrid();
                if constexpr (FieldT::Dimension == 1) {
                    density_scale_uniform<FieldT, OnDevice>(fld, scale / g.template del<0>());
                } else if constexpr (FieldT::Dimension == 2) {
                    if (axisymmetric) {
                        const double d0 = g.template del<0>(), d1 = g.template del<1>();
                        if constexpr (OnDevice) {
                            fld.for_each([d0, d1, scale](sycl::handler& h) {
                                return [d0, d1, scale](size_t, typename FieldT::Value& v,
                                                       const auto& pos) {
                                    const double r1 = pos[1] > 0.5 * d1 ? pos[1] - 0.5 * d1 : 0.0;
                                    const double r2 = pos[1] + 0.5 * d1;
                                    v = typename FieldT::Scalar(
                                        double(v) * scale / (d0 * M_PI * (r2 * r2 - r1 * r1)));
                                };
                            });
                        } else {
                            fld.for_each([d0, d1, scale](size_t, typename FieldT::Value& v,
                                                         const auto& pos) {
                                const double r1 = pos[1] > 0.5 * d1 ? pos[1] - 0.5 * d1 : 0.0;
                                const double r2 = pos[1] + 0.5 * d1;
                                v = typename FieldT::Scalar(
                                    double(v) * scale / (d0 * M_PI * (r2 * r2 - r1 * r1)));
                            });
                        }
                    } else {
                        density_scale_uniform<FieldT, OnDevice>(
                            fld, scale / (g.template del<0>() * g.template del<1>()));
                    }
                } else {
                    density_scale_uniform<FieldT, OnDevice>(
                        fld, scale / (g.template del<0>() * g.template del<1>() * g.template del<2>()));
                }
                return st_ok;
            }
        }

        template <typename F>
        static manager_status host_get_node(F& f, const int* idx, double* value) {
            constexpr int Dim = F::Dimension;
            bool in_range = true;
            [&]<int... D>(std::integer_sequence<int, D...>) {
                ((in_range = in_range && idx[D] >= 0 &&
                    idx[D] < f.getGrid().template numCells<D>() + 1), ...);
            }(std::make_integer_sequence<int, Dim>{});
            if (!in_range) return st_bad_arg;
            std::array<int, Dim> a;
            for (int d = 0; d < Dim; d++) a[d] = idx[d];
            size_t linear = f.getGrid().n2i({a});
            if constexpr (F::QuantitySize == 1)
                *value = double(f.data()[linear]);
            else
                for (int c = 0; c < F::QuantitySize; c++)
                    value[c] = double(f.data()[linear][c]);
            return st_ok;
        }    };

} // namespace psum_runtime

#endif
