#ifndef PSUM_RUNTIME_FIELD_MANAGER_HPP
#define PSUM_RUNTIME_FIELD_MANAGER_HPP

// The grid/field part of the engine

#include <psum/psum.hpp>

namespace psum_runtime {

    using namespace psum::prelude;

    // field_manager is parameterized by the universe
    // Universe::enable_float_fields will determine if float fields are enabled.
    template <typename Universe>
    class field_manager {
    public:
        field_manager(serialization::object_manager& mng, sycl::queue& q) : mng_(mng), q_(q) {}

        template <int Dim>
        using grid_t = simple_grid<Dim>;
        template <int Dim, int QuantitySize, typename Scalar = double>
        using field_t = psum::field::device_field<Dim, psum::field::var_loc::nodeCentered, Scalar, QuantitySize>;
        template <int Dim, int QuantitySize, typename Scalar = double>
        using host_field_t = host_field<Dim, psum::field::var_loc::nodeCentered, Scalar, QuantitySize>;

        int create_grid(const char* name, int dim, const double* lower, const double* upper, const int* n) {
            if (!name || !lower || !upper || !n || dim < 1 || dim > 3) return -1;
            std::vector<double> lo(lower, lower + dim), up(upper, upper + dim);
            std::vector<int> size(n, n + dim);
            switch (dim) {
            case 1: mng_.template obj<grid_t<1>>(name) = grid_t<1>(lo, up, size); return 0;
            case 2: mng_.template obj<grid_t<2>>(name) = grid_t<2>(lo, up, size); return 0;
            case 3: mng_.template obj<grid_t<3>>(name) = grid_t<3>(lo, up, size); return 0;
            default: return -1;
            }
        }

        int create_field(const char* name, const char* grid_name, int quantity_size, int scalar_kind = 0) {
            if (!name || !grid_name) return -1;
            if (scalar_kind != 0 && scalar_kind != 1) return -1;
            if (scalar_kind == 1 && !Universe::enable_float_fields) return -1;
            if (mng_.template contains<grid_t<1>>(grid_name))
                return create_field_on<1>(name, mng_.template obj<grid_t<1>>(grid_name), quantity_size, scalar_kind);
            if (mng_.template contains<grid_t<2>>(grid_name))
                return create_field_on<2>(name, mng_.template obj<grid_t<2>>(grid_name), quantity_size, scalar_kind);
            if (mng_.template contains<grid_t<3>>(grid_name))
                return create_field_on<3>(name, mng_.template obj<grid_t<3>>(grid_name), quantity_size, scalar_kind);
            return -1;   // no such grid
        }

        int fill_field(const char* name, double value) {
            if (!name) return -1;
            return dispatch_field(name, [&](auto& f) {
                using F = std::remove_cvref_t<decltype(f)>;
                if constexpr (F::QuantitySize == 1)
                    f.setConstant(typename F::Scalar(value));
                else
                    f.setConstant(F::Value::Constant(typename F::Scalar(value)));
                return 0;
            });
        }

        // node value(s) on the host as doubles (indices has one entry per
        // grid dim, range-checked; value receives QuantitySize components)
        int field_get_node(const char* name, const int* node_indices, double* value) {
            if (!name || !node_indices || !value) return -1;
            return dispatch_field(name, [&](auto& f) {
                using F = std::remove_cvref_t<decltype(f)>;
                std::array<int, F::Dimension> idx;
                bool in_range = true;
                [&]<int... D>(std::integer_sequence<int, D...>) {
                    ((in_range = in_range && node_indices[D] >= 0 &&
                        node_indices[D] < f.getGrid().template numCells<D>() + 1), ...);
                }(std::make_integer_sequence<int, F::Dimension>{});
                if (!in_range) return -1;
                for (int d = 0; d < F::Dimension; d++) idx[d] = node_indices[d];
                auto host = f.getContent().to_host();
                size_t linear = f.getGrid().n2i({idx});
                if constexpr (F::QuantitySize == 1)
                    *value = double(host[linear]);
                else
                    for (int c = 0; c < F::QuantitySize; c++)
                        value[c] = double(host[linear][c]);
                return 0;
            });
        }

        // node_counts has room for 3 ints (only dim of them are written)
        int field_shape(const char* name, int* dim, int* node_counts, int* quantity_size, int* scalar_kind) {
            if (!name || !dim || !node_counts || !quantity_size || !scalar_kind) return -1;
            return dispatch_field(name, [&](auto& f) {
                using F = std::remove_cvref_t<decltype(f)>;
                *dim = F::Dimension;
                [&]<int... D>(std::integer_sequence<int, D...>) {
                    ((node_counts[D] = f.getGrid().template numCells<D>() + 1), ...);
                }(std::make_integer_sequence<int, F::Dimension>{});
                *quantity_size = F::QuantitySize;
                *scalar_kind = std::is_same_v<typename F::Scalar, float> ? 1 : 0;
                return 0;
            });
        }

        // ---------- host side: mirrors the device window ----------
        int host_field_create(const char* name, const char* grid_name, int qs, int scalar_kind = 0) {
            if (scalar_kind == 1 && !Universe::enable_float_fields) return -1;
            if (!name || !grid_name || (scalar_kind != 0 && scalar_kind != 1)) return -1;
            if (mng_.template contains<grid_t<1>>(grid_name))
                return host_field_create_on<1>(name, mng_.template obj<grid_t<1>>(grid_name), qs, scalar_kind);
            if (mng_.template contains<grid_t<2>>(grid_name))
                return host_field_create_on<2>(name, mng_.template obj<grid_t<2>>(grid_name), qs, scalar_kind);
            if (mng_.template contains<grid_t<3>>(grid_name))
                return host_field_create_on<3>(name, mng_.template obj<grid_t<3>>(grid_name), qs, scalar_kind);
            return -1;
        }

        int host_field_fill(const char* name, double value) {
            if (!name) return -1;
            return dispatch_host_field(name, [&](auto& f) {
                using F = std::remove_cvref_t<decltype(f)>;
                if constexpr (F::QuantitySize == 1)
                    f.setConstant(typename F::Scalar(value));
                else
                    f.setConstant(F::Value::Constant(typename F::Scalar(value)));
                return 0;
            });
        }

        int host_field_get_node(const char* name, const int* node_indices, double* value) {
            if (!name || !node_indices || !value) return -1;
            return dispatch_host_field(name, [&](auto& f) {
                return host_get_node(f, node_indices, value);
            });
        }

        int host_field_shape(const char* name, int* dim, int* node_counts, int* quantity_size, int* scalar_kind) {
            if (!name || !dim || !node_counts || !quantity_size || !scalar_kind) return -1;
            return dispatch_host_field(name, [&](auto& f) {
                using F = std::remove_cvref_t<decltype(f)>;
                *dim = F::Dimension;
                [&]<int... D>(std::integer_sequence<int, D...>) {
                    ((node_counts[D] = f.getGrid().template numCells<D>() + 1), ...);
                }(std::make_integer_sequence<int, F::Dimension>{});
                *quantity_size = F::QuantitySize;
                *scalar_kind = std::is_same_v<typename F::Scalar, float> ? 1 : 0;
                return 0;
            });
        }

        // zeroing (cheaper name for fill(0), kept as its own port)
        int zero_field(const char* name) {
            if (!name) return -1;
            return dispatch_field(name, [&](auto& fld) { fld.setZero(); return 0; });
        }
        int host_zero_field(const char* name) {
            if (!name) return -1;
            return dispatch_host_field(name, [&](auto& fld) {
                using F = std::remove_cvref_t<decltype(fld)>;
                if constexpr (F::QuantitySize == 1)
                    fld.setConstant(typename F::Scalar(0));
                else
                    fld.setConstant(F::Value::Constant(typename F::Scalar(0)));
                return 0;
            });
        }

        // deposited charge -> density: multiplies by scale / cell measure.
        int deposit_to_density(const char* name, int axisymmetric, double scale) {
            if (!name) return -1;
            int r = dispatch_field(name, [&](auto& fld) {
                using F = std::remove_cvref_t<decltype(fld)>;
                return density_one<F, true>(fld, axisymmetric, scale);
            });
            if (r != -1) return r;   // not a device field -> try the host side
            return dispatch_host_field(name, [&](auto& fld) {
                using F = std::remove_cvref_t<decltype(fld)>;
                return density_one<F, false>(fld, axisymmetric, scale);
            });
        }

        // ---------- copies: same grid dim, same QS, same side, scalar
        // conversion in either direction (double <-> float) ----------
        // in-place += value on every node (the canonical use is adding
        // the neutralizing background charge after a density pass)
        int add_field(const char* name, double value) {
            if (!name) return -1;
            return dispatch_field(name, [&](auto& fld) {
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
                return 0;
            });
        }

        int host_add_field(const char* name, double value) {
            if (!name) return -1;
            return dispatch_host_field(name, [&](auto& fld) {
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
                return 0;
            });
        }

        // dst += src element-wise (both device fields; dims/QS must match)
        int add_field_from(const char* dst_name, const char* src_name) {
            if (!dst_name || !src_name) return -1;
            return dispatch_field(dst_name, [&](auto& dst) {
                return dispatch_field(src_name, [&](auto& src) {
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
                        return 0;
                    } else
                        return -1;
                });
            });
        }

        // host twin; the source may live in EITHER window (host or
        // device) — the device fallback stages through the host
        int host_add_field_from(const char* dst_name, const char* src_name) {
            if (!dst_name || !src_name) return -1;
            return dispatch_host_field(dst_name, [&](auto& dst) {
                int r = dispatch_host_field(src_name, [&](auto& src) {
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
                        return 0;
                    } else
                        return -1;
                });
                if (r != -1) return r;
                return dispatch_field(src_name, [&](auto& src) {
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
                        return 0;
                    } else
                        return -1;
                });
            });
        }

        int field_copy(const char* dst_name, const char* src_name) {
            if (!dst_name || !src_name) return -1;
            return dispatch_field(dst_name, [&](auto& dst) {
                // the source may live in EITHER window (device or host)
                int r = dispatch_field(src_name, [&](auto& src) {
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
                        return 0;
                    } else
                        return -1;
                });
                if (r != -1) return r;
                return dispatch_host_field(src_name, [&](auto& src) {
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
                        return 0;
                    } else
                        return -1;
                });
            });
        }

        int host_field_copy(const char* dst_name, const char* src_name) {
            if (!dst_name || !src_name) return -1;
            return dispatch_host_field(dst_name, [&](auto& dst) {
                // the source may live in EITHER window (host or device)
                int r = dispatch_host_field(src_name, [&](auto& src) {
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
                        return 0;
                    } else
                        return -1;
                });
                if (r != -1) return r;
                return dispatch_field(src_name, [&](auto& src) {
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
                        return 0;
                    } else
                        return -1;
                });
            });
        }

        template <typename F>
        int dispatch_field(const char* name, F&& fn) {
            if (mng_.template contains<field_t<1, 1, double>>(name)) return fn(mng_.template obj<field_t<1, 1, double>>(name));
            if (mng_.template contains<field_t<2, 1, double>>(name)) return fn(mng_.template obj<field_t<2, 1, double>>(name));
            if (mng_.template contains<field_t<2, 2, double>>(name)) return fn(mng_.template obj<field_t<2, 2, double>>(name));
            if (mng_.template contains<field_t<2, 3, double>>(name)) return fn(mng_.template obj<field_t<2, 3, double>>(name));
            if (mng_.template contains<field_t<3, 1, double>>(name)) return fn(mng_.template obj<field_t<3, 1, double>>(name));
            if (mng_.template contains<field_t<3, 2, double>>(name)) return fn(mng_.template obj<field_t<3, 2, double>>(name));
            if (mng_.template contains<field_t<3, 3, double>>(name)) return fn(mng_.template obj<field_t<3, 3, double>>(name));
            if constexpr (Universe::enable_float_fields) {
                if (mng_.template contains<field_t<1, 1, float>>(name)) return fn(mng_.template obj<field_t<1, 1, float>>(name));
                if (mng_.template contains<field_t<2, 1, float>>(name)) return fn(mng_.template obj<field_t<2, 1, float>>(name));
                if (mng_.template contains<field_t<2, 2, float>>(name)) return fn(mng_.template obj<field_t<2, 2, float>>(name));
                if (mng_.template contains<field_t<2, 3, float>>(name)) return fn(mng_.template obj<field_t<2, 3, float>>(name));
                if (mng_.template contains<field_t<3, 1, float>>(name)) return fn(mng_.template obj<field_t<3, 1, float>>(name));
                if (mng_.template contains<field_t<3, 2, float>>(name)) return fn(mng_.template obj<field_t<3, 2, float>>(name));
                if (mng_.template contains<field_t<3, 3, float>>(name)) return fn(mng_.template obj<field_t<3, 3, float>>(name));
            }
            return -1;
        }

        template <typename F>
        int dispatch_host_field(const char* name, F&& fn) {
            if (mng_.template contains<host_field_t<1, 1, double>>(name)) return fn(mng_.template obj<host_field_t<1, 1, double>>(name));
            if (mng_.template contains<host_field_t<2, 1, double>>(name)) return fn(mng_.template obj<host_field_t<2, 1, double>>(name));
            if (mng_.template contains<host_field_t<2, 2, double>>(name)) return fn(mng_.template obj<host_field_t<2, 2, double>>(name));
            if (mng_.template contains<host_field_t<2, 3, double>>(name)) return fn(mng_.template obj<host_field_t<2, 3, double>>(name));
            if (mng_.template contains<host_field_t<3, 1, double>>(name)) return fn(mng_.template obj<host_field_t<3, 1, double>>(name));
            if (mng_.template contains<host_field_t<3, 2, double>>(name)) return fn(mng_.template obj<host_field_t<3, 2, double>>(name));
            if (mng_.template contains<host_field_t<3, 3, double>>(name)) return fn(mng_.template obj<host_field_t<3, 3, double>>(name));
            if constexpr (Universe::enable_float_fields) {
                if (mng_.template contains<host_field_t<1, 1, float>>(name)) return fn(mng_.template obj<host_field_t<1, 1, float>>(name));
                if (mng_.template contains<host_field_t<2, 1, float>>(name)) return fn(mng_.template obj<host_field_t<2, 1, float>>(name));
                if (mng_.template contains<host_field_t<2, 2, float>>(name)) return fn(mng_.template obj<host_field_t<2, 2, float>>(name));
                if (mng_.template contains<host_field_t<2, 3, float>>(name)) return fn(mng_.template obj<host_field_t<2, 3, float>>(name));
                if (mng_.template contains<host_field_t<3, 1, float>>(name)) return fn(mng_.template obj<host_field_t<3, 1, float>>(name));
                if (mng_.template contains<host_field_t<3, 2, float>>(name)) return fn(mng_.template obj<host_field_t<3, 2, float>>(name));
                if (mng_.template contains<host_field_t<3, 3, float>>(name)) return fn(mng_.template obj<host_field_t<3, 3, float>>(name));
            }
            return -1;
        }

    private:
        serialization::object_manager& mng_;
        sycl::queue& q_;

        template <int Dim>
        int create_field_on(const char* name, grid_t<Dim>& g, int qs, int scalar_kind) {
            if (qs < 1 || qs > 3 || (Dim == 1 && qs != 1)) return -1;
            if (scalar_kind == 0) {
                switch (qs) {
                case 1: mng_.template obj<field_t<Dim, 1, double>>(name) = field_t<Dim, 1, double>(q_, g); return 0;
                case 2: mng_.template obj<field_t<Dim, 2, double>>(name) = field_t<Dim, 2, double>(q_, g); return 0;
                case 3: mng_.template obj<field_t<Dim, 3, double>>(name) = field_t<Dim, 3, double>(q_, g); return 0;
                }
            } else if constexpr (Universe::enable_float_fields) {
                switch (qs) {
                case 1: mng_.template obj<field_t<Dim, 1, float>>(name) = field_t<Dim, 1, float>(q_, g); return 0;
                case 2: mng_.template obj<field_t<Dim, 2, float>>(name) = field_t<Dim, 2, float>(q_, g); return 0;
                case 3: mng_.template obj<field_t<Dim, 3, float>>(name) = field_t<Dim, 3, float>(q_, g); return 0;
                }
            }
            return -1;
        }

        template <int Dim>
        int host_field_create_on(const char* name, grid_t<Dim>& g, int qs, int scalar_kind) {
            if (qs < 1 || qs > 3 || (Dim == 1 && qs != 1)) return -1;
            if (scalar_kind == 0) {
                switch (qs) {
                case 1: mng_.template obj<host_field_t<Dim, 1, double>>(name) = host_field_t<Dim, 1, double>(g); return 0;
                case 2: mng_.template obj<host_field_t<Dim, 2, double>>(name) = host_field_t<Dim, 2, double>(g); return 0;
                case 3: mng_.template obj<host_field_t<Dim, 3, double>>(name) = host_field_t<Dim, 3, double>(g); return 0;
                }
            } else if constexpr (Universe::enable_float_fields) {
                switch (qs) {
                case 1: mng_.template obj<host_field_t<Dim, 1, float>>(name) = host_field_t<Dim, 1, float>(g); return 0;
                case 2: mng_.template obj<host_field_t<Dim, 2, float>>(name) = host_field_t<Dim, 2, float>(g); return 0;
                case 3: mng_.template obj<host_field_t<Dim, 3, float>>(name) = host_field_t<Dim, 3, float>(g); return 0;
                }
            }
            return -1;
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
        static int density_one(FieldT& fld, int axisymmetric, double scale) {
            if constexpr (FieldT::QuantitySize != 1) {
                return -1;   // non-scalar fields cannot hold a density
            } else {
                if (axisymmetric && FieldT::Dimension != 2) return -2;
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
                return 0;
            }
        }

        template <typename F>
        static int host_get_node(F& f, const int* idx, double* value) {
            constexpr int Dim = F::Dimension;
            bool in_range = true;
            [&]<int... D>(std::integer_sequence<int, D...>) {
                ((in_range = in_range && idx[D] >= 0 &&
                    idx[D] < f.getGrid().template numCells<D>() + 1), ...);
            }(std::make_integer_sequence<int, Dim>{});
            if (!in_range) return -1;
            std::array<int, Dim> a;
            for (int d = 0; d < Dim; d++) a[d] = idx[d];
            size_t linear = f.getGrid().n2i({a});
            if constexpr (F::QuantitySize == 1)
                *value = double(f.data()[linear]);
            else
                for (int c = 0; c < F::QuantitySize; c++)
                    value[c] = double(f.data()[linear][c]);
            return 0;
        }
    };
}

#endif
