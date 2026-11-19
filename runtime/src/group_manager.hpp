#ifndef PSUM_RUNTIME_GROUP_MANAGER_HPP
#define PSUM_RUNTIME_GROUP_MANAGER_HPP

// The particle-group part of the engine

#include <psum/psum.hpp>

namespace psum_runtime {

    using namespace psum::prelude;

    // group_manager is parameterized by the universe
    // the particle type (position/velocity scalars and bindings) comes from it.
    template <typename Universe>
    class group_manager {
    public:
        group_manager(serialization::object_manager& mng, sycl::queue& q) : mng_(mng), q_(q) {}

        template <int Dim>
        using particle_t = typename Universe::template particle<Dim>;
        template <int Dim>
        using group_t = typename Universe::template group<Dim>;

        int create(const char* name, int dim) {
            if (!name || !Universe::has_dim(dim)) return -1;
            if constexpr (Universe::has_dim(1)) if (dim == 1) { mng_.template obj<group_t<1>>(name); return 0; }
            if constexpr (Universe::has_dim(2)) if (dim == 2) { mng_.template obj<group_t<2>>(name); return 0; }
            if constexpr (Universe::has_dim(3)) if (dim == 3) { mng_.template obj<group_t<3>>(name); return 0; }
            return -1;
        }

        int append(const char* name, const double* pos, const double* vel) {
            if (!name || !pos || !vel) return -1;
            if constexpr (Universe::has_dim(1)) if (mng_.template contains<group_t<1>>(name)) return append_one(mng_.template obj<group_t<1>>(name), pos, vel);
            if constexpr (Universe::has_dim(2)) if (mng_.template contains<group_t<2>>(name)) return append_one(mng_.template obj<group_t<2>>(name), pos, vel);
            if constexpr (Universe::has_dim(3)) if (mng_.template contains<group_t<3>>(name)) return append_one(mng_.template obj<group_t<3>>(name), pos, vel);
            return -1;
        }

        // batch append: pos/vel each hold dim*count doubles
        int append_n(const char* name, size_t count, const double* pos, const double* vel) {
            if (!name || (!pos && count) || (!vel && count)) return -1;
            auto on_dim = [&](auto dim_constant) {
                constexpr int Dim = decltype(dim_constant)::value;
                auto& grp = mng_.template obj<group_t<Dim>>(name);
                std::vector<particle_t<Dim>> ps(count);
                for (size_t k = 0; k < count; k++)
                    for (int i = 0; i < Dim; i++) {
                        psum::tag::get<pt::position>(ps[k])[i] = pos[k * Dim + i];
                        psum::tag::get<pt::velocity>(ps[k])[i] = vel[k * Dim + i];
                    }
                grp.insert(ps);
                return 0;
            };
            if constexpr (Universe::has_dim(1)) if (mng_.template contains<group_t<1>>(name)) return on_dim(std::integral_constant<int, 1>{});
            if constexpr (Universe::has_dim(2)) if (mng_.template contains<group_t<2>>(name)) return on_dim(std::integral_constant<int, 2>{});
            if constexpr (Universe::has_dim(3)) if (mng_.template contains<group_t<3>>(name)) return on_dim(std::integral_constant<int, 3>{});
            return -1;
        }

        // size() is the VALID count
        int count(const char* name, size_t* count_out) {
            if (!name || !count_out) return -1;
            if constexpr (Universe::has_dim(1)) if (mng_.template contains<group_t<1>>(name)) { *count_out = mng_.template obj<group_t<1>>(name).size(); return 0; }
            if constexpr (Universe::has_dim(2)) if (mng_.template contains<group_t<2>>(name)) { *count_out = mng_.template obj<group_t<2>>(name).size(); return 0; }
            if constexpr (Universe::has_dim(3)) if (mng_.template contains<group_t<3>>(name)) { *count_out = mng_.template obj<group_t<3>>(name).size(); return 0; }
            return -1;
        }

        int size(const char* name, size_t* count_out, int* dim) {
            if (!name || !count_out || !dim) return -1;
            if constexpr (Universe::has_dim(1)) if (mng_.template contains<group_t<1>>(name)) { *count_out = mng_.template obj<group_t<1>>(name).size(); *dim = 1; return 0; }
            if constexpr (Universe::has_dim(2)) if (mng_.template contains<group_t<2>>(name)) { *count_out = mng_.template obj<group_t<2>>(name).size(); *dim = 2; return 0; }
            if constexpr (Universe::has_dim(3)) if (mng_.template contains<group_t<3>>(name)) { *count_out = mng_.template obj<group_t<3>>(name).size(); *dim = 3; return 0; }
            return -1;
        }

        int compress(const char* name) {
            if (!name) return -1;
            if constexpr (Universe::has_dim(1)) if (mng_.template contains<group_t<1>>(name)) { mng_.template obj<group_t<1>>(name).compress(); return 0; }
            if constexpr (Universe::has_dim(2)) if (mng_.template contains<group_t<2>>(name)) { mng_.template obj<group_t<2>>(name).compress(); return 0; }
            if constexpr (Universe::has_dim(3)) if (mng_.template contains<group_t<3>>(name)) { mng_.template obj<group_t<3>>(name).compress(); return 0; }
            return -1;
        }

        // bulk read-back: pos/vel each hold dim*valid_count doubles (see size)
        int get_all_pv(const char* name, double* pos, double* vel) {
            if (!name || !pos || !vel) return -1;
            if constexpr (Universe::has_dim(1)) if (mng_.template contains<group_t<1>>(name)) return get_all_one(mng_.template obj<group_t<1>>(name), pos, vel);
            if constexpr (Universe::has_dim(2)) if (mng_.template contains<group_t<2>>(name)) return get_all_one(mng_.template obj<group_t<2>>(name), pos, vel);
            if constexpr (Universe::has_dim(3)) if (mng_.template contains<group_t<3>>(name)) return get_all_one(mng_.template obj<group_t<3>>(name), pos, vel);
            return -1;
        }

        // export the leave buffer of a group: the particles recorded (pre-invalidation copies) by the most recent gated kernel
        int leave_buffer(const char* name, size_t* count, int* dim, double* pos, double* vel) {
            if (!name || !count || !dim) return -1;
            const std::string ln = std::string(name) + "_leave_buffer";
            bool overflowed = false;
            auto export_one = [&](auto dim_constant) {
                constexpr int D = decltype(dim_constant)::value;
                using vec_t = psum::particle_container::device_vector<particle_t<D>>;
                if (!mng_.template contains<vec_t>(ln)) return false;
                auto& lv = mng_.template obj<vec_t>(ln);
                if (lv.overflowed()) {
                    // capacity exceeded: entries were dropped — report the
                    // loss as an error instead of handing back truncated data
                    *count = lv.capacity();
                    *dim = D;
                    overflowed = true;
                    return true;
                }
                else { *count = lv.size(); }
                *dim = D;
                if (pos && vel) {
                    auto host = lv.to_host();
                    for (size_t k = 0; k < host.size(); k++) {
                        for (int i = 0; i < D; i++) {
                            pos[k * D + i] = psum::tag::get<pt::position>(host[k])[i];
                            vel[k * D + i] = psum::tag::get<pt::velocity>(host[k])[i];
                        }
                    }
                }
                return true;
            };
            if (export_one(std::integral_constant<int, 1>{})) return overflowed ? -1 : 0;
            if (export_one(std::integral_constant<int, 2>{})) return overflowed ? -1 : 0;
            if (export_one(std::integral_constant<int, 3>{})) return overflowed ? -1 : 0;
            return -2;   // no leave buffer (no gated kernel ran yet)
        }

        // dimension dispatch over the named group: fn(dim_constant, group) is invoked with the group's concrete dimension
        template <typename F>
        int dispatch_group(const char* name, F&& fn) {
            auto on_dim = [&](auto dim_constant) {
                constexpr int Dim = decltype(dim_constant)::value;
                return fn(std::integral_constant<int, Dim>{}, mng_.template obj<group_t<Dim>>(name));
            };
            if constexpr (Universe::has_dim(1)) if (mng_.template contains<group_t<1>>(name)) return on_dim(std::integral_constant<int, 1>{});
            if constexpr (Universe::has_dim(2)) if (mng_.template contains<group_t<2>>(name)) return on_dim(std::integral_constant<int, 2>{});
            if constexpr (Universe::has_dim(3)) if (mng_.template contains<group_t<3>>(name)) return on_dim(std::integral_constant<int, 3>{});
            return -1;
        }

        // reads the index-th VALID particle (skips lazily deleted slots)
        int get(const char* name, size_t index, double* pos, double* vel) {
            if (!name || !pos || !vel) return -1;
            if constexpr (Universe::has_dim(1)) if (mng_.template contains<group_t<1>>(name)) return get_one(mng_.template obj<group_t<1>>(name), index, pos, vel);
            if constexpr (Universe::has_dim(2)) if (mng_.template contains<group_t<2>>(name)) return get_one(mng_.template obj<group_t<2>>(name), index, pos, vel);
            if constexpr (Universe::has_dim(3)) if (mng_.template contains<group_t<3>>(name)) return get_one(mng_.template obj<group_t<3>>(name), index, pos, vel);
            return -1;
        }

    private:
        serialization::object_manager& mng_;
        sycl::queue& q_;

        template <int Dim>
        int append_one(group_t<Dim>& grp, const double* pos, const double* vel) {
            particle_t<Dim> p;
            for (int i = 0; i < Dim; i++) {
                psum::tag::get<pt::position>(p)[i] = pos[i];
                psum::tag::get<pt::velocity>(p)[i] = vel[i];
            }
            std::vector<particle_t<Dim>> dv({p});
            grp.insert(dv);
            return 0;
        }

        template <int Dim>
        int get_one(group_t<Dim>& grp, size_t index, double* pos, double* vel) {
            auto host = grp.get_content().to_host();
            size_t valid_seen = 0;
            for (auto& p : host) {
                if (!group_t<Dim>::validator::is_valid(p)) continue;
                if (valid_seen++ == index) {
                    for (int i = 0; i < Dim; i++) {
                        pos[i] = psum::tag::get<pt::position>(p)[i];
                        vel[i] = psum::tag::get<pt::velocity>(p)[i];
                    }
                    return 0;
                }
            }
            return -1;   // index beyond the valid count
        }

        template <int Dim>
        int get_all_one(group_t<Dim>& grp, double* pos, double* vel) {
            auto host = grp.get_content().to_host();
            size_t k = 0;
            for (auto& p : host) {
                if (!group_t<Dim>::validator::is_valid(p)) continue;
                for (int i = 0; i < Dim; i++) {
                    pos[k * Dim + i] = psum::tag::get<pt::position>(p)[i];
                    vel[k * Dim + i] = psum::tag::get<pt::velocity>(p)[i];
                }
                k++;
            }
            return 0;
        }
    };

} // namespace psum_runtime

#endif
