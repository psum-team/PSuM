#ifndef PSUM_RUNTIME_SPECIES_MANAGER_HPP
#define PSUM_RUNTIME_SPECIES_MANAGER_HPP

#include <psum/psum.hpp>
#include "family_manager.hpp"
#include "families.hpp"
#include "species.hpp"
#include "desc/species_desc.hpp"
#include <vector>
#include <memory>
#include <cstring>
#include <unordered_map>
#include <optional>
#include <string>

namespace psum_runtime {

    using namespace psum::prelude;
    namespace pt = psum::property;   // this header stands on its own

    // species_manager is parameterized by the universe
    // the particle type (position/velocity scalars and bindings) comes from it.
    template <typename Universe>
    class species_manager {
    public:
        // one status set for the whole manager layer (family_manager.hpp)
        using status = manager_status_of<300>;   // the species segment

        // the species family: one species<Universe, Dim> per enabled dim.
        // Every query / access / create below goes through it.
        template <int Dim>
        using particle_t = typename Universe::template particle<Dim>;
        template <int Dim>
        using species_t = species<Universe, Dim>;
        using family = species_family<Universe>;

        // the leave buffers: one device_vector per ENABLED dim, a family of
        // their own, keyed "<species>_leave_buffer"
        template <int D>
        using leave_list = std::conditional_t<Universe::has_dim(D),
            type_list<keyed_type<D, psum::particle_container::device_vector<particle_t<D>>>>, type_list<>>;
        using leave_types = typename type_list_cat<
            typename type_list_cat<leave_list<1>, leave_list<2>>::type, leave_list<3>>::type;

        species_manager(serialization::object_manager& mng, sycl::queue& q, last_failure& last)
            : q_(q), species_(mng, &last, 300), leave_(mng, &last, 300) {}

        // ---------------- query / access ----------------

        bool exists(const char* name) const { return species_.exists(name); }

        // ---------------- descriptor interface ----------------

        bool desc_get(const char* name, psum_obj_species_desc* out) const { return species_.desc_get(name, out); }
        status check(const psum_obj_species_desc& d) const { return status(species_.check(d)); }
        status create(psum_obj_species_desc* d) const { return status(species_.create(d)); }

        template <typename F>
        status with_species(const char* name, F&& fn) {
            return species_.dispatch(name, status(st_missing), [&](auto& grp) {
                using G = std::remove_cvref_t<decltype(grp)>;
                return fn(std::integral_constant<int, G::dim_value>{}, grp);
            });
        }

        status append(const char* name, const double* pos, const double* vel) {
            if (!name || !pos || !vel) return st_bad_arg;
            return with_species(name, [&](auto, auto& grp) { return append_one(grp, pos, vel); });
        }

        // batch append: pos/vel each hold dim*count doubles
        status append_n(const char* name, size_t count, const double* pos, const double* vel) {
            if (!name || (!pos && count) || (!vel && count)) return st_bad_arg;
            return with_species(name, [&](auto dim_constant, auto& grp) {
                constexpr int Dim = decltype(dim_constant)::value;
                std::vector<particle_t<Dim>> ps(count);
                for (size_t k = 0; k < count; k++)
                    for (int i = 0; i < Dim; i++) {
                        psum::tag::get<pt::position>(ps[k])[i] = pos[k * Dim + i];
                        psum::tag::get<pt::velocity>(ps[k])[i] = vel[k * Dim + i];
                    }
                grp.insert(ps);
                return st_ok;
            });
        }

        // size() is the VALID count
        status count(const char* name, size_t* count_out) {
            if (!name || !count_out) return st_bad_arg;
            return with_species(name, [&](auto, auto& grp) { *count_out = grp.size(); return st_ok; });
        }

        status size(const char* name, size_t* count_out, int* dim) {
            if (!name || !count_out || !dim) return st_bad_arg;
            return with_species(name, [&](auto dim_constant, auto& grp) {
                *count_out = grp.size();
                *dim = decltype(dim_constant)::value;
                return st_ok;
            });
        }

        status compress(const char* name) {
            if (!name) return st_bad_arg;
            return with_species(name, [&](auto, auto& grp) { grp.compress(); return st_ok; });
        }

        // bulk read-back: pos/vel each hold dim*valid_count doubles (see size)
        status get_all_pv(const char* name, double* pos, double* vel) {
            if (!name || !pos || !vel) return st_bad_arg;
            return with_species(name, [&](auto, auto& grp) { return get_all_one(grp, pos, vel); });
        }

        // export the leave buffer of a species: the particles recorded (pre-invalidation copies) by the most recent gated kernel
        // The buffers are a family of their own (one device_vector per dim) keyed "<species>_leave_buffer".
        status leave_buffer(const char* name, size_t* count, int* dim, double* pos, double* vel) {
            if (!name || !count || !dim) return st_bad_arg;
            const std::string ln = std::string(name) + "_leave_buffer";
            status r = st_ok;
            const bool found = leave_.find(ln.c_str(), [&](auto& lv, auto entry) {
                // the entry found carries its dim as the key
                constexpr int D = decltype(entry)::type::key;
                if (lv.overflowed()) {
                    // capacity exceeded: entries were dropped — report the
                    // loss as an error instead of handing back truncated data
                    *count = lv.capacity();
                    *dim = D;
                    r = st_overflow;
                    return;
                }
                *count = lv.size();
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
            });
            if (!found) return st_missing;   // no leave buffer (no gated kernel ran yet)
            return r;
        }

        // dimension dispatch over the named species: fn(dim_constant, species) is invoked with its concrete dimension
        template <typename F>
        status dispatch_species(const char* name, F&& fn) {
            return with_species(name, [&](auto dim_constant, auto& grp) { return fn(dim_constant, grp); });
        }

        // reads the index-th VALID particle (skips lazily deleted slots)
        status get(const char* name, size_t index, double* pos, double* vel) {
            if (!name || !pos || !vel) return st_bad_arg;
            return with_species(name, [&](auto, auto& grp) { return get_one(grp, index, pos, vel); });
        }

    private:
        family_manager<family> species_;
        family_manager<leave_types> leave_;
        sycl::queue& q_;

        template <int Dim>
        manager_status append_one(species_t<Dim>& grp, const double* pos, const double* vel) {
            particle_t<Dim> p;
            for (int i = 0; i < Dim; i++) {
                psum::tag::get<pt::position>(p)[i] = pos[i];
                psum::tag::get<pt::velocity>(p)[i] = vel[i];
            }
            std::vector<particle_t<Dim>> dv({p});
            grp.insert(dv);
            return st_ok;
        }

        template <int Dim>
        manager_status get_one(species_t<Dim>& grp, size_t index, double* pos, double* vel) {
            auto host = grp.get_content().to_host();
            size_t valid_seen = 0;
            for (auto& p : host) {
                if (!species_t<Dim>::validator::is_valid(p)) continue;
                if (valid_seen++ == index) {
                    for (int i = 0; i < Dim; i++) {
                        pos[i] = psum::tag::get<pt::position>(p)[i];
                        vel[i] = psum::tag::get<pt::velocity>(p)[i];
                    }
                    return st_ok;
                }
            }
            return st_bad_arg;   // index beyond the valid count
        }

        template <int Dim>
        manager_status get_all_one(species_t<Dim>& grp, double* pos, double* vel) {
            auto host = grp.get_content().to_host();
            size_t k = 0;
            for (auto& p : host) {
                if (!species_t<Dim>::validator::is_valid(p)) continue;
                for (int i = 0; i < Dim; i++) {
                    pos[k * Dim + i] = psum::tag::get<pt::position>(p)[i];
                    vel[k * Dim + i] = psum::tag::get<pt::velocity>(p)[i];
                }
                k++;
            }
            return st_ok;
        }
    };

    // species  <->  the species<Universe, Dim> family
    template <typename Universe>
    struct desc_family_protocol<psum_obj_species_desc, species_family<Universe>> {
        using protocol = void;

        // the descriptor's TYPE invariant (dim) IS the family key
        static constexpr int key(const psum_obj_species_desc& d) { return d.dim; }

        template <typename Obj>
        static void extract(const Obj& s, psum_obj_species_desc& d) {
            d.dim = Obj::dim_value;
            d.has_global_mass = s.global_mass.has_value();
            d.global_mass = s.global_mass.value_or(0.0);
            d.has_global_charge = s.global_charge.has_value();
            d.global_charge = s.global_charge.value_or(0.0);
            d.has_global_weight = s.global_weight.has_value();
            d.global_weight = s.global_weight.value_or(0.0);
        }

        static equality equal(const psum_obj_species_desc& real, const psum_obj_species_desc& d) {
            if (real.dim != d.dim) return equality::type_diff;
            if (real.has_global_mass != d.has_global_mass || real.has_global_charge != d.has_global_charge
                || real.has_global_weight != d.has_global_weight
                || (d.has_global_mass && real.global_mass != d.global_mass)
                || (d.has_global_charge && real.global_charge != d.global_charge)
                || (d.has_global_weight && real.global_weight != d.global_weight))
                return equality::value_diff;
            return equality::same;
        }

        template <typename Obj>
        static void build(const psum_obj_species_desc& d, Obj& box) {
            if (d.has_global_mass) box.global_mass = d.global_mass;
            if (d.has_global_charge) box.global_charge = d.global_charge;
            if (d.has_global_weight) box.global_weight = d.global_weight;
        }
    };

} // namespace psum_runtime

#endif
