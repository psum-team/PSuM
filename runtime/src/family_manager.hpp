#ifndef PSUM_RUNTIME_FAMILY_MANAGER_HPP
#define PSUM_RUNTIME_FAMILY_MANAGER_HPP

#include <psum/psum.hpp>
#include "family.hpp"
#include "status.hpp"
#include "last_failure.hpp"
#include <stdexcept>
#include <string>
#include <type_traits>
#include <cstddef>
#include <utility>
#include <vector>

namespace psum_runtime {

    using namespace psum::prelude;

    template <typename Desc, typename Family>
    struct desc_family_protocol;          // not defined: opt-in per (Desc, Family)

    template <typename Desc, typename Family>
    concept descriptor_of = requires { typename desc_family_protocol<Desc, Family>::protocol; };

    // what a protocol's equal(real, d) reports
    enum class equality : int { same = 0, type_diff = 1, value_diff = 2 };

    enum manager_status : int {
        st_ok = PSUM_OK,
        st_bad_arg = PSUM_ERR_ARG,
        st_missing = PSUM_ERR_NOT_FOUND,
        st_unsupported = PSUM_ERR_UNSUPPORTED,
        st_overflow = PSUM_ERR_OVERFLOW,
        // outcomes of a descriptor-driven create on an object that EXISTS
        st_same = PSUM_ERR_EXISTS,                   // ...and matches the descriptor exactly
        st_conflict_type = PSUM_ERR_CONFLICT_TYPE,   // ...but its TYPE invariants differ
        st_conflict_value = PSUM_ERR_CONFLICT_VALUE  // ...TYPE invariants equal, VALUE invariants differ
    };

    template <int Segment>
    struct manager_status_of {
        int code;                                   // the code, complete
        static constexpr int segment = Segment;
        constexpr manager_status_of(manager_status base) : code(base ? int(base) - Segment : 0) {}
        constexpr manager_status_of(int complete) : code(complete) {}
        constexpr operator int() const { return code; }
        // comparisons are made on base codes, against base codes
        constexpr bool operator==(int b) const { return code == (b ? b - Segment : 0); }
        constexpr bool operator!=(int b) const { return !(*this == b); }
    };

    template <typename Family>
    class family_manager {
        serialization::object_manager& mng_;
        last_failure* last_;     // the engine's last failure; may be null
        const int segment_;

        // this manager's own verdict, written down with its segment
        manager_status report(manager_status r, const char* name) const {
            if (last_ && r != st_ok)
                last_->write(psum_code(int(r), segment_), name);
            return r;
        }

    public:
        using family = Family;                 // the protocol key
        using types = family_types_t<Family>;   // the entries
        static_assert(!std::is_same_v<types, type_list<>>, "a family manager needs at least one type");
        // every entry is keyed_type<Key, T> ...
        static_assert([] <typename... Ts> (type_list<Ts...>) {
            return (is_keyed<Ts> && ...);
        }(types{}), "a family must consist of keyed_type<Key, T> entries only");
        // ... and no two of them share a key (see no_duplicate_keys)
        static_assert(no_duplicate_keys<types>, "duplicate key in the family");
        static constexpr std::size_t size = [] {
            return [] <typename... Ts> (type_list<Ts...>) { return sizeof...(Ts); }(types{});
        }();

        // last: the engine's last failure (null: none); segment: the family's, see status.hpp
        family_manager(serialization::object_manager& mng, last_failure* last, int segment)
            : mng_(mng), last_(last), segment_(segment) {}

        // ---------------- query ----------------
        // does ANY type of the family manage an object under `name`?
        bool exists(const char* name) const {
            if (!name) return false;
            bool found = false;
            [&] <typename... Ts> (type_list<Ts...>) {
                ([&] { if (!found) found = mng_.template contains<member_t<Ts>>(name); }(), ...);
            }(types{});
            return found;
        }

        // ---------------- object access ----------------
        // hand the family member that exists under `name` to fn(obj).
        template <typename F>
        bool find(const char* name, F&& fn) const {
            if (!name) return false;
            bool found = false;
            [&] <typename... Ts> (type_list<Ts...>) {
                ([&] {
                    if (found || !mng_.template contains<member_t<Ts>>(name)) return;
                    fn(mng_.template obj<member_t<Ts>>(name), entry_tag<Ts>{});
                    found = true;
                }(), ...);
            }(types{});
            return found;
        }
        // dispatch: run fn(obj) on the family member that exists under `name` and return its result.
        template <typename R, typename F>
        R dispatch(const char* name, R not_found, F&& fn) const {
            R r = not_found;
            if (find(name, [&](auto& obj, auto) { r = R(fn(obj)); })) return r;
            if (last_ && int(r) != 0) last_->write(int(r), name);
            return r;
        }
        // the same, restricted to the family members whose TYPE satisfies Pred<T>::value
        template <template <typename> class Pred, typename R, typename F>
        R dispatch_if(const char* name, R not_found, F&& fn) const {
            if (!name) return not_found;
            R r = not_found;
            bool done = false;
            [&] <typename... Ts> (type_list<Ts...>) {
                ([&] {
                    // the filter must be constexpr: a member the predicate
                    // rejects is never even instantiated, so fn is never
                    // compiled against a type it was not written for
                    if constexpr (Pred<member_t<Ts>>::value) {
                        if (done || !mng_.template contains<member_t<Ts>>(name)) return;
                        r = R(fn(mng_.template obj<member_t<Ts>>(name)));
                        done = true;
                    }
                }(), ...);
            }(types{});
            if (done) return r;
            if (last_ && int(r) != 0) last_->write(int(r), name);   // no member of the window under the name
            return r;
        }
        // the object of EXACTLY type T under `name`; throws when the family does not hold it (never creates)
        template <typename T>
        T& get(const char* name) const {
            static_assert(family_member<Family, T>, "get<T>: T is not one of this family's types");
            if (!name || !mng_.template contains<T>(name))
                throw std::runtime_error(std::string("family_manager: no object named '")
                                         + (name ? name : "") + "' of the requested type");
            return mng_.template obj<T>(name);
        }

        // ---------------- descriptor protocol ----------------

        // TYPE + VALUE invariants of the object under `name` (or NOT_FOUND-like
        // false when the family does not hold it). Never creates.
        template <descriptor_of<Family> Desc>
        bool desc_get(const char* name, Desc* out) const {
            using Protocol = desc_family_protocol<Desc, Family>;
            if (!name || !out) return false;
            Desc d = *out;
            d.name = name;
            const bool found = find(name, [&](const auto& obj, auto) { Protocol::extract(obj, d); });
            if (found) *out = d;
            return found;
        }

        // Authenticity of a descriptor against what the family holds:
        //   st_missing        no object under that name
        //   st_same           an object exists and matches d exactly
        //   st_conflict_type  an object exists, its TYPE invariants differ
        //   st_conflict_value TYPE invariants equal, VALUE invariants differ
        template <descriptor_of<Family> Desc>
        manager_status check(const Desc& d) const {
            using Protocol = desc_family_protocol<Desc, Family>;
            Desc real = {};
            if (!desc_get(d.name, &real)) return st_missing;
            switch (Protocol::equal(real, d)) {
                case equality::same:      return st_same;
                case equality::type_diff: return st_conflict_type;
                case equality::value_diff: return st_conflict_value;
            }
            return st_conflict_type;
        }

        // judge d, create through the family only when absent.
        //
        // Returns the status of d AS PASSED IN: st_ok when it was created,
        // st_same / st_conflict_* when an object already existed (d is made
        // authentic in every case, so a conflicting caller can read back
        // what the real object is like).
        template <descriptor_of<Family> Desc, typename... Args>
        manager_status create(Desc* d, Args&&... args) const {
            using Protocol = desc_family_protocol<Desc, Family>;
            if (!d || !d->name) return report(st_bad_arg, nullptr);
            const manager_status before = check(*d);
            if (before == st_missing) {
                // the TYPE invariants pick the type: the protocol spells them
                // as the family's key, and the family is looked up by it
                const auto key = Protocol::key(*d);
                bool built = false;
                [&] <typename... Ts> (type_list<Ts...>) {
                    ([&] {
                        if (Ts::key == key) {
                            Protocol::build(*d, std::forward<Args>(args)..., mng_.template obj<member_t<Ts>>(d->name));
                            built = true;
                        }
                    }(), ...);
                }(types{});
                if (!built) return report(st_unsupported, d->name);   // no such type in this build
                desc_get(d->name, d);
                return st_ok;
            }
            desc_get(d->name, d);   // d is now authentic
            return report(before, d->name);
        }

    };

} // namespace psum_runtime

#endif
