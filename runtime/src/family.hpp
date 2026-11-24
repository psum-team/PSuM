#ifndef PSUM_RUNTIME_FAMILY_HPP
#define PSUM_RUNTIME_FAMILY_HPP

#include <concepts>
#include <cstddef>
#include <type_traits>

namespace psum_runtime {

    // family is a compile-time determined finited set of {(k_i, Type_i)}.
    // k_i is an instance of type Key and Key is comparable.
    // so it is possible to choose a Type_j with given k_in, making k_j == k_in

    // for compile-time type 'family' defination
    template <typename... Ts>
    struct type_list {};
    template <typename A, typename B> struct type_list_cat;
    template <typename... As, typename... Bs>
    struct type_list_cat<type_list<As...>, type_list<Bs...>> { using type = type_list<As..., Bs...>; };

    // one entry of a keyed family: a comparable key and the type it selects
    template <auto Key, typename T>
    struct keyed_type { static constexpr auto key = Key; using type = T; };

    // an empty tag naming a family entry, handed to find() callbacks
    template <typename E> struct entry_tag { using type = E; };

    // the key is a compile-time constant: an integer, or a constant of a type that can be tested for equality
    template <typename E>
    concept is_keyed = requires {
        typename E::type;
        { E::key } -> std::equality_comparable;
        typename std::bool_constant<(E::key == E::key)>::type;
    };

    // how many entries of a family carry the same key as E
    template <typename E, typename List> struct same_key_count;
    template <typename E> struct same_key_count<E, type_list<>> : std::integral_constant<int, 0> {};
    template <typename E, typename T, typename... Rest>
    struct same_key_count<E, type_list<T, Rest...>> : std::integral_constant<int,
        (std::is_same_v<decltype(E::key), decltype(T::key)> && E::key == T::key ? 1 : 0)
        + same_key_count<E, type_list<Rest...>>::value> {};
    // true when no two entries of the family share a key: a duplicate would
    // make "which type does this key build" ambiguous
    template <typename List>
    concept no_duplicate_keys = requires {
        requires [] <typename... Ts> (type_list<Ts...>) {
            return ((same_key_count<Ts, type_list<Ts...>>::value == 1) && ...);
        }(List{});
    };

    // the concrete C++ type a family ENTRY manages: the type its key selects
    template <typename T> struct member_of { using type = T; };
    template <auto K, typename T> struct member_of<keyed_type<K, T>> { using type = T; };
    template <typename T> using member_t = typename member_of<T>::type;


    template <typename Family> struct family_types;
    template <typename... Ts> struct family_types<type_list<Ts...>> { using type = type_list<Ts...>; };
    template <typename Family> struct family_types : family_types<typename Family::base_list> {};
    template <typename Family> using family_types_t = typename family_types<Family>::type;

    // is T one of the family's types?
    template <typename Family, typename T>
    concept family_member = requires {
        requires [] <typename... Ts> (type_list<Ts...>) {
            return (std::is_same_v<T, member_t<Ts>> || ...);
        }(family_types_t<Family>{});
    };

} // namespace psum_runtime

#endif
