#ifndef PSUM_TAG_FOUNDATION_HPP
#define PSUM_TAG_FOUNDATION_HPP

#include <tuple>
#include <concepts>
#include <string>
#include <vector>

namespace psum {

namespace tag {

namespace foundation {

    template <typename Item, typename Tuple>
    struct index_in_tuple;
    
    template <typename Item>
    struct index_in_tuple<Item, std::tuple<>> {
        static constexpr int value = -1;
    };
    
    template <typename Item, typename First, typename... Rest>
    struct index_in_tuple<Item, std::tuple<First, Rest...>> {
        static constexpr int temp = index_in_tuple<Item, std::tuple<Rest...>>::value;
        static constexpr int value = std::is_same_v<Item, First> ? 0 : (temp == -1 ? -1 : temp + 1);
    };

    /// Base type for all tags in the tag system.
    struct abstract_tag{};
    /// Base type for <tag, type> pair bound by 'tag_bind'.
    struct concrete_tag{};

    template <typename Type>
    concept has_tag_name = requires {
        { Type::tag_name } -> std::convertible_to<std::string>;
    };

    template <typename Type>
    concept a_tag_concept = std::is_base_of_v<abstract_tag, Type> && has_tag_name<Type>;

    template <typename Type>
    concept c_tag_concept = std::is_base_of_v<concrete_tag, Type>;
    
    template <typename Tag, typename Type>
    concept has_check = requires {
        { Tag::template check<Type>() } -> std::convertible_to<bool>;
    };
        
    template <typename Tag, typename Type>
    constexpr bool try_check() {
        if constexpr (has_check<Tag, Type>) {
            return Tag::template check<Type>();
        }
        return true;
    }
        
    template <typename... Args>
    struct dual_tuple;
    
    template <>
    struct dual_tuple<> {
        using tuple_front = std::tuple<>;
        using tuple_back = std::tuple<>;
    };
        
    template <c_tag_concept First, typename... Rest>
    struct dual_tuple<First, Rest...> {
    private:
        using rest = dual_tuple<Rest...>;
        using first_front = typename First::front;
        using first_back = typename First::back;
    
    public:
        using tuple_front = decltype(std::tuple_cat(std::tuple<first_front>{}, typename rest::tuple_front()));
        using tuple_back = decltype(std::tuple_cat(std::tuple<first_back>{}, typename rest::tuple_back()));
    };

    template <typename DualTuple, typename Front>
    struct dual_tuple_map {
        static constexpr int index = index_in_tuple<Front, typename DualTuple::tuple_front>::value;
        using type = decltype(std::get<index>(typename DualTuple::tuple_back{}));
    };

    template <template<typename...> class Container, typename Type>
    struct tuple_of_containers {
        using types = void;
        constexpr static bool match_with_tuple = false;
    };

    template <template<typename...> class Container, typename... TypeList>
    struct tuple_of_containers<Container, std::tuple<TypeList...>> {
        using types = std::tuple<Container<TypeList>...>;
        constexpr static bool match_with_tuple = true;
    };

    template <typename Type>
    concept has_tuple_of_type_and_tag = requires {
        typename Type::tuple_of_type;
        typename Type::tuple_of_tag;
    };

    template <has_tuple_of_type_and_tag Tags_and_Types>
    static std::vector<std::string> get_tag_name_vector() {
        return []() {
            typename Tags_and_Types::tuple_of_tag v_tags;
            std::vector<std::string> tag_names;
            std::apply(
                [&tag_names](const auto&... args) {
                    size_t current_index = 0;
                    auto process_single_element = [&](auto element) {
                        tag_names.push_back(decltype(element)::tag_name);
                        current_index++;
                    };
                    (process_single_element(args), ...); 
                }, 
                v_tags
            );
            return tag_names;
        }();
    }
}

}

}

#endif