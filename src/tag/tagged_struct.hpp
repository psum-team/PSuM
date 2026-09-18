#ifndef PSUM_TAG_TAGGER_STRUCT_HPP
#define PSUM_TAG_TAGGER_STRUCT_HPP

#include "foundation.hpp"
#include <iostream>
#include <vector>

namespace psum {

namespace tag {

    template <foundation::a_tag_concept Tag, typename Type>
    struct tag_bind: public foundation::concrete_tag
    {
        using front  = Tag;
        using back = Type;
        static_assert(foundation::try_check<Tag, Type>(), "Tag check failed.");
    };

    template <typename... TagsAndTypes>
    struct tagged_struct : public foundation::dual_tuple<TagsAndTypes...>::tuple_back
    {
    public:

        using tuple_of_tag = typename foundation::dual_tuple<TagsAndTypes...>::tuple_front;
        using tuple_of_type = typename foundation::dual_tuple<TagsAndTypes...>::tuple_back;
        using tuple_of_type::tuple_of_type;

        template <typename Tag>
        constexpr static bool contains()
        {
            constexpr int temp = foundation::index_in_tuple<Tag, tuple_of_tag>::value;
            return temp != -1;
        }
    };

    // Helper functions for tagged_struct
    template <typename T>
    struct is_tagged_struct : std::false_type {};

    template <typename... Ts>
    struct is_tagged_struct<tag::tagged_struct<Ts...>> : std::true_type {};

    template <foundation::a_tag_concept Tag, typename... TagsAndTypes>
    decltype(auto) get(tagged_struct<TagsAndTypes...>& in) {
        using TaggedStruct = tagged_struct<TagsAndTypes...>;
        static constexpr int temp = foundation::index_in_tuple<Tag, typename TaggedStruct::tuple_of_tag>::value;
        static_assert(temp != -1, "Invalid tag in psum::get.");
        return std::get<temp>(in);
    }

    template <foundation::a_tag_concept Tag, typename... TagsAndTypes>
    decltype(auto) get(const tagged_struct<TagsAndTypes...>& in) {
        using TaggedStruct = tagged_struct<TagsAndTypes...>;
        static constexpr int temp = foundation::index_in_tuple<Tag, typename TaggedStruct::tuple_of_tag>::value;
        static_assert(temp != -1, "Invalid tag in psum::get.");
        return std::get<temp>(in);
    }

    template <foundation::a_tag_concept Tag, typename... TagsAndTypes>
    decltype(auto) get(tagged_struct<TagsAndTypes...>&& in) {
        using TaggedStruct = tagged_struct<TagsAndTypes...>;
        static constexpr int temp = foundation::index_in_tuple<Tag, typename TaggedStruct::tuple_of_tag>::value;
        static_assert(temp != -1, "Invalid tag in psum::get.");
        return std::get<temp>(std::move(in));
    }
}

}

#endif