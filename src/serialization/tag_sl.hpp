#ifndef PSUM_SERIALIZATION_TAG_SL_HPP
#define PSUM_SERIALIZATION_TAG_SL_HPP

#include "mas.hpp"
#include "../tag/tagged_struct.hpp"

namespace psum {

namespace serialization {

    template <typename... TagsAndTypes>
    void save(mas_file& fp, const std::string& name, const tag::tagged_struct<TagsAndTypes...>& obj) {
        using namespace tag;
        typedef tagged_struct<TagsAndTypes...> TaggedStruct;

        static std::vector<std::string> tag_name_vec = tag::foundation::get_tag_name_vector<TaggedStruct>();

        std::apply(
            [&](const auto&... args) {
                size_t current_index = 0;
                auto process_single_element = [&](auto element) {
                    save(fp, name + "." + tag_name_vec[current_index], element);
                    current_index++;
                };
                (process_single_element(args), ...); 
            }, 
            typename TaggedStruct::tuple_of_type(obj)
        );
    }

    template <typename... TagsAndTypes>
    void load(mas_file& fp, const std::string& name, tag::tagged_struct<TagsAndTypes...>& obj) {
        using namespace tag;
        typedef tagged_struct<TagsAndTypes...> TaggedStruct;

        static std::vector<std::string> tag_name_vec = tag::foundation::get_tag_name_vector<TaggedStruct>();
        
        typename TaggedStruct::tuple_of_tag v_tags;

        std::apply(
            [&](const auto&... args) {
                size_t current_index = 0;
                auto process_single_element = [&](auto& v_tag) {
                    typedef std::remove_cvref_t<decltype(v_tag)> tag_type;
                    load(fp, name + "." + tag_name_vec[current_index], tag::get<tag_type>(obj));
                    current_index++;
                };
                (process_single_element(args), ...); 
            }, 
            v_tags
        );
    }

    template <typename T>
    concept IsDeviceContainer = requires {
        typename std::remove_cvref_t<T>::is_device_container_t;
        requires (std::remove_cvref_t<T>::is_device_container_t::value); 
    };

    template <template<typename...> class Container, typename... TagsAndTypes>
    requires (!IsDeviceContainer<Container<tag::tagged_struct<TagsAndTypes...>>>)
    void save(mas_file& fp, const std::string& name, const Container<tag::tagged_struct<TagsAndTypes...>>& obj) {
        typedef tag::tagged_struct<TagsAndTypes...> TaggedType;
        typedef typename TaggedType::tuple_of_type TupleOfTypes;
        typedef typename TaggedType::tuple_of_tag TupleOfTags;
        typedef typename tag::foundation::tuple_of_containers<Container, TupleOfTypes>::types TupleOfContainers;

        TupleOfTags v_tags;
        TupleOfContainers v_containers;
        for (auto& item : obj) {
            std::apply(
                [&](const auto&... args) {
                    auto process_single_tag = [&](auto v_tag) {
                        typedef std::remove_cvref_t<decltype(v_tag)> tag_type;
                        auto &component_container =
                            std::get<tag::foundation::index_in_tuple<tag_type, TupleOfTags>::value>(v_containers);
                        component_container.push_back(get<tag_type>(item));
                    };
                    (process_single_tag(args), ...); 
                }, 
                v_tags
            );
        }
        std::apply(
            [&](const auto&... args) {
                auto process_single_tag = [&](auto v_tag) {
                    typedef std::remove_cvref_t<decltype(v_tag)> tag_type;
                    auto &component_container =
                        std::get<tag::foundation::index_in_tuple<tag_type, TupleOfTags>::value>(v_containers);
                    save(fp, name + "." + decltype(v_tag)::tag_name, component_container);
                };
                (process_single_tag(args), ...); 
            }, 
            v_tags
        );
    }

    template <template<typename...> class Container, typename... TagsAndTypes>
    requires (!IsDeviceContainer<Container<tag::tagged_struct<TagsAndTypes...>>>)
    void load(mas_file& fp, const std::string& name, Container<tag::tagged_struct<TagsAndTypes...>>& obj) {
        typedef tag::tagged_struct<TagsAndTypes...> TaggedType;
        typedef typename TaggedType::tuple_of_type TupleOfTypes;
        typedef typename TaggedType::tuple_of_tag TupleOfTags;
        typedef typename tag::foundation::tuple_of_containers<Container, TupleOfTypes>::types TupleOfContainers;

        TupleOfTags v_tags;
        TupleOfContainers v_containers;

        std::apply(
            [&](const auto&... args) {
                auto process_single_tag = [&](auto v_tag) {
                    typedef std::remove_cvref_t<decltype(v_tag)> tag_type;
                    auto &component_container =
                        std::get<tag::foundation::index_in_tuple<tag_type, TupleOfTags>::value>(v_containers);
                    load(fp, name + "." + decltype(v_tag)::tag_name, component_container);
                };
                (process_single_tag(args), ...); 
            }, 
            v_tags
        );
        obj.resize(std::get<0>(v_containers).size());
        for (size_t i = 0; i < obj.size(); i++) {
            std::apply(
                [&](const auto&... args) {
                    auto process_single_tag = [&](auto v_tag) {
                        typedef std::remove_cvref_t<decltype(v_tag)> tag_type;
                        auto &component_container =
                            std::get<tag::foundation::index_in_tuple<tag_type, TupleOfTags>::value>(v_containers);
                        get<tag_type>(obj[i]) = component_container[i];
                    };
                    (process_single_tag(args), ...); 
                }, 
                v_tags
            );
        }
    }
}

}

#endif