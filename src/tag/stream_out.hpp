#ifndef PSUM_TAG_STREAM_OUT_HPP
#define PSUM_TAG_STREAM_OUT_HPP

#include <iostream>
#include "tagged_struct.hpp"

namespace psum {

namespace tag {

    struct ostream_settings {

    	enum mode
    	{
            tag_newline,
            all_one_line,
    		json_like
    	};
    	mode current_mode;

        static ostream_settings& get_instance() {
            static ostream_settings instance;
            return instance;
        }

        inline static mode& get_mode() {
            return ostream_settings::get_instance().current_mode;
        }

        inline static void set_newline_by_property() {
            get_mode() = tag_newline;
        }

        inline static void set_one_line() {
            get_mode() = all_one_line;
        }

        inline static void set_like_json() {
            get_mode() = json_like;
        }
    };

    template <typename... TagsAndTypes>
    std::ostream& operator<<(std::ostream& os, const tagged_struct<TagsAndTypes...>& obj) {

        typedef tagged_struct<TagsAndTypes...> TaggedStruct;

        std::string begin_str, end_str, tag_affix_front, tag_affix_back, bridge, delimiter;
        
        /// [begin_str]
        /// [tag_affix_front], tag_name, [tag_affix_back], [bridge], content, [delimiter]
        /// [end_str]

        if (ostream_settings::get_mode() == ostream_settings::mode::tag_newline) {
            begin_str = "{\n";
            end_str = "\n}\n";
            bridge = " =\n  ";
            delimiter = ";\n";
        }

        if (ostream_settings::get_mode() == ostream_settings::mode::all_one_line) {
            begin_str = "{ ";
            end_str = " }";
            delimiter = "; ";
            bridge = "=";
        }

        if (ostream_settings::get_mode() == ostream_settings::mode::json_like) {
            begin_str = "{\n";
            end_str = "\n}\n";
            tag_affix_front = "  \"";
            tag_affix_back = "\"";
            bridge = " : ";
            delimiter = ",\n";
        }

        static std::vector<std::string> tag_name_vec = foundation::get_tag_name_vector<TaggedStruct>();

        os << begin_str;

        std::apply(
            [&](const auto&... args) {
                size_t current_index = 0;
                auto process_single_element = [&](auto element) {
                    os << (tag_affix_front + tag_name_vec[current_index] + tag_affix_back + bridge);
                    os << element;
                    current_index++;
                    if (current_index != tag_name_vec.size())
                        os << delimiter;
                };
                (process_single_element(args), ...); 
            }, 
            typename TaggedStruct::tuple_of_type(obj)
        );

        os << end_str;
        return os;
    }
}

}

#endif