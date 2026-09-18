#ifndef PSUM_PARTICLE_COLLISION_SPECIES_TAGS_HPP
#define PSUM_PARTICLE_COLLISION_SPECIES_TAGS_HPP

#include "../../include/psum/tag.hpp"

namespace psum {

namespace particle_collision {

// Never using this namespace!
namespace species_tags {

    struct electron : psum::tag::foundation::abstract_tag { inline const static std::string tag_name = "electron"; };
    struct Hydrogen : psum::tag::foundation::abstract_tag { inline const static std::string tag_name = "Hydrogen"; };
    struct Hydrogen_pos_1 : psum::tag::foundation::abstract_tag { inline const static std::string tag_name = "Hydrogen_pos_1"; };
    struct Helium : psum::tag::foundation::abstract_tag { inline const static std::string tag_name = "Helium"; };
    struct Helium_pos_1 : psum::tag::foundation::abstract_tag { inline const static std::string tag_name = "Helium_pos_1"; };
    struct Helium_pos_2 : psum::tag::foundation::abstract_tag { inline const static std::string tag_name = "Helium_pos_2"; };
    struct Argon : psum::tag::foundation::abstract_tag { inline const static std::string tag_name = "Argon"; };
    struct Argon_pos_1 : psum::tag::foundation::abstract_tag { inline const static std::string tag_name = "Argon_pos_1"; };
    struct Argon_pos_2 : psum::tag::foundation::abstract_tag { inline const static std::string tag_name = "Argon_pos_2"; };
    struct Krypton : psum::tag::foundation::abstract_tag { inline const static std::string tag_name = "Krypton"; };
    struct Krypton_pos_1 : psum::tag::foundation::abstract_tag { inline const static std::string tag_name = "Krypton_pos_1"; };
    struct Krypton_pos_2 : psum::tag::foundation::abstract_tag { inline const static std::string tag_name = "Krypton_pos_2"; };
    struct Xenon : psum::tag::foundation::abstract_tag { inline const static std::string tag_name = "Xenon"; };
    struct Xenon_pos_1 : psum::tag::foundation::abstract_tag { inline const static std::string tag_name = "Xenon_pos_1"; };
    struct Xenon_pos_2 : psum::tag::foundation::abstract_tag { inline const static std::string tag_name = "Xenon_pos_2"; };
    struct Nitrogen : psum::tag::foundation::abstract_tag { inline const static std::string tag_name = "Nitrogen"; };

}

}

}

#endif