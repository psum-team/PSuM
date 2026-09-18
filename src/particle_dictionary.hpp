#ifndef PSUM_PARTICLE_DICTIONARY_HPP
#define PSUM_PARTICLE_DICTIONARY_HPP

#include <array>
#include <string>
#include <stdexcept>
#include <cmath>
#include <iostream>
#include "tag/property.hpp"
#include "serialization/mas.hpp"

namespace psum {

namespace particle_dictionary {

enum class particle_kind : int {
    NotDefined,
    Electron,
    Hydrogen,
    HydrogenIon,
    Helium,
    HeliumIon,
    HeliumDoubleIon,
    Argon,
    ArgonIon,
    Krypton,
    KryptonIon,
    Xenon,
    XenonIon,
};

inline double get_charge(particle_kind particle) {
    switch (particle) {
        case particle_kind::Electron:
            return -1.602176634e-19;
        case particle_kind::Hydrogen:
        case particle_kind::Helium:
        case particle_kind::Argon:
        case particle_kind::Krypton:
        case particle_kind::Xenon:
            return 0.0;
        case particle_kind::HydrogenIon:
        case particle_kind::HeliumIon:
        case particle_kind::ArgonIon:
        case particle_kind::KryptonIon:
        case particle_kind::XenonIon:
            return 1.602176634e-19;
        case particle_kind::HeliumDoubleIon:
            return 2.0 * 1.602176634e-19;
        default:
            throw std::runtime_error("Unknown particle type");
    }
}

inline double get_mass(particle_kind particle) {
    switch (particle) {
        case particle_kind::Electron:
            return 9.1093837015e-31;
        case particle_kind::Hydrogen:
        case particle_kind::HydrogenIon:
            return 1.6735575e-27;
        case particle_kind::Helium:
        case particle_kind::HeliumIon:
        case particle_kind::HeliumDoubleIon:
            return 6.646477e-27;
        case particle_kind::Argon:
        case particle_kind::ArgonIon:
            return 6.634e-26;
        case particle_kind::Krypton:
        case particle_kind::KryptonIon:
            return 1.392e-25;
        case particle_kind::Xenon:
        case particle_kind::XenonIon:
            return 2.18e-25;
        default:
            throw std::runtime_error("Unknown particle type");
    }
}

inline std::ostream& operator<<(std::ostream& os, particle_kind particle) {
    switch (particle) {
        case particle_kind::NotDefined:
            os << "NotDefined";
            break;
        case particle_kind::Electron:
            os << "e-";
            break;
        case particle_kind::Hydrogen:
            os << "H";
            break;
        case particle_kind::HydrogenIon:
            os << "H+";
            break;
        case particle_kind::Helium:
            os << "He";
            break;
        case particle_kind::HeliumIon:
            os << "He+";
            break;
        case particle_kind::HeliumDoubleIon:
            os << "He++";
            break;
        case particle_kind::Argon:
            os << "Ar";
            break;
        case particle_kind::ArgonIon:
            os << "Ar+";
            break;
        case particle_kind::Krypton:
            os << "Kr";
            break;
        case particle_kind::KryptonIon:
            os << "Kr+";
            break;
        case particle_kind::Xenon:
            os << "Xe";
            break;
        case particle_kind::XenonIon:
            os << "Xe+";
            break;
        default:
            os << "UnknownParticle(" << static_cast<int>(particle) << ")";
    }
    return os;
}

}

namespace tag {

template <foundation::a_tag_concept Tag>
inline auto get(particle_dictionary::particle_kind particle) -> decltype(auto) {
    if constexpr (std::is_same_v<Tag, property::mass>) {
        return particle_dictionary::get_mass(particle);
    } else if constexpr (std::is_same_v<Tag, property::charge>) {
        return particle_dictionary::get_charge(particle);
    } else {
        throw std::runtime_error("Unsupported tag type");
    }
}

}

namespace serialization {

    void save(mas_file& fp, const std::string& name, const particle_dictionary::particle_kind& obj) {
        save(fp, name, static_cast<int>(obj));
    }

    void load(mas_file& fp, const std::string& name, particle_dictionary::particle_kind& obj) {
        int value;
        load(fp, name, value);
        obj = static_cast<particle_dictionary::particle_kind>(value);
    }


}

}

#endif
