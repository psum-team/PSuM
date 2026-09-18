#ifndef PSUM_MAGNET_NOZZLE_2D_SPECIES_HPP
#define PSUM_MAGNET_NOZZLE_2D_SPECIES_HPP

#include "types.hpp"
#include <sycl/sycl.hpp>

namespace magnet_nozzle_2d {

class Species : public ParticleGroup {
    double m_mass, m_charge, m_weight;

public:
    Species(sycl::queue& q, double mass, double charge, double weight, size_t capacity = 100000)
        : ParticleGroup(q, capacity), m_mass(mass), m_charge(charge), m_weight(weight) {}

    double mass() const { return m_mass; }
    double charge() const { return m_charge; }
    double weight() const { return m_weight; }

    double& mass() { return m_mass; }
    double& charge() { return m_charge; }
    double& weight() { return m_weight; }
};

}

namespace psum::serialization {

template<>
inline void save<magnet_nozzle_2d::Species>(mas_file& fp, const std::string& name, const magnet_nozzle_2d::Species& obj) {
    save(fp, name + ".particles", obj.get_content());
    save(fp, name + ".mass", obj.mass());
    save(fp, name + ".charge", obj.charge());
    save(fp, name + ".weight", obj.weight());
}

template<>
inline void load<magnet_nozzle_2d::Species>(mas_file& fp, const std::string& name, magnet_nozzle_2d::Species& obj) {
    sycl::queue q{sycl::default_selector_v};
    psum::particle_container::device_vector<magnet_nozzle_2d::Particle> d_vec(q);
    load(fp, name + ".particles", d_vec);
    obj.clear();
    obj.template insert<false>(d_vec);

    double mass, charge, weight;
    load(fp, name + ".mass", mass);
    load(fp, name + ".charge", charge);
    load(fp, name + ".weight", weight);
    obj.mass() = mass;
    obj.charge() = charge;
    obj.weight() = weight;
}

}

#endif
