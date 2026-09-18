#include <psum/tag.hpp>
#include <psum/serialization.hpp>
#include "../../src/particle_dictionary.hpp"
#include <iostream>
#include <vector>

using namespace psum;
using namespace psum::tag;
using namespace psum::serialization;
using namespace psum::particle_dictionary;

using Particle = tagged_struct<
    tag_bind<property::species_id, particle_kind>,
    tag_bind<property::mass, double>,
    tag_bind<property::charge, double>
>;

bool check_particle_same(const Particle& p1, const Particle& p2) {
    bool id_same = get<property::species_id>(p1) == get<property::species_id>(p2);
    bool mass_same = get<property::mass>(p1) == get<property::mass>(p2);
    bool charge_same = get<property::charge>(p1) == get<property::charge>(p2);
    return id_same && mass_same && charge_same;
}

int main() {
    psum::tag::ostream_settings::set_one_line();

    std::cout << "=== Particle Dictionary and Tag System Test ===" << std::endl;

    std::cout << "\n=== Test 1: Direct access via tag::get on particle_kind ===" << std::endl;
    
    auto mass = get<property::mass>(particle_kind::Electron);
    auto charge = get<property::charge>(particle_kind::Electron);
    std::cout << "Electron: mass=" << mass 
              << ", charge=" << charge << std::endl;

    mass = get<property::mass>(particle_kind::XenonIon);
    charge = get<property::charge>(particle_kind::XenonIon);
    std::cout << "Xenon Ion: mass=" << mass 
              << ", charge=" << charge << std::endl;

    std::cout << "\n=== Test 2: operator<< for particle_kind ===" << std::endl;
    std::cout << "Particle symbols: " 
              << particle_kind::Electron << ", " 
              << particle_kind::Hydrogen << ", " 
              << particle_kind::Helium << ", " 
              << particle_kind::HeliumDoubleIon << ", "
              << particle_kind::Xenon << ", "
              << particle_kind::XenonIon << std::endl;

    std::cout << "\n=== Test 3: Particle tagged_struct ===" << std::endl;
    
    Particle p1;
    get<property::species_id>(p1) = particle_kind::Electron;
    get<property::mass>(p1) = get<property::mass>(get<property::species_id>(p1));
    get<property::charge>(p1) = get<property::charge>(get<property::species_id>(p1));
    std::cout << p1 << std::endl;

    Particle p2;
    get<property::species_id>(p2) = particle_kind::XenonIon;
    get<property::mass>(p2) = get<property::mass>(get<property::species_id>(p2));
    get<property::charge>(p2) = get<property::charge>(get<property::species_id>(p2));
    std::cout << p2 << std::endl;

    std::cout << "\n=== Test 4: MAS serialization for single Particle ===" << std::endl;
    
    mas_file fp("particle_test.mas", mas_file::replaceMode);
    save(fp, "electron", p1);
    
    Particle p1_loaded;
    load(fp, "electron", p1_loaded);
    
    if (check_particle_same(p1, p1_loaded)) {
        std::cout << "Single Particle S/L: pass" << std::endl;
    } else {
        std::cout << "Single Particle S/L: failed" << std::endl;
    }

    std::cout << "\n=== Test 5: MAS serialization for multiple Particles ===" << std::endl;
    
    std::vector<Particle> particles_orig;
    for (int i = 0; i < 5; ++i) {
        Particle p;
        get<property::species_id>(p) = static_cast<particle_kind>(i % static_cast<int>(particle_kind::XenonIon) + 1);
        get<property::mass>(p) = get<property::mass>(get<property::species_id>(p));
        get<property::charge>(p) = get<property::charge>(get<property::species_id>(p));
        particles_orig.push_back(p);
    }
    
    save(fp, "particles", particles_orig);
    
    std::vector<Particle> particles_loaded;
    load(fp, "particles", particles_loaded);
    
    bool all_particles_same = (particles_orig.size() == particles_loaded.size());
    for (size_t i = 0; i < particles_orig.size() && all_particles_same; ++i) {
        if (!check_particle_same(particles_orig[i], particles_loaded[i])) {
            all_particles_same = false;
        }
    }
    
    if (all_particles_same) {
        std::cout << "Multiple Particles S/L: pass" << std::endl;
    } else {
        std::cout << "Multiple Particles S/L: failed" << std::endl;
    }
    
    std::filesystem::remove("particle_test.mas");
    return 0;
}
