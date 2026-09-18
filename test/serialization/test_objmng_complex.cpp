#include <psum/tag.hpp>
#include <psum/serialization.hpp>
#include <iostream>

using namespace std;
using namespace psum::tag;
using namespace psum::tag::property;
using namespace psum::serialization;

struct metadata: psum::tag::foundation::abstract_tag { inline const static std::string tag_name = "metadata"; };
struct id: psum::tag::foundation::abstract_tag { inline const static std::string tag_name = "id"; };
struct name: psum::tag::foundation::abstract_tag { inline const static std::string tag_name = "name"; };
struct tags: psum::tag::foundation::abstract_tag { inline const static std::string tag_name = "tags"; };
struct system_name: psum::tag::foundation::abstract_tag { inline const static std::string tag_name = "system_name"; };
struct particles: psum::tag::foundation::abstract_tag { inline const static std::string tag_name = "particles"; };
struct parameters: psum::tag::foundation::abstract_tag { inline const static std::string tag_name = "parameters"; };
struct subsystem: psum::tag::foundation::abstract_tag { inline const static std::string tag_name = "subsystem"; };
struct sub_name: psum::tag::foundation::abstract_tag { inline const static std::string tag_name = "sub_name"; };
struct sub_particles: psum::tag::foundation::abstract_tag { inline const static std::string tag_name = "sub_particles"; };

// basic particle definition
using Particle = tagged_struct<
    tag_bind<position, Eigen::RowVector3d>,
    tag_bind<velocity, Eigen::RowVector3d>,
    tag_bind<weight, double>
>;

// complex particle definition - nested Tagged_Struct
using ComplexParticle = tagged_struct<
    tag_bind<position, Eigen::RowVector3d>,
    tag_bind<velocity, Eigen::RowVector3d>,
    tag_bind<weight, double>,
    tag_bind<metadata, tagged_struct<
        tag_bind<id, int>,
        tag_bind<name, std::string>,
        tag_bind<tags, std::vector<std::string>>
    >>
>;

// physical system definition
using PhysicalSystem = tagged_struct<
    tag_bind<system_name, std::string>,
    tag_bind<particles, std::vector<Particle>>,
    tag_bind<parameters, std::map<std::string, double>>,
    tag_bind<subsystem, std::vector<tagged_struct<
        tag_bind<sub_name, std::string>,
        tag_bind<sub_particles, std::vector<Particle>>
    >>>
>;

auto check_same_particle = [](const Particle& p1, const Particle& p2) {
    try {
        assert(get<position>(p1) == get<position>(p2));
        assert(get<velocity>(p1) == get<velocity>(p2));
        assert(get<weight>(p1) == get<weight>(p2));
        return true;
    } catch (const std::exception& e) {
        return false;
    }
};

auto check_same_complex_particle = [](const ComplexParticle& cp1, const ComplexParticle& cp2) {
    try {
        assert(get<position>(cp1) == get<position>(cp2));
        assert(get<velocity>(cp1) == get<velocity>(cp2));
        assert(get<weight>(cp1) == get<weight>(cp2));
        assert(get<id>(get<metadata>(cp1)) == get<id>(get<metadata>(cp2)));
        assert(get<name>(get<metadata>(cp1)) == get<name>(get<metadata>(cp2)));
        assert(get<tags>(get<metadata>(cp1)) == get<tags>(get<metadata>(cp2)));
        return true;
    } catch (const std::exception& e) {
        return false;
    }
};

template<typename T>
bool check_same_vector(const std::vector<T>& v1, const std::vector<T>& v2) {
    if (v1.size() != v2.size()) return false;
    for (size_t i = 0; i < v1.size(); ++i) {
        if constexpr (std::is_same_v<T, Particle>) {
            if (!check_same_particle(v1[i], v2[i])) return false;
        } else if constexpr (std::is_same_v<T, ComplexParticle>) {
            if (!check_same_complex_particle(v1[i], v2[i])) return false;
        } else {
            if (v1[i] != v2[i]) return false;
        }
    }
    return true;
}

template<typename K, typename V>
bool check_same_map(const std::map<K, V>& m1, const std::map<K, V>& m2) {
    if (m1.size() != m2.size()) return false;
    for (const auto& [key, value] : m1) {
        if (m2.find(key) == m2.end()) return false;
        if (m2.at(key) != value) return false;
    }
    return true;
}

int main() {
    object_manager mng("");

    // basic test
    auto& matrix = mng.obj<Eigen::MatrixXd>("my_matrix");
    auto& config = mng.obj<std::map<std::string, double>>("config");
    auto& counter = mng.obj<int>("counter");
    auto& p1 = mng.obj<Particle>("p1");

    matrix = Eigen::MatrixXd::Random(3, 3);
    config["threshold"] = 0.5;
    config["max_iter"] = 100;
    config["alpha"] = 0.1;
    counter = 999;
    get<position>(p1) = Eigen::RowVector3d(1, 2, 3);
    get<velocity>(p1) = Eigen::RowVector3d(4, 5, 6);
    get<weight>(p1) = 0.1;

    // complex particle test (nested Tagged_Struct)
    auto& complex_particle = mng.obj<ComplexParticle>("complex_particle");
    get<position>(complex_particle) = Eigen::RowVector3d(10, 20, 30);
    get<velocity>(complex_particle) = Eigen::RowVector3d(40, 50, 60);
    get<weight>(complex_particle) = 2.5;
    get<id>(get<metadata>(complex_particle)) = 12345;
    get<name>(get<metadata>(complex_particle)) = "test_particle";
    get<tags>(get<metadata>(complex_particle)) = {"physics", "simulation", "test"};

    // vector<Particle> test 
    auto& particle_vector = mng.obj<std::vector<Particle>>("particle_vector");
    particle_vector.resize(3);
    for (int i = 0; i < 3; ++i) {
        get<position>(particle_vector[i]) = Eigen::RowVector3d(i, i*2, i*3);
        get<velocity>(particle_vector[i]) = Eigen::RowVector3d(i*4, i*5, i*6);
        get<weight>(particle_vector[i]) = i * 0.5;
    }

    // map<string, ComplexParticle> test
    auto& complex_particle_map = mng.obj<std::map<std::string, ComplexParticle>>("complex_particle_map");
    complex_particle_map["particle1"] = complex_particle;
    complex_particle_map["particle2"] = complex_particle;
    get<name>(get<metadata>(complex_particle_map["particle2"])) = "second_particle";

    // vector<ComplexParticle> test
    auto& complex_particle_vector = mng.obj<std::vector<ComplexParticle>>("complex_particle_vector");
    complex_particle_vector.resize(2);
    for (int i = 0; i < 2; ++i) {
        get<position>(complex_particle_vector[i]) = Eigen::RowVector3d(i*10, i*20, i*30);
        get<velocity>(complex_particle_vector[i]) = Eigen::RowVector3d(i*40, i*50, i*60);
        get<weight>(complex_particle_vector[i]) = i * 1.5;
        get<id>(get<metadata>(complex_particle_vector[i])) = 1000 + i;
        get<name>(get<metadata>(complex_particle_vector[i])) = "vector_particle_" + std::to_string(i);
        get<tags>(get<metadata>(complex_particle_vector[i])) = {"tag" + std::to_string(i)};
    }

    // tagged_struct<..., vector, map, ...> test
    auto& physical_system = mng.obj<PhysicalSystem>("physical_system");
    get<system_name>(physical_system) = "TestPhysicalSystem";
    get<particles>(physical_system) = particle_vector;
    get<parameters>(physical_system) = config;
    
    auto& subsystems = get<subsystem>(physical_system);
    subsystems.resize(2);
    for (int i = 0; i < 2; ++i) {
        get<sub_name>(subsystems[i]) = "subsystem_" + std::to_string(i);
        get<sub_particles>(subsystems[i]) = std::vector<Particle>{particle_vector[0], particle_vector[1]};
    }

    // map of map test
    auto& nested_map = mng.obj<std::map<std::string, std::map<std::string, Particle>>>("nested_map");
    nested_map["group1"]["particle1"] = p1;
    nested_map["group1"]["particle2"] = p1;
    nested_map["group2"]["particle1"] = p1;
    get<weight>(nested_map["group2"]["particle1"]) = 0.9;

    // save data
    mng.save("tmp_complex_saved.mas");

    // load and verify
    object_manager mng2("tmp_complex_saved.mas");
    
    if (mng2.obj<Eigen::MatrixXd>("my_matrix") == matrix) 
        cout << "S/L matrix: pass\n"; 
    else 
        cout << "S/L matrix: fail\n";
        
    if (mng2.obj<std::map<std::string, double>>("config") == config) 
        cout << "S/L config: pass\n"; 
    else 
        cout << "S/L config: fail\n";
        
    if (mng2.obj<int>("counter") == counter) 
        cout << "S/L counter: pass\n"; 
    else 
        cout << "S/L counter: fail\n";
        
    if (check_same_particle(mng2.obj<Particle>("p1"), p1)) 
        cout << "S/L p1: pass\n"; 
    else 
        cout << "S/L p1: fail\n";

    if (check_same_complex_particle(mng2.obj<ComplexParticle>("complex_particle"), complex_particle))
        cout << "S/L complex_particle: pass\n";
    else
        cout << "S/L complex_particle: fail\n";

    if (check_same_vector(mng2.obj<std::vector<Particle>>("particle_vector"), particle_vector))
        cout << "S/L particle_vector: pass\n";
    else
        cout << "S/L particle_vector: fail\n";

    if (check_same_map(mng2.obj<std::map<std::string, ComplexParticle>>("complex_particle_map"), complex_particle_map))
        cout << "S/L complex_particle_map: pass\n";
    else
        cout << "S/L complex_particle_map: fail\n";

    if (check_same_vector(mng2.obj<std::vector<ComplexParticle>>("complex_particle_vector"), complex_particle_vector))
        cout << "S/L complex_particle_vector: pass\n";
    else
        cout << "S/L complex_particle_vector: fail\n";

    // complex tagged_struct varify
    auto& loaded_system = mng2.obj<PhysicalSystem>("physical_system");
    if (get<system_name>(loaded_system) == get<system_name>(physical_system))
        cout << "S/L physical_system name: pass\n";
    else
        cout << "S/L physical_system name: fail\n";

    if (check_same_vector(get<particles>(loaded_system), get<particles>(physical_system)))
        cout << "S/L physical_system particles: pass\n";
    else
        cout << "S/L physical_system particles: fail\n";

    if (check_same_map(get<parameters>(loaded_system), get<parameters>(physical_system)))
        cout << "S/L physical_system parameters: pass\n";
    else
        cout << "S/L physical_system parameters: fail\n";

    // nested map varify
    auto& loaded_nested_map = mng2.obj<std::map<std::string, std::map<std::string, Particle>>>("nested_map");
    bool nested_map_pass = true;
    for (const auto& [group, particles] : nested_map) {
        if (loaded_nested_map.find(group) == loaded_nested_map.end()) {
            nested_map_pass = false;
            break;
        }
        for (const auto& [name, particle] : particles) {
            if (loaded_nested_map[group].find(name) == loaded_nested_map[group].end() || 
                !check_same_particle(loaded_nested_map[group][name], particle)) {
                nested_map_pass = false;
                break;
            }
        }
    }
    if (nested_map_pass)
        cout << "S/L nested_map: pass\n";
    else
        cout << "S/L nested_map: fail\n";

    return 0;
}