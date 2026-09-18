#include <iostream>
#include <map>
#include <string>
#include <Eigen/Dense>
#include "../../include/psum/tag.hpp"
#include "../../src/particle_collision/foundation.hpp"

using namespace psum;
using namespace tag;
using namespace tag::property;
using namespace psum::particle_collision::foundation;

using particle = tagged_struct<
    tag_bind<position, Eigen::Vector3d>,
    tag_bind<velocity, Eigen::Vector3d>,
    tag_bind<random_seed, uint32_t>
>;

std::map<std::string, int> stats;

struct Xe_Elastic {
    static double collision_cross_section(double v2) { return 1.0e-20; }
    static void collide(particle& p) { stats["Xe_Elastic"]++; }
};
struct Xe_Ionization {
    static double collision_cross_section(double v2) { return 0.5e-20; }
    static void collide(particle& p) { stats["Xe_Ionization"]++; }
};

struct XePlus_Excitation {
    static double collision_cross_section(double v2) { return 0.2e-20; }
    static void collide(particle& p) { stats["XePlus_Excite"]++; }
};

struct AtomDensity {
    static double compute(const Eigen::Vector3d& pos) { return 1e20; }
};
struct IonDensity {
    static double compute(const Eigen::Vector3d& pos) { return 1e18; }
};

int main() {
    particle p{ {0,0,0}, {1000, 0, 0}, 1234567 };
    double dt = 1e-6;
    const int num_tests = 1000000;

    using GlobalMCC = sequence_executor<
        collision_processor<
            AtomDensity, std::tuple<Xe_Elastic, Xe_Ionization>
        >,
        collision_processor<
            IonDensity, std::tuple<XePlus_Excitation>
        >
    >;

    std::cout << "Starting Monte Carlo Statistical Test (" << num_tests << " iterations)...\n";

    for(int i = 0; i < num_tests; ++i) {
        GlobalMCC::execute(p, dt);
    }

    double v_mag = psum::tag::get<velocity>(p).norm();
    double n_atom = AtomDensity::compute(psum::tag::get<position>(p));
    double n_ion = IonDensity::compute(psum::tag::get<position>(p));

    std::cout << "\n--- Statistics ---\n";
    for (auto const& [name, count] : stats) {
        std::cout << name << ": " << count << " events\n";
    }

    double expected_elastic = Xe_Elastic::collision_cross_section(0) * v_mag * n_atom * dt * num_tests;
    std::cout << "\nTheory Check (Xe_Elastic):\n";
    std::cout << "  Expected: ~" << (int)expected_elastic << "\n";
    std::cout << "  Measured:  " << stats["Xe_Elastic"] << "\n";

    return 0;
}