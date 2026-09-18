#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <sycl/sycl.hpp>
#include <psum/particle_boundary.hpp>
#include <psum/particle_container.hpp>
#include <psum/tag.hpp>
#include "../../src/random/rander.hpp"
#include "../../src/random/rand_functions.hpp"

using namespace psum::tag;
using namespace psum::tag::property;
using namespace psum::particle_boundary;
using namespace psum::particle_container;
using namespace psum::random::RandFunction3D;

using particle = tagged_struct<
    tag_bind<position, Eigen::RowVector3d>,
    tag_bind<velocity, Eigen::RowVector3d>,
    tag_bind<random_seed, uint32_t>
    >;
using ParticleGroup = particle_group<particle, pos_x_nan_is_invalid>;

using MaterialSet = basic_material_set<particle, pos_x_nan_is_invalid>;
using Router = boundary_router<MaterialSet, particle>;
using material_type = MaterialSet::material_type;

void test_boundary_router() {
    std::cout << "==========================================" << std::endl;
    std::cout << "check: boundary_router demonstration" << std::endl;

    sycl::queue q;
    std::cout << "Running on " << q.get_device().get_info<sycl::info::device::name>() << std::endl;

    Router router(q);

    auto outer_cube = mesh_generator::axis_aligned_cube(-5, -5, -5, 5, 5, 10);
    auto inner_nozzle = mesh_generator::revolve_zr_curve({
        {-4, 1.0},
        {-3.5, 1.0},
        {-3, 0.9},
        {-2.5, 0.8},
        {-2, 0.6},
        {-1.5, 0.5},
        {-1, 0.6},
        {-0.5, 0.8},
        {0, 1.0},
        {0.5, 1.2},
        {1, 1.4},
        {1.5, 1.6},
        {2, 1.8},
        {2.5, 1.9},
        {3, 2.0},
        {3.5, 2.0},
        {4, 2.0}
    }, 64);
    auto reflective_plane = mesh_generator::axis_aligned_plane(-5, -5, -4.0, 5, 5, -4.0);

    std::vector<triangle_mesh_with_id<material_type>> meshes = {
        {outer_cube, material_type::absorb},
        {inner_nozzle, material_type::specular},
        {reflective_plane, material_type::diffuse}
    };

    router.set(meshes, 100, 100, 100);

    
    const int num_particles = 20000;
    std::vector<particle> particles(num_particles);

    psum::random::rander R;
    for (int i = 0; i < num_particles; ++i) {
        double r = sqrt(R()) * 0.7;
        double theta = 2 * M_PI * R();
        get<position>(particles[i]) = {r * cos(theta), r * sin(theta), -4 + 0.1 * R()};
        get<velocity>(particles[i]) = RandV_Maxwell(R, 1, 1e-23);
        get<random_seed>(particles[i]) = R() * RAND_MAX;
    }

    particle_group<particle, pos_x_nan_is_invalid> particles_d(q);
    particles_d.insert(particles);

    double dt = 0.005;
    int max_iterations = 10000;
    int iteration = 0;
    int frame_interval = 100;

    while (iteration < max_iterations) {
        particles_d.for_each([&](sycl::handler& h) {
            auto router_acc = router.get_access(h);
            return [=](particle& p) {
                double p1x = get<position>(p).x();
                double p1y = get<position>(p).y();
                double p1z = get<position>(p).z();

                get<position>(p) += get<velocity>(p) * dt;

                double p2x = get<position>(p).x();
                double p2y = get<position>(p).y();
                double p2z = get<position>(p).z();

                router_acc.deal(p1x, p1y, p1z, p2x, p2y, p2z, p);
            };
        });

        iteration++;

        if (iteration % frame_interval == 0) {
            auto host_particles = particles_d.get_content().to_host();
            int active = 0;
            for (const auto& p : host_particles) {
                if (ParticleGroup::validator::is_valid(p)) {
                    active++;
                }
            }
            std::cout << "Iteration " << iteration << ": Active particles = " << active << std::endl;

            std::ostringstream filename;
            filename << "frame_" << std::setw(4) << std::setfill('0') << iteration << ".txt";

            std::ofstream out_file(filename.str());
            out_file << num_particles << std::endl;
            for (const auto& p : host_particles) {
                if (ParticleGroup::validator::is_valid(p)) {
                    auto pos = get<position>(p);
                    out_file << pos.x() << " " << pos.y() << " " << pos.z() << std::endl;
                }
            }
            out_file.close();

            if (active == 0) {
                break;
            }
        }
    }

    std::cout << "==========================================" << std::endl;
    std::cout << "Total iterations: " << iteration << std::endl;
    std::cout << "✅ Test completed! Frames saved." << std::endl;
}

int main() {
    test_boundary_router();
    return 0;
}
