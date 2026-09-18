#include <Eigen/Core>
#include <psum/particle_container.hpp>
#include <psum/tag.hpp>
#include "../../src/random/rander.hpp"

using namespace psum::tag;
using namespace psum::particle_container;
using property::position;
using property::velocity;
using psum::random::rander;

using Particle =
    tagged_struct<
        tag_bind<position, Eigen::RowVector3f>,
        tag_bind<velocity, Eigen::RowVector3f>>;

using pContainer =
    particle_group<Particle, pos_x_nan_is_invalid>;

int main() {
    sycl::queue q{ sycl::default_selector_v };

    std::cout << "Running particle_group test on device: "
              << q.get_device().get_info<sycl::info::device::name>() << std::endl;

    pContainer container(q);

    {
        pContainer empty_container(q);
        if (empty_container.size() != 0) { std::cout << "FAIL: empty container size != 0\n"; return 1; }
        empty_container.for_each([&](sycl::handler&) {
            return [=](Particle& p) { (void)p; };
        });
        if (empty_container.size() != 0) { std::cout << "FAIL: empty container size changed\n"; return 1; }
        std::cout << "empty-container for_each regression PASSED\n";
    }

    std::cout << "initial capacity = " << container.capacity() << std::endl;
    container.reserve(10000);
    std::cout << "capacity after reserve = " << container.capacity() << std::endl;

    int num_particles = 1e6;
    rander R;
    std::vector<Particle> p_vec;
    for (int i = 0; i < num_particles; ++i) {
        Particle p;
        get<position>(p) = {R(), R(), R()};
        get<velocity>(p) = {R(), R(), R()};
        p_vec.push_back(p);
    }
    int remove_count = 0;
    for (auto p : p_vec) {
        if (get<position>(p).x() > 0.5 && get<position>(p).y() < 0.6)
            remove_count++;
    }

    container.insert(p_vec);

    std::cout << "After first insert, size = " << container.size() << std::endl;
    assert(container.size() == num_particles);

    container.for_each([&](sycl::handler&) {
        return [=](Particle& p) {
            if (get<position>(p).x() > 0.5 && get<position>(p).y() < 0.6) {
                pContainer::validator::make_invalid(p);
            }
        };
    });

    std::cout << "After marking invalid particles, size should be " << num_particles - remove_count << std::endl;
    std::cout << "After marking invalid particles, size = " << container.size();
    if (container.size() == num_particles - remove_count) std::cout << " ✅" << std::endl;
    else std::cout << " ❌" << std::endl;

    //another loop
    container.for_each([&](sycl::handler&) {
        return [](Particle& p) {
            if (get<position>(p).x() > 0.5 && get<position>(p).y() < 0.6) {
                pContainer::validator::make_invalid(p);
            }
        };
    });
    std::cout << "After marking invalid particles again, size = " << container.size();
    if (container.size() == num_particles - remove_count) std::cout << " ✅" << std::endl;
    else std::cout << " ❌" << std::endl;

    std::vector<Particle> p_vec_2;
    for (int i = 0; i < num_particles * 0.2; ++i) {
        Particle p;
        get<position>(p) = {R(), R(), R()};
        get<velocity>(p) = {R(), R(), R()};
        p_vec_2.push_back(p);
    }

    container.insert(p_vec_2);
    
    std::cout << "After second insert, size should be " << num_particles - remove_count + p_vec_2.size() << std::endl;
    std::cout << "After second insert, size = " << container.size();
    if (container.size() == num_particles - remove_count + p_vec_2.size()) std::cout << " ✅" << std::endl;

    std::cout << "capacity = " << container.capacity();
    if (container.capacity() == num_particles) std::cout << " ✅" << std::endl;
    else std::cout << " ❌" << std::endl;

    container.shrink(container.size());
    std::cout << "After shrink, capacity = " << container.capacity();
    if (container.capacity() == container.size()) std::cout << " ✅" << std::endl;
    else std::cout << " ❌" << std::endl;

    return 0;
}
