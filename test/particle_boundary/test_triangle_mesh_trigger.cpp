#include <iostream>
#include <vector>
#include <cmath>
#include <sycl/sycl.hpp>
#include "../../src/random/rander.hpp"
#include "../../src/particle_boundary/triangle_mesh_trigger.hpp"
#include "../../src/particle_boundary/geometry.hpp"
#include "../../src/field/device_array.hpp"
#include <Eigen/Dense>

using namespace psum::random;
using namespace psum::particle_boundary;
using namespace psum::field;

typedef int8_t MeshId;

struct ray_data {
    double p1x, p1y, p1z;
    double p2x, p2y, p2z;
    int hit;
    MeshId mesh_id;
    double k, nx, ny, nz;
};

bool verify_trigger_by_direct_check(const std::vector<triangle_with_id<MeshId>>& tris,
                                  double p1x, double p1y, double p1z,
                                  double p2x, double p2y, double p2z) {
    if (tris.empty()) {
        return false;
    }
    
    double min_lambda = 1.0;
    bool found = false;
    
    for (const auto& trig : tris) {
        double lambda;
        bool hit = geometry::line_seg_tri_intersect_test(
            trig.data_.p1x, trig.data_.p1y, trig.data_.p1z,
            trig.data_.p2x, trig.data_.p2y, trig.data_.p2z,
            trig.data_.p3x, trig.data_.p3y, trig.data_.p3z,
            p1x, p1y, p1z, p2x, p2y, p2z,
            lambda
        );
        
        if (hit && lambda < min_lambda) {
            min_lambda = lambda;
            found = true;
        }
    }
    
    return found;
}

void test_triangle_mesh_trigger_random() {
    std::cout << "==========================================" << std::endl;
    std::cout << "check: triangle_mesh_trigger random test" << std::endl;

    sycl::queue q;
    std::cout << "Running on " << q.get_device().get_info<sycl::info::device::name>() << std::endl;

    std::vector<triangle_with_id<MeshId>> triangles;

    rander seed;
    auto R = [&seed](){return seed() * 2;};

    for (MeshId i = 0; i < 10; ++i) {
        geometry::triangle_data trig;
        trig.p1x = R(); trig.p1y = R(); trig.p1z = R();
        trig.p2x = R(); trig.p2y = R(); trig.p2z = R();
        trig.p3x = R(); trig.p3y = R(); trig.p3z = R();
        triangles.push_back({trig, i});
    }

    triangle_mesh_trigger<MeshId> trigger(q);
    trigger.set(triangles, 0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 2, 2, 2);
    trigger.set_restrict();

    int intersected = 0;
    int total = 1000;
    int passed = 0;

    std::vector<ray_data> rays(total);
    for (int i = 0; i < total; ++i) {
        rays[i].p1x = R(); rays[i].p1y = R(); rays[i].p1z = R();
        rays[i].p2x = R(); rays[i].p2y = R(); rays[i].p2z = R();
    }

    device_array<ray_data> rays_d(q, rays);

    rays_d.for_each([&](sycl::handler& h) {
        auto tmt_acc = trigger.get_access(h);
        return [=](ray_data& ray) {
            bool hit = tmt_acc.detect(
                ray.p1x, ray.p1y, ray.p1z,
                ray.p2x, ray.p2y, ray.p2z,
                ray.k, ray.nx, ray.ny, ray.nz,
                ray.mesh_id
            );
            ray.hit = hit ? 1 : 0;
        };
    });

    rays = rays_d.to_host();

    for (int i = 0; i < total; ++i) {
        bool hit = (rays[i].hit == 1);
        double k = rays[i].k;

        if (hit) {
            intersected++;
            if (k <= 0.0 || k > 1.0) {
                std::cout << "Invalid k: " << k << std::endl;
            }
        }

        bool hit_direct = verify_trigger_by_direct_check(triangles,
            rays[i].p1x, rays[i].p1y, rays[i].p1z,
            rays[i].p2x, rays[i].p2y, rays[i].p2z);
        if (hit_direct == hit) {
            passed++;
        }
    }

    double hit_rate = static_cast<double>(intersected) / total * 100.0;
    std::cout << "Random test: " << passed << "/" << total << " passed" << std::endl;
    std::cout << "hit cases: " << intersected << "/" << total << " (" << hit_rate << "%)" << std::endl;
    if (passed == total && intersected > 0) {
        std::cout << "✅ Random test passed!" << std::endl;
    } else {
        std::cout << "❌ Random test failed!" << std::endl;
    }
}

int main() {
    test_triangle_mesh_trigger_random();
    return 0;
}
