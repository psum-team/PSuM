#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <fstream>
#include <sycl/sycl.hpp>
#include <psum/particle_boundary.hpp>
#include "../../src/timer.hpp"
#include "../../src/random/rander.hpp"
#include "../../src/random/rand_functions.hpp"
#include "../../src/field/device_array.hpp"

using namespace psum::random;
using namespace psum::particle_boundary;
using namespace psum::field;

struct ray_data {
    double p1x, p1y, p1z;
    double p2x, p2y, p2z;
    int hit;
    int mesh_id;
    double k, nx, ny, nz;
};

void test_stl_load_and_detect() {
    std::cout << "==========================================" << std::endl;
    std::cout << "check: STL load and collision detection" << std::endl;

    sycl::queue q;
    std::cout << "Running on " << q.get_device().get_info<sycl::info::device::name>() << std::endl;

    std::string stl_file = "example.stl";

    Tic("build_mesh")
    triangle_mesh_trigger<int> trigger(q);
    trigger.set({{stl_loader::load(stl_file), 0}}, 200, 200, 200);
    Toc

    std::cout << "Grid: " << trigger.get_I() << "x" << trigger.get_J() << "x" << trigger.get_K() << std::endl;
    std::cout << "Bounds: [" << trigger.get_x0() << "," << trigger.get_y0() << "," << trigger.get_z0() << "] to ["
              << trigger.get_x1() << "," << trigger.get_y1() << "," << trigger.get_z1() << "]" << std::endl;

    const int num_rays = 1000000;
    const double ray_length = std::min({
        trigger.get_x1() - trigger.get_x0(),
        trigger.get_y1() - trigger.get_y0(),
        trigger.get_z1() - trigger.get_z0()
    }) * 0.002;

    std::vector<ray_data> rays(num_rays);

    rander seed;
    auto R_x = [&seed, &trigger](){return trigger.get_x0() + trigger.span_x() * seed();};
    auto R_y = [&seed, &trigger](){return trigger.get_y0() + trigger.span_y() * seed();};
    auto R_z = [&seed, &trigger](){return trigger.get_z0() + trigger.span_z() * seed();};

    for (int i = 0; i < num_rays; ++i) {
        double p1x = R_x();
        double p1y = R_y();
        double p1z = R_z();

        auto dir = psum::random::RandFunction3D::RandV_spherical(seed);

        rays[i].p1x = p1x;
        rays[i].p1y = p1y;
        rays[i].p1z = p1z;
        rays[i].p2x = p1x + dir.x() * ray_length;
        rays[i].p2y = p1y + dir.y() * ray_length;
        rays[i].p2z = p1z + dir.z() * ray_length;
    }

    device_array<ray_data> rays_d(q, rays);

    int n_loop = 100;

    Tic("warmup")
    for (int i = 0; i < n_loop / 10; ++i) {
        rays_d.for_each([&](sycl::handler& h) {
            auto tmt_acc = trigger.get_access(h);
            return [=](ray_data& ray) {
                bool hit = tmt_acc.detect(
                    ray.p1x, ray.p1y, ray.p1z,
                    ray.p2x, ray.p2y, ray.p2z,
                    ray.k, ray.nx, ray.ny, ray.nz,
                    ray.mesh_id
                );
            };
        });
    }
    Toc

    Tic("detect")
    for (int i = 0; i < n_loop; ++i) {
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
    }
    Toc

    rays = rays_d.to_host();

    int hit_count = 0;
    std::vector<double> hit_points;
    std::vector<double> hit_normals;

    for (int i = 0; i < num_rays; ++i) {
        if (rays[i].hit == 1) {
            hit_count++;
            double hit_x = rays[i].p1x + rays[i].k * (rays[i].p2x - rays[i].p1x);
            double hit_y = rays[i].p1y + rays[i].k * (rays[i].p2y - rays[i].p1y);
            double hit_z = rays[i].p1z + rays[i].k * (rays[i].p2z - rays[i].p1z);

            hit_points.push_back(hit_x);
            hit_points.push_back(hit_y);
            hit_points.push_back(hit_z);

            hit_normals.push_back(rays[i].nx);
            hit_normals.push_back(rays[i].ny);
            hit_normals.push_back(rays[i].nz);
        }
    }

    std::cout << "Hit count: " << hit_count << " / " << num_rays << std::endl;

    std::ofstream out_file("hit_data.txt");
    out_file << hit_count << std::endl;
    for (size_t i = 0; i < hit_points.size(); i += 3) {
        out_file << hit_points[i] << " " << hit_points[i+1] << " " << hit_points[i+2] << " "
                 << hit_normals[i] << " " << hit_normals[i+1] << " " << hit_normals[i+2] << std::endl;
    }
    out_file.close();

    std::cout << "✅ STL load and detect test completed!" << std::endl;
    PrintTimer
    std::cout << num_rays * n_loop / TimeUsed("detect") << " rays per second" << std::endl;
}

int main() {
    test_stl_load_and_detect();
    return 0;
}
