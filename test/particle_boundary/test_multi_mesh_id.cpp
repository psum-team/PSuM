#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <fstream>
#include <map>
#include <sycl/sycl.hpp>
#include "../../src/timer.hpp"
#include "../../src/random/rander.hpp"
#include "../../src/random/rand_functions.hpp"
#include "../../src/particle_boundary/triangle_mesh_trigger.hpp"
#include "../../src/particle_boundary/stl_loader.hpp"
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

inline auto translate(std::vector<geometry::triangle_data>&& triangles_, double dx, double dy, double dz) {
    std::vector<geometry::triangle_data> triangles = triangles_;
    for (auto& trig : triangles) {
        trig.p1x += dx;
        trig.p1y += dy;
        trig.p1z += dz;
        trig.p2x += dx;
        trig.p2y += dy;
        trig.p2z += dz;
        trig.p3x += dx;
        trig.p3y += dy;
        trig.p3z += dz;
    }
    return triangles;
}

void test_multi_mesh_id() {
    std::cout << "==========================================" << std::endl;
    std::cout << "check: multi mesh_id demonstration" << std::endl;

    sycl::queue q;
    std::cout << "Running on " << q.get_device().get_info<sycl::info::device::name>() << std::endl;

    Tic("build_mesh")
    triangle_mesh_trigger<int> trigger(q);
    trigger.set(
        {
            {translate(stl_loader::load("race.stl"), 0,0,0), 0},
            {translate(stl_loader::load("suv.stl"), 1,0,0), 1},
            {translate(stl_loader::load("firetruck.stl"), 2,0,0), 2},
        }, 150, 150, 150);
    trigger.set_restrict();
    Toc

    std::cout << "Grid: " << trigger.get_I() << "x" << trigger.get_J() << "x" << trigger.get_K() << std::endl;
    std::cout << "Bounds: [" << trigger.get_x0() << "," << trigger.get_y0() << "," << trigger.get_z0() << "] to ["
              << trigger.get_x1() << "," << trigger.get_y1() << "," << trigger.get_z1() << "]" << std::endl;

    const int num_rays = 8000000;
    const double ray_length = std::min({
        trigger.get_x1() - trigger.get_x0(),
        trigger.get_y1() - trigger.get_y0(),
        trigger.get_z1() - trigger.get_z0()
    }) * 0.02;

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

    Tic("detect")
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
    Toc

    rays = rays_d.to_host();

    std::map<int, std::vector<double>> hit_points_by_mesh;
    int total_hit_count = 0;

    for (int i = 0; i < num_rays; ++i) {
        if (rays[i].hit == 1) {
            total_hit_count++;
            int mesh_id = rays[i].mesh_id;

            double hit_x = rays[i].p1x + rays[i].k * (rays[i].p2x - rays[i].p1x);
            double hit_y = rays[i].p1y + rays[i].k * (rays[i].p2y - rays[i].p1y);
            double hit_z = rays[i].p1z + rays[i].k * (rays[i].p2z - rays[i].p1z);

            hit_points_by_mesh[mesh_id].push_back(hit_x);
            hit_points_by_mesh[mesh_id].push_back(hit_y);
            hit_points_by_mesh[mesh_id].push_back(hit_z);
        }
    }

    std::cout << "Total hit count: " << total_hit_count << " / " << num_rays << std::endl;
    for (const auto& [mesh_id, points] : hit_points_by_mesh) {
        std::cout << "Mesh " << mesh_id << ": " << points.size() / 3 << " hits" << std::endl;
    }

    for (const auto& [mesh_id, points] : hit_points_by_mesh) {
        std::string filename = "hit_data_mesh_" + std::to_string(mesh_id) + ".txt";
        std::ofstream out_file(filename, std::ios::out);
        int count = points.size() / 3;
        out_file << count << std::endl;
        for (size_t i = 0; i < points.size(); i += 3) {
            out_file << points[i] << " " << points[i+1] << " " << points[i+2] << std::endl;
        }
        out_file.close();
        std::cout << "Saved " << filename << std::endl;
    }

    std::cout << "✅ Multi mesh_id test completed!" << std::endl;
    PrintTimer
    std::cout << num_rays / TimeUsed("detect") << " rays per second" << std::endl;
}

int main() {
    test_multi_mesh_id();
    return 0;
}
