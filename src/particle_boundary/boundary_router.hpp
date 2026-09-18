#ifndef PSUM_PARTICLE_BOUNDARY_BOUNDARY_ROUTER_HPP
#define PSUM_PARTICLE_BOUNDARY_BOUNDARY_ROUTER_HPP

#include <cmath>
#include <vector>
#include <stdexcept>
#include <sycl/sycl.hpp>
#include "triangle_mesh_trigger.hpp"
#include "geometry.hpp"
#include "../field/device_array.hpp"
#include "../tag/property.hpp"

namespace psum {

using namespace tag;
using namespace tag::property;

namespace particle_boundary {

    template<typename MaterialSet, typename ParticleType>
    class boundary_router_acc {
        using material_type = typename MaterialSet::material_type;
        using trigger_type = triangle_mesh_trigger<material_type>;
    public:
        boundary_router_acc(
                 const typename trigger_type::acc_type& trigger_acc,
                 const typename MaterialSet::acc_type& material_set_acc)
            :trigger_acc_(trigger_acc), material_set_acc_(material_set_acc) {}

        bool deal(double p1x, double p1y, double p1z,
                 double p2x, double p2y, double p2z,
                 ParticleType& p) const {
            double k, nx, ny, nz;
            material_type mat;

            bool hit = trigger_acc_.detect(
                p1x, p1y, p1z,
                p2x, p2y, p2z,
                k, nx, ny, nz,
                mat
            );

            if (!hit) {
                return false;
            }

            double hit_x = p1x + k * (p2x - p1x);
            double hit_y = p1y + k * (p2y - p1y);
            double hit_z = p1z + k * (p2z - p1z);

            material_set_acc_.action(mat, p, hit_x, hit_y, hit_z, nx, ny, nz);

            return true;
        }

    private:
        typename trigger_type::acc_type trigger_acc_;
        typename MaterialSet::acc_type material_set_acc_;
    };

    template<typename MaterialSet, typename ParticleType>
    class boundary_router {
    public:
        using material_type = typename MaterialSet::material_type;
        using trigger_type = triangle_mesh_trigger<material_type>;
        using acc_type = boundary_router_acc<MaterialSet, ParticleType>;

    private:
        trigger_type trigger_;
        MaterialSet material_set_;
        sycl::queue q_;

    public:
        boundary_router(const sycl::queue& q) : trigger_(q), q_(q) {}

        void set(const std::vector<triangle_mesh_with_id<material_type>>& meshes,
                 int I, int J, int K) {
            trigger_.set(meshes, I, J, K);
            trigger_.set_restrict();
        }

        auto& get_trigger() { return trigger_; }

        auto& get_material_set() { return material_set_; }

        acc_type get_access(sycl::handler& h) const {
            return acc_type(
                trigger_.get_access(h),
                material_set_.get_access(h)
            );
        }
    };

}

}

#endif
