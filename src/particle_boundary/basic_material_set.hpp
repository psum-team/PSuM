#ifndef PSUM_PARTICLE_BOUNDARY_BASIC_MATERIAL_SET_HPP
#define PSUM_PARTICLE_BOUNDARY_BASIC_MATERIAL_SET_HPP

#include <Eigen/Dense>
#include <sycl/sycl.hpp>
#include "../random/rander.hpp"
#include "../random/rand_functions.hpp"
#include "../tag/property.hpp"
#include "../particle_container/validator.hpp"

namespace psum {

namespace particle_boundary {

    template<typename ParticleType, template<typename> typename Validator>
    class basic_material_set {
    public:
        enum class material_type {
            diffuse,
            specular,
            maxwell_reflect,
            absorb
        };

        class acc_type {
        public:
            acc_type(double default_temperature, double default_mass)
                : default_temperature_(default_temperature), default_mass_(default_mass) {}

            void action(material_type mat, ParticleType& p,
                       double hit_x, double hit_y, double hit_z,
                       double nx, double ny, double nz) const {
                switch(mat) {
                    case material_type::diffuse:
                        reflect_diffuse_(p, hit_x, hit_y, hit_z, nx, ny, nz);
                        break;
                    case material_type::specular:
                        reflect_specular_(p, hit_x, hit_y, hit_z, nx, ny, nz);
                        break;
                    case material_type::maxwell_reflect:
                        reflect_maxwell_(p, hit_x, hit_y, hit_z, nx, ny, nz);
                        break;
                    case material_type::absorb:
                        absorb_(p);
                        break;
                }
            }

        private:
            double default_temperature_;
            double default_mass_;

            void reflect_diffuse_(ParticleType& p, double hit_x, double hit_y, double hit_z,
                                  double nx, double ny, double nz) const {
                auto& pos = tag::get<tag::property::position>(p);
                auto& vel = tag::get<tag::property::velocity>(p);
                auto R = random::view_as_rander(tag::get<tag::property::random_seed>(p));

                Eigen::RowVector3d norm(nx, ny, nz);
                Eigen::RowVector3d hit(hit_x, hit_y, hit_z);
                Eigen::RowVector3d overrush = pos - hit;
                Eigen::Vector3d randv = random::RandFunction3D::RandV_Cosine(R.get_rander(), norm);
                Eigen::RowVector3d new_dir(randv.x(), randv.y(), randv.z());

                pos = hit + new_dir * overrush.norm() * 0.001;
                vel = new_dir * vel.norm();
            }

            void reflect_specular_(ParticleType& p, double hit_x, double hit_y, double hit_z,
                                   double nx, double ny, double nz) const {
                auto& pos = tag::get<tag::property::position>(p);
                auto& vel = tag::get<tag::property::velocity>(p);

                Eigen::RowVector3d norm(nx, ny, nz);
                Eigen::RowVector3d hit(hit_x, hit_y, hit_z);
                Eigen::RowVector3d overrush = pos - hit;

                pos = hit + (overrush - 2.0 * overrush.dot(norm) * norm) * 0.001;
                vel = vel - 2.0 * vel.dot(norm) * norm;
            }

            void reflect_maxwell_(ParticleType& p, double hit_x, double hit_y, double hit_z,
                                  double nx, double ny, double nz) const {
                auto& pos = tag::get<tag::property::position>(p);
                auto& vel = tag::get<tag::property::velocity>(p);
                auto R = random::view_as_rander(tag::get<tag::property::random_seed>(p));

                Eigen::RowVector3d hit(hit_x, hit_y, hit_z);
                double overrush_t = (pos - hit).norm()/vel.norm();

                Eigen::RowVector3d norm(nx, ny, nz);
                norm.normalize();

                Eigen::RowVector3d new_vel = random::RandFunction3D::RandV_halfMaxw(R.get_rander(), norm, default_temperature_, default_mass_);

                pos = hit + new_vel * overrush_t;
                vel = new_vel;
            }

            void absorb_(ParticleType& p) const {
                Validator<ParticleType>::make_invalid(p);
            }
        };

        basic_material_set() = default;

        double get_default_temperature() const { return default_temperature_; }
        double get_default_mass() const { return default_mass_; }
        void set_default_temperature(double temp) { default_temperature_ = temp; }
        void set_default_mass(double mass) { default_mass_ = mass; }

        acc_type get_access(sycl::handler& h) const {
            return acc_type(default_temperature_, default_mass_);
        }

    private:
        double default_temperature_ = 273.15;
        double default_mass_ = 9.11e-31;
    };

}

}

#endif
