#ifndef CUSTOM_MATERIAL_HPP
#define CUSTOM_MATERIAL_HPP

#include <psum/psum.hpp>

template<typename ParticleType, template<typename> typename Validator>
class simple_material_set {
public:
    enum class material_type {
        specular,
        maxwell_reflect,
        absorb
    };

    class acc_type {
    public:
        acc_type(double default_temperature, double default_mass, double default_sigma)
            : default_temperature_(default_temperature), default_mass_(default_mass), default_sigma_(default_sigma) {}

        void action(material_type mat, ParticleType &p,
                    double hit_x, double hit_y, double hit_z,
                    double nx, double ny, double nz) const
        {
            switch (mat) {
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
        double default_sigma_;

        void reflect_specular_(ParticleType &p, double hit_x, double hit_y, double hit_z,
                               double nx, double ny, double nz) const
        {
            using namespace psum;
            auto &pos = tag::get<tag::property::position>(p);
            auto &vel = tag::get<tag::property::velocity>(p);

            Eigen::Vector3d norm(nx, ny, nz);
            Eigen::Vector3d hit(hit_x, hit_y, hit_z);
            Eigen::Vector3d overrush = pos - hit;

            pos = hit + (overrush - 2.0 * overrush.dot(norm) * norm) * 0.001;
            vel = vel - 2.0 * vel.dot(norm) * norm;
        }

        void reflect_maxwell_(ParticleType &p, double hit_x, double hit_y, double hit_z,
                              double nx, double ny, double nz) const
        {
            using namespace psum;
            auto &pos = tag::get<tag::property::position>(p);
            auto &vel = tag::get<tag::property::velocity>(p);
            auto R = random::view_as_rander(tag::get<tag::property::random_seed>(p));

            Eigen::Vector3d hit(hit_x, hit_y, hit_z);
            double overrush_t = (pos - hit).norm() / vel.norm();

            Eigen::Vector3d norm(nx, ny, nz);
            norm.normalize();
            if (norm.dot(vel) > 0.0)
                norm *= -1;

            Eigen::Vector3d new_vel;

            if (R() < default_sigma_)
                new_vel = random::RandFunction3D::RandV_halfMaxw(R.get_rander(), norm, default_temperature_, default_mass_);
            else
                new_vel = vel - 2.0 * vel.dot(norm) * norm;

            pos = hit + new_vel * overrush_t;
            vel = new_vel;
        }

        void absorb_(ParticleType &p) const
        {
            Validator<ParticleType>::make_invalid(p);
        }
    };

    simple_material_set() = default;

    double get_default_temperature() const { return default_temperature_; }
    double get_default_mass() const { return default_mass_; }
    void set_default_temperature(double temp) { default_temperature_ = temp; }
    void set_default_mass(double mass) { default_mass_ = mass; }

    acc_type get_access(sycl::handler &h) const
    {
        return acc_type(default_temperature_, default_mass_, default_sigma_);
    }

private:
    double default_temperature_ = 273.15;
    double default_mass_ = 9.11e-31;
    double default_sigma_ = 0.8;
};

#endif