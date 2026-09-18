#include <iostream>
#include <vector>
#include <cmath>
#include <sycl/sycl.hpp>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <psum/tag.hpp>
#include <psum/field.hpp>
#include <psum/particle_container.hpp>
#include <psum/particle_boundary.hpp>
#include "../../src/random/rander.hpp"
#include "../../src/random/rand_functions.hpp"
#include "../../src/timer.hpp"

using namespace psum;
using namespace psum::tag;
using namespace psum::tag::property;
using namespace psum::particle_boundary;
using namespace psum::particle_container;
using namespace psum::field;
using namespace psum::random::RandFunction3D;
using namespace psum::field::interp_tools;

using Vec3 = Eigen::RowVector3d;

struct init_pos : tag::foundation::abstract_tag {
    inline const static std::string tag_name = "init_pos";
};

using particle = tagged_struct<
    tag_bind<position, Vec3>,
    tag_bind<velocity, Vec3>,
    tag_bind<random_seed, uint32_t>,
    tag_bind<init_pos, std::pair<double, double>>,
    tag_bind<weight, float>
>;
using ParticleGroup = particle_group<particle, pos_x_nan_is_invalid>;

using Grid2D = simple_grid<2>;
using cf = device_field<2, var_loc::cellCentered, double, 3>;

struct CameraParams {
    double distance;
    double focal_length;
    double azimuth;
    double elevation;
    int width;
    int height;

    Vec3 get_look_dir() const {
        double az = azimuth * M_PI / 180.0;
        double el = elevation * M_PI / 180.0;
        Eigen::AngleAxisd r_az(az, Eigen::Vector3d::UnitZ());
        Eigen::AngleAxisd r_el(-el, Eigen::Vector3d::UnitX());
        return (r_az * r_el * Eigen::Vector3d::UnitY()).normalized();
    }

    Vec3 get_position(const Vec3& center) const {
        return center - get_look_dir() * distance;
    }

    Vec3 get_direction(double u, double v) const {
        double hw = width / 2.0;
        double hh = height / 2.0;
        double tan_h = (u - hw) / hw * 18 / focal_length;
        double tan_v = (v - hh) / hw * 18 / focal_length;
        double az = azimuth * M_PI / 180.0;
        double el = elevation * M_PI / 180.0;
        Eigen::AngleAxisd r_az(az, Eigen::Vector3d::UnitZ());
        Eigen::AngleAxisd r_el(-el, Eigen::Vector3d::UnitX());
        return (r_az * r_el * Eigen::Vector3d(tan_h, 1, tan_v).normalized()).normalized();
    }
};

struct LightParams {
    Vec3 direction;
    double intensity;
    LightParams() : direction(Vec3(1, 1, -1).normalized()), intensity(1.0) {}
};

template<typename ParticleType>
class material_set_ray_tracing {
public:
    enum class material_type { model_surface, sky_wall, ground_surface };
    using cf_acc = cf::acc_type;

    class material_set_acc {
    public:
        cf_acc img_acc;
        Vec3 light_dir;
        double light_intensity;

        material_set_acc() : light_dir(0, 0, 1), light_intensity(1.0) {}

        material_set_acc(const cf_acc& img, const Vec3& light, double intensity)
            : img_acc(img), light_dir(light), light_intensity(intensity) {}

        Eigen::RowVector3d get_sky_color(const Vec3& dir) const {
            double t = 0.5 * (dir.z() + 1.0);
            t = std::max(0.0, std::min(1.0, t));
            return (1.0 - t) * Eigen::RowVector3d(0.85, 0.9, 0.95) + t * Eigen::RowVector3d(0.3, 0.5, 0.85);
        }

        void action(material_type mat, ParticleType& p,
                   double hit_x, double hit_y, double hit_z,
                   double nx, double ny, double nz) const {
            switch(mat) {
                case material_type::model_surface:
                    action_model(p, hit_x, hit_y, hit_z, nx, ny, nz);
                    break;
                case material_type::sky_wall:
                    action_sky(p, nx, ny, nz);
                    break;
                case material_type::ground_surface:
                    action_ground(p, hit_x, hit_y, hit_z, nx, ny, nz);
                    break;
            }
        }

    private:
        void action_model(ParticleType& p, double hx, double hy, double hz, double nx, double ny, double nz) const {
            auto& pos = get<position>(p);
            auto& vel = get<velocity>(p);
            auto& w = get<weight>(p);
            auto R = psum::random::view_as_rander(get<random_seed>(p));
            Vec3 norm(nx, ny, nz);
            Vec3 randv = RandV_Cosine(R.get_rander(), norm);
            pos = Vec3(hx, hy, hz) + randv * 1e-6;
            vel = randv * vel.norm();
            w *= 0.9f;
            if (w < 0.001f) pos_x_nan_is_invalid<ParticleType>::make_invalid(p);
        }

        void action_sky(ParticleType& p, double nx, double ny, double nz) const {
            auto& vel = get<velocity>(p);
            auto& w = get<weight>(p);
            auto init = get<init_pos>(p);
            Vec3 dir = vel.normalized();
            double cos_theta = std::max(0.0, -dir.dot(light_dir));
            Eigen::RowVector3d color = get_sky_color(dir) * cos_theta * w * light_intensity;
            add_back_nearest(Eigen::RowVector2d(init.first, init.second), color, img_acc);
            pos_x_nan_is_invalid<particle>::make_invalid(p);
        }

        void action_ground(ParticleType& p, double hx, double hy, double hz, double nx, double ny, double nz) const {
            auto& pos = get<position>(p);
            auto& vel = get<velocity>(p);
            auto& w = get<weight>(p);
            auto R = psum::random::view_as_rander(get<random_seed>(p));
            Vec3 norm(nx, ny, nz);
            Vec3 hit(hx, hy, hz);
            double r = R();
            if (r < 0.4) {
                Vec3 randv = RandV_Cosine(R.get_rander(), norm);
                pos = hit + randv * 1e-6;
                vel = randv * vel.norm();
            } else {
                Vec3 overrush = pos - hit;
                pos = hit + (overrush - 2.0 * overrush.dot(norm) * norm) * 1e-6;
                vel = vel - 2.0 * vel.dot(norm) * norm;
            }
            w *= 0.8f;
            if (w < 0.001f) pos_x_nan_is_invalid<ParticleType>::make_invalid(p);
        }
    };

    using acc_type = material_set_acc;
    cf* img_field;
    LightParams light_params_;

    material_set_ray_tracing() : img_field(nullptr) {}

    void set_fields(cf& img) { img_field = &img; }

    void set_light(const LightParams& light) { light_params_ = light; }

    material_set_acc get_access(sycl::handler& h) const {
        if (!img_field) throw std::runtime_error("fields not set");
        return material_set_acc(img_field->get_access(h), light_params_.direction, light_params_.intensity);
    }
};

using MaterialSet = material_set_ray_tracing<particle>;
using Router = boundary_router<MaterialSet, particle>;
using material_type = MaterialSet::material_type;

void run_ray_tracing(const std::string& prefix, int seed, bool use_udf) {

    std::string timer_name = use_udf ? "rendering_udf" : "rendering_baseline";
    std::string timer_name_detect = use_udf ? "detect_udf" : "detect_baseline";
    std::string timer_name_build = use_udf ? "build_udf" : "build_baseline";

    sycl::queue q;

    Tic(timer_name_build)
    auto triangles = stl_loader::load("example.stl");
    auto [lp, hp] = stl_loader::box_of_triangles(triangles);
    auto [xmin, ymin, zmin] = lp;
    auto [xmax, ymax, zmax] = hp;
    Vec3 center((xmin + xmax) / 2, (ymin + ymax) / 2, (zmin + zmax) / 2);
    Vec3 size(xmax - xmin, ymax - ymin, zmax - zmin);

    CameraParams cam;
    cam.distance = 3.0 * size.norm();
    cam.focal_length = 85.0;
    cam.azimuth = 30.0;
    cam.elevation = 15.0;
    cam.width = 640;
    cam.height = 480;

    LightParams light;
    light.direction = Vec3(-1, -1, 1).normalized();
    light.intensity = 1.0;

    Grid2D grid({0, 0}, {(double)cam.width, (double)cam.height}, {cam.width, cam.height});
    cf img(q, grid);
    Vec3 p0 = cam.get_position(center);
    double diameter = size.norm();

    double zx0 = std::min(xmin, p0.x()), zx1 = std::max(xmax, p0.x());
    double zy0 = std::min(ymin, p0.y()), zy1 = std::max(ymax, p0.y());
    double zz0 = std::min(zmin, p0.z()), zz1 = std::max(zmax, p0.z());

    auto sky_box = mesh_generator::axis_aligned_cube(zx0, zy0, zz0, zx1, zy1, zz1);
    mesh_generator::rescale_triangles(sky_box, 1.2);
    auto ground = mesh_generator::axis_aligned_plane(zx0, zy0, zz0, zx1, zy1, zz0);
    mesh_generator::rescale_triangles(ground, 1.2);

    Router router(q);
    router.set({
        {sky_box, material_type::sky_wall},
        {ground, material_type::ground_surface},
        {triangles, material_type::model_surface}
    }, 250, 250, 250);

    router.get_trigger().set_acceleration_UDF(use_udf);
    router.get_material_set().set_fields(img);
    router.get_material_set().set_light(light);

    psum::random::rander R(seed);
    int spp = 20, nloop = 5;
    std::vector<particle> particles;
    particles.resize(img.size() * spp);
    img.setConstant(Eigen::RowVector3d(0, 0, 0));
    Toc
    long long int n_detect = 0;
    Tic(timer_name)
    for (int i = 0; i < nloop; i++) {
        ParticleGroup particles_d(q);

        for (int j = 0; j < img.size(); j++) {
            for (int k = 0; k < spp; k++) {
                auto pos = grid.cellCorner(grid.i2c(j));
                double alpha = pos.x() + R() * grid.del<0>();
                double beta = pos.y() + R() * grid.del<1>();
                Vec3 vel = cam.get_direction(alpha, beta) * diameter * 0.02;

                particle p;
                get<position>(p) = p0;
                get<velocity>(p) = vel;
                get<random_seed>(p) = R() * RAND_MAX;
                get<init_pos>(p) = std::make_pair(alpha, beta);
                get<weight>(p) = 1.0f;
                particles[j * spp + k] = p;
            }
        }

        particles_d.insert(particles);
        int iter = 0;
        while (particles_d.size() > img.size() * spp * 1e-6 && iter < 1000) {
            n_detect += particles_d.size();
            Tic(timer_name_detect)
            particles_d.for_each([&](sycl::handler& h) {
                auto acc = router.get_access(h);
                return [=](particle& p) {
                    auto& pos = get<position>(p);
                    double p1x = pos.x(), p1y = pos.y(), p1z = pos.z();
                    pos += get<velocity>(p);
                    acc.deal(p1x, p1y, p1z, pos.x(), pos.y(), pos.z(), p);
                };
            });
            Toc
            if (iter % 100 == 0) particles_d.compress();
            iter++;
            if (iter % 50 == 0) std::cout << "rendering iteration " << iter << "; remaining particles: " << particles_d.size() << std::endl;
        }
    }
    std::cout << n_detect / TimeUsed(timer_name_detect) << " ray/s" << std::endl;
    Toc
    int total = spp * nloop;
    img.for_each([&](sycl::handler &h) {
        return [=](size_t i, typename cf::Value& val) {
            val /= total;
        };
    });
    img.plot(prefix + ".plt", "R,G,B");
    std::cout << (use_udf ? "UDF" : "Baseline") << " done" << std::endl;
}

int main() {
    std::cout << "=== UDF Ray Tracing Test ===" << std::endl;

    run_ray_tracing("ray_baseline", 12345, false);
    run_ray_tracing("ray_udf", 12345, true);

    double t0 = TimeUsed("detect_baseline");
    double t1 = TimeUsed("detect_udf");
    std::cout << "Baseline: " << t0 << "s, UDF: " << t1 << "s";
    if (t1 > 0) std::cout << ", Speedup: " << t0 / t1 << "x";
    std::cout << std::endl;
    PrintTimer
    system("python3 plot_compare.py ray_baseline.plt ray_udf.plt");

    return 0;
}
