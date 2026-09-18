#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <sycl/sycl.hpp>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <psum/tag.hpp>
#include <psum/field.hpp>
#include <psum/particle_container.hpp>
#include <psum/particle_boundary.hpp>
#include "../../src/random/rander.hpp"
#include "../../src/random/rand_functions.hpp"

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
using cf = device_field<2, var_loc::cellCentered, double, 3>;  // RGB channels

struct CameraParams {
    double distance_from_center;  // Distance from model center
    double focal_length_mm;       // Focal length (determines FOV)
    double azimuth;               // Rotation angle around Y axis (degrees)
    double elevation;             // Pitch angle around X axis (degrees)
    int image_width;
    int image_height;

    Vec3 get_look_direction() const {
        double az = azimuth * M_PI / 180.0;
        double el = elevation * M_PI / 180.0;

        Eigen::AngleAxisd r_az(az, Eigen::Vector3d::UnitZ());
        Eigen::AngleAxisd r_el(-el, Eigen::Vector3d::UnitX());

        return (r_az * r_el * Eigen::Vector3d::UnitY()).normalized();
    }

    Vec3 get_position(const Vec3& center) const {
        return center - get_look_direction() * distance_from_center;
    }

    // u, v: pixel coordinates
    Vec3 get_direction(double u, double v) const {
        double half_width = image_width / 2.0;
        double half_height = image_height / 2.0;
        // 18mm is half the width of the full-frame image field
        double tan_h = (u - half_width) / half_width * 18 / focal_length_mm;
        double tan_v = (v - half_height) / half_width * 18 / focal_length_mm;

        double az = azimuth * M_PI / 180.0;
        double el = elevation * M_PI / 180.0;
        Eigen::AngleAxisd r_az(az, Eigen::Vector3d::UnitZ());
        Eigen::AngleAxisd r_el(-el, Eigen::Vector3d::UnitX());

        Eigen::Vector3d dir = r_az * r_el * Eigen::Vector3d(tan_h, 1, tan_v).normalized();

        return dir.normalized();
    }
};

struct SamplingParams {
    int particles_per_pixel;
    int max_iterations;
    int n_loop;

    SamplingParams()
        : particles_per_pixel(100),
        n_loop(10), max_iterations(10000) {}
};

struct LightParams {
    Vec3 direction;
    double intensity;

    LightParams()
        : direction(Vec3(1, 1, -1).normalized())
        , intensity(1.0) {}
};

struct SceneParams {
    std::string stl_file;
    double skybox_scale;
    int grid_resolution;
    double ground_height_offset;

    SceneParams()
        : stl_file("example.stl")
        , skybox_scale(4.0)
        , grid_resolution(400)
        , ground_height_offset(0.01) {}
};

template<typename ParticleType>
class material_set_ray_tracing {
public:
    enum class material_type {
        model_surface,
        sky_wall,
        ground_surface
    };

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
            double up = dir.z();
            double t = 0.5 * (up + 1.0);
            t = std::max(0.0, std::min(1.0, t));
            Eigen::RowVector3d horizon_color(0.85, 0.9, 0.95);
            Eigen::RowVector3d zenith_color(0.3, 0.5, 0.85);
            Eigen::RowVector3d sky_color = (1.0 - t) * horizon_color + t * zenith_color;
            return sky_color;
        }

        void action(material_type mat, ParticleType& p,
                   double hit_x, double hit_y, double hit_z,
                   double nx, double ny, double nz) const {
            switch(mat) {
                case material_type::model_surface:
                    action_model_surface_(p, hit_x, hit_y, hit_z, nx, ny, nz);
                    break;
                case material_type::sky_wall:
                    action_sky_wall_(p, nx, ny, nz);
                    break;
                case material_type::ground_surface:
                    action_ground_surface_(p, hit_x, hit_y, hit_z, nx, ny, nz);
                    break;
            }
        }

    private:
        void action_model_surface_(ParticleType& p, double hit_x, double hit_y, double hit_z,
                                   double nx, double ny, double nz) const {
            auto& pos = get<position>(p);
            auto& vel = get<velocity>(p);
            auto& w = get<weight>(p);

            auto R = psum::random::view_as_rander(get<random_seed>(p));
            Vec3 norm(nx, ny, nz);
            Vec3 hit(hit_x, hit_y, hit_z);

            Vec3 randv = RandV_Cosine(R.get_rander(), norm);
            pos = hit + randv * 0.001;
            vel = randv * vel.norm();
            w *= 0.9f;
            if (w < 0.001f) {
                pos_x_nan_is_invalid<ParticleType>::make_invalid(p);
            }
        }

        void action_sky_wall_(ParticleType& p, double nx, double ny, double nz) const {
            auto& vel = get<velocity>(p);
            auto& w = get<weight>(p);
            auto init = get<init_pos>(p);

            Vec3 norm(nx, ny, nz);
            Vec3 dir = vel.normalized();

            double cos_theta = std::max(0.0, -dir.dot(light_dir));

            Eigen::RowVector3d sky_color = get_sky_color(dir);

            Eigen::RowVector3d color_contribution =
                sky_color * cos_theta * w * light_intensity;

            double u = init.first;
            double v = init.second;

            add_back_nearest(Eigen::RowVector2d(u, v), color_contribution, img_acc);

            pos_x_nan_is_invalid<particle>::make_invalid(p);
        }

        void action_ground_surface_(ParticleType& p, double hit_x, double hit_y, double hit_z,
                                    double nx, double ny, double nz) const {
            auto& pos = get<position>(p);
            auto& vel = get<velocity>(p);
            auto& w = get<weight>(p);

            auto R = psum::random::view_as_rander(get<random_seed>(p));
            Vec3 norm(nx, ny, nz);
            Vec3 hit(hit_x, hit_y, hit_z);

            double r = R();
            if (r < 0.4) {
                Vec3 randv = RandV_Cosine(R.get_rander(), norm);
                pos = hit + randv * 0.001;
                vel = randv * vel.norm();
            } else {
                Vec3 overrush = pos - hit;
                pos = hit + (overrush - 2.0 * overrush.dot(norm) * norm) * 0.001;
                vel = vel - 2.0 * vel.dot(norm) * norm;
            }
            w *= 0.8f;
            if (w < 0.001f) {
                pos_x_nan_is_invalid<ParticleType>::make_invalid(p);
            }
        }
    };

    using acc_type = material_set_acc;

    cf* img_field;
    LightParams light_params_;

    material_set_ray_tracing() : img_field(nullptr) {}

    void set_fields(cf& img) {
        img_field = &img;
    }

    void set_light(const LightParams& light) {
        light_params_ = light;
    }

    material_set_acc get_access(sycl::handler& h) const {
        if (!img_field) {
            throw std::runtime_error("Error: fields not set in material_set_ray_tracing");
        }
        return material_set_acc(
            img_field->get_access(h),
            light_params_.direction,
            light_params_.intensity
        );
    }
};

using MaterialSet = material_set_ray_tracing<particle>;
using Router = boundary_router<MaterialSet, particle>;
using material_type = MaterialSet::material_type;

void test_ray_tracing() {
    std::cout << "==========================================" << std::endl;
    std::cout << "Test: Ray Tracing Rendering" << std::endl;

    sycl::queue q;
    std::cout << "Running on " << q.get_device().get_info<sycl::info::device::name>() << std::endl;

    SceneParams scene;
    scene.stl_file = "example.stl";

    std::cout << "\n=== Loading Scene ===" << std::endl;
    std::cout << "Loading STL file: " << scene.stl_file << "..." << std::endl;

    auto model_triangles = stl_loader::load(scene.stl_file);
    std::cout << "Loaded " << model_triangles.size() << " triangles" << std::endl;

    auto [lp, hp] = stl_loader::box_of_triangles(model_triangles);
    auto [xmin, ymin, zmin] = lp;
    auto [xmax, ymax, zmax] = hp;
    Vec3 model_center((xmin + xmax) / 2, (ymin + ymax) / 2, (zmin + zmax) / 2);
    Vec3 model_size(xmax - xmin, ymax - ymin, zmax - zmin);

    std::cout << "Model bounding box:" << std::endl;
    std::cout << "  X: [" << xmin << ", " << xmax << "]" << std::endl;
    std::cout << "  Y: [" << ymin << ", " << ymax << "]" << std::endl;
    std::cout << "  Z: [" << zmin << ", " << zmax << "]" << std::endl;
    std::cout << "  Center: (" << model_center.x() << ", " << model_center.y() << ", " << model_center.z() << ")" << std::endl;

    CameraParams camera;
    camera.distance_from_center = 3.0 * model_size.norm();
    camera.focal_length_mm = 85.0;
    camera.azimuth = 30.0;
    camera.elevation = 15.0;
    camera.image_width = 1024;
    camera.image_height = 768;

    SamplingParams sampling;
    sampling.particles_per_pixel = 20;
    sampling.max_iterations = 200;
    sampling.n_loop = 20;

    LightParams light;
    light.direction = Vec3(-1, -1, 1).normalized();
    light.intensity = 1.0;

    std::cout << "\n=== Configuration ===" << std::endl;
    std::cout << "Camera:" << std::endl;
    std::cout << "  Distance: " << camera.distance_from_center << std::endl;
    std::cout << "  Focal length: " << camera.focal_length_mm << std::endl;
    std::cout << "  Azimuth: " << camera.azimuth << " degrees" << std::endl;
    std::cout << "  Elevation: " << camera.elevation << " degrees" << std::endl;
    std::cout << "  Resolution: " << camera.image_width << "x" << camera.image_height << std::endl;
    std::cout << "Sampling:" << std::endl;
    std::cout << "  Particles per pixel: " << sampling.particles_per_pixel << std::endl;
    std::cout << "  Max iterations: " << sampling.max_iterations << std::endl;
    std::cout << "Light:" << std::endl;
    std::cout << "  Direction: (" << light.direction.x() << ", " << light.direction.y() << ", " << light.direction.z() << ")" << std::endl;
    std::cout << "  Intensity: " << light.intensity << std::endl;

    Grid2D grid({0, 0}, {(double)camera.image_width, (double)camera.image_height},
                 {camera.image_width, camera.image_height});
    cf img(q, grid);

    double model_diameter = model_size.norm();

    Vec3 p0 = camera.get_position(model_center);
    std::cout << "\nCamera position: (" << p0.x() << ", " << p0.y() << ", " << p0.z() << ")" << std::endl;

    double zone_x0 = std::min(xmin, p0.x());
    double zone_x1 = std::max(xmax, p0.x());
    double zone_y0 = std::min(ymin, p0.y());
    double zone_y1 = std::max(ymax, p0.y());
    double zone_z0 = std::min(zmin, p0.z());
    double zone_z1 = std::max(zmax, p0.z());
    
    auto sky_box = mesh_generator::axis_aligned_cube(zone_x0, zone_y0, zone_z0, zone_x1, zone_y1, zone_z1);
    mesh_generator::rescale_triangles(sky_box, 1.2);
    double ground_height = zone_z0 + scene.ground_height_offset * (zone_z1 - zone_z0);
    auto ground_plane = mesh_generator::axis_aligned_plane(zone_x0, zone_y0, ground_height, zone_x1, zone_y1, ground_height);
    mesh_generator::rescale_triangles(ground_plane, 1.2);

    Router router(q);
    router.set({
        {sky_box, material_type::sky_wall},
        {ground_plane, material_type::ground_surface},
        {model_triangles, material_type::model_surface}
    }, scene.grid_resolution, scene.grid_resolution, scene.grid_resolution);
    router.get_trigger().set_acceleration_UDF(true);

    router.get_material_set().set_fields(img);
    router.get_material_set().set_light(light);

    psum::random::rander R;
    std::vector<particle> particles;
    particles.resize(img.size() * sampling.particles_per_pixel);

    img.setConstant(Eigen::RowVector3d(0.0, 0.0, 0.0));

    for (int i = 0; i < sampling.n_loop; i++) {

        ParticleGroup particles_d(q);

        std::cout << "\n=== Generating Particles ===" << std::endl;

        for (int i = 0; i < img.size(); i++) {
            for (int k = 0; k < sampling.particles_per_pixel; k++) {
                auto pos = grid.cellCorner(grid.i2c(i));
                double alpha = pos.x() + R() * grid.del<0>();
                double beta = pos.y() + R() * grid.del<1>();
                Vec3 vel = camera.get_direction(alpha, beta);


                Vec3 vel_scaled = vel * model_diameter * 0.1;

                particle p;
                get<position>(p) = p0;
                get<velocity>(p) = vel_scaled;
                get<random_seed>(p) = R() * RAND_MAX;
                get<init_pos>(p) = std::make_pair(alpha, beta);
                get<weight>(p) = 1.0f;
                particles[i * sampling.particles_per_pixel + k] = p;
            }
        }

        std::cout << "Generated " << particles.size() << " particles" << std::endl;

        particles_d.insert(particles);

        int iteration = 0;

        std::cout << "\n=== Rendering (" << i + 1 << "/" << sampling.n_loop << ") ===" << std::endl;

        while (iteration < sampling.max_iterations && particles_d.size() > 0) {
            particles_d.for_each([&](sycl::handler& h) {
                auto router_acc = router.get_access(h);
                return [=](particle& p) {
                    double p1x = get<position>(p).x();
                    double p1y = get<position>(p).y();
                    double p1z = get<position>(p).z();

                    get<position>(p) += get<velocity>(p) * 1.0;

                    double p2x = get<position>(p).x();
                    double p2y = get<position>(p).y();
                    double p2z = get<position>(p).z();

                    router_acc.deal(p1x, p1y, p1z, p2x, p2y, p2z, p);
                };
            });

            iteration++;

            if (iteration % 100 == 0) {
                auto host_particles = particles_d.get_content().to_host();
                int active = 0;
                for (const auto& p : host_particles) {
                    if (ParticleGroup::validator::is_valid(p)) {
                        active++;
                    }
                }
                std::cout << "Iteration " << iteration << ": Active particles = " << active << std::endl;
                particles_d.compress();
            }
        }
        std::cout << "Total iterations: " << iteration << std::endl;
    }

    int total_samples = sampling.particles_per_pixel * sampling.n_loop;
    std::cout << "Total samples per pixel: " << total_samples << std::endl;

    img.for_each([&](sycl::handler &h) {
        return [=](size_t i, typename cf::Value& val) {
            val /= total_samples;
        };
    });
    img.plot("ray_tracing_img.plt", "R,G,B");

    std::cout << "\n✅ Rendering completed!" << std::endl;
}

int main() {
    test_ray_tracing();
    return 0;
}
