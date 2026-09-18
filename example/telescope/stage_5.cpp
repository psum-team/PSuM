#include <cmath>
#include <iostream>
#include <vector>
#include <sycl/sycl.hpp>
#include <Eigen/Core>
#include <psum/psum.hpp>

using namespace std;
using namespace psum::prelude;
using namespace Eigen;

using Photon = tagged_struct<
    tag_bind<property::position, Eigen::RowVector2d>,
    tag_bind<property::velocity, Eigen::RowVector2d>
>;

using PhotonGroup = particle_group<Photon, pos_x_nan_is_invalid>;

class OpticalMaterialSet {
public:
    enum class material_type { absorb, specular, refract };

    class acc_type {
    public:
        acc_type(double glass_ior) : glass_ior_(glass_ior) {}

        void action(material_type mat, Photon& p, double hit_x, double hit_y, double, double nx, double ny, double) const {
            RowVector2d hit(hit_x, hit_y);
            RowVector2d n(nx, ny);
            n.normalize();

            RowVector2d dir = get<property::velocity>(p).normalized();
            double speed = get<property::velocity>(p).norm();
            if (mat == material_type::absorb) {
                PhotonGroup::validator::make_invalid(p);
            } else if (mat == material_type::specular) {
                n = dir.dot(n) > 0.0 ? -n : n;
                RowVector2d out = dir - 2.0 * dir.dot(n) * n;
                out.normalize();
                get<property::velocity>(p) = out * speed;
                get<property::position>(p) = hit + n * 1e-6;
            } else if (mat == material_type::refract) {
                double inc_ior = 1.0 / speed;  // ior is reciprocal to speed when c=1
                bool enter = dir.dot(n) < 0.0;
                RowVector2d norm = enter ? n : -n;
                double t_ior = enter ? glass_ior_ : 1.0;
                double eta = inc_ior / t_ior, ci = -dir.dot(norm);
                double sin2 = eta * eta * (1.0 - ci * ci);
                if (sin2 > 1.0) {
                    // 全反射（离开玻璃时）
                    RowVector2d out = dir - 2.0 * dir.dot(norm) * norm;
                    out.normalize();
                    get<property::velocity>(p) = out * speed;
                    get<property::position>(p) = hit + out * 1e-6;
                } else {
                    double ct = sycl::sqrt(1.0 - sin2);
                    RowVector2d td = eta * dir + (eta * ci - ct) * norm;
                    td.normalize();
                    get<property::velocity>(p) = td / t_ior;
                    get<property::position>(p) = hit + td * 1e-6;
                }
            }
        }
    private:
        double glass_ior_;
    };
    void set_glass_ior(double v) { ior_ = v; }

    acc_type get_access(sycl::handler&) const { return acc_type(ior_); }
private:
    double ior_ = 1.5;
};

using Router = boundary_router<OpticalMaterialSet, Photon>;
using material_type = OpticalMaterialSet::material_type;

vector<Photon> make_photons(int count, int num_way, double ds, double width, random::rander& R) {
    vector<Photon> photons(count);
    for (int i = 0; i < count; ++i) {
        double y = -0.095 + 0.190 * (int(R()*num_way) + 0.5) / num_way;
        y += width * (R() - 0.5);
        RowVector2d dir(1.0, 0.0);

        get<property::velocity>(photons[i]) = dir;
        get<property::position>(photons[i]) = RowVector2d(-0.095, y) + dir * (ds * R());
    }
    return photons;
}

vector<pair<double, double>> arc_surface(RowVector2d top, RowVector2d curv_vector, double half_angle, bool norm_out = true) {
    vector<pair<double, double>> pts;
    int seg = 80;
    RowVector2d center = top - curv_vector;
    double radius = curv_vector.norm();
    double base_angle = atan2(curv_vector.y(), curv_vector.x());
    for (int i = 0; i <= seg; ++i) {
        double t = -1.0 + 2.0 * i / seg;
        if (norm_out == false) t = -1.0 + 2.0 * (seg - i) / seg;
        double angle = base_angle + t * half_angle;
        pts.push_back({center.x() + radius * cos(angle), center.y() + radius * sin(angle)});
    }
    return pts;
}

int main() {
    sycl::queue q{sycl::default_selector_v};
    cout << "device: " << q.get_device().get_info<sycl::info::device::name>() << endl;
    grid2D grid({-0.11, -0.18}, {0.7, 0.18}, {540, 240});
    node_field2D<double> density(q, grid);
    PhotonGroup photons(q);
    Router router(q);
    auto central_obstruction = mesh_generator::extrude_xy_line_to_width(
        arc_surface({-0.001, 0.0}, {0.2686151, 0.0}, 0.13, false), 0.2);
    auto corrector_1 = mesh_generator::extrude_xy_line_to_width(
        arc_surface({0.0, 0.0}, {0.2686151, 0.0}, 0.38, false), 0.2);
    auto corrector_2 = mesh_generator::extrude_xy_line_to_width(
        arc_surface({0.03, 0.0}, {0.2861193, 0.0}, 0.37, true), 0.2);
    auto secondary = mesh_generator::extrude_xy_line_to_width(
        arc_surface({0.0301, 0.0}, {0.2861193, 0.0}, 0.08), 0.2);
    auto primary_up = mesh_generator::extrude_xy_line_to_width(
        arc_surface({0.4808846, 0.0693117}, {1.1094470, 0.0693117}, 0.035), 0.2);
    auto primary_down = mesh_generator::extrude_xy_line_to_width(
        arc_surface({0.4808846, -0.0693117}, {1.1094470, -0.0693117}, 0.035), 0.2);
    router.set({{central_obstruction, material_type::absorb},
                {corrector_1, material_type::refract},
                {corrector_2, material_type::refract},
                {secondary, material_type::specular},
                {primary_up, material_type::specular},
                {primary_down, material_type::specular}}, 160, 80, 20);
    router.get_material_set().set_glass_ior(1.45);
    double ds = 0.002;
    int steps = 1200, pps = 1000, interval = 30;
    random::rander R;
    for (int step = 0; step < steps; ++step) {
        photons.insert(make_photons(pps, 20, ds, grid.span<1>() * 0.01, R));
        photons.for_each([&](sycl::handler& h) {
            auto router_acc = router.get_access(h);
            return [=](Photon &p) {
                RowVector2d old_p = get<property::position>(p);
                get<property::position>(p) += get<property::velocity>(p) * ds;
                RowVector2d new_p = get<property::position>(p);
                router_acc.deal(old_p.x(), old_p.y(), 0.0, new_p.x(), new_p.y(), 0.0, p);
            };
        });
        if (step % interval == 0) {
            density.setZero();
            photons.for_each([&](sycl::handler& h) {
                auto da = density.get_access(h);
                return [=](Photon& p) {
                    if (da.getGrid().inGrid(get<property::position>(p)))
                        add_back(get<property::position>(p), 1.0, da);
                    else PhotonGroup::validator::make_invalid(p);
                };
            });
            density.plot("output/density_step_" + to_string(step) + ".plt", "density", step);
            cout << "step = " << step << ", alive = " << photons.size() << endl;
        }
    }
    cout << "Ray tracing 2D: wrote photon density snapshots." << endl;
}
