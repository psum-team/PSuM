#ifndef PSUM_MAGNET_NOZZLE_2D_PARTICLE_OPS_HPP
#define PSUM_MAGNET_NOZZLE_2D_PARTICLE_OPS_HPP

#include "types.hpp"
#include "species.hpp"
#include <cmath>
#include <functional>
#include <vector>
#include <algorithm>
#include <utility>

using namespace psum::field::interp_tools;
using namespace psum::random::RandFunction3D;

namespace magnet_nozzle_2d {

template<typename B_acc_t, typename Phi_acc_t>
inline void push_particles(Particle& p, const Phi_acc_t& phi_acc, const B_acc_t& B_acc,
                           double q_mass_rate, double dt, double start_pos, double end_pos, double r_length){
    const Eigen::RowVector3d pos_old = get<position>(p);
    const Eigen::RowVector2d pos_yz = {pos_old.y(), pos_old.z()};
    const double r_old = pos_yz.norm();
    const Eigen::RowVector2d norm_yz = (r_old == 0.0) ? Eigen::RowVector2d{1.0, 0.0} : pos_yz / r_old;

    double interp_x = std::min(std::max(pos_old.x(), start_pos), end_pos - 1e-10);
    double interp_r = std::min(std::max(r_old, 0.0), r_length - 1e-10);

    Eigen::RowVector3d Bvec, Evec;
    auto b_val = interp(Eigen::RowVector2d{interp_x, interp_r}, B_acc);
    Bvec << b_val.x(), b_val.y() * norm_yz;

    auto [ez, er] = field::interp_diff(Eigen::RowVector2d{interp_x, interp_r}, phi_acc);
    double E_max = 1e8;
    double e_mag = sqrt(ez * ez + er * er);
    if (e_mag > E_max) { ez *= E_max / e_mag; er *= E_max / e_mag; }
    Evec << -ez, -er * norm_yz;
    const double kk = dt/2 * q_mass_rate;
    const Eigen::RowVector3d tao = Bvec * kk;
    const Eigen::RowVector3d s = tao * (2.0 / (1.0 + tao.squaredNorm()));
    const Eigen::RowVector3d v_minus = get<velocity>(p) + Evec * kk;
    const Eigen::RowVector3d v_plus = v_minus + (v_minus + v_minus.cross(tao)).cross(s);
    get<velocity>(p) = v_plus + Evec * kk;
    get<position>(p) += dt * get<velocity>(p);
    double new_x = get<position>(p).x();
    const double new_y = get<position>(p).y();
    const double new_z = get<position>(p).z();
    double new_r = sqrt(new_y * new_y + new_z * new_z);
    if (new_x < start_pos || new_x != new_x || new_r != new_r || new_r > 2 * r_length){
        Species::validator::make_invalid(p);
    }
}

template<typename IonCreator, typename EleCreator>
inline std::pair<size_t, size_t> inject_particles(
    Species& ele, Species& ion,
    IonCreator&& ion_creator,
    EleCreator&& ele_creator,
    double ion_inject_num,
    cell_field1D<double>& num_field,
    double r_inlet
){
    size_t ion_inj_num = ion_inject_num > 0 ? (size_t)ion_inject_num : 0;
    if (psum::random::global_random::rand() < ion_inject_num - ion_inj_num) ion_inj_num++;

    std::vector<Particle> particles_e, particles_i;
    particles_i.reserve(ion_inj_num);
    
    for (size_t i = 0; i < ion_inj_num; i++){
        particles_i.emplace_back(ion_creator(psum::random::global_random::rand_uint()));
    }
    ion.insert(particles_i);

    num_field.setZero();
    ion.for_each([&](sycl::handler& h) {
        auto num_acc = num_field.get_access(h);
        return [=](Particle& p) {
            const double py = get<position>(p).y();
            const double pz = get<position>(p).z();
            double r = sqrt(py * py + pz * pz);
            Eigen::Vector<double, 1> pos{get<position>(p).x()};
            if (num_acc.getGrid().inGrid(pos) && r < r_inlet) {
                add_back_nearest(pos, 1.0, num_acc);
            }
        };
    });
    ele.for_each([&](sycl::handler& h) {
        auto num_acc = num_field.get_access(h);
        return [=](Particle& p) {
            const double py = get<position>(p).y();
            const double pz = get<position>(p).z();
            double r = sqrt(py * py + pz * pz);
            Eigen::Vector<double, 1> pos{get<position>(p).x()};
            if (num_acc.getGrid().inGrid(pos) && r < r_inlet) {
                add_back_nearest(pos, -1.0, num_acc);
            }
        };
    });

    auto num_host = num_field.getContent().to_host();
    double ele_inject_num = 0;
    for (auto& v : num_host) ele_inject_num += v;
    ele_inject_num = std::max(ele_inject_num, 0.0);

    size_t ele_inj_num = static_cast<size_t>(ele_inject_num);
    if (psum::random::global_random::rand() < ele_inject_num - ele_inj_num) ele_inj_num++;
    particles_e.reserve(ele_inj_num);
    for (size_t i = 0; i < ele_inj_num; i++){
        particles_e.emplace_back(ele_creator(psum::random::global_random::rand_uint()));
    }
    ele.insert(particles_e);

    return {ion_inj_num, ele_inj_num};
}

inline void count_charge(Species& ele, Species& ion, DNf<1>& charge_density){
    charge_density.setZero();
    const double ion_q_weight = ion.weight() * ion.charge();
    const double ele_q_weight = ele.weight() * ele.charge();
    auto g = charge_density.getGrid();
    double x_max = g.upperBound<0>() - 1e-10;
    double r_max = g.upperBound<1>() - 1e-10;

    ele.for_each([&](sycl::handler& h) {
        auto charge_acc = charge_density.get_access(h);
        return [=](Particle& p) {
            const double py = get<position>(p).y();
            const double pz = get<position>(p).z();
            double r = sqrt(py * py + pz * pz);
            double cx = std::min(get<position>(p).x(), x_max);
            double cr = std::min(r, r_max);
            add_back(Eigen::RowVector2d{cx, cr}, ele_q_weight, charge_acc);
        };
    });

    ion.for_each([&](sycl::handler& h) {
        auto charge_acc = charge_density.get_access(h);
        return [=](Particle& p) {
            const double py = get<position>(p).y();
            const double pz = get<position>(p).z();
            double r = sqrt(py * py + pz * pz);
            double cx = std::min(get<position>(p).x(), x_max);
            double cr = std::min(r, r_max);
            add_back(Eigen::RowVector2d{cx, cr}, ion_q_weight, charge_acc);
        };
    });

    charge_density.for_each([&](sycl::handler& h){
        auto g = charge_density.getGrid();
        return [=](size_t idx, double& charge_val, const auto& pos) {
            double dx = g.del<0>(), dy = g.del<1>();
            double r = pos.y();
            auto [i, j] = g.i2n(idx).indices;
            if (j == 0 || j == g.numCells<1>()) {dy *= 0.5; r += (j == 0) ? 0.5*dy : -0.5*dy;}
            if (i == 0 || i == g.numCells<0>()) {dx *= 0.5; }
            charge_val /= 2 * M_PI * r * dx * dy;
        };
    });
}

}

#endif
