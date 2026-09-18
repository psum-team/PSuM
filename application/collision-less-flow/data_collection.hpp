#ifndef PSUM_MAGNET_NOZZLE_2D_DATA_COLLECTION_HPP
#define PSUM_MAGNET_NOZZLE_2D_DATA_COLLECTION_HPP

#include "types.hpp"
#include "species.hpp"
#include <cmath>
#include <stdexcept>
#include <algorithm>

namespace magnet_nozzle_2d {

constexpr int moment10_components = 10;
using Moment10 = Eigen::RowVector<double, moment10_components>;

inline Moment10 make_weighted_moment10(double weight, const Eigen::RowVector3d& velocity, double y, double z) {
    const double radius = sycl::sqrt(y * y + z * z);
    const double cy = radius > 1.0e-30 ? y / radius : 1.0;
    const double cz = radius > 1.0e-30 ? z / radius : 0.0;
    const double vx = velocity.x();
    const double vr = cy * velocity.y() + cz * velocity.z();
    const double vt = -cz * velocity.y() + cy * velocity.z();
    Moment10 value;
    value << weight,
        weight * vx, weight * vr, weight * vt,
        weight * vx * vx, weight * vr * vr, weight * vt * vt,
        weight * vx * vr, weight * vx * vt, weight * vr * vt;
    return value;
}

template<int var_num>
class mean_smoother {
    HCf<var_num> acc_;
    size_t count_;

public:
    mean_smoother(const typename HCf<var_num>::Grid& grid) : acc_(grid), count_(0) {}

    void inject(const HCf<var_num>& data) {
        if (count_ == 0) {
            acc_.copy(data.getContent());
        } else {
            for (size_t j = 0; j < acc_.size(); j++) acc_(j) += data(j);
        }
        count_++;
    }

    size_t count() const { return count_; }

    HCf<var_num> pop() {
        if (count_ == 0) throw std::runtime_error("mean_smoother::pop - no data injected");
        HCf<var_num> result(acc_.getGrid());
        result.copy(acc_.getContent());
        for (size_t j = 0; j < result.size(); j++) result(j) /= static_cast<double>(count_);
        count_ = 0;
        acc_.setZero();
        return result;
    }
};

inline void collect_data(
    Species& ps, DCf<7>& data,
    mean_smoother<7>& data_smooth
) {
    data.setZero();

    ps.for_each([&](sycl::handler& h) {
        auto data_acc = data.get_access(h);
        return [=](Particle& p) {
            const double py = get<position>(p).y();
            const double pz = get<position>(p).z();
            double r = sqrt(py * py + pz * pz);
            Eigen::RowVector<double, 7> data_val;
            double vx = get<velocity>(p).x();
            double vy = get<velocity>(p).y();
            double vz = get<velocity>(p).z();
            data_val << 1.0, vx, vy, vz, vx*vx + vy*vy + vz*vz, vx*vx, vy*vy + vz*vz;
            add_back_nearest(Eigen::RowVector2d{get<position>(p).x(), r}, data_val, data_acc);
        };
    });

    HCf<7> data_host(data.getGrid());
    data_host.copy(data.getContent().to_host());
    data_smooth.inject(data_host);
}

template<bool ENABLE_MOMENTS>
inline void collect_data_with_moments(
    Species& ps,
    DCf<7>& data,
    mean_smoother<7>& data_smooth,
    DCf<10>* moment_data,
    mean_smoother<10>* moment_smooth
) {
    data.setZero();
    if constexpr (ENABLE_MOMENTS) moment_data->setZero();

    const double particle_weight = ps.weight();
    ps.for_each([&](sycl::handler& h) {
        auto data_acc = data.get_access(h);
        if constexpr (ENABLE_MOMENTS) {
            auto moment_acc = moment_data->get_access(h);
            return [=](Particle& p) {
                const double py = get<position>(p).y();
                const double pz = get<position>(p).z();
                const double radius = sycl::sqrt(py * py + pz * pz);
                const Eigen::RowVector2d axis_pos{get<position>(p).x(), radius};
                const Eigen::RowVector3d vel = get<velocity>(p);
                const double vx = vel.x();
                const double vy = vel.y();
                const double vz = vel.z();
                Eigen::RowVector<double, 7> legacy_val;
                legacy_val << 1.0, vx, vy, vz, vx * vx + vy * vy + vz * vz, vx * vx, vy * vy + vz * vz;
                add_back_nearest(axis_pos, legacy_val, data_acc);
                add_back_nearest(axis_pos, make_weighted_moment10(particle_weight, vel, py, pz), moment_acc);
            };
        } else {
            return [=](Particle& p) {
                const double py = get<position>(p).y();
                const double pz = get<position>(p).z();
                const double radius = sycl::sqrt(py * py + pz * pz);
                const Eigen::RowVector2d axis_pos{get<position>(p).x(), radius};
                const Eigen::RowVector3d vel = get<velocity>(p);
                const double vx = vel.x();
                const double vy = vel.y();
                const double vz = vel.z();
                Eigen::RowVector<double, 7> legacy_val;
                legacy_val << 1.0, vx, vy, vz, vx * vx + vy * vy + vz * vz, vx * vx, vy * vy + vz * vz;
                add_back_nearest(axis_pos, legacy_val, data_acc);
            };
        }
    });

    HCf<7> data_host(data.getGrid());
    data_host.copy(data.getContent().to_host());
    data_smooth.inject(data_host);
    if constexpr (ENABLE_MOMENTS) {
        HCf<10> moment_host(moment_data->getGrid());
        moment_host.copy(moment_data->getContent().to_host());
        moment_smooth->inject(moment_host);
    }
}

inline HCf<7> pop_data(
    Species& ps,
    mean_smoother<7>& data_smooth
) {
    const double particle_weight = ps.weight();
    const double mass = ps.mass();
    HCf<7> data_host = data_smooth.pop();
    const auto& grid = data_host.getGrid();
    const double dx = grid.del<0>();
    const double dr = grid.del<1>();

    for (size_t i = 0; i < data_host.size(); i++) {
        auto& val = data_host(i);
        const double count = val(0);
        if (count <= 0.0) continue;

        const double inv_count = 1.0 / count;
        const double ux = val(1) * inv_count;
        const double uy = val(2) * inv_count;
        const double uz = val(3) * inv_count;
        const double v2_mean = val(4) * inv_count;
        const double vx2_mean = val(5) * inv_count;
        const double vyz2_mean = val(6) * inv_count;

        const double t_parallel = std::max(0.0, vx2_mean - ux * ux) * mass / Q;
        const double t_perp = std::max(0.0, vyz2_mean - uy * uy - uz * uz) * 0.5 * mass / Q;
        const double t_total = std::max(0.0, v2_mean - ux * ux - uy * uy - uz * uz) * mass / (3.0 * Q);
        const double anisotropy = t_perp / (t_parallel + 1e-30);

        const double r = std::max(grid.cellCenter(grid.i2c(i)).y(), 0.5 * dr);
        const double volume = 2.0 * M_PI * r * dx * dr;

        val(0) = count * particle_weight / volume;
        val(1) = ux;
        val(2) = uy;
        val(3) = uz;
        val(4) = t_parallel;
        val(5) = t_perp;
        val(6) = anisotropy;
        (void)t_total;
    }

    return data_host;
}

}

#endif
