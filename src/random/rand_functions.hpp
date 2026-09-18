#ifndef PSUM_RANDOM_RANDFUNCTIONS_HPP
#define PSUM_RANDOM_RANDFUNCTIONS_HPP

#include <sycl/sycl.hpp>
#include "rander.hpp"
#include <cmath>
#include <Eigen/Dense>

namespace psum {

namespace random {

namespace RandFunction3D {

    inline Eigen::Vector3d RandV_spherical(rander &R) {
        double R1 = 2 * R() - 1;
        double R2 = 2 * 3.1415926535897932 * R();
        return Eigen::Vector3d(R1, sycl::sqrt(1 - R1 * R1) * sycl::sin(R2), sycl::sqrt(1 - R1 * R1) * sycl::cos(R2));
    }

    inline Eigen::Vector3d RandV_Maxwell(rander &R, double Temprt, double mass) {
        return Eigen::Vector3d(
                   sycl::sin(2 * 3.1415926535897932 * R()) * sycl::sqrt(-sycl::log(R())),
                   sycl::sin(2 * 3.1415926535897932 * R()) * sycl::sqrt(-sycl::log(R())),
                   sycl::sin(2 * 3.1415926535897932 * R()) * sycl::sqrt(-sycl::log(R()))) *
               sycl::sqrt(2.0 * 1.38065e-23 * Temprt / mass);
    }

    // axis.length() is supposed to be 1
    inline Eigen::Vector3d RandV_Cosine(rander &R, const Eigen::Vector3d &axis) {
        Eigen::Vector3d randv = RandV_spherical(R);
        Eigen::Vector3d biAxis = axis.cross(randv);
        biAxis *= (1 / biAxis.norm());
        double R1 = sycl::sqrt(R());
        double R2 = sycl::sqrt(1 - R1 * R1);
        return axis * R1 + biAxis * R2;
    }

    // axis.length() is supposed to be 1
    inline Eigen::Vector3d RandV_Semisphere(rander &R, const Eigen::Vector3d &axis) {
        Eigen::Vector3d randv = RandV_spherical(R);
        Eigen::Vector3d biAxis = axis.cross(randv);
        biAxis *= (1 / biAxis.norm());
        double R1 = R();
        double R2 = sycl::sqrt(1 - R1 * R1);
        return axis * R1 + biAxis * R2;
    }

    // axis.length() is supposed to be 1
    inline Eigen::Vector3d RandV_halfMaxw(rander &R, const Eigen::Vector3d &axis, double Temprt, double mass) {
        Eigen::Vector3d randv = RandV_spherical(R);
        Eigen::Vector3d biAxis = axis.cross(randv);
        biAxis *= (1 / biAxis.norm());
        return (biAxis * sycl::sqrt(-sycl::log(R())) + axis * sycl::sqrt(-sycl::log(R()))) * sycl::sqrt(2.0 * 1.3805e-23 * Temprt / mass);
    }

    // axis.length() is supposed to be 1
    // cosPhi: cosine of the angle between the direction and the axis
    inline Eigen::Vector3d RandV_Phi(rander &R, const Eigen::Vector3d &axis, double cosPhi) {
        Eigen::Vector3d randv = RandV_spherical(R);
        Eigen::Vector3d biAxis = axis.cross(randv);
        biAxis *= (1 / biAxis.norm());
        return biAxis * sycl::sqrt(1 - cosPhi * cosPhi) + axis * cosPhi;
    }
}

namespace RandFunction2D
{
    inline Eigen::Vector2d RandV_Maxwell(rander &R, double Temprt, double mass) {
        return Eigen::Vector2d(sycl::sin(2 * 3.1415926535897932 * R()) * sycl::sqrt(-sycl::log(R())),
                    sycl::sin(2 * 3.1415926535897932 * R()) * sycl::sqrt(-sycl::log(R())))
                     * sycl::sqrt(2.0 * 1.3805e-23 * Temprt / mass);
    }

    inline Eigen::Vector2d RandV_spherical(rander &R) {
        double R1 = 2 * 3.1415926535897932 * R();
        return Eigen::Vector2d(sycl::cos(R1), sycl::sin(R1));
    }

    // axis.length() is supposed to be 1
    inline Eigen::Vector2d RandV_Cosine(rander &R, const Eigen::Vector2d &axis) {
        Eigen::Vector2d randv = RandV_spherical(R);
        Eigen::Vector2d biAxis = Eigen::Vector2d(-axis.y(), axis.x());
        double R1 = -1 + 2*R();
        return axis * sycl::sqrt(1 - R1 * R1) + biAxis * R1;
    }

    // axis.length() is supposed to be 1
    inline Eigen::Vector2d RandV_CosHalfMaxw(rander &R, const Eigen::Vector2d &axis, double Temprt, double mass) {
        Eigen::Vector2d direction = RandV_Cosine(R, axis);

        return direction * sycl::sqrt(-sycl::log(R())) * sycl::sqrt(2.0 * 1.3805e-23 * Temprt / mass);
    }
};

namespace RandFunction1D
{
    inline double RandV_halfMaxw(rander &R, const double &axis, double Temprt, double mass) {
        double v = sycl::sqrt(2.0 * 1.3805e-23 * Temprt / mass)*sycl::sqrt(-sycl::log(R())) * sycl::sin(3.1415926535898/2.0*R());
        return axis > 0 ? v : v * -1;
    }
};

}

}
#endif
