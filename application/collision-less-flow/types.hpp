#ifndef PSUM_MAGNET_NOZZLE_2D_TYPES_HPP
#define PSUM_MAGNET_NOZZLE_2D_TYPES_HPP

#include <psum/psum.hpp>
#include <Eigen/Dense>

using namespace psum::prelude;
using property::position;
using property::velocity;
using property::random_seed;

namespace magnet_nozzle_2d {

constexpr double Q = 1.602e-19;
constexpr double m_e = 9.109e-31;
constexpr double m_H = 1.67e-27;
constexpr double epsilon_0 = 8.85e-12;

using Particle = tagged_struct<
    tag_bind<position, Eigen::RowVector3d>,
    tag_bind<velocity, Eigen::RowVector3d>,
    tag_bind<random_seed, uint32_t>
>;

using ParticleGroup = particle_group<Particle, pos_x_nan_is_invalid>;

template<int var_num>
using DNf = node_field2D<double, var_num>;
template<int var_num>
using DCf = cell_field2D<double, var_num>;
template<int var_num>
using HNf = host_node_field2D<double, var_num>;
template<int var_num>
using HCf = host_cell_field2D<double, var_num>;

using Grid = grid2D;

}

#endif
