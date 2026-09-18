#ifndef PSUM_MAGNET_NOZZLE_2D_MOMENT_OUTPUT_HPP
#define PSUM_MAGNET_NOZZLE_2D_MOMENT_OUTPUT_HPP

#include "data_collection.hpp"

#include <array>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <string>

namespace magnet_nozzle_2d {

struct Moment10Metadata {
    std::string source_sha = "unprovided";
    std::string binary_sha = "unprovided";
    std::string config_sha = "unprovided";
    std::string species;
    double mass = 0.0;
    double charge = 0.0;
    double weight = 0.0;
    int collect_step = 0;
    int output_step = 0;
    size_t samples = 0;
};

inline void write_moment10_header(std::ofstream& out, const Moment10Metadata& meta) {
    out << "# schema=magnet-nozzle.moments10.v1\n"
        << "# source_sha=" << meta.source_sha << "\n"
        << "# binary_sha=" << meta.binary_sha << "\n"
        << "# config_sha=" << meta.config_sha << "\n"
        << "# species=" << meta.species << "\n"
        << std::setprecision(17)
        << "# mass=" << meta.mass << "\n"
        << "# charge=" << meta.charge << "\n"
        << "# particle_weight=" << meta.weight << "\n"
        << "# collect_step=" << meta.collect_step << "\n"
        << "# output_step=" << meta.output_step << "\n"
        << "# collect_samples=" << meta.samples << "\n"
        << "# coordinate_basis=(x,r,t), r=sqrt(y*y+z*z), t=(-z/r)y+(y/r)z\n"
        << "# normalization=weighted_raw_moment_density; raw sums divided by 2*pi*r*dx*dr\n"
        << "# empty_bucket=M0..M2rt are zero; use M0==0 as invalid\n"
        << "# axis_r0=use y as radial and z as tangent basis\n"
        << "VARIABLES=x,r,M0,M1x,M1r,M1t,M2xx,M2rr,M2tt,M2xr,M2xt,M2rt\n";
}

inline void write_moment10_axisymmetric(
    const std::string& path,
    const HCf<10>& data,
    const Moment10Metadata& meta
) {
    std::ofstream out(path);
    if (!out) throw std::runtime_error("cannot open moment output: " + path);
    write_moment10_header(out, meta);
    out << "ZONE I=" << data.size() << ", DATAPACKING=POINT\n";
    const auto& grid = data.getGrid();
    const double dx = grid.del<0>();
    const double dr = grid.del<1>();
    for (size_t i = 0; i < data.size(); i++) {
        const auto pos = grid.cellCenter(grid.i2c(i));
        const double radius = std::max(pos.y(), 0.5 * dr);
        const double volume = 2.0 * M_PI * radius * dx * dr;
        const double scale = volume > 0.0 ? 1.0 / volume : 0.0;
        out << std::setprecision(17) << pos.x() << ' ' << pos.y();
        for (int j = 0; j < 10; j++) out << ' ' << data(i)(j) * scale;
        out << '\n';
    }
}

}

#endif
