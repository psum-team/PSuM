#ifndef PSUM_MAGNET_NOZZLE_2D_CHECKPOINT_HPP
#define PSUM_MAGNET_NOZZLE_2D_CHECKPOINT_HPP

#include "types.hpp"
#include "species.hpp"
#include <psum/psum.hpp>

namespace magnet_nozzle_2d {

inline void save_checkpoint(
    Species& ele, Species& ion,
    int charge_out_num, double time,
    const std::string& filename
) {
    ele.compress();
    ion.compress();

    psum::serialization::mas_file fp(filename, psum::serialization::mas_file::replaceMode);
    psum::serialization::save(fp, "electrons", ele);
    psum::serialization::save(fp, "ions", ion);
    psum::serialization::save(fp, "charge_out_num", charge_out_num);
    psum::serialization::save(fp, "physical_time", time);
}

inline void load_checkpoint(
    Species& ele, Species& ion,
    int& charge_out_num, double& time,
    const std::string& filename
) {
    psum::serialization::mas_file fp(filename, psum::serialization::mas_file::autoMode);

    psum::serialization::load(fp, "electrons", ele);
    psum::serialization::load(fp, "ions", ion);
    psum::serialization::load(fp, "charge_out_num", charge_out_num);
    psum::serialization::load(fp, "physical_time", time);
}

}

#endif
