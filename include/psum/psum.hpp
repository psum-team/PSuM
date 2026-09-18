#ifndef PSUM_HPP
#define PSUM_HPP

#include "tag.hpp"
#include "serialization.hpp"
#include "particle_container.hpp"
#include "field.hpp"
#include "particle_boundary.hpp"
#include "field_solver.hpp"
#include "particle_collision.hpp"
#include "../../src/particle_dictionary.hpp"
#include "../../src/random.hpp"
#include "../../src/timer.hpp"
#include "../../src/utils_sycl.hpp"

namespace psum::prelude {

    using namespace psum;
    using namespace psum::tag;
    using namespace psum::serialization;
    using namespace psum::particle_container;
    using namespace psum::field;
    using namespace psum::particle_boundary;
    using namespace psum::field_solver;
    using namespace psum::particle_collision;
    using namespace psum::random;
    using namespace psum::utils_sycl;
    using namespace psum::field::multi_patch;
}

#endif