#ifndef PSUM_PARTICLE_COLLISION_FOUNDATION_HPP
#define PSUM_PARTICLE_COLLISION_FOUNDATION_HPP

#include <concepts>
#include <tuple>
#include <type_traits>
#include <cmath>
#include <sycl/sycl.hpp>
#include "../../include/psum/tag.hpp"
#include "../random.hpp"

namespace psum {

namespace particle_collision {

namespace foundation {

    template <typename... CollisionCrossSections>
    struct random_choose {
        // sum of collision cross section, and random choosed index.
        template <typename... Args>
        static auto compute(double r, Args&&... args) {
            double areas[sizeof...(CollisionCrossSections)] = {
                CollisionCrossSections::collision_cross_section(std::forward<Args>(args)...)... 
            };
            
            double sum = 0;
            for (double s : areas) sum += s;

            int hit_idx = -1;
            if (sum > 0) {
                double threshold = r * sum;
                double acc = 0;
                for (int i = 0; i < (int)sizeof...(CollisionCrossSections); ++i) {
                    acc += areas[i];
                    if (threshold <= acc) {
                        hit_idx = i;
                        break;
                    }
                }
            }
            return std::make_pair(sum, hit_idx);
        }
    };

    template <typename... Actions>
    struct static_switch {
        template <typename... Args>
        static void deal(int idx, Args&&... args) {
            if (idx < 0) return;
            dispatch<0>(idx, std::forward<Args>(args)...);
        }

    private:
        template <std::size_t I, typename... Args>
        static void dispatch(int idx, Args&&... args) {
            if constexpr (I < sizeof...(Actions)) {
                if (I == (std::size_t)idx) {
                    using CurrentType = std::tuple_element_t<I, std::tuple<Actions...>>;
                    CurrentType::collide(std::forward<Args>(args)...);
                } else {
                    dispatch<I + 1>(idx, std::forward<Args>(args)...);
                }
            }
        }
    };

    template <typename DensityInterpolator, typename CollisionTuple>
    struct collision_processor;

    template <typename DensityInterpolator, typename... Collisions>
    struct collision_processor<DensityInterpolator, std::tuple<Collisions...>> {

        using Chooser = random_choose<Collisions...>;
        using Dispatcher = static_switch<Collisions...>;
        using col_struct_tuple = std::tuple<Collisions...>;

        template <typename P, typename... Args>
        static void execute(P& p, double dt, Args&&... args) {
            auto& pos = psum::tag::get<psum::tag::property::position>(p);
            auto& seed = psum::tag::get<psum::tag::property::random_seed>(p);
            auto& vel = psum::tag::get<psum::tag::property::velocity>(p);

            double dens = DensityInterpolator::compute(pos, std::forward<Args>(args)...);
            auto R = psum::random::view_as_rander(seed);
            
            double v2 = vel.squaredNorm();
            double v_mag = std::sqrt(v2);
            
            auto [total_ccs, idx] = Chooser::compute(R(), v2);
            
            if (idx != -1 && R() < total_ccs * dt * dens * v_mag) {
                R.sync_seed();
                Dispatcher::deal(idx, p, std::forward<Args>(args)...);
            }
        }
    };

    template <typename... Tasks>
    struct sequence_executor {
        using col_struct_tuple = decltype(std::tuple_cat(std::declval<typename Tasks::col_struct_tuple>()...));
        template <typename... Args>
        static void execute(Args&&... args) {
            (Tasks::execute(std::forward<Args>(args)...), ...);
        }
    };

}

}

}

#endif