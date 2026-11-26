#ifndef PSUM_PARTICLE_COLLISION_COLLISION_EXECUTOR_HPP
#define PSUM_PARTICLE_COLLISION_COLLISION_EXECUTOR_HPP
#include <cmath>
#include <tuple>
#include <type_traits>
#include <utility>
#include <sycl/sycl.hpp>
#include "foundation.hpp"
#include "context/species_context.hpp"
#include "../random.hpp"

namespace psum::particle_collision {
namespace foundation {

template<class Tuple>struct collision_executor;
template<class... CollisionTypes>struct collision_executor<std::tuple<CollisionTypes...>> {
private:
    using channels = std::tuple<CollisionTypes...>;
    using first = std::tuple_element_t<0,channels>;

    template<class Validator,size_t I=0,class Particle,class ContextAcc>
    static void dispatch(size_t selected,Particle& p,const ContextAcc& ctx) {
        if constexpr(I < sizeof...(CollisionTypes)) {
            if(I != selected) {
                dispatch<Validator,I+1>(selected,p,ctx);
                return;
            }

            using collision = std::tuple_element_t<I,channels>;
            if constexpr(std::is_base_of_v<reaction_collision_tag,collision>) {
                using reaction_access = reaction_context<collision,ContextAcc>;
                auto collide = [&](const auto& access) {
                    static_assert(mcc_collision<collision,Particle,std::remove_cvref_t<decltype(access)>>,
                        "MCC reaction must provide collide(p,ctx)");
                    collision::collide(p,access);
                };
                if constexpr(has_particle_erasure<collision>) {
                    static_assert(!std::is_void_v<Validator>,
                        "An MCC reaction that consumes the incident particle needs a particle validator");
                    erasure_requests erasures;
                    collide(reaction_access{ctx,erasures});
                    if(erasures.erase_first) Validator::make_invalid(p);
                } else {
                    collide(reaction_access{ctx});
                }
            } else {
                static_assert(mcc_collision<collision,Particle,ContextAcc>,"MCC collision must provide collide(p,ctx)");
                collision::collide(p,ctx);
            }
        }
    }

public:
    template<class Validator=void,class Particle,class ContextAcc>
    static void execute(Particle& p,double dt,const ContextAcc& ctx) {
        auto v2=psum::tag::get<psum::tag::property::velocity>(p).squaredNorm();
        double rates[]{CollisionTypes::collision_cross_section(v2)...};
        double total=0;for(double rate:rates)total+=rate;
        auto density=collision_density<typename first::incident_species_tag,typename first::background_species_tag>(
            psum::tag::get<psum::tag::property::position>(p),ctx);
        size_t selected=sizeof...(CollisionTypes);bool accepted=false;
        {
            auto random=psum::random::view_as_rander(psum::tag::get<psum::tag::property::random_seed>(p));
            double threshold=random()*total,sum=0;
            for(size_t i=0;i<sizeof...(CollisionTypes);++i){sum+=rates[i];if(sum>0&&threshold<=sum){selected=i;break;}}
            accepted=selected<sizeof...(CollisionTypes)&&random()<total*dt*density*std::sqrt(v2);
        }
        if(accepted) dispatch<Validator>(selected,p,ctx);
    }
};
template<class Tuple>struct collision_request_executor;
template<class... CollisionTypes>struct collision_request_executor<std::tuple<CollisionTypes...>> {
    template<size_t I=0,class Particle,class ContextAcc>
    static void enqueue(size_t selected,Particle& p,const ContextAcc& ctx,double probability) {
        if constexpr(I<sizeof...(CollisionTypes)) {
            if(I==selected) {
                using Collision=std::tuple_element_t<I,std::tuple<CollisionTypes...>>;
                auto buffer=ctx.template get<Collision>();
                auto field=ctx.template get<typename Collision::background_species_tag>().field_acc;
                const auto& grid=field.getGrid();
                auto pos=projected_position<decltype(field)::Dimension,typename Collision::background_species_tag>(
                    psum::tag::get<psum::tag::property::position>(p));
                if(grid.inGrid(pos))
                    buffer.push_back({uint32_t(grid.c2i(grid.nC(pos))),{&p,nullptr,probability}});
            } else enqueue<I+1>(selected,p,ctx,probability);
        }
    }
    template<class Particle,class ContextAcc>
    static void execute(Particle& p,const ContextAcc& ctx) {
        static_assert((requires {
            { CollisionTypes::request_probability(p,ctx) } -> std::same_as<double>;
        } && ...), "DPMCC collision must provide request_probability(p,ctx) returning double");
        double probabilities[]{CollisionTypes::request_probability(p,ctx)...};
        double total=0;
        for(double probability:probabilities) {
            if(!sycl::isfinite(probability) || probability<0 || probability>1)return;
            total+=probability;
        }
        // Probabilities are absolute, not weights. Invalid groups generate no requests.
        if(!(total>0) || total>1)return;
        double r;
        {auto random=psum::random::view_as_rander(psum::tag::get<psum::tag::property::random_seed>(p));r=random();}
        double sum=0;
        for(size_t i=0;i<sizeof...(CollisionTypes);++i) {
            sum+=probabilities[i];
            if(r<sum) {
                enqueue(i,p,ctx,probabilities[i]);
                return;
            }
        }
    }
};
template<class CollisionType,bool CheckValidity,class FirstValidator,class SecondValidator,class Payload,class ContextAcc,class RecordInvalid>
void execute_pair(Payload& pair,const ContextAcc& ctx,const sycl::stream& warnings,
    RecordInvalid record_invalid) {
    constexpr bool reaction = std::is_base_of_v<reaction_collision_tag,CollisionType>;
    static_assert(collision_channel<CollisionType> && requires {
        { CollisionType::pair_collision_cross_section(pair,ctx) } -> std::same_as<double>;
    }, "DPMCC collision must provide pair_collision_cross_section(pair,ctx) returning double");
    if constexpr(CheckValidity || reaction) {
        if(!FirstValidator::is_valid(*pair.first) || !SecondValidator::is_valid(*pair.second)) return;
    }
    const double sigma=CollisionType::pair_collision_cross_section(pair,ctx);
    if(!sycl::isfinite(sigma) || sigma<0) {
        warnings << "warning: invalid DPMCC pair cross section\n";
        return;
    }
    const double speed=(psum::tag::get<psum::tag::property::velocity>(*pair.first)
        -psum::tag::get<psum::tag::property::velocity>(*pair.second)).norm();
    using incident=typename CollisionType::incident_species_tag;
    using background=typename CollisionType::background_species_tag;
    const double density=collision_density<incident,background>(
        psum::tag::get<psum::tag::property::position>(*pair.first),ctx);
    double factor=1.0;
    if constexpr(std::is_same_v<incident,background>) {
        auto counts=ctx.template get<incident>().particle_counts;
        auto pos=projected_position<decltype(counts)::Dimension,incident>(
            psum::tag::get<psum::tag::property::position>(*pair.first));
        const auto& grid=counts.getGrid();
        const uint32_t N=counts(grid.c2i(grid.nC(pos)));
        if(N<2)return;
        factor=0.5*(1.0-1.0/double(N));
    }
    double p=factor*sigma*speed*density*ctx.dt/pair.request_probability;
    if(!sycl::isfinite(p)||p<0)warnings << "warning: invalid DPMCC pair probability\n";
    else {
        if(p>1){warnings << "warning: DPMCC pair probability exceeds 1; clipped p=" << p << "\n";p=1;}
        bool accepted;
        {auto random=psum::random::view_as_rander(psum::tag::get<psum::tag::property::random_seed>(*pair.first));auto r=random();accepted=p>=1||r<p;}
        if(accepted) {
            auto collide = [&](const auto& access) {
                static_assert(requires { { CollisionType::pair_collide(pair,access) } -> std::same_as<void>; },
                    "pair_collide must return void");
                CollisionType::pair_collide(pair,access);
            };
            if constexpr(has_particle_erasure<CollisionType>) {
                erasure_requests erasures{pair.first,pair.second};
                auto& first = *pair.first;
                auto& second = *pair.second;
                collide(reaction_context<CollisionType,ContextAcc>{ctx,erasures});
                if(erasures.erase_first) FirstValidator::make_invalid(first);
                if(erasures.erase_second) SecondValidator::make_invalid(second);
                // Record both the requests and any directly invalidated declared endpoints.
                record_invalid(!FirstValidator::is_valid(first),!SecondValidator::is_valid(second));
            } else if constexpr(reaction) {
                collide(reaction_context<CollisionType,ContextAcc>{ctx});
            } else {
                collide(ctx);
            }
        }
    }
}
}
}
#endif
