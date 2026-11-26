#ifndef PSUM_PARTICLE_COLLISION_COLLISION_CHANNEL_CONTEXT_HPP
#define PSUM_PARTICLE_COLLISION_COLLISION_CHANNEL_CONTEXT_HPP

#include <algorithm>
#include <array>
#include <tuple>
#include <type_traits>
#include <utility>
#include "../foundation.hpp"
#include "../pairing_engine.hpp"

namespace psum::particle_collision {
    namespace foundation {
        template<class Channel>
        struct same_species_channel : std::is_same<incident_species_of<Channel>,background_species_of<Channel>> {};

        template<class Tuple>struct deferred_pairing_filter;
        template<class... CollisionTypes>struct deferred_pairing_filter<std::tuple<CollisionTypes...>> {
            using type=decltype(std::tuple_cat(std::declval<std::conditional_t<
                std::is_base_of_v<deferred_pairing_collision_tag,CollisionTypes>,std::tuple<CollisionTypes>,std::tuple<>>>()...));
        };

        template<class CollisionTypesTuple>struct dpmcc_model_types {
            using collision_types_tuple=typename deferred_pairing_filter<CollisionTypesTuple>::type;
            using species_tuple=typename collision_model_types<CollisionTypesTuple>::species_tuple;
            template<class CollisionType>static constexpr int channel_index=
                psum::tag::foundation::index_in_tuple<CollisionType,collision_types_tuple>::value;
            template<class C>static constexpr bool no_reverse=[]<class... Other>(std::tuple<Other...>*) {
                return ((!std::is_same_v<typename C::incident_species_tag,typename Other::background_species_tag>
                    ||!std::is_same_v<typename C::background_species_tag,typename Other::incident_species_tag>
                    ||std::is_same_v<typename C::incident_species_tag,typename C::background_species_tag>)&&...);
            }(static_cast<collision_types_tuple*>(nullptr));
            static_assert([]<class... C>(std::tuple<C...>*){return (no_reverse<C>&&...);}(static_cast<collision_types_tuple*>(nullptr)),
                "reverse duplicate pair requires an explicit counting policy");
        };
    }

    template<class CollisionType,class ParticleGroupType,class GridType>
    struct collision_channel_context {
        using engine_type=pairing_engine<ParticleGroupType,GridType,typename CollisionType::background_species_tag>;
        using payload_type=typename engine_type::payload_type;
        using request_buffer_type=psum::particle_container::device_vector<std::pair<uint32_t,payload_type>>;
        using config_type=std::tuple<sycl::queue,GridType,size_t>;
        using acc_type=typename request_buffer_type::acc_type;
        request_buffer_type buffer;
        engine_type engine;
        explicit collision_channel_context(config_type c)
            :buffer(std::get<0>(c),std::get<2>(c)),engine(std::get<0>(c),std::get<1>(c)){}
        collision_channel_context(sycl::queue q,const GridType& grid,size_t capacity)
            :collision_channel_context(config_type{q,grid,capacity}){}
        acc_type get_access(sycl::handler& h){return buffer.get_access(h);}
    };

    // Collection layer: collision types identify their channel resources.
    template<class ModelType,class ParticleGroupType,class SpeciesContextType> class collision_channel_contexts {
        using model_types=foundation::dpmcc_model_types<typename ModelType::collision_types_tuple>;
        template<class CollisionType> using background_resource_type=std::remove_reference_t<decltype(std::declval<SpeciesContextType&>().template get<typename CollisionType::background_species_tag>())>;
        template<class CollisionType> using grid_type=std::remove_cvref_t<decltype(std::declval<background_resource_type<CollisionType>&>().density->getGrid())>;
        template<class CollisionType> using channel_context_type=collision_channel_context<CollisionType,ParticleGroupType,grid_type<CollisionType>>;
        template<class CollisionType> static grid_type<CollisionType> background_grid(SpeciesContextType& species) {
            return species.template get<typename CollisionType::background_species_tag>().density->getGrid();
        }
    public:
        using tags_tuple=typename model_types::collision_types_tuple;
        using storage_type=typename foundation::map_tuple<tags_tuple,channel_context_type>::type;
        using device_tuple=typename foundation::access_tuple<storage_type>::type;
        struct acc_type {
            device_tuple objects;
            template<class CollisionType> requires(model_types::template channel_index<CollisionType> >= 0)
            decltype(auto) get()const{return std::get<model_types::template channel_index<CollisionType>>(objects);}
        };
        storage_type objects;
        collision_channel_contexts(sycl::queue q,SpeciesContextType& species)
            :collision_channel_contexts(q,species,[&]<class SpeciesTag>{
                return std::max(size_t{8192},size_t(species.template get<SpeciesTag>().get_group().size()*.2));
            }){}
        template<class CapacityGetter> collision_channel_contexts(sycl::queue q,SpeciesContextType& species,CapacityGetter capacity)
            :objects([&]<class... CollisionType>(std::tuple<CollisionType...>*){
                return storage_type{typename channel_context_type<CollisionType>::config_type{
                    q,background_grid<CollisionType>(species),
                    capacity.template operator()<typename CollisionType::incident_species_tag>()}...};
            }(static_cast<tags_tuple*>(nullptr))) {}
        template<class CollisionType> requires(model_types::template channel_index<CollisionType> >= 0)
        decltype(auto) get(){return std::get<model_types::template channel_index<CollisionType>>(objects);}
        acc_type get_access(sycl::handler& h){
            return {std::apply([&](auto&... x){return std::tuple{x.get_access(h)...};},objects)};
        }
    };
}
#endif
