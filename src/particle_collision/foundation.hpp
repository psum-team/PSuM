#ifndef PSUM_PARTICLE_COLLISION_FOUNDATION_HPP
#define PSUM_PARTICLE_COLLISION_FOUNDATION_HPP

#include <concepts>
#include <tuple>
#include <type_traits>
#include <utility>
#include <sycl/sycl.hpp>
#include "../../include/psum/tag.hpp"
#include "context/context_composition.hpp"

namespace psum {

namespace particle_collision {

    struct deferred_pairing_collision_tag {};
    // Reaction channels declare consumed_species_tags and product_species_tags.
    struct reaction_collision_tag {};

    template <typename T>
    concept has_collisions = requires {
        typename std::remove_cvref_t<T>::collision_types_tuple;
    };

namespace foundation {

    template<class Tuple,template<class> class Element> struct map_tuple;
    template<class... Elements,template<class> class Element>
    struct map_tuple<std::tuple<Elements...>,Element> { using type=std::tuple<Element<Elements>...>; };

    // Map host resources to their declared acc_type tuple without changing order.
    template<class Tuple> struct access_tuple;
    template<class... ResourceTypes>
    struct access_tuple<std::tuple<ResourceTypes...>> { using type=std::tuple<typename ResourceTypes::acc_type...>; };

    template<class T> inline constexpr bool species_tag_tuple=false;
    template<class... Tags> inline constexpr bool species_tag_tuple<std::tuple<Tags...>> =
        (tag::foundation::a_tag_concept<Tags>&&...);

    template<class Channel>
    inline constexpr bool has_particle_erasure = [] {
        if constexpr(std::is_base_of_v<reaction_collision_tag,Channel>) {
            return std::tuple_size_v<typename Channel::consumed_species_tags> != 0;
        } else {
            return false;
        }
    }();

    template<class Channel,class SpeciesTag>
    inline constexpr bool consumes_species = [] {
        if constexpr(std::is_base_of_v<reaction_collision_tag,Channel>) {
            return tag::foundation::index_in_tuple<SpeciesTag,typename Channel::consumed_species_tags>::value >= 0;
        } else {
            return false;
        }
    }();

    template<class Channel>
    consteval bool valid_reaction_species() {
        if constexpr(!std::is_base_of_v<reaction_collision_tag,Channel>) {
            return true;
        } else if constexpr(requires {
            requires species_tag_tuple<typename Channel::consumed_species_tags>;
            requires species_tag_tuple<typename Channel::product_species_tags>;
        }) {
            return []<class... Species>(std::tuple<Species...>*) {
                return ((std::is_same_v<Species,typename Channel::incident_species_tag> ||
                         (std::is_base_of_v<deferred_pairing_collision_tag,Channel> &&
                          std::is_same_v<Species,typename Channel::background_species_tag>)) && ...);
            }(static_cast<typename Channel::consumed_species_tags*>(nullptr));
        } else {
            return false;
        }
    }

    template<class Channel> struct product_species { using type=std::tuple<>; };
    template<class Channel>
    requires (std::is_base_of_v<reaction_collision_tag,Channel> &&
              requires { requires species_tag_tuple<typename Channel::product_species_tags>; })
    struct product_species<Channel> { using type=typename Channel::product_species_tags; };

    // Repeated product tags declare the maximum insertions per accepted event.
    template<class ProductTag,class Channels>
    inline constexpr size_t max_product_count=[] {
        size_t maximum=0;
        tag::foundation::for_each_type<Channels>([&]<class Channel> {
            size_t count=0;
            tag::foundation::for_each_type<typename product_species<Channel>::type>([&]<class Tag> {
                if constexpr(std::is_same_v<Tag,ProductTag>) ++count;
            });
            if(count>maximum) maximum=count;
        });
        return maximum;
    }();

    template<class CollisionType> concept collision_channel=requires {
        typename CollisionType::incident_species_tag;
        typename CollisionType::background_species_tag;
    } && valid_reaction_species<CollisionType>() &&
        (std::is_base_of_v<deferred_pairing_collision_tag,CollisionType> || requires {
            {CollisionType::collision_cross_section(0.0)} -> std::convertible_to<double>;
        });
    template<class CollisionType,class ParticleType,class ContextAcc>
    concept mcc_collision=collision_channel<CollisionType> && requires(ParticleType& p,const ContextAcc& ctx) {
        {CollisionType::collide(p,ctx)} -> std::same_as<void>;
    };
    template<class CollisionType,class PayloadType,class ContextAcc>
    concept dpmcc_collision=collision_channel<CollisionType> && requires(PayloadType& pair,const ContextAcc& ctx) {
        {CollisionType::request_probability(*pair.first,ctx)} -> std::same_as<double>;
        {CollisionType::pair_collision_cross_section(pair,ctx)} -> std::same_as<double>;
        {CollisionType::pair_collide(pair,ctx)} -> std::same_as<void>;
    };
    template<class CollisionType> using incident_species_of=typename CollisionType::incident_species_tag;
    template<class CollisionType> using background_species_of=typename CollisionType::background_species_tag;
    template<class CollisionTypesTuple>struct collision_model_types;
    template<class... CollisionTypes>struct collision_model_types<std::tuple<CollisionTypes...>> {
        static_assert((collision_channel<CollisionTypes>&&...),
            "Each collision needs incident/background tags; MCC needs collision_cross_section(double). "
            "Reactions need consumed_species_tags and product_species_tags as std::tuple types of species tags. "
            "MCC can consume the incident species only. DPMCC can consume either pair endpoint.");
        using channels=std::tuple<CollisionTypes...>;
        static_assert(std::tuple_size_v<typename unique_tuple<channels>::type> == sizeof...(CollisionTypes),
            "collision types must be unique");
        using incident_tuple=typename unique_tuple<std::tuple<incident_species_of<CollisionTypes>...>>::type;
        using background_tuple=typename unique_tuple<std::tuple<background_species_of<CollisionTypes>...>>::type;
        template<class Tag>
        static constexpr bool has_same_species_dpmcc=((
            std::is_base_of_v<deferred_pairing_collision_tag,CollisionTypes> &&
            std::is_same_v<Tag,incident_species_of<CollisionTypes>> &&
            std::is_same_v<Tag,background_species_of<CollisionTypes>>) || ...);
        using product_tuple=typename unique_tuple<decltype(std::tuple_cat(std::declval<typename product_species<CollisionTypes>::type>()...))>::type;
        using species_tuple=typename unique_tuple<decltype(std::tuple_cat(
            std::declval<std::tuple<incident_species_of<CollisionTypes>...,background_species_of<CollisionTypes>...>>(),
            std::declval<product_tuple>()))>::type;
    };

    template<class A,class B> inline constexpr bool same_collision_group=
        std::is_same_v<incident_species_of<A>,incident_species_of<B>> &&
        std::is_same_v<background_species_of<A>,background_species_of<B>> &&
        (std::is_base_of_v<deferred_pairing_collision_tag,A> == std::is_base_of_v<deferred_pairing_collision_tag,B>);
    template<class Channel,class Tuple> struct matching_channels;
    template<class Channel,class... CollisionTypes>struct matching_channels<Channel,std::tuple<CollisionTypes...>> {
        using type=decltype(std::tuple_cat(std::declval<std::conditional_t<same_collision_group<Channel,CollisionTypes>,
            std::tuple<CollisionTypes>,std::tuple<>>>()...));
    };
    template<class SpeciesTag,class Tuple> struct incident_collision_groups;
    template<class SpeciesTag,class... Channel>
    struct incident_collision_groups<SpeciesTag,std::tuple<Channel...>> {
        using channels=std::tuple<Channel...>;
        using type=typename unique_tuple<decltype(std::tuple_cat(std::declval<std::conditional_t<
            std::is_same_v<SpeciesTag,incident_species_of<Channel>>,
            std::tuple<typename matching_channels<Channel,channels>::type>,std::tuple<>>>()...))>::type;
    };

}

    template<typename... CollisionTypes>
    struct collision_model {
        using collision_types_tuple = std::tuple<CollisionTypes...>;
        using types=foundation::collision_model_types<collision_types_tuple>;
        using species_tuple=typename types::species_tuple;
    };

}

}

#endif
