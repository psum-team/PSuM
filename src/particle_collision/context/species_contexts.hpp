#ifndef PSUM_PARTICLE_COLLISION_SPECIES_CONTEXTS_HPP
#define PSUM_PARTICLE_COLLISION_SPECIES_CONTEXTS_HPP

#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <unordered_set>
#include "species_context.hpp"

namespace psum::particle_collision {

namespace foundation {
template<class Model>
using species_model_types=collision_model_types<typename Model::collision_types_tuple>;

template<class Tag,class Model,class ParticleGroupType,class DensityFieldType>
using species_resource=species_context<ParticleGroupType,DensityFieldType,
    (tag::foundation::index_in_tuple<Tag,typename species_model_types<Model>::product_tuple>::value >= 0),
    (tag::foundation::index_in_tuple<Tag,typename species_model_types<Model>::background_tuple>::value >= 0),
    species_model_types<Model>::template has_same_species_dpmcc<Tag>>;

template<class Model,class ParticleGroupType,class DensityFieldType,class... Tags>
auto species_storage_for(std::tuple<Tags...>*) -> tagged_context<
    tag::tag_bind<Tags,species_resource<Tags,Model,ParticleGroupType,DensityFieldType>>...>;

template<class Model,class ParticleGroupType,class DensityFieldType>
using species_storage=decltype(species_storage_for<Model,ParticleGroupType,DensityFieldType>(
    static_cast<typename species_model_types<Model>::species_tuple*>(nullptr)));
}

// Bind in species_tuple order. The model selects the background fields and product buffers.
template<class Model,class ParticleGroupType,class DensityFieldType>
class species_contexts : public foundation::species_storage<Model,ParticleGroupType,DensityFieldType> {
    using storage_type=foundation::species_storage<Model,ParticleGroupType,DensityFieldType>;
    using resource_types=typename storage_type::tuple_of_type;

    template<class First,class... Rest>
    static void check_bindings(const First& first,const Rest&... rest) {
        const ParticleGroupType* group=first.group;
        if(!group) throw std::invalid_argument("unbound collision species");
        const auto queue=group->get_content().get_queue();
        std::unordered_set<const ParticleGroupType*> groups{group};
        const auto check=[&](const auto& binding) {
            if(!binding.group) throw std::invalid_argument("unbound collision species");
            if(!groups.insert(binding.group).second)
                throw std::invalid_argument("different species tags must bind different particle groups");
            if(binding.group->get_content().get_queue()!=queue)
                throw std::invalid_argument("MCC bindings must use one queue");
        };
        (check(rest),...);
    }

    template<class... Resources,class... Bindings>
    requires (sizeof...(Resources)==sizeof...(Bindings) &&
              (!std::is_same_v<Bindings,Resources> && ...))
    static storage_type make_storage(std::tuple<Resources...>*,const Bindings&... bindings) {
        if constexpr(sizeof...(Bindings)>0) check_bindings(bindings...);
        return storage_type(Resources(bindings)...);
    }

public:
    template<class... Bindings>
    requires requires(const Bindings&... bindings) {
        make_storage(static_cast<resource_types*>(nullptr),bindings...);
    }
    explicit species_contexts(const Bindings&... bindings)
        : storage_type(make_storage(static_cast<resource_types*>(nullptr),bindings...)) {}
};

template <typename SpeciesTagsTuple, typename ContextType>
void flush_product_buffers(ContextType& ctx) {
    std::apply([&](auto&&... args) {
        auto one_clean_buffer = [&](auto v_tag) {
            typedef std::remove_cvref_t<decltype(v_tag)> species_tag;
            auto& resource=ctx.template get<species_tag>();
            if constexpr(requires { resource.clean_buffer(); resource.get_group(); resource.buffer; })
                resource.clean_buffer();
        };
        (one_clean_buffer(args),...);
    }, SpeciesTagsTuple{});
}

}
#endif
