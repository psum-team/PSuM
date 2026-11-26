#ifndef PSUM_PARTICLE_COLLISION_CONTEXT_COMPOSITION_HPP
#define PSUM_PARTICLE_COLLISION_CONTEXT_COMPOSITION_HPP

#include <concepts>
#include <tuple>
#include <type_traits>
#include <utility>
#include <sycl/sycl.hpp>
#include "../../tag/tagged_struct.hpp"

namespace psum::particle_collision {
namespace foundation {
    template<class Input,class Output=std::tuple<>> struct unique_tuple;
    template<class... Out>
    struct unique_tuple<std::tuple<>,std::tuple<Out...>> { using type=std::tuple<Out...>; };
    template<class T,class... Rest,class... Out>
    struct unique_tuple<std::tuple<T,Rest...>,std::tuple<Out...>>
        : unique_tuple<std::tuple<Rest...>,std::conditional_t<(std::is_same_v<T,Out>||...),std::tuple<Out...>,std::tuple<Out...,T>>> {};

    template<class ContextType,class ResourceTag>
    concept context_entry=requires(ContextType& host,const typename ContextType::acc_type& device) {
        typename std::remove_cvref_t<decltype(host.template get<ResourceTag>())>::acc_type;
        requires std::is_lvalue_reference_v<decltype(host.template get<ResourceTag>())>;
        requires std::same_as<std::remove_cvref_t<decltype(device.template get<ResourceTag>())>,
            typename std::remove_cvref_t<decltype(host.template get<ResourceTag>())>::acc_type>;
    };
    template<class ContextType,class Tuple> inline constexpr bool context_entries=false;
    template<class ContextType,class... ResourceTags>
    inline constexpr bool context_entries<ContextType,std::tuple<ResourceTags...>> = (context_entry<ContextType,ResourceTags>&&...);
    template<class ContextType,class... ResourceTags>
    concept context=requires(ContextType& host,sycl::handler& h) {
        typename ContextType::acc_type;
        {host.get_access(h)} -> std::same_as<typename ContextType::acc_type>;
    } && [] {
        if constexpr(sizeof...(ResourceTags)>0)return context_entries<ContextType,std::tuple<ResourceTags...>>;
        else if constexpr(requires{typename ContextType::tags_tuple;})return context_entries<ContextType,typename ContextType::tags_tuple>;
        else return false;
    }();

    // Locate the owning context at compile time; do not copy resources or perform device-side lookup.
    template<class ResourceTag,class... ContextTypes> inline constexpr int context_owner_index=[] {
        int result=-1,index=0;
        ((psum::tag::foundation::index_in_tuple<ResourceTag,typename ContextTypes::tags_tuple>::value>=0
            ? (result=index,++index) : ++index),...);
        return result;
    }();
}

// Own tagged resources and expose their device accessors by tag.
template<class... ResourceBindings> class tagged_context : public tag::tagged_struct<ResourceBindings...> {
    using storage_type=tag::tagged_struct<ResourceBindings...>;
public:
    using tags_tuple=typename storage_type::tuple_of_tag;
    using device_tuple=std::tuple<typename ResourceBindings::back::acc_type...>;
    struct acc_type : device_tuple {
        using device_tuple::device_tuple;
        acc_type(const device_tuple& resources) : device_tuple(resources) {}

        template<class ResourceTag> requires(psum::tag::foundation::index_in_tuple<ResourceTag,tags_tuple>::value>=0)
        decltype(auto) get()const {
            return std::get<psum::tag::foundation::index_in_tuple<ResourceTag,tags_tuple>::value>(*this);
        }
    };
    using storage_type::storage_type;
    template<class ResourceTag> requires(psum::tag::foundation::index_in_tuple<ResourceTag,tags_tuple>::value>=0)
    decltype(auto) get(){return tag::get<ResourceTag>(*static_cast<storage_type*>(this));}
    template<class ResourceTag> static constexpr bool exist(){
        return psum::tag::foundation::index_in_tuple<ResourceTag,tags_tuple>::value>=0;
    }
    acc_type get_access(sycl::handler& h){return acc_type(tag::get<typename ResourceBindings::front>(*this).get_access(h)...);}
};

// Non-owning composition: host contexts must outlive this object; device views contain only their acc_type.
// Reject duplicate tags so later contexts cannot silently shadow existing resources.
template<foundation::context... ContextTypes> class context_composition {
public:
    using tags_tuple=decltype(std::tuple_cat(std::declval<typename ContextTypes::tags_tuple>()...));
    using device_tuple=std::tuple<typename ContextTypes::acc_type...>;
    static_assert(std::tuple_size_v<typename foundation::unique_tuple<tags_tuple>::type>
        ==std::tuple_size_v<tags_tuple>,"duplicate tag in context composition");
    struct acc_type {
        device_tuple contexts;

        template<class ResourceTag> requires(foundation::context_owner_index<ResourceTag,ContextTypes...> >= 0)
        decltype(auto) get()const {
            const auto& owner=std::get<foundation::context_owner_index<ResourceTag,ContextTypes...>>(contexts);
            return owner.template get<ResourceTag>();
        }
    };
private:
    std::tuple<ContextTypes*...> contexts_;
public:
    explicit context_composition(ContextTypes&... contexts):contexts_(&contexts...){}
    template<class ResourceTag> requires(foundation::context_owner_index<ResourceTag,ContextTypes...> >= 0)
    decltype(auto) get(){return std::get<foundation::context_owner_index<ResourceTag,ContextTypes...>>(contexts_)->template get<ResourceTag>();}
    acc_type get_access(sycl::handler& h){
        return {std::apply([&](auto*... c){return device_tuple{c->get_access(h)...};},contexts_)};
    }
};
template<class... ContextTypes> context_composition(ContextTypes&...)->context_composition<ContextTypes...>;
}
#endif
