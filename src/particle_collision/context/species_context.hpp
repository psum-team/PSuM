#ifndef PSUM_PARTICLE_COLLISION_SPECIES_CONTEXT_HPP
#define PSUM_PARTICLE_COLLISION_SPECIES_CONTEXT_HPP

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include "../foundation.hpp"
#include "../../field/device_field.hpp"
#include "../../particle_container/device_vector.hpp"

namespace psum::particle_collision {

// Map shared physical particle coordinates onto this background species' field grid.
template<class SpeciesTag> struct position_projection {
    template<class Position> static auto project(const Position& pos) {return pos;}
};

namespace foundation {
template<size_t Dimension,class SpeciesTag,class Position>
auto projected_position(const Position& pos) {
    auto projected=position_projection<SpeciesTag>::project(pos);
    using projected_type=std::remove_cvref_t<decltype(projected)>;
    if constexpr(requires{std::tuple_size<projected_type>::value;})
        static_assert(std::tuple_size_v<projected_type> == Dimension,"projection dimension must match the field/grid");
    else if constexpr(requires{projected_type::SizeAtCompileTime;})
        static_assert(projected_type::SizeAtCompileTime == int(Dimension),"projection dimension must match the field/grid");
    std::array<double,Dimension> coordinates{};
    for(size_t i=0;i<Dimension;++i)coordinates[i]=projected[i];
    return coordinates;
}
template<class IncidentTag,class BackgroundTag,class Position,class ContextAcc>
double collision_density(const Position& pos,const ContextAcc& ctx) {
    if constexpr(requires { ctx.unbound_access(); }) {
        // Keep density specializations for the original device context.
        return collision_density<IncidentTag,BackgroundTag>(pos,ctx.unbound_access());
    } else {
        auto field=ctx.template get<BackgroundTag>().field_acc;
        auto coordinates=projected_position<decltype(field)::Dimension,BackgroundTag>(pos);
        if constexpr(decltype(field)::Location==psum::field::var_loc::cellCentered)
            return psum::field::interp_nearest(coordinates,field);
        else return psum::field::interp(coordinates,field);
    }
}

// Return extent for an endpoint outside the particle storage.
template<class Particle>
size_t endpoint_slot(const Particle* data,size_t extent,const Particle* particle) {
    const auto base=reinterpret_cast<std::uintptr_t>(data);
    const auto addr=reinterpret_cast<std::uintptr_t>(particle);
    if(!particle || addr<base)return extent;
    const auto offset=addr-base;
    const auto index=offset/sizeof(Particle);
    return offset%sizeof(Particle)==0 && index<extent ? index : extent;
}

struct no_product_buffer {};
struct no_density_field {};
struct no_particle_counts {};
struct no_incident_marks {};

// MCC uses the first flag for its current particle. DPMCC also supplies both endpoint addresses.
struct erasure_requests {
    const void* first=nullptr;
    const void* second=nullptr;
    bool erase_first=false;
    bool erase_second=false;
};

enum class erasure_scope { none, current, first, second, both };

template<class Tag,class Collision>
consteval erasure_scope species_erasure_scope() {
    if constexpr(!consumes_species<Collision,Tag>) {
        return erasure_scope::none;
    } else if constexpr(!std::is_base_of_v<deferred_pairing_collision_tag,Collision>) {
        return erasure_scope::current;
    } else {
        constexpr bool first=std::is_same_v<Tag,incident_species_of<Collision>>;
        constexpr bool second=std::is_same_v<Tag,background_species_of<Collision>>;
        if constexpr(first && second) return erasure_scope::both;
        else if constexpr(first) return erasure_scope::first;
        else if constexpr(second) return erasure_scope::second;
        else return erasure_scope::none;
    }
}

template<class Tag,class Collision>
consteval bool can_insert_product() {
    if constexpr(std::is_base_of_v<reaction_collision_tag,Collision>) {
        return tag::foundation::index_in_tuple<Tag,typename Collision::product_species_tags>::value >= 0;
    } else {
        return false;
    }
}

template<class Access,class Tag>
using species_particle_t=typename std::remove_cvref_t<decltype(std::declval<const Access&>().template get<Tag>())>::particle_type;

// Add the selected reaction's permissions to the device context used by the callback.
template<class Collision,class Access>
struct reaction_context : Access {
    explicit reaction_context(const Access& access) requires (!has_particle_erasure<Collision>)
        : Access(access) {}
    reaction_context(const Access& access,erasure_requests& requests) requires has_particle_erasure<Collision>
        : Access(access), requests_(&requests) {}

    const Access& unbound_access() const { return *this; }

    template<class Tag>
    requires (can_insert_product<Tag,Collision>() && requires(const Access& access,const species_particle_t<Access,Tag>& p) {
        access.template get<Tag>().insert_particle(p);
    })
    void insert_particle(const species_particle_t<Access,Tag>& p) const {
        this->template get<Tag>().insert_particle(p);
    }

    template<class Tag>
    requires (species_erasure_scope<Tag,Collision>() == erasure_scope::current &&
              requires { typename species_particle_t<Access,Tag>; })
    void erase_particle() const {
        this->template get<Tag>().erase_particle(*requests_);
    }

    template<class Tag>
    requires (std::is_base_of_v<deferred_pairing_collision_tag,Collision> &&
              species_erasure_scope<Tag,Collision>() != erasure_scope::none)
    void erase_particle(const species_particle_t<Access,Tag>* particle) const {
        constexpr auto scope=species_erasure_scope<Tag,Collision>();
        this->template get<Tag>().template erase_particle<scope>(particle,*requests_);
    }

private:
    struct no_requests {};
    [[no_unique_address]] std::conditional_t<has_particle_erasure<Collision>,erasure_requests*,no_requests> requests_{};
};
}

// Borrow a species' particle group and any required density; own a product buffer when requested.
template<class ParticleGroupType,class DensityFieldType,bool WithProductBuffer=false,bool WithDensity=true,bool WithParticleCounts=false>
struct species_context {
    static_assert(!WithParticleCounts || WithDensity);
    using particle_type=typename ParticleGroupType::value_type;
    using product_buffer_type=particle_container::device_vector<particle_type>;
    using buffer_type=std::conditional_t<WithProductBuffer,product_buffer_type,foundation::no_product_buffer>;
    using product_access=std::conditional_t<WithProductBuffer,typename product_buffer_type::acc_type,foundation::no_product_buffer>;
    using density_access=std::conditional_t<WithDensity,typename DensityFieldType::acc_type,foundation::no_density_field>;
    using particle_count_field=field::device_field<DensityFieldType::Dimension,field::var_loc::cellCentered,uint32_t>;
    using count_type=std::conditional_t<WithParticleCounts,particle_count_field,foundation::no_particle_counts>;
    using count_access=std::conditional_t<WithParticleCounts,typename particle_count_field::acc_type,foundation::no_particle_counts>;

    struct acc_type {
        using particle_type=typename species_context::particle_type;
        [[no_unique_address]] density_access field_acc;
        [[no_unique_address]] count_access particle_counts;

        acc_type(density_access field,product_access products={},count_access counts={})
            : field_acc(field), particle_counts(counts), buffer_acc_(products) {}

    private:
        template<class,class> friend struct foundation::reaction_context;

        void insert_particle(const particle_type& p) const requires WithProductBuffer {
            buffer_acc_.push_back(p);
        }

        static void erase_particle(foundation::erasure_requests& requests) { requests.erase_first=true; }

        template<foundation::erasure_scope Scope>
        static void erase_particle(const particle_type* particle,foundation::erasure_requests& requests) {
            using foundation::erasure_scope;
            if constexpr(Scope == erasure_scope::first || Scope == erasure_scope::both) {
                if(particle==requests.first) requests.erase_first=true;
            }
            if constexpr(Scope == erasure_scope::second || Scope == erasure_scope::both) {
                if(particle==requests.second) requests.erase_second=true;
            }
        }

        [[no_unique_address]] product_access buffer_acc_;
    };

    ParticleGroupType* const group;
    DensityFieldType* const density;
    [[no_unique_address]] buffer_type buffer;
    // Snapshot after MCC, before DPMCC execution; one count per pairing cell.
    [[no_unique_address]] count_type particle_counts;

    template<class BindingType>
    explicit species_context(const BindingType& binding)
        : group(binding.group), density(bound_density(binding)),
          buffer(make_buffer(checked_group())), particle_counts(make_particle_counts()) {}

    auto get_particle_access(sycl::handler& h) {
        using particle = typename ParticleGroupType::value_type;
        using validator = typename ParticleGroupType::validator;
        struct particle_view {
            const particle* data;
            size_t extent;
            bool is_valid(size_t index) const { return index < extent && validator::is_valid(data[index]); }
        };
        return particle_view{group->get_content().get_access(h).data(),group->extent()};
    }

    acc_type get_access(sycl::handler& h) {
        auto field = [&] {
            if constexpr(WithDensity) {
                return density->get_access(h);
            } else {
                return foundation::no_density_field{};
            }
        }();
        auto counts = [&] {
            if constexpr(WithParticleCounts) return particle_counts.get_access(h);
            else return foundation::no_particle_counts{};
        }();
        if constexpr(WithProductBuffer) return {field,buffer.get_access(h),counts};
        else return {field,{},counts};
    }

    auto& get_group() { return *group; }

    // Called once per nonempty same-species segment, after filtering its first live requests.
    uint8_t* prepare_incident_marks(size_t extent) requires WithParticleCounts {
        auto q=group->get_content().get_queue();
        if(!incident_marks_ || incident_marks_->size()<extent) {
            const size_t capacity=incident_marks_?incident_marks_->size():0;
            incident_marks_=std::make_unique<field::device_array<uint8_t>>(
                q,std::max(extent,capacity+capacity/2));
        }
        auto* marked=incident_marks_->data();
        // Clear active slots on every segment, including slots reused since the last call.
        q.submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<1>(extent),[=](sycl::id<1> i) { marked[i]=0; });
        }).wait();
        return marked;
    }


    void clean_buffer() requires WithProductBuffer {
        auto& particles=get_group();
        const size_t count=buffer.size();
        if(count==0)return;
        particles.insert(buffer);
        buffer.clear();
    }

private:
    [[no_unique_address]] std::conditional_t<WithParticleCounts,
        std::unique_ptr<field::device_array<uint8_t>>,foundation::no_incident_marks> incident_marks_;

    template<class BindingType>
    static DensityFieldType* bound_density(const BindingType& binding) {
        if constexpr(WithDensity) return binding.density;
        else return nullptr;
    }

    const ParticleGroupType& checked_group() const {
        if(!group || (WithDensity && !density)) throw std::invalid_argument("unbound collision species");
        if constexpr(WithDensity) {
            if(density->getQueue()!=group->get_content().get_queue())
                throw std::invalid_argument("MCC bindings must use one queue");
        }
        return *group;
    }

    count_type make_particle_counts() const {
        if constexpr(WithParticleCounts)
            return particle_count_field(group->get_content().get_queue(),density->getGrid());
        else return {};
    }

    static buffer_type make_buffer(const ParticleGroupType& particles) {
        if constexpr(WithProductBuffer)
            return product_buffer_type(particles.get_content().get_queue(),std::max(size_t{8192},size_t(particles.size()*.2)));
        else return {};
    }
};

}
#endif
