#pragma once
#include <psum/psum.hpp>

// IST-Lisbon electron–Ar cross sections (processes 182789 and 182827).
// Energy in eV, cross section in m². The fingerprint identifies the source XML.
namespace ar_data {
inline constexpr char sha256[] = "66ff4aa70920585e927cd703be567fb729f4efa1f720185c3f9a0f767bb8d8e7";
inline constexpr double elastic_energy[] = {
    0.0, 0.001, 0.002, 0.003, 0.005, 0.007, 0.0085, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.07,
    0.1, 0.12, 0.15, 0.17, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.7, 1.0, 1.2, 1.3, 1.5, 1.7, 1.9,
    2.1, 2.2, 2.5, 2.8, 3.0, 3.3, 3.6, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 15.0, 17.0,
    20.0, 25.0, 30.0, 50.0, 75.0, 100.0, 150.0, 200.0, 300.0, 500.0, 700.0, 1000.0
};
inline constexpr double elastic_sigma[] = {
    7.5e-20, 7.5e-20, 7.1e-20, 6.7e-20, 6.1e-20, 5.4e-20, 5.05e-20, 4.6e-20, 3.75e-20, 3.25e-20,
    2.5e-20, 2.05e-20, 1.73e-20, 1.13e-20, 5.9e-21, 4e-21, 2.3e-21, 1.6e-21, 1.03e-21, 9.1e-22,
    1.53e-21, 2.35e-21, 3.3e-21, 5.1e-21, 8.6e-21, 1.38e-20, 1.66e-20, 1.82e-20, 2.1e-20,
    2.3e-20, 2.5e-20, 2.8e-20, 2.9e-20, 3.3e-20, 3.8e-20, 4.1e-20, 4.5e-20, 4.9e-20, 5.4e-20,
    6.1e-20, 6.7e-20, 8.1e-20, 9.6e-20, 1.17e-19, 1.5e-19, 1.45e-19, 1.37e-19, 1.1e-19, 9.2e-20,
    6.8e-20, 5.5e-20, 3.2e-20, 2.15e-20, 1.6e-20, 1.1e-20, 8.8e-21, 6e-21, 3.7e-21, 2.6e-21,
    1.7e-21
};
inline constexpr double ionization_energy[] = {
    15.76, 16.0, 16.5, 17.0, 18.0, 19.0, 20.0, 22.0, 24.0, 26.0, 28.0, 30.0, 34.0, 40.0, 45.0,
    50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 150.0, 200.0, 250.0, 300.0, 400.0, 500.0, 1000.0
};
inline constexpr double ionization_sigma[] = {
    0.0, 2.024e-22, 6.688e-22, 1.3376e-21, 2.9392e-21, 4.6024e-21, 6.2744e-21, 9.328e-21,
    1.1792e-20, 1.408e-20, 1.6016e-20, 1.804e-20, 2.112e-20, 2.3936e-20, 2.4904e-20, 2.5344e-20,
    2.6576e-20, 2.772e-20, 2.8424e-20, 2.86e-20, 2.8512e-20, 2.684e-20, 2.3936e-20, 2.1736e-20,
    1.98e-20, 1.6808e-20, 1.4608e-20, 9.152e-21
};
}

namespace argon {
using namespace psum::prelude;
using namespace psum::particle_collision;
using Electron = species_tags::electron;
using Atom = species_tags::Argon;
using Ion = species_tags::Argon_pos_1;
constexpr double e = 1.602176634e-19, me = 9.109383632e-31, ma = 66.3e-27;
constexpr double reduced_mass = me * ma / (me + ma);

template<size_t N>
double interpolate(double energy, const double (&x)[N], const double (&y)[N]) {
    if (energy <= x[0]) return y[0];
    if (energy >= x[N-1]) return y[N-1];
    size_t lo = 0, hi = N-1;
    while (hi - lo > 1) {
        size_t mid = (hi + lo) / 2;
        if (energy < x[mid]) hi = mid;
        else lo = mid;
    }
    return y[lo] + (energy-x[lo]) / (x[hi]-x[lo]) * (y[hi]-y[lo]);
}

using Particle = tagged_struct<tag_bind<property::position,Eigen::Vector<double,1>>,
    tag_bind<property::velocity,Eigen::Vector3d>, tag_bind<property::weight,double>,
    tag_bind<property::random_seed,uint32_t>>;
using Group = particle_group<Particle,pos_x_nan_is_invalid>;
using Field = cell_field1D<double>;
using Vec = Eigen::Vector3d;
inline auto& weight(auto& p) { return get<property::weight>(p); }
inline auto& velocity(auto& p) { return get<property::velocity>(p); }

enum Counter { requested, matched, clipped, accepted, counter_count };
inline void increment(uint64_t* counter) {
    sycl::atomic_ref<uint64_t,sycl::memory_order::relaxed,sycl::memory_scope::device,
        sycl::access::address_space::global_space>(*counter).fetch_add(1);
}

// Count actual enqueues at the normal request buffer. The library still owns all pairing.
template<class Resource>
struct ObservedChannel {
    decltype(Resource::buffer)& buffer;
    decltype(Resource::engine)& engine;
    uint64_t* counters;
    double atom_count;
    double* produced{};
    size_t* live{};
    struct acc_type {
        typename Resource::acc_type output;
        uint64_t* counters;
        double atom_count;
        double* produced{};
        size_t* live{};
        void push_back(const std::pair<uint32_t,typename Resource::payload_type>& request) const {
            output.push_back(request);
            increment(counters+requested);
        }
    };
    acc_type get_access(sycl::handler& h) { return {buffer.get_access(h),counters,atom_count,produced,live}; }
};
template<class R>
auto observe(R& r,uint64_t* counters) { return ObservedChannel<R>{r.buffer,r.engine,counters,0}; }

struct ArgonChannel : deferred_pairing_collision_tag, reaction_collision_tag, psum::tag::foundation::abstract_tag {
    using background_species_tag = Atom;
    template<class Channel>
    static double effective(double sigma,const auto& pair,const auto& ctx) {
        const auto& a=ctx.template get<Channel>();
        const auto& gas=ctx.template get<Atom>();
        const double density=gas.field_acc(0);
        double count=a.atom_count;
        if constexpr(requires { gas.particle_counts(0); })
            count=gas.particle_counts(0);
        double scaled=sigma*(sycl::fmax(weight(*pair.first),weight(*pair.second))*count/density);
        increment(a.counters+matched);
        double g=(velocity(*pair.first)-velocity(*pair.second)).norm();
        double probability=scaled*g*density*ctx.dt/pair.request_probability;
        if constexpr(std::is_same_v<typename Channel::incident_species_tag,Atom>)
            probability*=0.5*(1.-1./count);
        if(probability>1) increment(a.counters+clipped);
        return scaled;
    }
    template<class Tag>
    static void update_count(size_t* live,int delta) {
        if(!live)return;
        constexpr size_t index=std::is_same_v<Tag,Electron>?0:(std::is_same_v<Tag,Atom>?1:2);
        sycl::atomic_ref<size_t,sycl::memory_order::relaxed,sycl::memory_scope::device,
            sycl::access::address_space::global_space>(live[index]).fetch_add(delta);
    }
    template<class Tag>
    static void split(Particle& p,double reacting,const auto& ctx,auto& rng,size_t* live) {
        if (weight(p)>reacting) {
            auto rest=p;
            weight(rest)-=reacting;
            get<property::random_seed>(rest)=rng.gen_seed();
            ctx.template insert_particle<Tag>(rest);
            update_count<Tag>(live,1);
            weight(p)=reacting;
        }
    }
};

struct ElectronArgon : ArgonChannel {
    static double request_probability(const auto&, const auto&) { return 0.1; }
    inline static const std::string tag_name="electron-argon";
    using incident_species_tag=Electron;
    static double energy(const auto& pair) {
        return 0.5*reduced_mass/e*(velocity(*pair.first)-velocity(*pair.second)).squaredNorm();
    }
    static Vec center(const auto& pair) {
        return (me*velocity(*pair.first)+ma*velocity(*pair.second))/(me+ma);
    }
};

struct Elastic : ElectronArgon {
    using consumed_species_tags=std::tuple<>;
    using product_species_tags=std::tuple<Electron,Atom>;
    static double pair_collision_cross_section(const auto& pair,const auto& ctx) {
        return effective<Elastic>(interpolate(energy(pair),ar_data::elastic_energy,ar_data::elastic_sigma),pair,ctx);
    }
    static void pair_collide(auto& pair,const auto& ctx) {
        const auto& channel=ctx.template get<Elastic>();
        increment(channel.counters+accepted);
        auto& el=*pair.first;
        auto& ar=*pair.second;
        double w=sycl::fmin(weight(el),weight(ar));
        auto random=psum::random::view_as_rander(get<property::random_seed>(el));
        auto& rng=random.get_rander();
        split<Electron>(el,w,ctx,rng,channel.live);
        split<Atom>(ar,w,ctx,rng,channel.live);
        Vec com=center(pair);
        Vec relative=(velocity(el)-velocity(ar)).norm()*psum::random::RandFunction3D::RandV_spherical(rng);
        velocity(el)=com+ma/(me+ma)*relative;
        velocity(ar)=com-me/(me+ma)*relative;
    }
};

struct Ionization : ElectronArgon {
    using consumed_species_tags=std::tuple<Atom>;
    using product_species_tags=std::tuple<Electron,Electron,Ion>;
    static double pair_collision_cross_section(const auto& pair,const auto& ctx) {
        return effective<Ionization>(interpolate(energy(pair),ar_data::ionization_energy,ar_data::ionization_sigma),pair,ctx);
    }
    static void pair_collide(auto& pair,const auto& ctx) {
        const auto& channel=ctx.template get<Ionization>();
        increment(channel.counters+accepted);
        auto& el=*pair.first;
        auto& ar=*pair.second;
        double w=sycl::fmin(weight(el),weight(ar));
        auto random=psum::random::view_as_rander(get<property::random_seed>(el));
        auto& rng=random.get_rander();
        split<Electron>(el,w,ctx,rng,channel.live);
        auto child=el, ion=ar;
        const Vec com=center(pair);
        const double speed=sycl::sqrt((energy(pair)-ar_data::ionization_energy[0])*e/me);
        Vec v1=speed*psum::random::RandFunction3D::RandV_spherical(rng);
        Vec v2=speed*psum::random::RandFunction3D::RandV_spherical(rng);
        velocity(el)=com+v1;
        velocity(child)=com+v2;
        velocity(ion)=com-(me/ma)*(v1+v2);
        weight(ion)=w;
        if(auto* produced=ctx.template get<Ionization>().produced)
            sycl::atomic_ref<double,sycl::memory_order::relaxed,sycl::memory_scope::device,
                sycl::access::address_space::global_space>(*produced).fetch_add(w);
        get<property::random_seed>(child)=rng.gen_seed();
        ctx.template insert_particle<Electron>(child);
        ctx.template insert_particle<Ion>(ion);
        update_count<Electron>(channel.live,1);
        update_count<Ion>(channel.live,1);
        if(weight(ar)==w){
            ctx.template erase_particle<Atom>(&ar);
            update_count<Atom>(channel.live,-1);
        }
        else weight(ar)-=w;
    }
};

struct ArgonElastic : ArgonChannel {
    static double request_probability(const auto&, const auto&) { return 0.05; }
    inline static const std::string tag_name="argon-argon";
    using incident_species_tag=Atom;
    using consumed_species_tags=std::tuple<>;
    using product_species_tags=std::tuple<Atom>;
    static double pair_collision_cross_section(const auto& pair,const auto& ctx) {
        double g=(velocity(*pair.first)-velocity(*pair.second)).norm();
        constexpr double diameter=4.11e-10, boltzmann=1.380649e-23;
        double sigma=g>0 ? std::acos(-1.)*diameter*diameter*
            sycl::pow(4*boltzmann*273/ma,.31)*sycl::pow(g,-.62)/sycl::tgamma(1.69) : 0.;
        return effective<ArgonElastic>(sigma,pair,ctx);
    }
    static void pair_collide(auto& pair,const auto& ctx) {
        const auto& channel=ctx.template get<ArgonElastic>();
        increment(channel.counters+accepted);
        auto& first=*pair.first;
        auto& second=*pair.second;
        auto random=psum::random::view_as_rander(get<property::random_seed>(first));
        auto& rng=random.get_rander();
        double w=sycl::fmin(weight(first),weight(second));
        split<Atom>(first,w,ctx,rng,channel.live);
        split<Atom>(second,w,ctx,rng,channel.live);
        Vec center=(velocity(first)+velocity(second))/2;
        Vec half=0.5*(velocity(first)-velocity(second)).norm()*
            psum::random::RandFunction3D::RandV_spherical(rng);
        velocity(first)=center+half;
        velocity(second)=center-half;
    }
};

}
