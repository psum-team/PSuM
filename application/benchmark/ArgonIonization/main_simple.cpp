// Standalone 0D argon benchmark. Channel physics follows mpdt-magnetic-nozzle-3d
// (psumpreview a3095aa); fixed species weights and diagnostics use the PSuM DPMCC API.
#include <psum/psum.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

namespace nozzle {
using namespace psum::prelude;
using namespace psum::particle_collision;
using Electron = species_tags::electron;
using Atom = species_tags::Argon;
using Ion = species_tags::Argon_pos_1;
constexpr double e = 1.602176634e-19, me = 9.109383632e-31, ma = 66.3e-27;
using Vec = Eigen::RowVector3d;
using Particle = tagged_struct<tag_bind<property::position,Vec>,
    tag_bind<property::velocity,Vec>, tag_bind<property::random_seed,uint32_t>>;
using Group = particle_group<Particle,pos_x_nan_is_invalid>;
using Field = cell_field3D<double>;
inline const Vec& velocity(const Particle& p) { return get<property::velocity>(p); }
inline void set_velocity(Particle& p,const Vec& v) { get<property::velocity>(p)=v; }

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
    double* produced{};
    double plasma_weight{}, neutral_update_probability{};
    struct acc_type {
        typename Resource::acc_type output;
        uint64_t* counters;
        double* produced{};
        double plasma_weight{}, neutral_update_probability{};
        void push_back(const std::pair<uint32_t,typename Resource::payload_type>& request) const {
            output.push_back(request);
            increment(counters+requested);
        }
    };
    acc_type get_access(sycl::handler& h) { return {buffer.get_access(h),counters,produced,plasma_weight,neutral_update_probability}; }
};
template<class R>
auto observe(R& r,uint64_t* counters) { return ObservedChannel<R>{r.buffer,r.engine,counters}; }

struct ArgonChannel : deferred_pairing_collision_tag, psum::tag::foundation::abstract_tag {
    using background_species_tag = Atom;
    template<class Channel>
    static double effective(double sigma,const auto& pair,const auto& ctx) {
        const auto& a=ctx.template get<Channel>();
        const auto& gas=ctx.template get<Atom>();
        const double density=gas.field_acc(0);
        const double count=gas.particle_counts(0);
        increment(a.counters+matched);
        double g=(velocity(*pair.first)-velocity(*pair.second)).norm();
        double probability=sigma*g*density*ctx.dt/pair.request_probability;
        if constexpr(std::is_same_v<typename Channel::incident_species_tag,Atom>)
            probability*=0.5*(1.-1./count);
        if(probability>1) increment(a.counters+clipped);
        return sigma;
    }

};

struct ElectronArgon : ArgonChannel {
    static double request_probability(const auto&, const auto&) { return 0.1; }
    inline static const std::string tag_name="electron-argon";
    using incident_species_tag=Electron;
};

using Generator=std::mt19937_64;
constexpr double kb=1.380649e-23, dt=5e-14, neutral0=1e23, electron0=1e16;
constexpr size_t atom_target=4000;

struct Moments {
    double total=0;
    Vec mean=Vec::Zero(), variance=Vec::Zero();
};
template<class T> using DeviceArray=psum::field::device_array<T>;

// Sample the Maxwell distribution with equal-weight representatives of each species.
Moments initialize_group(Group& group,size_t count,double thermal_sigma,double macro_weight,Generator& rng) {
    std::normal_distribution<double> gaussian(0.,thermal_sigma);
    std::vector<Particle> particles(count);
    Moments m;
    for(auto& p:particles) {
        get<property::position>(p).setConstant(0.5);
        Vec v;
        for(int d=0;d<3;++d)v[d]=gaussian(rng);
        set_velocity(p,v);
        get<property::random_seed>(p)=uint32_t(rng());
        m.mean+=velocity(p);
    }
    m.mean/=double(count);
    for(const auto& p:particles)m.variance+=(velocity(p)-m.mean).array().square().matrix();
    m.variance/=double(count);
    m.total=count*macro_weight;
    group.insert<false>(particles);
    return m;
}

// Ionization creates one electron and one ion. Sharing the Bernoulli draws
// preserves equal charged counts, even when the groups have different holes.
void thin_charged_groups(Group& electrons,Group& ions,Generator& rng,
                         std::array<double,3>& weights,int step) {
    constexpr size_t threshold=12000, goal=6000;
    if(electrons.extent()<=threshold) return;
    const size_t n=electrons.size();
    if(n<=threshold) return;
    if(ions.size()!=n) throw std::logic_error("unequal charged counts before thinning");
    const double new_weight=weights[0]*(double(n)/goal);
    if(new_weight>weights[1])
        throw std::runtime_error("thinned plasma weight would exceed neutral weight");
    const double probability=double(goal)/n;
    std::bernoulli_distribution retain(probability);
    std::vector<uint8_t> keep(n);
    size_t survivors=0;
    for(auto& draw:keep) { draw=retain(rng); survivors+=draw; }
    auto q=electrons.get_content().get_queue();
    DeviceArray<uint8_t> mask(q,keep);
    DeviceArray<size_t> cursors(q,std::vector<size_t>(2,0));
    auto thin=[&](Group& group,size_t species) {
        auto* cursor=cursors.data()+species;
        auto* draws=mask.data();
        group.for_each([=](sycl::handler&) {
            return [=](Particle& p) {
                sycl::atomic_ref<size_t,sycl::memory_order::relaxed,sycl::memory_scope::device,
                    sycl::access::address_space::global_space> ticket(*cursor);
                if(!draws[ticket.fetch_add(1)]) Group::validator::make_invalid(p);
            };
        });
    };
    thin(electrons,0);
    thin(ions,1);
    weights[0]=weights[2]=new_weight;
    std::cout<<std::setprecision(17)<<"thinning step="<<step<<" before="<<n
             <<" after="<<survivors<<" keep_probability="<<probability
             <<" plasma_weight="<<new_weight<<" neutral_update_probability="<<new_weight/weights[1]<<'\n';
}

struct RunStats {
    Moments species[3];
    double produced=0,rate=0,temperature=0,sum_rate=0,sum_temperature=0;
    double histogram[101]{},spectrum[101]{};
    unsigned snapshots=0;
};

struct PartialMoments {
    double mass=0;
    Vec first=Vec::Zero(), second=Vec::Zero();
};

class DeviceDiagnostics {
    DeviceArray<PartialMoments> partials;
public:
    explicit DeviceDiagnostics(sycl::queue q):partials(q,1){}
    void advance(Group& el,Group& ar,Group& ion,RunStats* state,double* density,double kick,std::array<double,3> weights,
                 int step,int start,int steps){
        constexpr size_t width=128;
        std::array<Particle*,3> data{el.get_content().data(),ar.get_content().data(),ion.get_content().data()};
        std::array<size_t,3> extent{el.extent(),ar.extent(),ion.extent()};
        std::array<size_t,4> offset{};
        for(int s=0;s<3;++s)offset[s+1]=offset[s]+(extent[s]+width-1)/width;
        auto q=el.get_content().get_queue();
        if(offset[3]>partials.size())partials=DeviceArray<PartialMoments>(q,offset[3]*2);
        auto* out=partials.data();
        bool sample=step>=start && step<steps && (step==start || step%1000==0);
        auto partials_ready=q.parallel_for(sycl::nd_range<1>{offset[3]*width,width},[=](sycl::nd_item<1> item){
            auto block=item.get_group_linear_id(),lane=item.get_local_linear_id();
            int s=block<offset[1]?0:(block<offset[2]?1:2);
            size_t i=(block-offset[s])*width+lane;
            bool live=i<extent[s] && Group::validator::is_valid(data[s][i]);
            double w=0;
            Vec delta=Vec::Zero();
            if(live){
                auto& p=data[s][i]; w=weights[s];
                if(s==0){
                    Vec v=velocity(p);
                    v.x()+=kick;
                    set_velocity(p,v);
                }
                delta=velocity(p)-state->species[s].mean;
                if(s==0 && sample){
                    int bin=int(sycl::fmin(100.,0.5*me/e*velocity(p).squaredNorm()));
                    sycl::atomic_ref<double,sycl::memory_order::relaxed,sycl::memory_scope::device,
                        sycl::access::address_space::global_space>(state->histogram[bin]).fetch_add(w);
                }
            }
            PartialMoments m;
            auto g=item.get_group();
            m.mass=sycl::reduce_over_group(g,w,sycl::plus<>());
            for(int d=0;d<3;++d){
                m.first[d]=sycl::reduce_over_group(g,w*delta[d],sycl::plus<>());
                m.second[d]=sycl::reduce_over_group(g,w*delta[d]*delta[d],sycl::plus<>());
            }
            if(lane==0)out[block]=m;
        });
        q.single_task(partials_ready,[=]{
            const double denominator=dt*state->species[0].total*state->species[1].total;
            state->rate=denominator>0 ? state->produced/denominator : 0.;
            state->produced=0;
            for(int s=0;s<3;++s){
                PartialMoments sum;
                for(size_t b=offset[s];b<offset[s+1];++b){
                    sum.mass+=out[b].mass;
                    sum.first+=out[b].first;sum.second+=out[b].second;
                }
                auto& m=state->species[s];
                if(sum.mass==0){ m=Moments{}; continue; }
                Vec drift=sum.first/sum.mass;
                m.mean+=drift;
                for(int d=0;d<3;++d)m.variance[d]=sycl::fmax(0.,sum.second[d]/sum.mass-drift[d]*drift[d]);
                m.total=sum.mass;
            }
            *density=state->species[1].total;
            state->temperature=state->species[0].variance.sum()*me/(3*kb*11605.);
            if(step>=start && step<steps-1)state->sum_rate+=state->rate;
            if(step>=start && step<steps)state->sum_temperature+=state->temperature;
            if(sample){
                ++state->snapshots;
                for(int i=0;i<=100;++i){
                    state->spectrum[i]+=state->histogram[i]/state->species[0].total;
                    state->histogram[i]=0;
                }
            }
        }).wait();
    }
};

// Electron cross-section fits from mpdt-magnetic-nozzle-3d/collision/custom_mccm.hpp.
// Input is the squared electron speed in the laboratory frame (m²/s²).
inline double horner2(double x, double a, double b, double c) {
    return (a * x + b) * x + c;
}

inline double horner3(double x, double a, double b, double c, double d) {
    return ((a * x + b) * x + c) * x + d;
}

struct ElecAtomElasCS {
    static double value(double v2) {
        double x = sycl::log(v2);
        double y;
        if (x < 2.399748711511477e+01) {
            y = horner2(x, -1.238105978742491e-01, 5.019206835293452e+00, -9.512538745780023e+01);
        } else if (x < 2.514676225569993e+01) {
            y = horner2(x, -2.031702982392320e-01, 7.744923283755391e+00, -1.148341171438619e+02);
        } else if (x < 2.722019223564859e+01) {
            y = horner2(x, -5.059331058266652e-01, 2.813503697918090e+01, -4.361244749816307e+02);
        } else if (x < 2.897038884349426e+01) {
            y = horner2(x, -2.509000052810705e-03, 1.288898113922888e+00, -7.837390000412563e+01);
        } else if (x < 2.983337392837280e+01) {
            y = horner2(x, -3.437779493911779e-01, 1.917245194244984e+01, -3.100460340545293e+02);
        } else if (x < 3.060117915769724e+01) {
            y = horner2(x, -5.831979267354939e-02, 2.379798554880749e+00, -6.313090659440627e+01);
        } else if (x < 3.180806720293279e+01) {
            y = horner2(x, 2.129912414315138e-01, -1.437769567173223e+01, 1.956037971845577e+02);
        } else if (x < 3.319972617626796e+01) {
            y = horner2(x, -3.180443119979717e-02, 1.183604033472321e+00, -5.169827935961904e+01);
        } else {
            y = -1.356819494909022e+00 * x - 2.412451828804758e+00;
        }
        return sycl::exp(y);
    }
};

struct ElecAtomIonizCS {
    static constexpr double threshold_eV = 15.76;
    static double value(double v2) {
        if (0.5 * me * v2 / e < threshold_eV) return 0.0;
        double x = sycl::log(v2);
        double y;
        if (x < 2.937368022790717e+01) {
            y = horner3(x, -4.911818920045614e+00, 2.913468802396953e+02, -4.314160906745761e+03, -2.186194023259742e+02);
        } else if (x < 2.942966445728310e+01) {
            y = horner3(x, -2.147088514165119e+00, 1.273429849614457e+02, -1.904925932938912e+03, 4.486747505109843e+02);
        } else if (x < 2.948828907216449e+01) {
            y = horner3(x, -2.984199045500145e-01, 1.840869481910122e+01, -2.962217991079538e+02, 3.325473367860277e+02);
        } else if (x < 2.957531976493135e+01) {
            y = horner3(x, 6.308111083873481e-02, -3.217248908083159e+00, 3.142568386350342e+01, 2.062804327334617e+02);
        } else if (x < 2.970554291078782e+01) {
            y = horner3(x, -8.370865764744019e-02, 5.300823261140293e+00, -8.967620535061469e+01, 1.345343708453751e+02);
        } else if (x < 2.986548611990852e+01) {
            y = horner3(x, -2.725180494701460e-02, 1.758497629929769e+00, -2.998894255307064e+01, 7.419242027472998e+00);
        } else if (x < 3.011370959767325e+01) {
            y = horner3(x, -1.935224610449061e-02, 1.245347611229328e+00, -2.093318549199284e+01, -1.576450381804871e+01);
        } else if (x < 3.208190765240784e+01) {
            y = horner3(x, -1.078523275638771e-02, 7.278542971280714e-01, -1.395564340938894e+01, 9.447579646790356e+00);
        } else if (x < 3.288842429046228e+01) {
            y = horner3(x, -2.743639422605870e-03, 2.062904964671185e-01, -5.307122649221014e+00, 3.269908353152004e+00);
        } else {
            y = horner2(x, 2.136075228024931e-02, -2.090080651006592e+00, -1.060042922924891e-01);
        }
        return sycl::exp(y);
    }
};


// The nozzle electron collision rate uses electron speed. DPMCC multiplies the
// supplied cross section by relative speed, so convert the rate kernel here.
template<class Channel,class Fit>
double electron_cross_section(const auto& pair,const auto& ctx) {
    const double v2=velocity(*pair.first).squaredNorm();
    const double g=(velocity(*pair.first)-velocity(*pair.second)).norm();
    const double sigma=(v2>0 && g>0) ? Fit::value(v2)*sycl::sqrt(v2)/g : 0.;
    return ElectronArgon::effective<Channel>(sigma,pair,ctx);
}

struct Elastic : ElectronArgon {
    static double pair_collision_cross_section(const auto& pair,const auto& ctx) {
        return electron_cross_section<Elastic,ElecAtomElasCS>(pair,ctx);
    }
    static void pair_collide(auto& pair,const auto& ctx) {
        auto& el=*pair.first;
        const auto& channel=ctx.template get<Elastic>();
        increment(channel.counters+accepted);
        auto random=psum::random::view_as_rander(get<property::random_seed>(el));
        auto& rng=random.get_rander();
        const double speed=velocity(el).norm();
        set_velocity(el,speed*psum::random::RandFunction3D::RandV_spherical(rng).transpose());
    }
};

struct Ionization : ElectronArgon, reaction_collision_tag {
    using consumed_species_tags=std::tuple<Atom>;
    using product_species_tags=std::tuple<Electron,Ion>;
    static double pair_collision_cross_section(const auto& pair,const auto& ctx) {
        return electron_cross_section<Ionization,ElecAtomIonizCS>(pair,ctx);
    }
    static void pair_collide(auto& pair,const auto& ctx) {
        auto& el=*pair.first;
        auto& atom=*pair.second;
        const double excess=0.5*me*velocity(el).squaredNorm()-e*ElecAtomIonizCS::threshold_eV;
        if(excess<=0) return;
        const auto& channel=ctx.template get<Ionization>();
        increment(channel.counters+accepted);
        auto random=psum::random::view_as_rander(get<property::random_seed>(el));
        auto& rng=random.get_rander();
        if(channel.neutral_update_probability==1. || rng()<channel.neutral_update_probability)
            ctx.template erase_particle<Atom>(&atom);
        auto child=el, ion=el;
        const double speed=sycl::sqrt(excess/me);
        set_velocity(el,speed*psum::random::RandFunction3D::RandV_spherical(rng).transpose());
        set_velocity(child,speed*psum::random::RandFunction3D::RandV_spherical(rng).transpose());
        set_velocity(ion,velocity(atom));
        get<property::random_seed>(child)=rng.gen_seed();
        get<property::random_seed>(ion)=rng.gen_seed();
        ctx.template insert_particle<Electron>(child);
        ctx.template insert_particle<Ion>(ion);
        sycl::atomic_ref<double,sycl::memory_order::relaxed,sycl::memory_scope::device,
            sycl::access::address_space::global_space>(*channel.produced).fetch_add(channel.plasma_weight);
    }
};

struct NeutralElastic : ArgonChannel {
    static double request_probability(const auto&, const auto&) { return 0.05; }
    inline static const std::string tag_name="argon-argon";
    using incident_species_tag=Atom;
    static double pair_collision_cross_section(const auto& pair,const auto& ctx) {
        const double g2=(velocity(*pair.first)-velocity(*pair.second)).squaredNorm();
        constexpr double diameter=4.17e-10, omega=.74;
        const double sigma=g2>0 ? std::acos(-1.)*diameter*diameter*
            sycl::pow(2*kb*273/(ma*g2),omega-.5)/sycl::tgamma(2.5-omega) : 0.;
        return effective<NeutralElastic>(sigma,pair,ctx);
    }
    static void pair_collide(auto& pair,const auto& ctx) {
        const auto& channel=ctx.template get<NeutralElastic>();
        increment(channel.counters+accepted);
        auto& first=*pair.first;
        auto& second=*pair.second;
        auto random=psum::random::view_as_rander(get<property::random_seed>(first));
        auto& rng=random.get_rander();
        Vec center=(velocity(first)+velocity(second))/2;
        Vec half=0.5*(velocity(first)-velocity(second)).norm()*
            psum::random::RandFunction3D::RandV_spherical(rng).transpose();
        set_velocity(first,center+half);
        set_velocity(second,center-half);
    }
};
} // namespace nozzle

int main(int argc,char** argv) try {
    using namespace nozzle;
    using Model=collision_model<Elastic,Ionization,NeutralElastic>;
    if(argc!=7) throw std::invalid_argument("usage: argon-simple steps start electron_target seed Td output.csv");
    int steps=std::stoi(argv[1]),start=std::stoi(argv[2]),target=std::stoi(argv[3]);
    auto seed=std::stoul(argv[4]);
    double td=std::stod(argv[5]);
    if(start<1 || steps<=start+1 || target<16 || target>10000000 || !(td>0 && std::isfinite(td)))
        throw std::invalid_argument("invalid run parameters");
    Generator rng(seed);
    sycl::queue q;
    const double plasma_weight=electron0/target, neutral_weight=neutral0/atom_target;
    std::array<double,3> weights{plasma_weight,neutral_weight,plasma_weight};
    if(!(plasma_weight>0 && plasma_weight<=neutral_weight))
        throw std::invalid_argument("expected 0 < plasma weight <= neutral weight");
    Group electrons(q,target),atoms(q,atom_target),ions(q,target);
    RunStats initial{};
    initial.species[1]=initialize_group(atoms,atom_target,std::sqrt(kb*300/ma),neutral_weight,rng);
    initial.species[2]=initialize_group(ions,target,std::sqrt(kb*300/ma),plasma_weight,rng);
    initial.species[0]=initialize_group(electrons,target,std::sqrt(2*e/me),plasma_weight,rng);
    DeviceArray<RunStats> stats(q,std::vector<RunStats>{initial});
    DeviceDiagnostics diagnostics(q);
    Field density(q,grid3D({0.,0.,0.},{1.,1.,1.},{1,1,1}));
    density.setConstant(initial.species[1].total);
    struct Binding { Group* group; Field* density; };
    species_contexts<Model,Group,Field> species{Binding{&electrons,nullptr},Binding{&atoms,&density},Binding{&ions,nullptr}};
    collision_channel_contexts<Model,Group,decltype(species)> channels(q,species);
    psum::field::device_array<uint64_t> counters(q,std::vector<uint64_t>(3*counter_count,0));
    auto elastic=observe(channels.get<Elastic>(),counters.data());
    auto ionization=observe(channels.get<Ionization>(),counters.data()+counter_count);
    auto neutral=observe(channels.get<NeutralElastic>(),counters.data()+2*counter_count);
    ionization.plasma_weight=plasma_weight;
    ionization.neutral_update_probability=plasma_weight/neutral_weight;
    ionization.produced=&stats.data()->produced;
    tagged_context<tag_bind<Elastic,decltype(elastic)>,tag_bind<Ionization,decltype(ionization)>,
                   tag_bind<NeutralElastic,decltype(neutral)>>
        observed{elastic,ionization,neutral};
    context_composition context{species,observed};

    std::ofstream out(argv[6]);
    out.exceptions(std::ios::badbit|std::ios::failbit);
    out<<"step,ne_m3,nAr_m3,nIon_m3,Te_eV,k_instant_m3_s,mean_k_m3_s,mean_Te_eV,"
          "electron_slots,atom_slots,ion_slots,ArAr_accepted,elastic_requests,elastic_matched,elastic_clipped,"
          "elastic_accepted,ionization_requests,ionization_matched,ionization_clipped,ionization_accepted,ArAr_requests,ArAr_matched,ArAr_clipped\n";
    out<<std::setprecision(15);
    std::cout<<"device="<<q.get_device().get_info<sycl::info::device::name>()<<" xs=mpdt-nozzle-fit"
             <<" steps="<<steps<<" start="<<start<<" sample_particles="<<target<<" seed="<<seed
             <<" E_N_Td="<<td<<" atom_target="<<atom_target<<" plasma_weight="<<plasma_weight<<" neutral_weight="<<neutral_weight
             <<" neutral_update_probability="<<ionization.neutral_update_probability<<" model=mpdt-simple"<<std::endl;
    const double kick=-e/me*td*neutral0*1e-21*dt;
    for(int step=1;step<=steps;++step) {
        execute_collision_model<Model>(context,dt);
        thin_charged_groups(electrons,ions,rng,weights,step);
        auto& ionization_channel=observed.get<Ionization>();
        ionization_channel.plasma_weight=weights[0];
        ionization_channel.neutral_update_probability=weights[0]/neutral_weight;
        diagnostics.advance(electrons,atoms,ions,stats.data(),density.data(),kick,weights,step,start,steps);
        if(step%1000==0 || step==steps){
            const auto state=stats.to_host().front();
            const auto counts=counters.to_host();
            out<<step<<','<<state.species[0].total<<','<<state.species[1].total<<','<<state.species[2].total
               <<','<<state.temperature<<','<<state.rate
               <<','<<state.sum_rate/std::max(1,std::min(step,steps-2)-start+1)
               <<','<<state.sum_temperature/std::max(1,std::min(step,steps-1)-start+1);
            for(int s=0;s<3;++s)out<<','<<std::llround(state.species[s].total/weights[s]);
            out<<','<<counts[2*counter_count+accepted];
            for(size_t i=0;i<counts.size()-1;++i)out<<','<<counts[i];
            out<<std::endl;
        }
    }
    const auto state=stats.to_host().front();
    std::ofstream distribution(std::string(argv[6])+".energy.csv");
    distribution.exceptions(std::ios::badbit|std::ios::failbit);
    distribution<<"energy_lower_eV,energy_upper_eV,probability\n"<<std::setprecision(15);
    for(size_t i=0;i<100;++i)distribution<<i<<','<<i+1<<','<<state.spectrum[i]/state.snapshots<<'\n';
    distribution<<"100,inf,"<<state.spectrum[100]/state.snapshots<<'\n';
    return 0;
} catch(const std::exception& error) {
    std::cerr<<error.what()<<'\n';
    return 1;
}
