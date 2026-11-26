#pragma once
#include "argon.hpp"
#include <algorithm>
#include <array>
#include <random>
#include <vector>

namespace argon {
using Generator=std::mt19937_64;
constexpr double kb=1.380649e-23, dt=5e-14, neutral0=1e23, electron0=1e16;
constexpr size_t atom_target=4000;

struct Moments {
    double total=0;
    Vec mean=Vec::Zero(), variance=Vec::Zero();
};
inline int octant(const Vec& v,const Vec& mid) {
    return (v.x()>mid.x())+2*(v.y()>mid.y())+4*(v.z()>mid.z());
}
template<class T> using DeviceArray=psum::field::device_array<T>;

// A bin owns a linked list of slot numbers, never a copy of its particles.
struct VelocityBin {
    int head=-1, count=0, depth=0, order=0;
    double mass=0;
    Vec lo=Vec::Zero(),hi=Vec::Zero();
    void prepend(int i,int* next,double w) {
        next[i]=head;
        head=i;
        mass+=w;
        ++count;
    }
    bool operator<(const VelocityBin& b) const {
        return mass<b.mass || (mass==b.mass && order<b.order);
    }
};

inline void finish_leaf(const VelocityBin& bin,Particle* particles,const int* next,psum::random::rander& rng) {
    if(bin.count<=2)return;
    Vec mean=Vec::Zero(),variance=Vec::Zero();
    for(int i=bin.head;i>=0;i=next[i])mean+=weight(particles[i])*velocity(particles[i]);
    mean/=bin.mass;
    for(int i=bin.head;i>=0;i=next[i]){
        auto& p=particles[i];
        variance+=weight(p)*(velocity(p)-mean).array().square().matrix();
        weight(p)=0; // Mark losers after computing this leaf's moments.
    }
    Vec spread=(variance/bin.mass).cwiseSqrt();
    for(int d=0;d<3;++d)if(rng.gen_seed()&1)spread[d]=-spread[d];
    int i=bin.head;
    for(int sign=-1;sign<=1;sign+=2){
        auto& p=particles[i];
        weight(p)=bin.mass/2;
        velocity(p)=mean+sign*spread;
        get<property::random_seed>(p)=rng.gen_seed();
        i=next[i];
    }
}

inline size_t merge_bins(Particle* data,size_t extent,size_t target,const Moments* ion_reference,
                         uint32_t seed,int* next,VelocityBin* heap) {
    psum::random::rander rng(seed);
    if(ion_reference){
        VelocityBin bins[9]{};
        double limit=3.5*sycl::sqrt(2./3*ion_reference->variance.sum());
        for(int i=0;i<int(extent);++i)if(Group::validator::is_valid(data[i])){
            auto& v=velocity(data[i]);
            bool outside=false;
            for(int d=0;d<3;++d)outside|=sycl::fabs(v[d]-ion_reference->mean[d])>limit;
            bins[outside?1+octant(v,ion_reference->mean):0].prepend(i,next,weight(data[i]));
        }
        size_t survivors=0;
        for(auto& bin:bins){
            finish_leaf(bin,data,next,rng);
            survivors+=std::min(2,bin.count);
        }
        return survivors;
    }
    VelocityBin root{};
    for(int i=0;i<int(extent);++i)if(Group::validator::is_valid(data[i])){
        auto& v=velocity(data[i]);
        if(root.count==0){root.lo=v;root.hi=v;}
        else for(int d=0;d<3;++d){
            root.lo[d]=sycl::fmin(root.lo[d],v[d]);
            root.hi[d]=sycl::fmax(root.hi[d],v[d]);
        }
        root.prepend(i,next,weight(data[i]));
    }
    heap[0]=root;
    int pending=1,order=0;
    size_t survivors=2;
    while(pending && survivors+14<=target){
        std::pop_heap(heap,heap+pending);
        auto parent=heap[--pending];
        Vec mid=(parent.lo+parent.hi)/2;
        VelocityBin children[8]{};
        for(int i=parent.head;i>=0;){
            int following=next[i];
            children[octant(velocity(data[i]),mid)].prepend(i,next,weight(data[i]));
            i=following;
        }
        survivors-=2;
        for(int k=0;k<8;++k){
            auto& bin=children[k];
            if(!bin.count)continue;
            survivors+=std::min(2,bin.count);
            bin.depth=parent.depth+1;
            if(bin.count<=2 || bin.depth>=10){finish_leaf(bin,data,next,rng);continue;}
            bin.order=++order;
            for(int d=0;d<3;++d){
                bin.lo[d]=(k&(1<<d))?mid[d]:parent.lo[d];
                bin.hi[d]=(k&(1<<d))?parent.hi[d]:mid[d];
            }
            heap[pending++]=bin;
            std::push_heap(heap,heap+pending);
        }
    }
    for(int i=0;i<pending;++i)finish_leaf(heap[i],data,next,rng);
    return survivors;
}

inline void merge_on_device(Group& group,size_t target,const Moments* ion_reference,uint32_t seed,
                            size_t* live_count=nullptr) {
    auto extent=group.extent();
    auto q=group.get_content().get_queue();
    DeviceArray<int> links(q,extent);
    DeviceArray<VelocityBin> work(q,ion_reference?1:extent/3+1);
    auto* next=links.data();
    auto* heap=work.data();
    auto* data=group.get_content().data();
    q.single_task([=]{
        size_t n=merge_bins(data,extent,target,ion_reference,seed,next,heap);
        if(live_count)*live_count=n;
    }).wait();
    // for_each records newly invalidated slots in the group's existing free list.
    group.for_each([](sycl::handler&){return [](Particle& p){
        if(weight(p)==0)Group::validator::make_invalid(p);
    };});
}

// The Cartesian Maxwell grid separates into three identical one-dimensional sums.
inline Moments initialize_group(Group& group,int nodes,double speed,double total,size_t target,
                                bool ions,uint32_t seed) {
    double norm=0,second=0;
    for(int i=0;i<nodes;++i){
        double x=(2.*i/(nodes-1)-1.)*3.5;
        double w=std::exp(-x*x);
        norm+=w;second+=w*x*x;
    }
    Moments reference;
    reference.total=total;
    reference.variance.setConstant(second/norm*speed*speed);
    double scale=total/(norm*norm*norm);
    size_t count=size_t(nodes)*nodes*nodes;
    std::vector<Particle> particles(count);
    for(size_t i=0;i<count;++i){
        Vec x=(Vec(i/(nodes*nodes),(i/nodes)%nodes,i%nodes)*(2./(nodes-1))-Vec::Ones())*3.5;
        auto& p=particles[i];
        weight(p)=scale*std::exp(-x.squaredNorm());
        velocity(p)=x*speed;
        get<property::position>(p).x()=.5;
        psum::random::rander rng(seed^uint32_t(i));
        get<property::random_seed>(p)=rng.gen_seed();
    }
    if(ions || count>target){
        std::vector<int> links(count);
        std::vector<VelocityBin> work(ions?1:count/3+1);
        merge_bins(particles.data(),count,target,ions?&reference:nullptr,seed,links.data(),work.data());
        std::erase_if(particles,[](const Particle& p){return weight(p)==0;});
    }
    group.insert<false>(particles);
    return reference;
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
    void advance(Group& el,Group& ar,Group& ion,RunStats* state,double* density,double kick,
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
                auto& p=data[s][i]; w=weight(p);
                if(s==0)velocity(p).x()+=kick;
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
            state->rate=state->produced/(dt*state->species[0].total*state->species[1].total);
            state->produced=0;
            for(int s=0;s<3;++s){
                PartialMoments sum;
                for(size_t b=offset[s];b<offset[s+1];++b){
                    sum.mass+=out[b].mass;
                    sum.first+=out[b].first;sum.second+=out[b].second;
                }
                auto& m=state->species[s];
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
}
