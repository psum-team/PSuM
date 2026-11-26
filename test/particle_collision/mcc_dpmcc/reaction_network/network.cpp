#include <array>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <sstream>
#include <vector>
#include <Eigen/Dense>
#include <psum/field.hpp>
#include <psum/particle_container.hpp>
#include "../../../../src/particle_collision/collision_model.hpp"

using namespace psum;
using namespace psum::tag;
using namespace psum::particle_collision;

struct A : tag::foundation::abstract_tag { static constexpr auto tag_name="A"; };
struct B : tag::foundation::abstract_tag { static constexpr auto tag_name="B"; };
struct C : tag::foundation::abstract_tag { static constexpr auto tag_name="C"; };
struct X : tag::foundation::abstract_tag { static constexpr auto tag_name="X"; };
struct lineage : tag::foundation::abstract_tag { static constexpr auto tag_name="lineage"; };
struct events : tag::foundation::abstract_tag { static constexpr auto tag_name="events"; };
using Particle=tagged_struct<tag_bind<property::position,Eigen::Vector3d>,
    tag_bind<property::velocity,Eigen::Vector3d>,tag_bind<property::random_seed,uint32_t>,
    tag_bind<lineage,int>,tag_bind<events,std::array<int,4>>>;
using Group=particle_container::particle_group<Particle,particle_container::pos_x_nan_is_invalid>;
using Field=field::cell_field3D<double>;

// Counters live on the unchanged catalyst. Each X can be matched at most once per step.
template<class From,class To,int Slot,int RatePercent,int ProposalPercent>
struct Reaction : deferred_pairing_collision_tag,reaction_collision_tag {
    using incident_species_tag=From;
    using background_species_tag=X;
    using consumed_species_tags=std::tuple<From>;
    using product_species_tags=std::tuple<To>;
    static double request_probability(const auto&,const auto&) { return ProposalPercent/100.; }
    static double pair_collision_cross_section(const auto&,const auto&) { return RatePercent/100.; }
    static void pair_collide(auto& pair,const auto& ctx) {
        ++get<events>(*pair.second)[Slot];
        ctx.template insert_particle<To>(*pair.first);
        ctx.template erase_particle<From>(pair.first);
    }
};
using AB=Reaction<A,B,0,60,25>;
using AC=Reaction<A,C,1,40,40>;
using BC=Reaction<B,C,2,150,50>;
using AB1=Reaction<A,B,0,24,10>;
using AB2=Reaction<A,B,3,36,15>;

struct Snapshot {
    std::array<int,3> populations{};
    std::array<int,4> callbacks{};
};

Snapshot observe(Group& a,Group& b,Group& c,Group& x,int n,int step) {
    Snapshot s;
    std::vector<bool> seen(n);
    std::array<Group*,3> groups{&a,&b,&c};
    for(size_t k=0;k<groups.size();++k) {
        for(const auto& p:groups[k]->get_content().to_host()) {
            if(!Group::validator::is_valid(p)) continue;
            const int id=get<lineage>(p);
            if(id<0 || id>=n || seen[id]) throw std::runtime_error("duplicate or invalid lineage");
            seen[id]=true;
            ++s.populations[k];
        }
        if(size_t(s.populations[k])!=groups[k]->size()) throw std::runtime_error("incorrect live count");
    }
    if(x.size()!=size_t(n)) throw std::runtime_error("catalyst count changed");
    for(const auto& p:x.get_content().to_host()) {
        if(!Group::validator::is_valid(p)) throw std::runtime_error("invalid catalyst");
        for(size_t j=0;j<s.callbacks.size();++j) s.callbacks[j]+=get<events>(p)[j];
    }
    const auto [ab1,ac,bc,ab2]=s.callbacks;
    if(s.populations!=std::array{n-ab1-ab2-ac,ab1+ab2-bc,ac+bc})
        throw std::runtime_error("callback ledger differs from live particle census");
    if(step==1 && bc!=0) throw std::runtime_error("B reacted during its birth step");
    return s;
}

template<class Model>
Snapshot simulate(sycl::queue& q,std::ostream& out,const char* order,int n,int seed,int scale) {
    Group a(q,n),b(q,n),c(q,n),x(q,n);
    std::vector<Particle> initial(n);
    std::mt19937 rng(seed);
    for(int i=0;i<n;++i) {
        auto& p=initial[i];
        get<lineage>(p)=i;
        get<property::position>(p).setConstant(.5);
        get<property::velocity>(p)=Eigen::Vector3d(1,0,0);
        get<property::random_seed>(p)=rng();
        get<events>(p)={};
    }
    a.insert<false>(initial);
    for(auto& p:initial) get<property::velocity>(p).setZero();
    x.insert<false>(initial);
    Field density(q,field::grid3D({0.,0.,0.},{1.,1.,1.},{1,1,1}));
    density.setConstant(1);
    struct Binding { Group* group; Field* density; };
    static_assert(std::is_same_v<typename Model::species_tuple,std::tuple<A,B,X,C>>);
    species_contexts<Model,Group,Field> species{Binding{&a,nullptr},Binding{&b,nullptr},
        Binding{&x,&density},Binding{&c,nullptr}};
    collision_channel_contexts<Model,Group,decltype(species)> channels(q,species);
    context_composition ctx{species,channels};
    Snapshot s;
    for(int step=0;step<=10*scale;++step) {
        if(step) execute_collision_model<Model>(ctx,.2/scale);
        if(step>1 && step!=scale && step%(2*scale)!=0) continue;
        s=observe(a,b,c,x,n,step);
        out<<order<<','<<seed<<','<<n<<','<<scale<<','<<step;
        for(auto k:s.populations) out<<','<<k;
        for(auto k:s.callbacks) out<<','<<k;
        out<<'\n';
    }
    return s;
}

int main(int argc,char** argv) {
    if(argc==1) {
        sycl::queue q;
        std::ostringstream counts;
        auto s=simulate<collision_model<AB1,AB2,AC,BC>>(q,counts,"split",1024,701,1);
        for(int count:s.callbacks)
            if(count==0) throw std::runtime_error("reaction channel never fired");
        return 0;
    }
    const bool confirm=argc==3 && std::string(argv[2])=="--confirm";
    if(argc!=2 && !confirm) {std::cerr<<"usage: network NEW_COUNTS_CSV [--confirm]\n"; return 2;}
    if(std::ifstream(argv[1]).good()) throw std::runtime_error("output already exists");
    std::ofstream out(argv[1]);
    out.exceptions(std::ios::failbit|std::ios::badbit);
    out<<"order,seed,n,scale,step,A,B,C,AB1,AC,BC,AB2\n";
    sycl::queue q;
    std::cout<<q.get_device().get_info<sycl::info::device::name>()<<std::endl;
    for(int seed=confirm?1701:701;seed<=(confirm?1748:712);++seed) {
        for(int scale:{1,2,4}) {
            simulate<collision_model<AB,AC,BC>>(q,out,"base",32768,seed,scale);
            simulate<collision_model<AC,AB,BC>>(q,out,"swap",32768,seed,scale);
            simulate<collision_model<AB1,AB2,AC,BC>>(q,out,"split",32768,seed,scale);
        }
        out.flush();
        std::cout<<"completed seed "<<seed<<std::endl;
    }
}
