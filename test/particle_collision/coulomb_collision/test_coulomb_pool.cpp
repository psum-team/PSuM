#include <psum/psum.hpp>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>

namespace frozen_pool {
using namespace psum::prelude;
using namespace psum::particle_collision;
struct Identity : psum::tag::foundation::abstract_tag { inline static const std::string tag_name="pool_id"; };
using Particle = tagged_struct<tag_bind<property::position,Eigen::Vector<double,1>>,
                               tag_bind<property::random_seed,uint32_t>,tag_bind<Identity,int>>;
using Group = particle_group<Particle,pos_x_nan_is_invalid>;
constexpr int cells=132, stride=40;
int occupancy(int c,int species,int minority) {
    if(c<128) return species==2 ? 24 : species==0 ? minority : 16-minority;
    constexpr int edge[4][3]={{1,2,3},{0,1,0},{0,0,5},{3,2,2}};
    return edge[c-128][species];
}
int species(int id,int minority) { return id%stride<(id/stride<128?minority:8) ? 0 : id%stride<16 ? 1 : 2; }
struct Record { int within=0, cross=0, partner=-1, within_partner=-1, errors=0; double time=0; };
struct Probe { int id; int minority; Record* records; };
template<class Random>
void scattering_NanBu(Probe& a,Probe& b,double ma,double mb,double qa,double qb,double n,
                      double limit,double,double dt,Random&) {
    const int c=a.id/stride, x=species(a.id,a.minority), y=species(b.id,a.minority);
    const int ni=occupancy(c,0,a.minority)+occupancy(c,1,a.minority), ne=occupancy(c,2,a.minority);
    const bool cross=(x==2)!=(y==2);
    const int count=cross ? (ni>ne?ni:ne) : ((x==2?ne:ni)-1);
    const double charge[]={1,2,-1};
    const auto cap=coulomb_cross_section_limit(std::max(occupancy(c,x,a.minority),occupancy(c,y,a.minority))*1e18);
    auto& p=a.records[a.id]; auto& q=a.records[b.id];
    if(c!=b.id/stride || a.id==b.id || ma!=1+x || mb!=1+y || qa!=charge[x] || qb!=charge[y] ||
       n!=count*1e18 || std::abs(limit-cap)>1e-12*cap) ++p.errors;
    if(cross) {
        if(x!=2 || (ne>1 && !p.within) || (ni>1 && !q.within) || dt!=1) ++p.errors;
        ++p.cross; ++q.cross; p.partner=b.id; q.partner=a.id;
    } else {
        if(p.cross || q.cross || (dt!=1 && dt!=0.5)) ++p.errors;
        ++p.within; ++q.within; p.time+=dt; q.time+=dt;
        p.within_partner=b.id; q.within_partner=a.id;
    }
}

int first_slot(int c,int s,int minority) { return s==1 && c<128 ? minority : 8*s; }

struct PairStatistics { double mixed_fraction, worst_partner_error; };

PairStatistics run(sycl::queue& q,unsigned seed,int minority,int samples) {
    grid1D grid({0.},{double(cells)},{cells});
    Group xe(q),xe2(q),electrons(q);
    Group* groups[]={&xe,&xe2,&electrons};
    for(int s=0;s<3;++s) {
        std::vector<Particle> input;
        for(int c=0;c<cells;++c) for(int j=0;j<occupancy(c,s,minority);++j) {
            Particle p;
            get<Identity>(p)=stride*c+first_slot(c,s,minority)+j;
            get<property::position>(p).x()=c+.5;
            get<property::random_seed>(p)=1+seed+get<Identity>(p);
            input.push_back(p);
        }
        groups[s]->insert<false>(input);
    }
    cell_field1D<double> temp(q,grid); temp.setConstant(1);
    using role=coulomb_role;
    coulomb_collision_solver<Particle,pos_x_nan_is_invalid,1> solver(q,grid,
        {{xe, role::ion, 1, 1, temp},
         {electrons, role::electron, 3, -1, temp},
         {xe2, role::ion, 2, 2, temp}},seed);
    psum::field::device_array<Record> log(q,cells*stride);
    unsigned mix=0, hist[16][16]={};
    for(int step=0;step<samples;++step) {
        log.fill(Record{});
        solver.solve(1.,[](size_t,size_t n){return n*1e18;},
            [ptr=log.data(),minority](const Particle& p){return Probe{get<Identity>(p),minority,ptr};},
            [](Particle&,const Probe&){});
        const auto records=log.to_host();
        for(int c=0;c<cells;++c) {
            int cross=0, half[2]={};
            const int ni=occupancy(c,0,minority)+occupancy(c,1,minority),ne=occupancy(c,2,minority);
            for(int s=0;s<3;++s) for(int j=0;j<occupancy(c,s,minority);++j) {
                const int id=c*stride+first_slot(c,s,minority)+j, total=s==2?ne:ni;
                const auto& r=records[id];
                if(r.errors || r.time!=(total>1?1:0) || r.within>2 || r.cross>1 ||
                   (r.cross && (r.partner/stride!=c || records.at(r.partner).partner!=id)))
                    throw std::runtime_error("invalid endpoint or scattering parameters");
                half[s==2]+=r.within==2;
                cross+=r.cross;
                if(c<128 && s!=2) {
                    if(r.within!=1 || r.within_partner/stride!=c || r.within_partner%stride>=16 || r.within_partner<0 ||
                       records.at(r.within_partner).within_partner!=id)
                        throw std::runtime_error("invalid ion partner");
                    ++hist[id%stride][r.within_partner%stride];
                    mix+=s!=species(r.within_partner,minority);
                }
            }
            if(cross!=2*std::min(ni,ne) || half[0]!=(ni>1 && ni%2?3:0) || half[1]!=(ne>1 && ne%2?3:0))
                throw std::runtime_error("incorrect pair count or odd half-step");
        }
    }
    double worst=0;
    for(int i=0;i<16;++i) for(int j=0;j<16;++j) if(i!=j)
        worst=std::max(worst,std::abs(hist[i][j]/double(128*samples)-1./15));
    const double fraction=mix/double(128*16*samples);
    return {fraction,worst};
}
}

int main(int argc,char** argv) {
    sycl::queue q;
    if(argc==2 && std::string(argv[1])=="--study") {
        std::cout<<"seed,mixed_fraction,max_partner_frequency_error\n";
        for(unsigned seed=101;seed<=108;++seed) {
            const auto r=frozen_pool::run(q,seed,8,128);
            std::cout<<seed<<','<<r.mixed_fraction<<','<<r.worst_partner_error<<'\n';
        }
        return 0;
    }
    if(argc!=1) return 2;
    constexpr int samples=512;
    constexpr double trials=128*samples;
    for(int minority : {4,8}) {
        // Under a uniform perfect matching K counts cross-label pairs in a cell.
        // P(two specified minority particles both match the majority) = p2.
        const int majority=16-minority;
        const double p=majority/15.0, p2=p*(majority-1)/13.0;
        const double variance=minority*p*(1-p)+minority*(minority-1)*(p2-p*p);
        const double expected=minority*p/8;
        const double tolerance=0.01*expected+4*std::sqrt(variance/trials)/8;
        const double partner_tolerance=0.01/15+5*std::sqrt(14./225/trials);
        for(unsigned seed : {101u,102u}) {
            const auto r=frozen_pool::run(q,seed,minority,samples);
            std::cout<<"minority="<<minority<<" seed="<<seed<<" mixed="<<r.mixed_fraction
                     <<" expected="<<expected<<" max partner error="<<r.worst_partner_error<<std::endl;
            if(std::abs(r.mixed_fraction-expected)>tolerance || r.worst_partner_error>partner_tolerance)
                throw std::runtime_error("biased ion pooling or partner sampling");
        }
    }
}
