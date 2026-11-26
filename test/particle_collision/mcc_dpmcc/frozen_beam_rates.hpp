// Frozen, unit-volume counter experiment derived from validation/case_checks/check_n2_collision_rate.cpp.
#pragma once
#include "../../../application/benchmark/DSMC/custom_dpmcc.hpp"
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace frozen_beams {
using namespace psum::prelude;
using namespace psum::particle_collision;
struct observation : tag::foundation::abstract_tag { static constexpr auto tag_name = "observation"; };
struct Record { int id, partner, channel; bool accepted; std::uintptr_t address{}; };
using Molecule = tagged_struct<tag_bind<property::position,Eigen::Vector3d>,
    tag_bind<property::velocity,Eigen::Vector3d>, tag_bind<property::random_seed,uint32_t>,
    tag_bind<observation,Record>>;

template<int C, int Channels>
struct Counter : N2_dsmc_model::col_N2_to_N2_vhs {
    static double request_probability(const auto&, const auto&) { return 0.1 / Channels; }
    static double pair_collision_cross_section(const auto& pair, const auto& ctx) {
        auto& r = get<observation>(*pair.first);
        r.partner = get<observation>(*pair.second).id;
        r.channel = C;
        return N2_dsmc_model::col_N2_to_N2_vhs::pair_collision_cross_section(pair,ctx) / Channels;
    }
    static void pair_collide(auto& pair, const auto&) { get<observation>(*pair.first).accepted = true; }
};

struct Count { uint32_t pairs=0, hits=0; };
}
#include "frozen_pair_order.hpp"
namespace frozen_beams {
template<int Channels>
void run(sycl::queue& q, int n, int trials, bool shuffled, const std::filesystem::path& directory,bool controls=false,
         std::optional<uint32_t> precision_seed=std::nullopt,int cells=1) {
    using Model = std::conditional_t<Channels==1, collision_model<Counter<0,1>>,
        collision_model<Counter<0,2>,Counter<1,2>>>;
    using Group = particle_group<Molecule,pos_x_nan_is_invalid>;
    using Field = cell_field3D<double>;
    Group group(q);
    Field density(q,grid3D({0.,0.,0.},{double(cells),1.,1.},{cells,1,1}));
    density.setConstant(1e21);
    std::vector<int> slots(n);
    std::iota(slots.begin(),slots.end(),0);
    if(shuffled) std::shuffle(slots.begin(),slots.end(),std::mt19937(7189));
    std::vector<Molecule> input(n);
    const int per_cell=n/cells;
    auto cell_of=[=](int id){ return (id%(n/2))/(per_cell/2); };
    for(int id=0;id<n;++id) {
        auto& p=input[slots[id]];
        get<property::position>(p).setConstant(.5);
        get<property::position>(p).x()+=cell_of(id);
        get<property::velocity>(p)=Eigen::Vector3d(id<n/2?500.:-500.,0.,0.);
        get<observation>(p)={id,-1,-1,false};
    }
    group.insert<false>(input);
    struct Binding { Group* group; Field* density; };
    species_contexts<Model,Group,Field> species{Binding{&group,&density}};
    collision_channel_contexts<Model,Group,decltype(species)> channels(q,species);
    context_composition context{species,channels};
    // Increase small-system acceptance to 1/2 for statistical resolution.
    const double dt=n==2 ? 0.2/(1e21*1000*n2::vhs_cross_section(1e6)) : 1e-8;
    auto prefix=directory/(std::to_string(per_cell)+(cells==1?"":"x"+std::to_string(cells))+"-"+std::to_string(Channels)+(shuffled?"-shuffled":"-ordered"));
    std::ofstream blocks, pairs;
    blocks.exceptions(std::ios::badbit|std::ios::failbit);
    pairs.exceptions(std::ios::badbit|std::ios::failbit);
    if(!directory.empty()) {
        blocks.open(prefix.string()+"-blocks.csv");
        blocks<<"block,trials,channel,requests,capacity_pairs,expected_cross,pairs,cross,hits,dt\n"<<std::setprecision(17);
    }
    std::vector<std::array<Count,Channels>> matrix;
    if(blocks.is_open() && !precision_seed) {
        matrix.resize(size_t(n)*(n-1)/2);
        pairs.open(prefix.string()+"-pairs.csv");
        pairs<<"i,j,channel,trials,pairs,hits\n";
    }
    std::optional<PairOrderStudy> study;
    if(controls)study.emplace(q);
    for(int block=0;block<20;++block) {
        std::mt19937 rng(precision_seed.value_or(19937)+block);
        std::array<long long,Channels> requests{},capacity{},matched{},cross{},hits{};
        std::array<double,Channels> expected_cross{};
        using Requests=std::array<std::array<int,2>,Channels>;
        std::vector<Requests> cell_requests(cells);
        for(int t=0;t<trials/20;++t) {
            std::fill(cell_requests.begin(),cell_requests.end(),Requests{});
            for(int id=0;id<n;++id) {
                auto seed=uint32_t(rng());
                get<property::random_seed>(input[slots[id]])=seed;
                psum::random::rander replay(seed);
                double draw=replay();
                if(draw<0.1) {
                    int c=(Channels==2 && draw>=0.05);
                    ++cell_requests[cell_of(id)][c][id>=n/2];
                }
            }
            q.memcpy(group.get_content().data(),input.data(),n*sizeof(Molecule)).wait();
            if constexpr(Channels==2) {
                if(study)capture_order<Model>(context,dt);
                else execute_collision_model<Model>(context,dt);
            } else execute_collision_model<Model>(context,dt);
            for(const auto& requested:cell_requests) {
                std::array<int,2> background{per_cell/2,per_cell/2};
                for(const auto& channel:requested)
                    for(int beam=0;beam<2;++beam)background[beam]-=channel[beam];
                const int bg=background[0]+background[1];
                int left=bg;
                for(int c=0;c<Channels;++c) {
                    int m=requested[c][0]+requested[c][1], k=std::min(left,m);
                    left-=k; requests[c]+=m; capacity[c]+=k;
                    if(k) expected_cross[c]+=double(k)*(requested[c][0]*background[1]+requested[c][1]*background[0])/(double(m)*bg);
                }
            }
            const auto output=group.get_content().to_host();
            if(study)study->trial(input,output);
            for(const auto& p:output) {
                const auto r=get<observation>(p);
                if(r.partner<0)continue;
                int i=std::min(r.id,r.partner), j=std::max(r.id,r.partner), c=r.channel;
                assert(i<j && c>=0 && c<Channels && cell_of(i)==cell_of(j));
                if(!matrix.empty()) {
                    auto& item=matrix[size_t(j)*(j-1)/2+i][c];
                    ++item.pairs; item.hits+=r.accepted;
                }
                ++matched[c]; cross[c]+=(i<n/2 && j>=n/2); hits[c]+=r.accepted;
                assert(!r.accepted || (i<n/2 && j>=n/2));
            }
        }
        for(int c=0;c<Channels;++c) {
            assert(matched[c]==capacity[c]);
            if(blocks.is_open())blocks<<block<<','<<trials/20<<','<<c<<','<<requests[c]<<','<<capacity[c]<<','<<expected_cross[c]<<','<<matched[c]<<','<<cross[c]<<','<<hits[c]<<','<<dt<<'\n';
        }
    }
    if(study)study->write(prefix,trials);
    if(matrix.empty())return;
    // Sparse upper triangle: omitted pair/channel combinations have zero counts.
    for(int j=1;j<n;++j)for(int i=0;i<j;++i)for(int c=0;c<Channels;++c) {
        auto item=matrix[size_t(j)*(j-1)/2+i][c];
        if(item.pairs)pairs<<i<<','<<j<<','<<c<<','<<trials<<','<<item.pairs<<','<<item.hits<<'\n';
    }
}
inline int rate_precision(const std::filesystem::path& directory,int cells=1) {
    if(!std::filesystem::create_directory(directory))throw std::runtime_error("output directory already exists");
    sycl::queue q;
    std::cout<<q.get_device().get_info<sycl::info::device::name>()<<std::endl;
    const uint32_t seed=(q.get_device().is_cpu()?310000:410000)+(cells>1?1000:0);
    for(bool shuffled:{false,true})
        run<2>(q,2048,20000,shuffled,directory,false,seed+(shuffled?100:0),cells);
    return 0;
}
enum class experiment_mode { full, pair_order, pair_rates_n16 };

inline int experiment(const std::filesystem::path& directory,experiment_mode mode) {
    if(!std::filesystem::create_directory(directory)) throw std::runtime_error("output directory already exists");
    sycl::queue q;
    std::cout<<q.get_device().get_info<sycl::info::device::name>()<<std::endl;
    if(mode==experiment_mode::full)
        run<1>(q,2,60000,false,directory);
    for(bool shuffled:{false,true}) {
        run<2>(q,16,20000,shuffled,directory,mode==experiment_mode::pair_order);
        if(mode==experiment_mode::full) {
            run<1>(q,2048,5000,shuffled,directory);
            run<2>(q,2048,5000,shuffled,directory);
        }
    }
    return 0;
}
}
