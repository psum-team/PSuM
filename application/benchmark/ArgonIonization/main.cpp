#include "population.hpp"
#include <memory>
#include <fstream>
#include <iomanip>
#include <iostream>

using namespace argon;
using Model=collision_model<Elastic,Ionization,ArgonElastic>;

int main(int argc,char** argv) try {
    if(argc!=7) throw std::invalid_argument("usage: argon steps start electron_target seed Td output.csv");
    int steps=std::stoi(argv[1]),start=std::stoi(argv[2]),target=std::stoi(argv[3]);
    auto seed=std::stoul(argv[4]);
    double td=std::stod(argv[5]);
    if(start<1 || steps<=start+1 || target<16 || target>10000000 || !(td>0 && std::isfinite(td)))
        throw std::invalid_argument("invalid run parameters");
    Generator rng(seed);
    sycl::queue q;
    // Bounds on live slots after both stages, before merging. Each cross-species
    // match claims one electron and one atom; an Ar-Ar split needs two atoms.
    const size_t max_el=2*size_t(target),max_ar=2*atom_target;
    const size_t cross=std::min(max_el,max_ar);
    Group electrons(q,max_el+2*cross),atoms(q,max_ar+max_ar/2+cross),ions(q,100+cross);
    RunStats initial{};
    initial.species[1]=initialize_group(atoms,20,std::sqrt(2*kb*300/ma),neutral0,atom_target,false,rng());
    initial.species[2]=initialize_group(ions,20,std::sqrt(2*kb*300/ma),electron0,0,true,rng());
    initial.species[0]=initialize_group(electrons,40,std::sqrt(2*kb*(11604.51812*2)/me),electron0,target,false,rng());
    DeviceArray<RunStats> stats(q,std::vector<RunStats>{initial});
    DeviceDiagnostics diagnostics(q);
    Field density(q,grid1D({0.},{1.},{1}));
    density.setConstant(initial.species[1].total);
    struct Binding { Group* group; Field* density; };
    species_contexts<Model,Group,Field> species{Binding{&electrons,nullptr},Binding{&atoms,&density},Binding{&ions,nullptr}};
    collision_channel_contexts<Model,Group,decltype(species)> channels(q,species);
    psum::field::device_array<uint64_t> counters(q,std::vector<uint64_t>(3*counter_count,0));
    auto elastic=observe(channels.get<Elastic>(),counters.data());
    auto ionization=observe(channels.get<Ionization>(),counters.data()+counter_count);
    auto neutral=observe(channels.get<ArgonElastic>(),counters.data()+2*counter_count);
    // Pinned host memory: callbacks update counts only on insert/erase. The host
    // reads them after the executor has completed, without querying device vectors.
    auto free_counts=[&](size_t* p){sycl::free(p,q);};
    std::unique_ptr<size_t[],decltype(free_counts)> live(sycl::malloc_host<size_t>(3,q),free_counts);
    if(!live)throw std::bad_alloc();
    live[0]=electrons.extent();live[1]=atoms.extent();live[2]=ions.extent();
    elastic.live=ionization.live=neutral.live=live.get();
    ionization.produced=&stats.data()->produced;
    tagged_context<tag_bind<Elastic,decltype(elastic)>,tag_bind<Ionization,decltype(ionization)>,
                   tag_bind<ArgonElastic,decltype(neutral)>>
        observed{elastic,ionization,neutral};
    context_composition context{species,observed};

    std::ofstream out(argv[6]);
    out.exceptions(std::ios::badbit|std::ios::failbit);
    out<<"step,ne_m3,nAr_m3,nIon_m3,Te_eV,k_instant_m3_s,mean_k_m3_s,mean_Te_eV,"
          "electron_slots,atom_slots,ion_slots,ArAr_accepted,elastic_requests,elastic_matched,elastic_clipped,"
          "elastic_accepted,ionization_requests,ionization_matched,ionization_clipped,ionization_accepted,ArAr_requests,ArAr_matched,ArAr_clipped\n";
    out<<std::setprecision(15);
    std::cout<<"device="<<q.get_device().get_info<sycl::info::device::name>()<<" xs_sha256="<<ar_data::sha256
             <<" steps="<<steps<<" start="<<start<<" sample_particles="<<target<<" seed="<<seed
             <<" E_N_Td="<<td<<" atom_target="<<atom_target<<" model=weighted-AR"<<std::endl;
    const double kick=-e/me*td*neutral0*1e-21*dt;
    for(int step=1;step<=steps;++step) {
        execute_collision_model<Model>(context,dt);
        if(live[1]>2*atom_target)merge_on_device(atoms,atom_target,nullptr,rng(),&live[1]);
        if(live[2]>100)merge_on_device(ions,0,&stats.data()->species[2],rng(),&live[2]);
        if(live[0]>2*size_t(target))merge_on_device(electrons,target,nullptr,rng(),&live[0]);
        diagnostics.advance(electrons,atoms,ions,stats.data(),density.data(),kick,step,start,steps);
        if(step%1000==0 || step==steps){
            const auto state=stats.to_host().front();
            const auto counts=counters.to_host();
            out<<step<<','<<state.species[0].total<<','<<state.species[1].total<<','<<state.species[2].total
               <<','<<state.temperature<<','<<state.rate
               <<','<<state.sum_rate/std::max(1,std::min(step,steps-2)-start+1)
               <<','<<state.sum_temperature/std::max(1,std::min(step,steps-1)-start+1);
            for(int s=0;s<3;++s)out<<','<<live[s];
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
} catch(const std::exception& error) {
    std::cerr<<error.what()<<'\n';
    return 1;
}
