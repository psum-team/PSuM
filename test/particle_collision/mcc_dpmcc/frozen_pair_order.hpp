// Included by the N=16 frozen-beam diagnostic, after the counter model definition.
namespace frozen_beams {
using Order = std::array<int,16>;
using PairCounts = std::array<std::array<Count,120>,2>;
constexpr std::array names{"native","feistel-rank","feistel-ticket","shifted-rank",
                           "shifted-ticket","uniform-rank","uniform-ticket"};

// Record the real per-channel payload order before the production collision body copies it.
template<class Model,class Context>
void capture_order(Context& ctx,double dt) {
    namespace pc = psum::particle_collision;
    using Channels = typename Model::collision_types_tuple;
    using Tag = N2_dsmc_model::N2_tag;
    auto& species = ctx.template get<Tag>();
    pc::foundation::collision_step_context<Context> step{ctx,dt};
    auto bind = [&]<class Channel>() {
        auto& channel = ctx.template get<Channel>();
        if(channel.buffer.capacity()<16) channel.buffer.reserve(16);
        return pc::bind_pairing_channel<true>(channel.engine,channel.buffer,species,[&](sycl::handler& h) {
            auto collide = pc::foundation::pair_collision_body<Channel,true>(h,step);
            return [=](auto& payload) {
                get<observation>(*payload.first).address = reinterpret_cast<std::uintptr_t>(&payload);
                collide(payload);
            };
        });
    };
    auto channels = std::tuple{bind.template operator()<Counter<0,2>>(),bind.template operator()<Counter<1,2>>()};
    pc::execute_incident_collisions<Channels>(step,dt);
    pc::deal(species,channels);
}

struct PairOrderStudy {
    sycl::queue q;
    std::array<PairCounts,7> counts{};
    std::array<std::array<unsigned,2>,3> changed{}, assigned{};
    std::array<unsigned,16> first_slot{};
    std::mt19937_64 permutations{902129};
    int* state;
    explicit PairOrderStudy(sycl::queue queue):q(queue),state(sycl::malloc_device<int>(18,q)) {
        if(!state) throw std::bad_alloc();
    }
    PairOrderStudy(const PairOrderStudy&) = delete;
    ~PairOrderStudy() { sycl::free(state,q); }
    void trial(const std::vector<Molecule>& input,const std::vector<Molecule>& output) {
        Order roles{}, ids{}, accept{}, requests{};
        std::array<std::vector<std::pair<std::uintptr_t,int>>,2> by_channel;
        const double probability=.5*(1.-1./16)*1e21*1000*n2::vhs_cross_section(1e6)*1e-8/.1;
        for(int slot=0;slot<16;++slot) {
            const auto& r=get<observation>(output[slot]);
            ids[slot]=r.id;
            psum::random::rander rng(get<property::random_seed>(input[slot]));
            const double draw=rng();
            roles[slot]=draw<.05 ? 0 : draw<.1 ? 1 : -1;
            accept[r.id]=rng()<probability;
            // For this fixed ensemble every request fits. Verify every ID and channel.
            if(roles[slot]!=r.channel) throw std::logic_error("request/channel mismatch");
            if(r.channel<0)continue;
            by_channel[r.channel].emplace_back(r.address,r.id);
            if(r.accepted!=bool(accept[r.id] && ((r.id<8)!=(r.partner<8))))
                throw std::logic_error("acceptance replay mismatch");
            add(0,r.channel,r.id,r.partner,r.accepted);
        }
        std::array<int,2> sizes{};
        int offset=0;
        for(int c=0;c<2;++c) {
            auto& list=by_channel[c];
            std::sort(list.begin(),list.end());
            sizes[c]=int(list.size());
            for(auto [address,id]:list) requests[offset++]=id;
        }
        auto seed=permutations();
        int shift=std::uniform_int_distribution<int>(0,15)(permutations);
        Order uniform{};
        std::iota(uniform.begin(),uniform.end(),0);
        std::shuffle(uniform.begin(),uniform.end(),permutations);
        psum::particle_container::detail::slot_permutation feistel(16,seed);
        ++first_slot[feistel(0)];
        if(!offset)return;
        for(int kind=0;kind<3;++kind) {
            auto slot_of=[=](int lane) {
                if(kind==2)return uniform[lane];
                return (int(feistel(lane))+(kind==1?shift:0))%16;
            };
            // Logical-lane rank skips incident slots; no competing ticket operations.
            Order ranked{};
            int rank=0;
            for(int lane=0;lane<16 && rank<offset;++lane) {
                int slot=slot_of(lane);
                if(roles[slot]>=0)continue;
                int i=requests[rank],j=ids[slot],c=rank<sizes[0]?0:1;
                ranked[rank]=j;
                add(1+2*kind,c,i,j,accept[i] && ((i<8)!=(j<8)));
                ++rank;
            }
            if(rank!=offset)throw std::logic_error("insufficient background slots");
            // Same launch width, channel priority and load/fetch_add as selection_body.
            auto* tickets=state;
            q.fill(tickets,0,18).wait();
            q.parallel_for(sycl::range<1>(16),[=](sycl::id<1> lane) {
                int slot=slot_of(lane[0]);
                if(roles[slot]>=0)return;
                for(int c=0;c<2;++c) {
                    sycl::atomic_ref<int,sycl::memory_order::relaxed,sycl::memory_scope::device,
                        sycl::access::address_space::global_space> ticket(tickets[c]);
                    if(!sizes[c] || ticket.load()>=sizes[c])continue;
                    int k=ticket.fetch_add(1);
                    if(k>=sizes[c])continue;
                    tickets[2+(c?sizes[0]:0)+k]=1+ids[slot];
                    return;
                }
            }).wait_and_throw();
            Order partners{};
            q.memcpy(partners.data(),tickets+2,offset*sizeof(int)).wait();
            for(int k=0;k<offset;++k) {
                int i=requests[k],j=partners[k]-1;
                if(j<0 || j==i)throw std::logic_error("invalid ticket mapping");
                int c=k<sizes[0]?0:1;
                ++assigned[kind][c];
                changed[kind][c]+=(j!=ranked[k]);
                add(2+2*kind,c,i,j,accept[i] && ((i<8)!=(j<8)));
            }
        }
    }
    void add(int mode,int c,int i,int j,bool accepted) {
        if(i>j)std::swap(i,j);
        auto& item=counts[mode][c][j*(j-1)/2+i];
        ++item.pairs;
        item.hits+=accepted;
    }
    void write(const std::filesystem::path& prefix,int trials) const {
        std::ofstream differences(prefix.string()+"-ticket-differences.csv");
        differences.exceptions(std::ios::failbit|std::ios::badbit);
        differences<<"permutation,channel,assigned,different_partner\n";
        for(int kind=0;kind<3;++kind)for(int c=0;c<2;++c)
            differences<<names[1+2*kind]<<','<<c<<','<<assigned[kind][c]<<','<<changed[kind][c]<<'\n';
        std::ofstream lane(prefix.string()+"-first-slot.csv");
        lane.exceptions(std::ios::failbit|std::ios::badbit);
        lane<<"slot,first_lane_trials\n";
        for(int slot=0;slot<16;++slot)lane<<slot<<','<<first_slot[slot]<<'\n';
        std::ofstream f(prefix.string()+"-controls.csv");
        f.exceptions(std::ios::failbit|std::ios::badbit);
        f<<"mode,i,j,channel,trials,pairs,hits\n";
        for(size_t m=0;m<names.size();++m)for(int c=0;c<2;++c)
            for(int j=1;j<16;++j)for(int i=0;i<j;++i) {
                auto item=counts[m][c][j*(j-1)/2+i];
                f<<names[m]<<','<<i<<','<<j<<','<<c<<','<<trials<<','<<item.pairs<<','<<item.hits<<'\n';
            }
    }
};
}
