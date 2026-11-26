#include "dpmcc_test_models.hpp"

struct manual_electron_channel : deferred_pairing_collision_tag {
    using incident_species_tag=species_tags::electron;
    using background_species_tag=species_tags::Xenon;
    static double request_probability(const auto&,const auto&) { return 0; }
    static double pair_collision_cross_section(const auto& pair,const auto&) { return unit_rate_cross_section(pair); }
    static void pair_collide(auto& pair,const auto&) { get<velocity>(*pair.first).z()=1; }
};

struct record_grid_pair : manual_electron_channel {
    static double request_probability(const auto&,const auto&) { return 1; }
    static void pair_collide(auto& pair,const auto&) {
        get<velocity>(*pair.first).y()=get<position>(*pair.second).x();
        ++get<velocity>(*pair.first).z();
        ++get<velocity>(*pair.second).z();
    }
};

void test_request_cell_snapshot() {
    sycl::queue q;
    ParticleGroup electrons(q),atoms(q);
    particle p{};
    get<position>(p)=Eigen::Vector3d(0.5,0.5,0);
    get<velocity>(p)=Eigen::Vector3d(1,0,0);
    get<random_seed>(p)=12345;
    particle outside=p;
    get<position>(outside).x()=2.5;
    electrons.insert<false>(std::vector<particle>{p,outside});
    get<velocity>(p).setZero();
    particle next_cell=p;
    get<position>(next_cell).x()=1.5;
    atoms.insert<false>(std::vector<particle>{p,next_cell});
    field::grid2D grid({0.,0.},{2.,1.},{2,1});
    Field density(q,grid);
    density.setConstant(1);
    using Model=collision_model<record_grid_pair>;
    struct Binding { ParticleGroup* group; Field* density; };
    species_contexts<Model,ParticleGroup,Field> species{
        Binding{&electrons,nullptr},Binding{&atoms,&density}};
    collision_channel_contexts<Model,ParticleGroup,decltype(species)> channels(q,species);
    context_composition ctx{species,channels};
    psum::particle_collision::foundation::collision_step_context<decltype(ctx)> step{ctx,1};
    execute_incident_collisions<typename Model::collision_types_tuple>(step,1);
    const auto requests=channels.get<record_grid_pair>().buffer.to_host();
    static_assert(std::is_same_v<typename decltype(requests)::value_type::first_type,uint32_t>);
    assert(requests.size()==1 && requests[0].first==0);
    electrons.for_each([](sycl::handler&) {
        return [](particle& p) { get<position>(p).x()=get<position>(p).x()<1?1.5:0.5; };
    });
    deal(species.get<species_tags::Xenon>(),
        psum::particle_collision::foundation::bind_pairing_channels<true>(
            step,static_cast<typename Model::collision_types_tuple*>(nullptr)));
    const auto result=electrons.get_content().to_host();
    assert(get<velocity>(result[0]).z()==1 && get<velocity>(result[0]).y()==0.5);
    assert(get<velocity>(result[1]).z()==0); // No request when it was outside the grid.
}

void test_request_reservation() {
    sycl::queue q;
    ParticleGroup electrons(q),atoms(q);
    particle p{};
    get<position>(p)=Eigen::Vector3d(0.5,0.5,0);
    get<velocity>(p)=Eigen::Vector3d(1,0,0);
    get<random_seed>(p)=12345;
    electrons.insert<false>(std::vector<particle>(100,p));
    particle stationary=p;
    get<velocity>(stationary).setZero();
    atoms.insert<false>(std::vector<particle>(1024,stationary));
    field::grid2D grid({0.,0.},{1.,1.},{1,1});
    Field density(q,grid);
    density.setConstant(1);
    using Model=collision_model<record_grid_pair>;
    struct Binding { ParticleGroup* group; Field* density; };
    species_contexts<Model,ParticleGroup,Field> species{
        Binding{&electrons,nullptr},Binding{&atoms,&density}};
    collision_channel_contexts<Model,ParticleGroup,decltype(species)> channels(
        q,species,[]<class>{return size_t{128};});
    context_composition ctx{species,channels};
    auto& buffer=channels.get<record_grid_pair>().buffer;
    auto* original=buffer.data();
    size_t populated=100;
    for(size_t count:{100u,128u,129u,130u}) {
        electrons.insert<false>(std::vector<particle>(count-populated,p));
        populated=count;
        // Reset markers so every live incident must execute exactly once this step.
        electrons.for_each([](sycl::handler&) {
            return [](particle& value){ get<velocity>(value).z()=0; };
        });
        execute_collision_model<Model>(ctx,1);
        assert(buffer.size()==0 && !buffer.overflowed());
        for(const auto& value:electrons.get_content().to_host())assert(get<velocity>(value).z()==1);
        if(count<=128)assert(buffer.data()==original && buffer.capacity()==128);
        else assert(buffer.capacity()>count); // Leave headroom after crossing the old limit.
    }
    auto* grown=buffer.data();
    const auto capacity=buffer.capacity();
    // Many holes must not force request storage to grow to the slot extent.
    electrons.insert<false>(std::vector<particle>(1024-populated,p));
    auto* data=electrons.get_content().data();
    electrons.for_each([&](sycl::handler&) {
        return [=](particle& value){
            if(&value-data>=10)ParticleGroup::validator::make_invalid(value);
        };
    });
    assert(electrons.extent()>capacity && electrons.size()==10);
    execute_collision_model<Model>(ctx,1);
    assert(buffer.data()==grown && buffer.capacity()==capacity && buffer.size()==0);
    // Reusing invalid slots changes the live count without changing extent.
    electrons.insert<false>(std::vector<particle>(100,p));
    execute_collision_model<Model>(ctx,1);
    assert(electrons.size()==110 && buffer.data()==grown && buffer.capacity()==capacity);
    electrons.clear();
    execute_collision_model<Model>(ctx,1);
    assert(buffer.size()==0 && buffer.capacity()==capacity);
}

void test_pairing_storage_growth() {
    sycl::queue q;
    ParticleGroup electrons(q),atoms(q);
    field::grid2D grid({0.,0.},{3.,1.},{3,1});
    Field density(q,grid);
    density.setConstant(1);
    using Model=collision_model<record_grid_pair>;
    struct Binding { ParticleGroup* group; Field* density; };
    species_contexts<Model,ParticleGroup,Field> species{
        Binding{&electrons,nullptr},Binding{&atoms,&density}};
    collision_channel_contexts<Model,ParticleGroup,decltype(species)> channels(q,species);
    context_composition ctx{species,channels};
    // Cross the initial payload capacity, then reuse it for smaller and empty batches.
    for(size_t count:{size_t{1},size_t{9001},size_t{7},size_t{0}}) {
        electrons.clear();
        atoms.clear();
        std::vector<particle> incident(count),background(std::max(count,size_t{1}));
        for(size_t i=0;i<background.size();++i) {
            particle p{};
            get<position>(p)=Eigen::Vector3d(i%2?0.5:2.5,0.5,0); // Empty middle cell.
            get<velocity>(p)=Eigen::Vector3d(1,0,0);
            get<random_seed>(p)=12345;
            if(i<count)incident[i]=p;
            get<velocity>(p).setZero();
            background[i]=p;
        }
        electrons.insert<false>(incident);
        atoms.insert<false>(background);
        execute_collision_model<Model>(ctx,1);
        for(const auto& p:electrons.get_content().to_host()) {
            assert(get<velocity>(p).z()==1);
            assert(get<velocity>(p).y()==get<position>(p).x());
        }
        for(const auto& p:atoms.get_content().to_host())
            assert(get<velocity>(p).z()==double(count>0));
        assert(channels.get<record_grid_pair>().buffer.size()==0);
    }
}

struct uncorrected_pair_cross_section : manual_electron_channel {
    static double pair_collision_cross_section(const auto&,const auto&) { return 1e-100; }
};

void test_pairing_without_random_gate_or_compensation() {
    sycl::queue q;
    constexpr size_t request_count=8;
    for(bool accept: {false,true}) {
        for(size_t background_count: {size_t{0},size_t{3},size_t{16}}) {
            ParticleGroup electrons(q),atoms(q);
            particle p{};
            get<position>(p)=Eigen::Vector3d(0.5,0.5,0);
            get<velocity>(p)=Eigen::Vector3d(1,0,0);
            get<random_seed>(p)=12345;
            electrons.insert<false>(std::vector<particle>(request_count,p));
            particle stationary=p;
            get<velocity>(stationary).setZero();
            atoms.insert<false>(std::vector<particle>(background_count,stationary));
            field::grid2D grid({0.,0.},{1.,1.},{1,1});
            Field density(q,grid);
            density.setConstant(1e100); // The old density-based gate would skip almost every trial.
            using Model=collision_model<uncorrected_pair_cross_section>;
            struct Binding { ParticleGroup* group; Field* density; };
            species_contexts<Model,ParticleGroup,Field> species{
                Binding{&electrons,nullptr},Binding{&atoms,&density}};
            collision_channel_contexts<Model,ParticleGroup,decltype(species)> channels(q,species);
            auto& channel=channels.get<uncorrected_pair_cross_section>();
            using Request=decltype(channel.buffer)::value_type;
            std::vector<Request> requests;
            for(size_t i=0;i<request_count;++i)
                requests.push_back({0,{electrons.get_content().data()+i,nullptr,1}});
            channel.buffer.copy(requests);
            context_composition ctx{species,channels};
            psum::random::rander random(12345);
            // Every selected incident has the same seed: this probability must reject.
            // The former M/K scaling would accept when 8 requests share 3 backgrounds.
            const double probability=accept?1.0:double(random())*0.5;
            execute_collision_model<Model>(ctx,probability);
            size_t collisions=0;
            for(const auto& result:electrons.get_content().to_host())
                collisions+=get<velocity>(result).z()==1;
            assert(collisions==(accept?std::min(request_count,background_count):0));
            for(const auto& background:atoms.get_content().to_host())
                assert(get<random_seed>(background)==12345);
            assert(channel.buffer.size()==0);
        }
    }
}

struct manual_ion_channel : manual_electron_channel {
    using incident_species_tag=species_tags::Xenon_pos_1;
    static void pair_collide(auto& pair,const auto&) {
        get<velocity>(*pair.first).z()=get<velocity>(*pair.first).z()*10+2;
    }
};

void test_duplicate_species_bindings_rejected() {
    sycl::queue q;
    ParticleGroup electrons(q),ions(q),atoms(q);
    field::grid2D grid({0.,0.},{1.,1.},{1,1});
    Field density(q,grid);
    struct Binding { ParticleGroup* group; Field* density; };
    auto reject=[&]<class Model>(const auto&... bindings) {
        bool rejected=false;
        try {
            species_contexts<Model,ParticleGroup,Field> species{bindings...};
        } catch(const std::invalid_argument& error) {
            rejected=std::string_view(error.what())=="different species tags must bind different particle groups";
        }
        assert(rejected);
    };
    using Model=collision_model<manual_electron_channel,manual_ion_channel>;
    reject.template operator()<Model>(
        Binding{&electrons,nullptr},Binding{&electrons,nullptr},Binding{&atoms,&density});
    reject.template operator()<Model>(
        Binding{&electrons,nullptr},Binding{&ions,nullptr},Binding{&electrons,&density});
    reject.template operator()<Model>(
        Binding{&electrons,nullptr},Binding{&ions,nullptr},Binding{&ions,&density});
    // The same restriction applies to pure MCC, products, and self-collision tags.
    reject.template operator()<collision_model<observe_mcc>>(
        Binding{&electrons,nullptr},Binding{&electrons,&density});
    reject.template operator()<collision_model<consume_pair>>(
        Binding{&electrons,nullptr},Binding{&atoms,&density},Binding{&electrons,nullptr});
    reject.template operator()<collision_model<same_species_pair<species_tags::electron>,
                                               same_species_pair<species_tags::Xenon>>>(
        Binding{&atoms,&density},Binding{&atoms,&density});
    species_contexts<Model,ParticleGroup,Field> species{
        Binding{&electrons,nullptr},Binding{&ions,nullptr},Binding{&atoms,&density}};
}

struct sparse_reader_a : manual_electron_channel {
    static void pair_collide(auto& pair,const auto&) {
        auto& v=get<velocity>(*pair.second);
        v.y()=v.y()*10+1;
    }
};

struct sparse_reader_b : sparse_reader_a {
    static void pair_collide(auto& pair,const auto&) {
        auto& v=get<velocity>(*pair.second);
        v.y()=v.y()*10+2;
    }
};

struct sparse_consumer : manual_ion_channel,reaction_collision_tag {
    using consumed_species_tags=std::tuple<background_species_tag>;
    using product_species_tags=std::tuple<>;
    static void pair_collide(auto& pair,const auto& ctx) {
        get<velocity>(*pair.first).z()=get<velocity>(*pair.second).y();
        ctx.template erase_particle<background_species_tag>(pair.second);
    }
};

void test_sparse_exclusive_requests() {
    sycl::queue q;
    ParticleGroup electrons(q),ions(q),atoms(q);
    constexpr size_t count=64,extent=8192;
    particle p{};
    get<position>(p)=Eigen::Vector3d(-1,0.5,0);
    get<velocity>(p).setZero();
    get<random_seed>(p)=12345;
    std::vector<particle> incident(count,p);
    for(size_t i=0;i<count;++i)get<position>(incident[i]).x()=i+0.5;
    ions.insert<false>(incident);
    const auto second_incidents=incident;
    incident.insert(incident.end(),second_incidents.begin(),second_incidents.end());
    electrons.insert<false>(incident);
    std::vector<particle> background(extent,p);
    for(size_t i=0;i<count;++i) {
        get<position>(background[127*i+3])=Eigen::Vector3d(i+0.5,0.5,0);
        get<velocity>(background[127*i+3]).x()=1;
    }
    atoms.insert<false>(background);
    field::grid2D grid({0.,0.},{double(count),1.},{count,1});
    Field density(q,grid);
    density.setConstant(1);
    // Each cell selects only the first channel that has a request.
    using Model=collision_model<sparse_reader_a,sparse_reader_b,sparse_consumer>;
    struct Binding { ParticleGroup* group; Field* density; };
    species_contexts<Model,ParticleGroup,Field> species{
        Binding{&electrons,nullptr},Binding{&ions,nullptr},Binding{&atoms,&density}};
    collision_channel_contexts<Model,ParticleGroup,decltype(species)> channels(q,species);
    auto& a=channels.get<sparse_reader_a>();
    auto& b=channels.get<sparse_reader_b>();
    auto& c=channels.get<sparse_consumer>();
    using Request=typename std::remove_reference_t<decltype(a)>::request_buffer_type::value_type;
    std::vector<Request> ra,rb,rc;
    auto* e=electrons.get_content().data();
    auto* ion=ions.get_content().data();
    for(size_t i=0;i<count;++i) {
        if(i%3==0)ra.push_back({uint32_t(i),{e+i,nullptr,1}});
        if(i%3!=2)rb.push_back({uint32_t(i),{e+count+i,nullptr,1}});
        rc.push_back({uint32_t(i),{ion+i,nullptr,1}});
    }
    a.buffer.copy(ra);b.buffer.copy(rb);c.buffer.copy(rc);
    context_composition ctx{species,channels};
    execute_collision_model<Model>(ctx,1);
    const auto background_result=atoms.get_content().to_host();
    for(size_t i=0;i<count;++i) {
        const auto& result=background_result[127*i+3];
        assert(get<velocity>(result).y()==(i%3==2?0:double(i%3+1)));
        assert(ParticleGroup::validator::is_valid(result)==(i%3!=2));
    }
    for(const auto& result:ions.get_content().to_host()) assert(get<velocity>(result).z()==0);
    assert(atoms.size()==extent-count/3 && atoms.get_invalid_indexes().size()==count/3);
    assert(a.buffer.size()==0 && b.buffer.size()==0 && c.buffer.size()==0);
    execute_collision_model<Model>(ctx,1);
    assert(atoms.size()==extent-count/3 && atoms.get_invalid_indexes().size()==count/3);

    // b left an unmatched request in cell 0. Now empty, it must not steal a partner.
    c.buffer.copy({{0,{ion,nullptr,1}}});
    execute_collision_model<Model>(ctx,1);
    assert(atoms.size()==extent-count/3-1);
    assert(get<velocity>(ions.get_content().to_host()[0]).z()==1);

    // A nonempty channel can also become empty after invalid-request filtering.
    b.buffer.copy({{1,{nullptr,nullptr,1}}});
    c.buffer.copy({{1,{ion+1,nullptr,1}}});
    execute_collision_model<Model>(ctx,1);
    assert(atoms.size()==extent-count/3-2);
    assert(get<velocity>(ions.get_content().to_host()[1]).z()==2);

    // The same channel must reset its counters and resume normally on a later batch.
    b.buffer.copy({{3,{e+count+3,nullptr,1}}});
    execute_collision_model<Model>(ctx,1);
    assert(get<velocity>(atoms.get_content().to_host()[127*3+3]).y()==12);
    assert(atoms.size()==extent-count/3-2);
}

template<size_t Index>
struct proposal_channel : electron_xenon_tags,deferred_pairing_collision_tag {
    static double request_probability(const auto&,const auto& ctx) { return ctx.probabilities[Index]; }
    static double pair_collision_cross_section(const auto&,const auto&) { return 0.2; }
    static void pair_collide(auto&,const auto&) {}
};

using proposal_payload=pairing_engine<ParticleGroup,field::grid2D,species_tags::Xenon>::payload_type;
struct proposal_context {
    using request=std::pair<uint32_t,proposal_payload>;
    struct buffer_access {
        using value_type=request;
        std::vector<request>& requests;
        void push_back(const request& value) const { requests.push_back(value); }
    };
    struct grid_access {
        static constexpr int Dimension=2;
        field::grid2D grid{{0.,0.},{1.,1.},{1,1}};
        const auto& getGrid() const { return grid; }
    };
    struct background_access { grid_access field_acc; };
    std::array<double,2> probabilities;
    mutable std::vector<request> first,second;
    template<class Tag> auto get() const {
        if constexpr(std::is_same_v<Tag,species_tags::Xenon>)return background_access{};
        else if constexpr(std::is_same_v<Tag,proposal_channel<0>>)return buffer_access{first};
        else return buffer_access{second};
    }
};

void test_explicit_request_probabilities() {
    using executor=psum::particle_collision::foundation::collision_request_executor<
        std::tuple<proposal_channel<0>,proposal_channel<1>>>;
    particle p{};
    get<position>(p)=Eigen::Vector3d(0.5,0.5,0);
    get<velocity>(p).setZero(); // Fixed proposals must include stationary incidents.
    for(const auto probabilities:std::vector<std::array<double,2>>{
            {0.125,0.25},{0,1},{0,0},{0.75,0.75},{-0.1,0.2},
            {std::numeric_limits<double>::quiet_NaN(),0.2}}) {
        for(uint32_t seed=1;seed<257;++seed) {
            proposal_context ctx{probabilities,{},{}};
            get<random_seed>(p)=seed*12345;
            psum::random::rander random(seed*12345);
            const double r=random();
            const bool valid=std::isfinite(probabilities[0]) && probabilities[0]>=0
                && probabilities[0]+probabilities[1]<=1;
            executor::execute(p,ctx);
            const bool first=valid && r<probabilities[0];
            const bool second=valid && !first && r<probabilities[0]+probabilities[1];
            assert(ctx.first.size()==size_t(first) && ctx.second.size()==size_t(second));
            if(first)assert(ctx.first[0].second.request_probability==probabilities[0]);
            if(second)assert(ctx.second[0].second.request_probability==probabilities[1]);
        }
    }
}

struct count_survivors_mcc : electron_xenon_tags,reaction_collision_tag {
    using consumed_species_tags=std::tuple<incident_species_tag>;
    using product_species_tags=std::tuple<>;
    static double collision_cross_section(double v2) { return v2>=4?1:0; }
    static void collide(auto&,const auto& ctx) { ctx.template erase_particle<incident_species_tag>(); }
};

void test_particle_counts_after_mcc() {
    sycl::queue q;
    ParticleGroup electrons(q),atoms(q);
    std::vector<particle> initial(8);
    for(size_t i=0;i<initial.size();++i) {
        get<position>(initial[i])=Eigen::Vector3d(i<4?0.5:1.5,0.5,0);
        get<velocity>(initial[i])=Eigen::Vector3d(i==3||i==5?2:1,0,0);
        get<random_seed>(initial[i])=12345;
    }
    get<velocity>(initial[6]).y()=-1;
    get<position>(initial[7]).x()=3.5; // Valid storage, outside the pairing grid.
    electrons.insert<false>(initial);
    electrons.for_each([](sycl::handler&) {
        return [](particle& p) {
            if(get<velocity>(p).y()<0) ParticleGroup::validator::make_invalid(p);
        };
    });
    field::grid2D grid({0.,0.},{3.,1.},{3,1});
    Field density(q,grid);
    density.setConstant(1);
    using first=same_species_pair<species_tags::electron>;
    struct second : first {};
    using cross=manual_electron_channel;
    // A same-species MCC channel alone must not allocate counts either.
    struct self_mcc : count_survivors_mcc { using background_species_tag=incident_species_tag; };
    using mcc_resource=psum::particle_collision::foundation::species_resource<species_tags::electron,
        collision_model<self_mcc>,ParticleGroup,Field>;
    static_assert(std::is_empty_v<decltype(std::declval<mcc_resource>().particle_counts)>);
    using cross_resource=psum::particle_collision::foundation::species_resource<species_tags::electron,
        collision_model<cross>,ParticleGroup,Field>;
    static_assert(std::is_empty_v<decltype(std::declval<cross_resource>().particle_counts)>);
    using Model=collision_model<first,cross,count_survivors_mcc,second>;
    struct Binding { ParticleGroup* group; Field* density; };
    species_contexts<Model,ParticleGroup,Field> species{
        Binding{&electrons,&density},Binding{&atoms,&density}};
    static_assert(std::is_empty_v<decltype(species.get<species_tags::Xenon>().particle_counts)>);
    auto& counts=species.get<species_tags::electron>().particle_counts;
    static_assert(std::is_same_v<typename std::remove_reference_t<decltype(counts)>::Value,uint32_t>);
    assert(counts.size()==3); // Cell centered even though density is node centered.
    assert(counts.getQueue()==q);
    collision_channel_contexts<Model,ParticleGroup,decltype(species)> channels(q,species);
    context_composition ctx{species,channels};
    for(int round=0;round<2;++round) {
        execute_collision_model<Model>(ctx,1);
        assert(counts.getContent().to_host()==std::vector<uint32_t>({3,1,0}));
    }
    electrons.for_each([](sycl::handler&) {
        return [](particle& p) { ParticleGroup::validator::make_invalid(p); };
    });
    execute_collision_model<Model>(ctx,1);
    assert(counts.getContent().to_host()==std::vector<uint32_t>({0,0,0}));
}

template<bool Same>
struct formula_channel : deferred_pairing_collision_tag {
    // The cross-species incident tag has no 2D projection; only the background grid needs one.
    using incident_species_tag=std::conditional_t<Same,species_tags::electron,species_tags::Helium>;
    using background_species_tag=std::conditional_t<Same,incident_species_tag,species_tags::Xenon>;
    static double request_probability(const auto&,const auto&) { return 0.25; }
    static double pair_collision_cross_section(const auto&,const auto&) { return 0.2; }
    static void pair_collide(auto& pair,const auto&) { get<velocity>(*pair.first).z()=1; }
};
struct formula_context {
    Field::acc_type field_acc;
    field::cell_field2D<uint32_t>::acc_type particle_counts;
    double dt;
    template<class Tag> const auto& get() const { return *this; }
};
struct missing_request_function : electron_xenon_tags,deferred_pairing_collision_tag {
    static double pair_collision_cross_section(const auto&,const auto&) { return 0.2; }
    static void pair_collide(auto&,const auto&) {}
};
struct missing_pair_cross_section : electron_xenon_tags,deferred_pairing_collision_tag {
    static double request_probability(const auto&,const auto&) { return 0.25; }
    static void pair_collide(auto&,const auto&) {}
};
static_assert(psum::particle_collision::foundation::dpmcc_collision<formula_channel<false>,proposal_payload,formula_context>);
static_assert(!psum::particle_collision::foundation::dpmcc_collision<missing_request_function,proposal_payload,formula_context>);
static_assert(!psum::particle_collision::foundation::dpmcc_collision<missing_pair_cross_section,proposal_payload,formula_context>);

template<bool Same>
void test_pair_cross_section_probability() {
    sycl::queue q;
    field::grid2D grid({0.,0.},{1.,1.},{1,1});
    Field density(q,grid);
    field::cell_field2D<uint32_t> particle_counts(q,grid);
    constexpr size_t count=64;
    constexpr auto counts_to_test=[] {
        if constexpr(Same) return std::array{0u,1u,2u,5u,128u};
        else return std::array{0u};
    }();
    for(uint32_t N:counts_to_test)
    for(const auto values:std::vector<std::array<double,4>>{
            {2,0.25,0.5,2},{2,0.25,1,2},{2,0.5,1,2},{0,0.25,1,2},{2,0.25,1,0}}) {
        const double n=values[0],dt=values[1],request_probability=values[2],speed=values[3];
        density.setConstant(n);
        particle_counts.setConstant(N);
        std::vector<particle> initial(2*count);
        for(size_t i=0;i<initial.size();++i) {
            get<position>(initial[i])=Eigen::Vector3d(0.5,0.5,0);
            get<velocity>(initial[i])=Eigen::Vector3d(i<count?0:speed,0,0);
            get<random_seed>(initial[i])=12345*(i+1);
        }
        ParticleGroup particles(q);
        particles.insert<false>(initial);
        q.submit([&](sycl::handler& h) {
            const formula_context ctx{density.get_access(h),particle_counts.get_access(h),dt};
            auto* data=particles.get_content().data();
            sycl::stream warnings(1024,128,h);
            h.parallel_for(sycl::range<1>(count),[=](sycl::id<1> i) {
                proposal_payload pair{data+i[0],data+count+i[0],request_probability};
                psum::particle_collision::foundation::execute_pair<formula_channel<Same>,true,
                    ParticleGroup::validator,ParticleGroup::validator>(pair,ctx,warnings,[](bool,bool){});
            });
        }).wait_and_throw();
        const double factor=Same?(N<2?0.0:0.5*double(N-1)/N):1.0;
        const double probability=factor*0.2*speed*n*dt/request_probability;
        const auto result=particles.get_content().to_host();
        for(size_t i=0;i<count;++i) {
            psum::random::rander random(uint32_t(12345*(i+1)));
            assert(get<velocity>(result[i]).z()==double(random()<probability));
        }
    }
}

template<bool Same,bool First>
struct rejected_match_channel : deferred_pairing_collision_tag {
    using incident_species_tag=species_tags::electron;
    using background_species_tag=std::conditional_t<Same,incident_species_tag,species_tags::Xenon>;
    static double request_probability(const auto&,const auto&) { return 0; }
    static double pair_collision_cross_section(const auto& pair,const auto&) {
        return unit_rate_cross_section(pair,(Same?3.0:1.0)*(First?0.25:1.0));
    }
    static void pair_collide(auto& pair,const auto&) { get<velocity>(*pair.first).z()=First?1:2; }
};

template<bool Same>
void test_rejected_match_stops_channels() {
    sycl::queue q;
    ParticleGroup incident(q),background(q);
    particle p{};
    get<position>(p)=Eigen::Vector3d(0.5,0.5,0);
    get<velocity>(p)=Eigen::Vector3d(1,0,0);
    uint32_t seed=1;
    while(psum::random::rander(seed)()<0.75)++seed;
    get<random_seed>(p)=seed;
    std::vector<particle> initial(Same?3:2,p);
    if constexpr(Same)get<velocity>(initial.back()).x()=-1;
    incident.insert<false>(initial);
    if constexpr(!Same) {
        get<velocity>(p).x()=-1;
        background.insert<false>(std::vector<particle>{p});
    }
    field::grid2D grid({0.,0.},{1.,1.},{1,1});
    Field density(q,grid);
    density.setConstant(1);
    using First=rejected_match_channel<Same,true>;
    using Second=rejected_match_channel<Same,false>;
    using Model=collision_model<First,Second>;
    struct Binding { ParticleGroup* group; Field* density; };
    auto species=[&] {
        if constexpr(Same)return species_contexts<Model,ParticleGroup,Field>{Binding{&incident,&density}};
        else return species_contexts<Model,ParticleGroup,Field>{Binding{&incident,nullptr},Binding{&background,&density}};
    }();
    collision_channel_contexts<Model,ParticleGroup,decltype(species)> channels(q,species);
    context_composition ctx{species,channels};
    auto* data=incident.get_content().data();
    channels.template get<First>().buffer.copy({{0,{data,nullptr,1}}});
    channels.template get<Second>().buffer.copy({{0,{data+1,nullptr,1}}});
    execute_collision_model<Model>(ctx,1);
    auto result=incident.get_content().to_host();
    // The first claim draws and rejects p=0.25. The certain second channel gets no partner.
    psum::random::rander random(seed);
    random();
    assert(get<random_seed>(result[0])==random.get_engine().x);
    assert(get<random_seed>(result[1])==seed);
    for(const auto& value:result)assert(get<velocity>(value).z()==0);
    assert(channels.template get<First>().buffer.size()==0);
    assert(channels.template get<Second>().buffer.size()==0);
}

int main() {
    test_rejected_match_stops_channels<false>();
    test_rejected_match_stops_channels<true>();
    test_request_cell_snapshot();
    test_pairing_storage_growth();
    test_request_reservation();
    test_particle_counts_after_mcc();
    test_explicit_request_probabilities();
    test_pair_cross_section_probability<false>();
    test_pair_cross_section_probability<true>();
    test_duplicate_species_bindings_rejected();
    test_sparse_exclusive_requests();
    test_pairing_without_random_gate_or_compensation();
}
