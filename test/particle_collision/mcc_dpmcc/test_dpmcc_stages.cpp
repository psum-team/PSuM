#include "dpmcc_test_models.hpp"

struct same_species_scattering : same_species_pair<species_tags::electron> {
    static double request_probability(const auto&,const auto&) { return 1; }
    static double pair_collision_cross_section(const auto& pair,const auto&) { return unit_rate_cross_section(pair,2); }
    static void pair_collide(auto& pair,const auto&) {
        for(auto* p:{pair.first,pair.second})
            ++get<velocity>(*p).z();
    }
};

struct same_species_products : same_species_scattering,reaction_collision_tag {
    using consumed_species_tags=std::tuple<>;
    using product_species_tags=std::tuple<species_tags::Xenon_pos_1>;
    static void pair_collide(auto& pair,const auto& ctx) {
        same_species_scattering::pair_collide(pair,ctx);
        ctx.template insert_particle<species_tags::Xenon_pos_1>(*pair.first);
    }
};

void test_same_species_products() {
    sycl::queue q;
    ParticleGroup particles(q),products(q);
    constexpr size_t count=2;
    std::vector<particle> initial(count);
    for(size_t i=0;i<count;++i) {
        get<position>(initial[i])=Eigen::Vector3d(0.5,0.5,0);
        get<velocity>(initial[i])=Eigen::Vector3d(i? -2:2,0,0);
        get<random_seed>(initial[i])=12345+i;
    }
    particles.insert<false>(initial);
    field::grid2D grid({0.,0.},{1.,1.},{1,1});
    Field density(q,grid);
    density.setConstant(1);
    using Channel=same_species_products;
    using Model=collision_model<Channel>;
    struct Binding { ParticleGroup* group; Field* density; };
    species_contexts<Model,ParticleGroup,Field> species{
        Binding{&particles,&density},Binding{&products,nullptr}};
    collision_channel_contexts<Model,ParticleGroup,decltype(species)> channels(q,species);
    context_composition ctx{species,channels};
    auto& channel=channels.template get<Channel>();
    auto* data=particles.get_content().data();
    species.template get<species_tags::electron>().particle_counts.setConstant(count);
    psum::particle_collision::foundation::collision_step_context<decltype(ctx)> step{ctx,0};
    for(int round=0;round<3;++round) {
        channel.buffer.copy({{0,{data,nullptr,1}}});
        step.dt=round?2:0; // A rejected pair must not affect the next round.
        deal(species.template get<species_tags::electron>(),
            psum::particle_collision::foundation::bind_pairing_channels<true>(
                step,static_cast<typename Model::collision_types_tuple*>(nullptr)));
        assert(channel.buffer.size()==0);
        for(const auto& p:particles.get_content().to_host())
            assert(get<velocity>(p).z()==round);
        assert(products.size()==0);
    }
    // An empty batch following nonempty batches must not replay stale requests.
    deal(species.template get<species_tags::electron>(),
        psum::particle_collision::foundation::bind_pairing_channels<true>(
            step,static_cast<typename Model::collision_types_tuple*>(nullptr)));
    flush_product_buffers<typename Model::species_tuple>(species);
    for(const auto& p:particles.get_content().to_host())
        assert(get<velocity>(p).z()==2);
    assert(products.size()==count);
    assert(particles.size()==count && particles.get_invalid_indexes().size()==0);
}

struct marked_self_channel : same_species_scattering {
    static double request_probability(const auto& p,const auto&) {
        return get<velocity>(p).x()>0?1.0:0.0;
    }
};

template<class Context>
concept has_incident_marks=requires(Context& ctx) { ctx.prepare_incident_marks(size_t{1}); };
static_assert(!has_incident_marks<species_context<ParticleGroup,Field>>);
static_assert(has_incident_marks<species_context<ParticleGroup,Field,false,true,true>>);

void test_incident_mark_reuse() {
    sycl::queue q;
    ParticleGroup particles(q);
    field::grid2D grid({0.,0.},{1.,1.},{1,1});
    Field density(q,grid);
    density.setConstant(1);
    using Model=collision_model<marked_self_channel>;
    struct Binding { ParticleGroup* group; Field* density; };
    species_contexts<Model,ParticleGroup,Field> species{Binding{&particles,&density}};
    collision_channel_contexts<Model,ParticleGroup,decltype(species)> channels(q,species);
    context_composition ctx{species,channels};
    auto& resource=species.get<species_tags::electron>();
    particle p{};
    get<position>(p)=Eigen::Vector3d(0.5,0.5,0);
    get<random_seed>(p)=12345;
    uint8_t* storage=nullptr;
    size_t largest=0;
    for(size_t count:{2u,4u,2u,16u,4u,0u,4u}) {
        particles.clear();
        std::vector<particle> initial(count,p);
        for(size_t i=0;i<count;++i)get<velocity>(initial[i])=Eigen::Vector3d(i<count/2?1:-1,0,0);
        particles.insert<false>(initial);
        execute_collision_model<Model>(ctx,count?double(count)/(count-1):1);
        for(const auto& value:particles.get_content().to_host())assert(get<velocity>(value).z()==1);
        if(count) {
            auto* current=resource.prepare_incident_marks(count);
            if(count<=largest)assert(current==storage);
            storage=current;
            largest=std::max(largest,count);
        }
    }
    // Mark slots 0 and 1, then delete and refill them as background endpoints.
    execute_collision_model<Model>(ctx,4.0/3.0);
    particles.for_each([](sycl::handler&) {
        return [](particle& value) {
            if(get<velocity>(value).x()>0)ParticleGroup::validator::make_invalid(value);
        };
    });
    get<velocity>(p)=Eigen::Vector3d(-1,0,0);
    particles.insert<false>(std::vector<particle>(2,p));
    assert(particles.extent()==4 && particles.size()==4);
    auto* data=particles.get_content().data();
    auto set_incidents=[&](int begin) {
        particles.for_each([&](sycl::handler&) {
            return [=](particle& value) {
                const auto slot=&value-data;
                get<velocity>(value)=Eigen::Vector3d(begin>=0 && slot>=begin && slot<begin+2?1:-1,0,0);
            };
        });
    };
    set_incidents(2);
    execute_collision_model<Model>(ctx,4.0/3.0);
    for(const auto& value:particles.get_content().to_host())assert(get<velocity>(value).z()==1);
    // An empty request batch retains storage, but its marks must not leak into the next batch.
    set_incidents(-1);
    execute_collision_model<Model>(ctx,4.0/3.0);
    for(const auto& value:particles.get_content().to_host())assert(get<velocity>(value).z()==0);
    set_incidents(0);
    execute_collision_model<Model>(ctx,4.0/3.0);
    for(const auto& value:particles.get_content().to_host())assert(get<velocity>(value).z()==1);
    assert(resource.prepare_incident_marks(4)==storage);
}

// Manual requests obey the normal generator contract: one request per incident.
struct disjoint_self_channel : same_species_scattering {
    static double request_probability(const auto&,const auto&) { return 0; }
};

void test_disjoint_incident_background_sets() {
    sycl::queue q;
    ParticleGroup particles(q);
    auto make_particle=[](double x) {
        particle p{};
        get<position>(p)=Eigen::Vector3d(x,0.5,0);
        get<velocity>(p).setZero();
        get<random_seed>(p)=12345;
        return p;
    };
    std::vector<particle> initial;
    for(size_t i=0;i<12;++i) {
        initial.push_back(make_particle(i<6?0.5:1.5));
        get<velocity>(initial.back()).x()=i+1;
    }
    particles.insert<false>(initial);
    field::grid2D grid({0.,0.},{2.,1.},{2,1});
    Field density(q,grid);
    density.setConstant(1);
    using first=disjoint_self_channel;
    struct second : first {};
    using Model=collision_model<first,second>;
    using species_type=species_contexts<Model,ParticleGroup,Field>;
    using resource=species_context<ParticleGroup,Field>;
    struct Binding { ParticleGroup* group; Field* density; };
    static_assert(!std::default_initializable<resource>);
    static_assert(!std::is_move_assignable_v<resource>);
    static_assert(std::constructible_from<resource,Binding>);
    static_assert(!std::constructible_from<resource,sycl::queue,Binding>);
    static_assert(!std::constructible_from<species_type,resource,resource>);
    species_type species{Binding{&particles,&density}};
    collision_channel_contexts<Model,ParticleGroup,species_type> channels(q,species);
    context_composition ctx{species,channels};
    auto queue_requests=[&](size_t first_slot) {
        auto* data=particles.get_content().data();
        channels.get<first>().buffer.copy({
            {2,{data+3,nullptr,1}}, // Out-of-grid requests must not mark background endpoints.
            {0,{data+first_slot,nullptr,1}},
            {1,{data+6,nullptr,1}},
            {1,{data+7,nullptr,1}},
            {1,{data+8,nullptr,1}}});
        channels.get<second>().buffer.copy({
            {0,{data+first_slot+1,nullptr,1}},
            {0,{data+first_slot+2,nullptr,1}},
            {1,{data+9,nullptr,1}},
            {1,{data+10,nullptr,1}},
            {1,{data+11,nullptr,1}}});
    };
    queue_requests(0);
    execute_collision_model<Model>(ctx,1);
    auto result=particles.get_content().to_host();
    // Both channels share the incident set; an all-incident cell has no partner.
    for(size_t i=0;i<result.size();++i)
        assert(get<velocity>(result[i]).z()==(i<6?1:0));
    assert(channels.get<first>().buffer.size()==0);
    assert(channels.get<second>().buffer.size()==0);
    execute_collision_model<Model>(ctx,1);
    auto unchanged=particles.get_content().to_host();
    for(size_t i=0;i<result.size();++i)
        assert(get<velocity>(unchanged[i]).z()==get<velocity>(result[i]).z());

    particles.reserve(particles.capacity()+1);
    particles.insert<false>(std::vector<particle>(4,make_particle(1.5)));
    queue_requests(3); // Previous background particles now become incidents.
    execute_collision_model<Model>(ctx,1);
    result=particles.get_content().to_host();
    size_t hits=0;
    for(size_t i=0;i<result.size();++i) {
        const auto n=get<velocity>(result[i]).z();
        if(i<6)assert(n==2);
        else {
            assert(n==0 || n==1);
            if(i>=12)assert(n==1);
            hits+=size_t(n);
        }
    }
    assert(hits==8); // Six requests, four remaining background particles.

    // Invalid requests are removed before constructing the incident set.
    auto* data=particles.get_content().data();
    channels.get<first>().buffer.copy({
        {0,{nullptr,nullptr,1}},{0,{data,nullptr,1}}});
    execute_collision_model<Model>(ctx,1);
    auto filtered=particles.get_content().to_host();
    hits=0;
    for(size_t i=0;i<filtered.size();++i) {
        const auto delta=get<velocity>(filtered[i]).z()-get<velocity>(result[i]).z();
        assert(delta==0 || delta==1);
        hits+=size_t(delta);
    }
    assert(hits==2);
}

struct first_background_consumer : pair_reaction {
    using consumed_species_tags = std::tuple<background_species_tag>;
    using product_species_tags = std::tuple<>;
    static double pair_collision_cross_section(const auto& pair,const auto&) {
        return get<position>(*pair.first).x()<1 ? unit_rate_cross_section(pair) : 0;
    }
    static void pair_collide(auto& pair,const auto& ctx) {
        ctx.template erase_particle<background_species_tag>(pair.second);
    }
};

struct second_background_consumer : first_background_consumer {
    using incident_species_tag = species_tags::Xenon_pos_1;
    static double request_probability(const auto& p,const auto&) { return get<velocity>(p).squaredNorm()>=4?1.0:0.0; }
    static double pair_collision_cross_section(const auto& pair,const auto&) { return unit_rate_cross_section(pair); }
};

struct shared_background_scattering : deferred_pairing_collision_tag {
    using incident_species_tag = species_tags::Xenon_pos_1;
    using background_species_tag = species_tags::Xenon;
    static double request_probability(const auto& p,const auto&) { return get<velocity>(p).squaredNorm()<4?1.0:0.0; }
    static double pair_collision_cross_section(const auto& pair,const auto&) { return unit_rate_cross_section(pair); }
    static void pair_collide(auto& pair,const auto&) { get<velocity>(*pair.first).z()=1; }
};

template<bool ScatteringFirst>
void test_all_channels_exclusive_background() {
    sycl::queue q;
    ParticleGroup electrons(q), atoms(q), ions(q);
    auto make_particle=[](double x,double speed) {
        particle p{};
        get<position>(p)=Eigen::Vector3d(x,0.5,0);
        get<velocity>(p)=Eigen::Vector3d(speed,0,0);
        get<random_seed>(p)=12345;
        return p;
    };
    const std::vector<particle> two_cells{make_particle(0.5,2),make_particle(1.5,2)};
    electrons.insert<false>(two_cells);
    atoms.insert<false>(std::vector<particle>{make_particle(0.5,0),make_particle(1.5,0)});
    ions.insert<false>(std::vector<particle>{two_cells[0],two_cells[1],make_particle(0.5,1)});
    field::grid2D grid({0.,0.},{2.,1.},{2,1});
    Field density(q,grid);
    density.setConstant(1);
    using Model=std::conditional_t<ScatteringFirst,
        collision_model<shared_background_scattering,first_background_consumer,second_background_consumer>,
        collision_model<first_background_consumer,second_background_consumer,shared_background_scattering>>;
    using Species=species_contexts<Model,ParticleGroup,Field>;
    struct Binding { ParticleGroup* group; Field* density; };
    auto species=[&] {
        if constexpr(ScatteringFirst)
            return Species{Binding{&ions,nullptr},Binding{&electrons,nullptr},Binding{&atoms,&density}};
        else
            return Species{Binding{&electrons,nullptr},Binding{&ions,nullptr},Binding{&atoms,&density}};
    }();
    collision_channel_contexts<Model,ParticleGroup,Species> channels(q,species);
    context_composition ctx{species,channels};
    execute_collision_model<Model>(ctx,1);

    const auto result=ions.get_content().to_host();
    assert(get<velocity>(result[2]).z()==(ScatteringFirst?1:0));
    assert(get<velocity>(result[0]).z()==0 && get<velocity>(result[1]).z()==0);
    // A scattering pair blocks later consumers. A consumer pair blocks scattering
    // and other consumers, even in cell 1 where its acceptance probability is zero.
    const auto erased=ScatteringFirst?std::vector<size_t>{}:std::vector<size_t>{0};
    assert(atoms.get_invalid_indexes().to_host()==erased);
}


struct consume_or_slow_incident : pair_reaction {
    using consumed_species_tags=std::tuple<incident_species_tag>;
    using product_species_tags=std::tuple<incident_species_tag>;
    static double pair_collision_cross_section(const auto& pair,const auto&) { return unit_rate_cross_section(pair); }
    static void pair_collide(auto& pair,const auto& ctx) {
        if(get<position>(*pair.first).x()<1) {
            particle product=*pair.first;
            get<velocity>(product).z()=7;
            ctx.template erase_particle<incident_species_tag>(pair.first);
            ctx.template insert_particle<incident_species_tag>(product);
        } else {
            get<velocity>(*pair.first).setZero();
        }
    }
};

struct mark_later_pair : observe_pair {
    static double pair_collision_cross_section(const auto& pair,const auto&) { return unit_rate_cross_section(pair); }
    static void pair_collide(auto& pair,const auto&) {
        get<velocity>(*pair.first).z()=1;
        get<velocity>(*pair.second).z()=1;
    }
};

void test_global_requests_and_deferred_products() {
    sycl::queue q;
    ParticleGroup electrons(q,2), atoms(q,2), ions(q,2);
    particle p{};
    get<position>(p)=Eigen::Vector3d(0.5,0.5,0);
    get<velocity>(p)=Eigen::Vector3d(2,0,0);
    get<random_seed>(p)=12345;
    particle other=p;
    get<position>(other).x()=1.5;
    const std::vector<particle> initial{p,other};
    electrons.insert<false>(initial);
    auto background=initial;
    for(auto& item:background)get<velocity>(item).x()=-1;
    atoms.insert<false>(background);
    ions.insert<false>(background);
    field::grid2D grid({0.,0.},{2.,1.},{2,1});
    Field density(q,grid);
    density.setConstant(1);
    using Model=collision_model<consume_or_slow_incident,mark_later_pair>;
    using Species=species_contexts<Model,ParticleGroup,Field>;
    struct Binding { ParticleGroup* group; Field* density; };
    Species species{Binding{&electrons,nullptr},Binding{&atoms,&density},Binding{&ions,&density}};
    collision_channel_contexts<Model,ParticleGroup,Species> channels(q,species);
    context_composition ctx{species,channels};
    execute_collision_model<Model>(ctx,1);

    const auto e=electrons.get_content().to_host();
    const auto i=ions.get_content().to_host();
    assert(electrons.size()==2);
    assert(get<velocity>(e[0]).z()==7); // Replacement stays buffered through the second batch.
    assert(get<velocity>(i[0]).z()==0);
    assert(get<velocity>(e[1]).x()==0 && get<velocity>(e[1]).z()==1); // Request predates slowing.
    assert(get<velocity>(i[1]).z()==1);

}

struct mcc_acceleration : electron_xenon_channel {
    static void collide(particle& p,const auto&) { get<velocity>(p).x()=2; }
};

struct after_mcc_pair : observe_pair {
    static double request_probability(const auto& p,const auto&) { return get<velocity>(p).squaredNorm()>=4?1.0:0.0; }
};

void test_requests_use_post_mcc_state() {
    sycl::queue q;
    ParticleGroup electrons(q), atoms(q), ions(q);
    particle p{};
    get<position>(p)=Eigen::Vector3d(0.5,0.5,0);
    get<velocity>(p)=Eigen::Vector3d(1,0,0);
    get<random_seed>(p)=12345;
    electrons.insert(std::vector<particle>{p});
    atoms.insert(std::vector<particle>{p});
    ions.insert(std::vector<particle>{p});
    field::grid2D grid({0.,0.},{1.,1.},{1,1});
    Field density(q,grid);
    density.setConstant(1);
    using Model=collision_model<after_mcc_pair,mcc_acceleration>;
    using Species=species_contexts<Model,ParticleGroup,Field>;
    struct Binding { ParticleGroup* group; Field* density; };
    Species species{Binding{&electrons,nullptr},Binding{&ions,&density},Binding{&atoms,&density}};
    collision_channel_contexts<Model,ParticleGroup,Species> channels(q,species);
    context_composition ctx{species,channels};
    execute_collision_model<Model>(ctx,2);
    const auto result=electrons.get_content().to_host();
    assert(get<velocity>(result[0]).x()==2 && get<velocity>(result[0]).z()==0.5);
}


struct mcc_produce_ion : electron_xenon_channel, reaction_collision_tag {
    using consumed_species_tags=std::tuple<>;
    using product_species_tags=std::tuple<species_tags::Xenon_pos_1>;
    static void collide(particle& p,const auto& ctx) {
        ctx.template insert_particle<species_tags::Xenon_pos_1>(p);
    }
};

void test_accumulated_product_capacity() {
    sycl::queue q;
    ParticleGroup electrons(q), atoms(q), ions(q);
    particle p{};
    get<position>(p)=Eigen::Vector3d(0.5,0.5,0);
    get<velocity>(p)=Eigen::Vector3d(2,0,0);
    get<random_seed>(p)=12345;
    constexpr size_t count=5000;
    const std::vector<particle> initial(count,p);
    electrons.insert(initial);
    auto background=initial;
    for(auto& item:background)get<velocity>(item).setZero();
    atoms.insert(background);
    field::grid2D grid({0.,0.},{1.,1.},{1,1});
    Field density(q,grid);
    density.setConstant(1);
    using Model=collision_model<produce_ion,mcc_produce_ion>;
    using Species=species_contexts<Model,ParticleGroup,Field>;
    struct Binding { ParticleGroup* group; Field* density; };
    Species species{Binding{&electrons,nullptr},Binding{&atoms,&density},Binding{&ions,nullptr}};
    assert(species.get<species_tags::Xenon_pos_1>().buffer.capacity()<2*count);
    collision_channel_contexts<Model,ParticleGroup,Species> channels(q,species);
    context_composition ctx{species,channels};
    execute_collision_model<Model>(ctx,1);
    assert(ions.size()==2*count);
    assert(species.get<species_tags::Xenon_pos_1>().buffer.capacity()==2*count);
    assert(species.get<species_tags::Xenon_pos_1>().buffer.size()==0);
}

struct grouped_products_a : electron_xenon_channel,reaction_collision_tag {
    using consumed_species_tags=std::tuple<>;
    using product_species_tags=std::tuple<species_tags::electron,
        species_tags::Xenon_pos_1,species_tags::Xenon_pos_1>;
    static double collision_cross_section(double v2) { return v2<2?1:0; }
    static void collide(particle& p,const auto& ctx) {
        ctx.template insert_particle<species_tags::electron>(p);
        ctx.template insert_particle<species_tags::Xenon_pos_1>(p);
        ctx.template insert_particle<species_tags::Xenon_pos_1>(p);
    }
};

struct grouped_products_b : grouped_products_a {
    using product_species_tags=std::tuple<species_tags::electron,
        species_tags::electron,species_tags::Xenon_pos_1>;
    static double collision_cross_section(double v2) { return v2>=2?1:0; }
    static void collide(particle& p,const auto& ctx) {
        ctx.template insert_particle<species_tags::electron>(p);
        ctx.template insert_particle<species_tags::electron>(p);
        ctx.template insert_particle<species_tags::Xenon_pos_1>(p);
    }
};

template<class MCC>
struct grouped_pair_products : deferred_pairing_collision_tag,reaction_collision_tag {
    using incident_species_tag=typename MCC::incident_species_tag;
    using background_species_tag=typename MCC::background_species_tag;
    using consumed_species_tags=typename MCC::consumed_species_tags;
    using product_species_tags=typename MCC::product_species_tags;
    static double request_probability(const auto& p,const auto&) {
        return MCC::collision_cross_section(get<velocity>(p).squaredNorm());
    }
    static double pair_collision_cross_section(const auto& pair,const auto&) { return unit_rate_cross_section(pair); }
    static void pair_collide(auto& pair,const auto& ctx) { MCC::collide(*pair.first,ctx); }
};

template<bool Deferred>
void test_grouped_product_capacity() {
    sycl::queue q;
    ParticleGroup electrons(q),atoms(q),ions(q);
    constexpr size_t count=5000;
    std::vector<particle> initial(count);
    for(size_t i=0;i<count;++i) {
        get<position>(initial[i])=Eigen::Vector3d(0.5,0.5,0);
        get<velocity>(initial[i])=Eigen::Vector3d(i%2?2:1,0,0);
        get<random_seed>(initial[i])=12345+i;
    }
    electrons.insert(initial);
    auto background=initial;
    for(auto& item:background)get<velocity>(item).setZero();
    atoms.insert(background);
    field::grid2D grid({0.,0.},{1.,1.},{1,1});
    Field density(q,grid);
    density.setConstant(1);
    using A=std::conditional_t<Deferred,grouped_pair_products<grouped_products_a>,grouped_products_a>;
    using B=std::conditional_t<Deferred,grouped_pair_products<grouped_products_b>,grouped_products_b>;
    using Model=collision_model<A,B>;
    struct Binding { ParticleGroup* group; Field* density; };
    species_contexts<Model,ParticleGroup,Field> species{
        Binding{&electrons,nullptr},Binding{&atoms,&density},Binding{&ions,nullptr}};
    collision_channel_contexts<Model,ParticleGroup,decltype(species)> channels(q,species);
    context_composition ctx{species,channels};
    execute_collision_model<Model>(ctx,1);
    // Each half selects a different channel; newly inserted electrons wait until the next step.
    assert(electrons.size()==count+3*count/2);
    assert(ions.size()==3*count/2);
    // Each product takes its own maximum, not the sum or one chosen channel.
    assert(species.template get<species_tags::electron>().buffer.capacity()==2*count);
    assert(species.template get<species_tags::Xenon_pos_1>().buffer.capacity()==2*count);
    assert(species.template get<species_tags::electron>().buffer.size()==0);
    assert(species.template get<species_tags::Xenon_pos_1>().buffer.size()==0);
}

// Manual requests make the stage order deterministic without sampling assumptions.
struct stage_self_scatter : deferred_pairing_collision_tag {
    using incident_species_tag=species_tags::Xenon;
    using background_species_tag=incident_species_tag;
    static double request_probability(const auto&,const auto&) { return 0; }
    static double pair_collision_cross_section(const auto& pair,const auto&) { return unit_rate_cross_section(pair,2); }
    static void pair_collide(auto& pair,const auto&) {
        get<velocity>(*pair.first).y()=1;
        get<velocity>(*pair.second).y()=1;
    }
};

struct stage_self_consumer : stage_self_scatter,reaction_collision_tag {
    using consumed_species_tags=std::tuple<incident_species_tag>;
    using product_species_tags=std::tuple<incident_species_tag>;
    static void pair_collide(auto& pair,const auto& ctx) {
        auto product=*pair.first;
        get<position>(product)=get<position>(*pair.second);
        get<velocity>(product).y()=9;
        ctx.template erase_particle<incident_species_tag>(pair.first);
        ctx.template erase_particle<incident_species_tag>(pair.second);
        ctx.template insert_particle<incident_species_tag>(product);
    }
};

struct stage_cross_observer : stage_self_scatter {
    using incident_species_tag=species_tags::electron;
    static double pair_collision_cross_section(const auto& pair,const auto&) { return unit_rate_cross_section(pair); }
    static void pair_collide(auto& pair,const auto&) {
        get<velocity>(*pair.first).z()=get<velocity>(*pair.second).y();
    }
};

void test_same_species_stage_before_cross_species() {
    sycl::queue q;
    ParticleGroup electrons(q),atoms(q);
    auto make_particle=[](double x) {
        particle p{};
        get<position>(p)=Eigen::Vector3d(x,0.5,0);
        get<velocity>(p)=Eigen::Vector3d(x,0,0);
        get<random_seed>(p)=12345;
        return p;
    };
    electrons.insert<false>(std::vector<particle>{make_particle(0.5),make_particle(1.5)});
    std::vector<particle> background{
        make_particle(0.5),make_particle(0.5),make_particle(1.5),make_particle(1.5)};
    get<velocity>(background[0]).x()=-1;
    get<velocity>(background[2]).x()=-1;
    atoms.insert<false>(background);
    field::grid2D grid({0.,0.},{2.,1.},{2,1});
    Field density(q,grid);
    density.setConstant(1);
    // Cross-species is declared first; all self channels must still run first.
    using Model=collision_model<stage_cross_observer,stage_self_scatter,stage_self_consumer>;
    struct Binding { ParticleGroup* group; Field* density; };
    species_contexts<Model,ParticleGroup,Field> species{
        Binding{&electrons,nullptr},Binding{&atoms,&density}};
    collision_channel_contexts<Model,ParticleGroup,decltype(species)> channels(q,species);
    auto* e=electrons.get_content().data();
    auto* a=atoms.get_content().data();
    channels.get<stage_cross_observer>().buffer.copy({
        {0,{e,nullptr,1}},{1,{e+1,nullptr,1}}});
    channels.get<stage_self_scatter>().buffer.copy({{0,{a,nullptr,1}}});
    channels.get<stage_self_consumer>().buffer.copy({
        {1,{a+2,nullptr,1}}});
    context_composition ctx{species,channels};
    execute_collision_model<Model>(ctx,1);
    auto result=electrons.get_content().to_host();
    assert(get<velocity>(result[0]).z()==1);
    // The replacement atom is inserted only after the cross-species stage.
    assert(get<velocity>(result[1]).z()==0);
    assert(atoms.size()==3 && atoms.get_invalid_indexes().size()==1);
    assert(species.get<species_tags::Xenon>().buffer.size()==0);
}

int main() {
    test_incident_mark_reuse();
    test_disjoint_incident_background_sets();
    test_same_species_products();
    test_all_channels_exclusive_background<false>();
    test_all_channels_exclusive_background<true>();
    test_same_species_stage_before_cross_species();
    test_global_requests_and_deferred_products();
    test_requests_use_post_mcc_state();
    test_accumulated_product_capacity();
    test_grouped_product_capacity<false>();
    test_grouped_product_capacity<true>();
}
