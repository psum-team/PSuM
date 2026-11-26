#include "dpmcc_test_models.hpp"

struct reject_foreign_erasure : pair_reaction {
    using consumed_species_tags = std::tuple<incident_species_tag,background_species_tag>;
    using product_species_tags = std::tuple<>;

    static double pair_collision_cross_section(const auto& pair,const auto&) { return unit_rate_cross_section(pair); }

    static void pair_collide(auto& pair,const auto& ctx) {
        const particle foreign = *pair.first;
        ctx.template erase_particle<incident_species_tag>(pair.second);
        ctx.template erase_particle<background_species_tag>(pair.first);
        ctx.template erase_particle<incident_species_tag>(nullptr);
        ctx.template erase_particle<incident_species_tag>(&foreign);
        get<velocity>(*pair.second).y() = 0.5;
    }
};

struct missing_consumed_species : pair_reaction { using product_species_tags=std::tuple<>; };
struct missing_product_species : pair_reaction { using consumed_species_tags=std::tuple<>; };
struct invalid_product_species : pair_reaction {
    using consumed_species_tags=std::tuple<>;
    using product_species_tags=std::tuple<int>;
};
struct no_species_changes : pair_reaction {
    using consumed_species_tags=std::tuple<>;
    using product_species_tags=std::tuple<>;
};
struct mcc_consumes_background : electron_xenon_channel, reaction_collision_tag {
    using consumed_species_tags = std::tuple<background_species_tag>;
    using product_species_tags = std::tuple<>;
};
static_assert(!psum::particle_collision::foundation::collision_channel<missing_consumed_species>);
static_assert(!psum::particle_collision::foundation::collision_channel<missing_product_species>);
static_assert(!psum::particle_collision::foundation::collision_channel<invalid_product_species>);
static_assert(psum::particle_collision::foundation::collision_channel<no_species_changes>);
static_assert(!psum::particle_collision::foundation::collision_channel<mcc_consumes_background>);

template<class Access,class Channel>
concept binds_erasure=std::constructible_from<psum::particle_collision::foundation::reaction_context<Channel,Access>,
    const Access&,psum::particle_collision::foundation::erasure_requests&>;

struct consume_current : electron_xenon_channel, reaction_collision_tag {
    using consumed_species_tags = std::tuple<incident_species_tag>;
    using product_species_tags = std::tuple<species_tags::Xenon_pos_1>;

    static double collision_cross_section(double v2) { return v2 >= 4 ? 1 : 0; }

    static void collide(particle& p,const auto& ctx) {
        static_assert(!requires { ctx.template erase_particle<incident_species_tag>(&p); });
        static_assert(!requires { ctx.template erase_particle<background_species_tag>(); });
        ctx.template erase_particle<incident_species_tag>();
        ctx.template erase_particle<incident_species_tag>();
        ctx.template insert_particle<species_tags::Xenon_pos_1>(p);
    }
};

void test_endpoint_erasure() {
    sycl::queue q;
    ParticleGroup electrons(q), atoms(q), ions(q);
    particle p{};
    get<position>(p) = Eigen::Vector3d(0.5,0.5,0);
    get<velocity>(p) = Eigen::Vector3d(2,0,0);
    get<random_seed>(p) = 12345;
    const std::vector<particle> one{p};
    particle stationary = p;
    get<velocity>(stationary).setZero();
    // Only slot 1 can generate a request; preserve insertion order.
    electrons.insert<false>(std::vector<particle>{stationary,p});
    atoms.insert(std::vector<particle>{stationary});

    field::grid2D grid({0.,0.},{1.,1.},{1,1});
    Field density(q,grid);
    density.setConstant(1);
    struct binding { ParticleGroup* group; Field* density; };
    struct group_binding { ParticleGroup* group; };
    using model = collision_model<consume_pair>;
    species_contexts<model,ParticleGroup,Field> species{
        group_binding{&electrons},binding{&atoms,&density},group_binding{&ions}};
    assert(species.get<species_tags::electron>().density==nullptr);
    assert(species.get<species_tags::Xenon_pos_1>().density==nullptr);
    {
        using product_model = collision_model<produce_ion>;
        collision_channel_contexts<product_model,ParticleGroup,decltype(species)> channels(q,species);
        context_composition ctx{species,channels};
        using step = psum::particle_collision::foundation::collision_step_context<decltype(ctx)>;
        static_assert(!binds_erasure<step::acc_type,produce_ion>);
        static_assert(binds_erasure<step::acc_type,consume_pair>);
        using ion_tag = species_tags::Xenon_pos_1;
        using ion_access = std::remove_reference_t<decltype(species.get<ion_tag>())>::acc_type;
        using device_context = decltype(ctx)::acc_type;
        using product_context = psum::particle_collision::foundation::reaction_context<produce_ion,device_context>;
        static_assert(std::same_as<decltype(std::declval<const device_context&>().get<ion_tag>()),const ion_access&>);
        static_assert(std::same_as<decltype(std::declval<const product_context&>().get<ion_tag>()),const ion_access&>);
        execute_collision_model<product_model>(ctx,1);
        assert(ions.size()==1 && electrons.size()==2 && atoms.size()==1);
    }
    {
        using rejection_model = collision_model<reject_foreign_erasure>;
        collision_channel_contexts<rejection_model,ParticleGroup,decltype(species)> channels(q,species);
        context_composition ctx{species,channels};
        execute_collision_model<rejection_model>(ctx,1);
        assert(get<velocity>(atoms.get_content().to_host()[0]).y()==0.5);
        assert(electrons.get_invalid_indexes().to_host().empty());
        assert(atoms.get_invalid_indexes().to_host().empty());
    }
    collision_channel_contexts<model,ParticleGroup,decltype(species)> channels(q,species);
    context_composition ctx{species,channels};
    electrons.reserve(electrons.capacity()+1);
    execute_collision_model<model>(ctx,1);
    assert(ions.size()==2 && electrons.size()==1 && atoms.size()==0);
    assert(species.get<species_tags::Xenon_pos_1>().buffer.size()==0);
    const std::vector<size_t> slot_zero{0};
    const std::vector<size_t> slot_one{1};
    assert(electrons.get_invalid_indexes().to_host()==slot_one);
    assert(atoms.get_invalid_indexes().to_host()==slot_zero);

    electrons.insert(one);
    assert(electrons.size()==2 && electrons.get_content().size()==2);
    using mcc_model = collision_model<observe_mcc,consume_current>;
    execute_incident_collisions<mcc_model::collision_types_tuple>(species,1);
    assert(electrons.size()==1 && electrons.get_invalid_indexes().to_host()==slot_one);
    flush_product_buffers<mcc_model::species_tuple>(species);
    assert(ions.size()==3);
}

template<bool EraseBoth>
struct consume_same_species : deferred_pairing_collision_tag, reaction_collision_tag {
    using incident_species_tag = species_tags::electron;
    using background_species_tag = incident_species_tag;
    using consumed_species_tags = std::tuple<incident_species_tag>;
    using product_species_tags = std::tuple<>;

    static double request_probability(const auto&,const auto&) { return 1; }
    static double pair_collision_cross_section(const auto& pair,const auto&) { return unit_rate_cross_section(pair,2); }

    static void pair_collide(auto& pair,const auto& ctx) {
        ctx.template erase_particle<incident_species_tag>(pair.first);
        if constexpr(EraseBoth) {
            ctx.template erase_particle<incident_species_tag>(pair.second);
        }
    }
};

template<bool EraseBoth>
void test_same_species_erasure() {
    sycl::queue q;
    ParticleGroup particles(q,2);
    particle p{};
    get<position>(p) = Eigen::Vector3d(0.5,0.5,0);
    get<velocity>(p) = Eigen::Vector3d(2,0,0);
    get<random_seed>(p) = 12345;
    particle other=p;
    get<velocity>(other).x()=-2;
    particles.insert<false>(std::vector<particle>{p,other});

    field::grid2D grid({0.,0.},{1.,1.},{1,1});
    Field density(q,grid);
    density.setConstant(1);
    struct binding { ParticleGroup* group; Field* density; };
    using channel = consume_same_species<EraseBoth>;
    using model = collision_model<channel>;
    species_contexts<model,ParticleGroup,Field> species{binding{&particles,&density}};
    species.template get<species_tags::electron>().particle_counts.setConstant(2);
    using validator = ParticleGroup::validator;
    psum::particle_collision::foundation::collision_step_context<decltype(species)> step{species,2};
    q.submit([&](sycl::handler& h) {
        const auto ctx = step.get_access(h);
        const auto slots = particles.get_invalid_indexes().get_access(h);
        auto* data = particles.get_content().get_access(h).data();
        sycl::stream warnings(1024,128,h);
        h.single_task([=] {
            typename pairing_engine<ParticleGroup,field::grid2D,species_tags::electron>::payload_type pair{data,data+1,1};
            psum::particle_collision::foundation::execute_pair<channel,true,validator,validator>(
                pair,ctx,warnings,[&](bool erase_first,bool erase_second) {
                    if(erase_first) slots.push_back(0);
                    if(erase_second) slots.push_back(1);
                });
        });
    }).wait_and_throw();

    const auto expected = EraseBoth ? std::vector<size_t>{0,1} : std::vector<size_t>{0};
    assert(particles.get_invalid_indexes().to_host()==expected);
    const auto contents = particles.get_content().to_host();
    assert(!validator::is_valid(contents[0]));
    assert(validator::is_valid(contents[1]) != EraseBoth);
}

void test_mcc_reaction_before_pairing() {
    sycl::queue q;
    ParticleGroup electrons(q), atoms(q), ions(q);
    particle slow{};
    get<position>(slow) = Eigen::Vector3d(0.5,0.5,0);
    get<velocity>(slow) = Eigen::Vector3d(1,0,0);
    get<random_seed>(slow) = 12345;
    particle fast = slow;
    get<velocity>(fast).x() = 2;
    electrons.insert<false>(std::vector<particle>{slow,fast});
    atoms.insert(std::vector<particle>{slow});

    field::grid2D grid({0.,0.},{1.,1.},{1,1});
    Field density(q,grid);
    density.setConstant(1);
    struct binding { ParticleGroup* group; Field* density; };
    using model = collision_model<observe_pair,observe_mcc,consume_current>;
    using species_type = species_contexts<model,ParticleGroup,Field>;
    using channel_contexts = collision_channel_contexts<model,ParticleGroup,species_type>;
    static_assert(std::is_same_v<channel_contexts::tags_tuple,std::tuple<observe_pair>>);
    // The ion created by MCC becomes available only after all DPMCC batches finish.
    species_type species{binding{&electrons,nullptr},binding{&ions,&density},binding{&atoms,&density}};
    channel_contexts channels(q,species);
    context_composition ctx{species,channels};

    execute_collision_model<model>(ctx,2);

    const std::vector<size_t> erased{1};
    assert(electrons.get_invalid_indexes().to_host()==erased);
    assert(electrons.size()==1 && atoms.size()==1 && ions.size()==1);
    const auto remaining = electrons.get_content().to_host();
    assert(get<velocity>(remaining[0]).y()==0.5);
    assert(get<velocity>(remaining[0]).z()==0);
}

int main() {
    test_endpoint_erasure();
    test_same_species_erasure<false>();
    test_same_species_erasure<true>();
    test_mcc_reaction_before_pairing();
}
