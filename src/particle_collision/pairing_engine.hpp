#ifndef PSUM_PARTICLE_COLLISION_PAIRING_ENGINE_HPP
#define PSUM_PARTICLE_COLLISION_PAIRING_ENGINE_HPP

#include <sycl/sycl.hpp>
#include <cstddef>
#include <cstdint>
#include <algorithm>
#include <tuple>
#include <utility>
#include "../../include/psum/particle_container.hpp"
#include "../field/device_array.hpp"
#include <AdaptiveCpp/algorithms/numeric.hpp>
#include "context/species_context.hpp"

namespace psum {

namespace particle_collision {

    template<class Background,class... Channel>
    void deal(Background&,std::tuple<Channel...>);

    // Both particle groups must keep their storage and slot identities stable
    // from request generation through the last collision callback.
    template<typename ParticleGroupType, typename GridType,typename BackgroundSpeciesTag>
    struct pairing_engine {
        using particle_type=typename ParticleGroupType::value_type;
        struct payload_type {particle_type* first; particle_type* second; double request_probability;};
        struct acc_type {
            psum::field::device_array_acc<uint32_t> prefix;
            payload_type* data;
            uint32_t total;
            uint32_t partition_offset(size_t i) const {return prefix[i];}
            uint32_t partition_size(size_t i) const {return prefix[i+1]-prefix[i];}
            auto& operator()(size_t i,size_t j) const {return data[partition_offset(i)+j];}
        };

    private:
        template<class Background,class... Channel>
        friend void deal(Background&,std::tuple<Channel...>);
        using background_species_tag=BackgroundSpeciesTag;
        psum::field::device_array<uint32_t> ticket_counters_;
        psum::field::device_array<payload_type> data_;
        psum::field::device_array<uint32_t> prefix_sum_;
        psum::field::device_array<uint32_t> partition_sizes_;
        acpp::algorithms::util::allocation_cache scan_cache_{
            acpp::algorithms::util::allocation_type::device};
        uint32_t total_size_ = 0;
        GridType grid_;

        template<class IncidentGetter>
        static auto request_filter(sycl::handler& h,IncidentGetter incident_getter) {
            auto incident=incident_getter(h);
            return [=](const payload_type& payload) {
                return incident.is_valid(foundation::endpoint_slot(incident.data,incident.extent,payload.first));
            };
        }

        template<class RequestBuffer,class IncidentGetter,class MarkerFactory>
        void prepare_pairing(RequestBuffer& input_data,IncidentGetter incident_getter,MarkerFactory make_marker) {
            static_assert(std::is_same_v<typename RequestBuffer::value_type,std::pair<uint32_t,payload_type>>,
                          "request buffer must store cell/payload pairs");
            auto q = input_data.get_queue();

            const size_t n = input_data.size();
            total_size_=0;
            if(n==0)return;
            size_t num_cells = partition_sizes_.size();

            auto* counts=partition_sizes_.data();
            auto* tickets=ticket_counters_.data();
            auto* prefix=prefix_sum_.data();
            if(!num_cells)return;
            q.submit([&](sycl::handler& h) {
                h.parallel_for(sycl::range<1>(num_cells),[=](sycl::id<1> i) {
                    counts[i]=0;
                    tickets[i]=0;
                    if(i[0]==0)prefix[0]=0;
                });
            }).wait();

            // Count only live requests immediately before pairing.
            auto counted=q.submit([&](sycl::handler& h) {
                auto requests=input_data.get_access(h);
                auto is_live=request_filter(h,incident_getter);
                h.parallel_for(sycl::range<1>(n),[=](sycl::id<1> i) {
                    const auto& request=requests[i];
                    if(request.first>=num_cells || !is_live(request.second))return;
                    const uint32_t cell=request.first;
                    sycl::atomic_ref<uint32_t,sycl::memory_order::relaxed,
                        sycl::memory_scope::device,sycl::access::address_space::global_space>(
                            counts[cell]).fetch_add(1);
                });
            });
            {
                acpp::algorithms::util::allocation_group scratch{&scan_cache_, q.get_device()};
                auto scanned=acpp::algorithms::inclusive_scan(q, scratch, counts,
                    counts+num_cells, prefix+1, {counted});
                q.memcpy(&total_size_,prefix+num_cells,sizeof(total_size_),scanned).wait();
            }

            if(total_size_==0)return;
            // Every live payload is overwritten; growth need not copy the old batch.
            if(total_size_>data_.size())
                data_=decltype(data_)(q,std::max(size_t(total_size_),data_.size()*2));
            auto mark_incident=make_marker();

            q.submit([&](sycl::handler& h) {
                auto in_acc = input_data.get_access(h);
                auto out_acc = data_.get_access(h);
                auto offsets = prefix_sum_.data();
                auto counts = partition_sizes_.data();
                auto is_live=request_filter(h,incident_getter);
                h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
                    if(in_acc[i].first>=num_cells || !is_live(in_acc[i].second))return;
                    const uint32_t gid=in_acc[i].first;
                    sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed,
                                    sycl::memory_scope::device,
                                    sycl::access::address_space::global_space> ref(counts[gid]);
                    const uint32_t local_idx = ref.fetch_sub(1) - 1;
                    out_acc[offsets[gid] + local_idx] = in_acc[i].second;
                    mark_incident(in_acc[i].second);
                });
            }).wait();
        }

        acc_type mapped_access(sycl::handler& h) {return {prefix_sum_.get_access(h),data_.data(),total_size_};}

        // Each background particle stops after its first successful claim.
        auto selection_body(sycl::handler& h) {
            auto mapped=mapped_access(h);
            auto tickets=ticket_counters_.data();
            return [=](particle_type& p,uint32_t gid) -> bool {
                // Empty channels may retain partitions from an earlier batch.
                if(mapped.total==0)return false;
                const uint32_t M=mapped.partition_size(gid);
                sycl::atomic_ref<uint32_t,sycl::memory_order::relaxed,sycl::memory_scope::device,
                    sycl::access::address_space::global_space> ticket(tickets[gid]);
                if(M==0||ticket.load()>=M)return false;
                auto i=ticket.fetch_add(1);
                if(i>=M)return false;
                auto& payload=mapped(gid,i);
                payload.second=&p;
                return true;
            };
        }

    public:
        pairing_engine(sycl::queue q, GridType grid): ticket_counters_(q),data_(q),prefix_sum_(q),partition_sizes_(q),grid_(grid) {
            size_t cells=1;
            for(auto n:grid.get_cell_num())cells*=n;
            prefix_sum_=psum::field::device_array<uint32_t>(q,cells+1);
            partition_sizes_=psum::field::device_array<uint32_t>(q,cells);
            ticket_counters_=psum::field::device_array<uint32_t>(q,cells);
        }
    };

}

}

#endif
