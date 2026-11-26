#include "frozen_beam_rates.hpp"
#include <string_view>

int main(int argc,char** argv) {
    using frozen_beams::experiment_mode;
    if(argc==1) {
        sycl::queue q;
        for(bool shuffled:{false,true})frozen_beams::run<2>(q,16,100,shuffled,{});
        return 0;
    }
    if(argc==3 && std::string_view(argv[1])=="--frozen-rates")
        return frozen_beams::experiment(argv[2],experiment_mode::full);
    if(argc==3 && std::string_view(argv[1])=="--pair-order")
        return frozen_beams::experiment(argv[2],experiment_mode::pair_order);
    if(argc==3 && std::string_view(argv[1])=="--rate-precision")
        return frozen_beams::rate_precision(argv[2]);
    if(argc==3 && std::string_view(argv[1])=="--rate-precision-n16")
        return frozen_beams::rate_precision(argv[2],128);
    if(argc==3 && std::string_view(argv[1])=="--pair-rates-n16")
        return frozen_beams::experiment(argv[2],experiment_mode::pair_rates_n16);
    return 2;
}
