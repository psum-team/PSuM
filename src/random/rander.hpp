#ifndef PSUM_RANDOM_RANDER_HPP
#define PSUM_RANDOM_RANDER_HPP

#include <math.h>
#include <climits>
#include <iostream>
#include <string>
#include <random>

namespace psum{

namespace random {
    
	struct global_random {
		std::mt19937_64 engine;

		inline static global_random& getInstance() {
			static global_random instance;
			return instance;
		}

		inline static double rand() {
			static std::uniform_real_distribution<double> distr(0.0, 1.0);
			return distr(getInstance().engine);
		}

		inline static uint rand_uint() {
			static std::uniform_int_distribution<uint> distr(0, UINT_MAX);
			return distr(getInstance().engine);
		}

		inline static double rand_normal() {
			static thread_local std::normal_distribution<double> distr(0.0, 1.0);
			return distr(getInstance().engine);
		}
	};

	template<uint32_t _a, uint32_t _c>
	class rander_core {
	    static constexpr uint32_t a = _a;
	    static constexpr uint32_t c = _c;
	public:
		static constexpr uint32_t DEFAULT_SEED = 1;
	    using result_type = uint32_t;
	    result_type x;

	    static constexpr result_type min() noexcept { return 0; }
	    static constexpr result_type max() noexcept { return UINT32_MAX; }

	    rander_core() : x(DEFAULT_SEED) {}
	    rander_core(result_type srand) : x(srand) {}
	    rander_core(int srand) : x(static_cast<result_type>(srand)) {}

	    inline void seed(result_type s = 1) { x = s; }
		inline void seed(int s) { x = static_cast<result_type>(s); }
	    inline void seed(double srand) {
	        if (srand < 0) srand = 0;
	        x = static_cast<result_type>(srand < 1 ? srand * RAND_MAX : srand);
	    }

	    inline result_type operator()() {
			x = x * a + c;
			return x;
	    }
	};

	using core166_10 = rander_core<1664525UL, 1013904223UL>;

	class rander {
		using Engine = core166_10;
	public:
		rander() { engine.seed(global_random::rand_uint()); }
		rander(int srand) { engine.seed(srand);}
		rander(uint32_t srand) { engine.seed(srand);}
		rander(double srand) { engine.seed(srand);}
		inline double operator() (){
			// return std::generate_canonical<double, std::numeric_limits<double>::digits>(engine);
			return engine() / double(UINT32_MAX); // safe for cuda
		}
		inline Engine& get_engine() { return engine; }
		uint32_t gen_seed() { 
			uint32_t raw = engine();
			raw = (raw ^ (raw >> 16)) * 0x85ebca6b;
			raw = (raw ^ (raw >> 13)) * 0xc2b2ae35;
			raw = raw ^ (raw >> 16);
			return raw;
		}
	private:
		Engine engine;
	};

	class rander_wrapper {
		uint32_t& outer_seed;
		rander R;
	public:
		rander_wrapper(uint32_t& s) : outer_seed(s), R(s) {}
		~rander_wrapper() {
			sync_seed();
		}
		inline double operator()() {
			return R();
		}
		inline rander& get_rander() {
			return R;
		}
		inline void sync_seed() {
			outer_seed = R.get_engine().x;
		}
	};

	inline rander_wrapper view_as_rander(uint32_t& s) {
		return rander_wrapper(s);
	}

}

}

#endif