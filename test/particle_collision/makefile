-include ../../config.mk

test1:test_foundation.cpp
	$(ACPP) -o test1 test_foundation.cpp -std=c++20 -O3
	./test1
	rm test1

test2:test_context.cpp
	$(ACPP) -o test2 test_context.cpp -std=c++20 -O3
	./test2
	rm test2

test3:test_pairing_engine.cpp
	$(ACPP) -o test3 test_pairing_engine.cpp -std=c++20 -O3
	./test3
	rm test3

test4:pairing_engine_performance.cpp
	$(ACPP) -o test4 pairing_engine_performance.cpp -std=c++20 -O3
	./test4
	rm test4

test5:test_grid_mapped_device_vector.cpp
	$(ACPP) -o test5 test_grid_mapped_device_vector.cpp -std=c++20 -O3
	./test5
	rm test5

test6:test_dpmcc_context.cpp
	$(ACPP) -o test6 test_dpmcc_context.cpp -std=c++20 -O3
	./test6
	rm test6

test7: test_simple_dsmc.cpp
	$(ACPP) -o test7 test_simple_dsmc.cpp -std=c++20 -O3 -I ../../include
	./test7
	rm test7

test8: test_exclusive_scan.cpp
	$(ACPP) -o test8 test_exclusive_scan.cpp -std=c++20 -O3 -I ../../include
	./test8
	rm test8

test9: test_coulomb_collision.cpp
	$(ACPP) -o test9 test_coulomb_collision.cpp -std=c++20 -O3 -I ../../include
	./test9
	rm test9

.PHONY: test1 test2 test3 test4 test5 test6 test7 test8 test9
