-include ../../config.mk

INCLUDES = -I ../../include -I ../../src -I ../../third_party

test: test_particle_dict.cpp
	$(ACPP) $(COMMON_FLAGS) $(INCLUDES) -o test test_particle_dict.cpp $(LIBS) $(LDFLAGS)
	./test
	rm test
