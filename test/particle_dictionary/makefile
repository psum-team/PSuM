-include ../../config.mk

CXX ?= g++
INCLUDES = -I ../../include -I ../../src -I ../../third_party

test: test_particle_dict.cpp
	$(CXX) -o test test_particle_dict.cpp -std=c++20 $(INCLUDES)
	./test
	rm test