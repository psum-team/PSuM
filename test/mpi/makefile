CC        = mpicc
CXX       = mpicxx
CFLAGS   ?= -O2 -Wall -Wextra -fPIC
CXXFLAGS ?= -O2 -Wall -std=c++20 -fopenmp
INCLUDES  = -I../../src/mpi
MPIRUN   ?= mpirun -np 2

all: test_comm_wrapper test_distributedvec test_mpi_edges

mpi_implement.o: ../../src/mpi/mpi_implement.c ../../src/mpi/interface.h
	$(CC) $(CFLAGS) -I../../src/mpi -c ../../src/mpi/mpi_implement.c -o $@

test_comm_wrapper: test_comm_wrapper.cpp mpi_implement.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ test_comm_wrapper.cpp mpi_implement.o

test_distributedvec: test_distributedvec.cpp mpi_implement.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ test_distributedvec.cpp mpi_implement.o

test_mpi_edges: test_mpi_edges.cpp mpi_implement.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ test_mpi_edges.cpp mpi_implement.o

run: all
	$(MPIRUN) ./test_comm_wrapper
	$(MPIRUN) ./test_distributedvec
	$(MPIRUN) ./test_mpi_edges

clean:
	rm -f test_comm_wrapper test_distributedvec test_mpi_edges mpi_implement.o
