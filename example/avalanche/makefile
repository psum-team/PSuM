-include ../../config.mk

INCLUDES = -I ../../include -I ../../src

STAGES = stage_1 stage_2 stage_3 stage_4 stage_5

.PHONY: avalanche $(STAGES) clean

# command to compile avalanche
#avalanche: avalanche.cpp
#	$(ACPP) $(COMMON_FLAGS) $(INCLUDES) -o avalanche avalanche.cpp $(LIBS) $(LDFLAGS)
#	mkdir -p output
#	./avalanche
#	python3 ../pltview.py output/atom_dens.plt
#	python3 ../pltview.py ke_step output
#	python3 ../pltview.py ion_step output
#	python3 ../pltview.py count output

stage_1: stage_1.cpp
	$(ACPP) $(COMMON_FLAGS) $(INCLUDES) -o stage_1 stage_1.cpp $(LIBS) $(LDFLAGS)
	mkdir -p output
	./stage_1

stage_2: stage_2.cpp
	$(ACPP) $(COMMON_FLAGS) $(INCLUDES) -o stage_2 stage_2.cpp $(LIBS) $(LDFLAGS)
	mkdir -p output
	./stage_2
	python3 ../pltview.py output/atom_dens.plt

stage_3: stage_3.cpp
	$(ACPP) $(COMMON_FLAGS) $(INCLUDES) -o stage_3 stage_3.cpp $(LIBS) $(LDFLAGS)
	mkdir -p output
	./stage_3
	python3 ../pltview.py output/atom_dens.plt

stage_4: stage_4.cpp
	$(ACPP) $(COMMON_FLAGS) $(INCLUDES) -o stage_4 stage_4.cpp $(LIBS) $(LDFLAGS)
	mkdir -p output
	./stage_4
	python3 ../pltview.py ke_step output

stage_5: stage_5.cpp
	$(ACPP) $(COMMON_FLAGS) $(INCLUDES) -o stage_5 stage_5.cpp $(LIBS) $(LDFLAGS)
	mkdir -p output
	./stage_5
	python3 ../pltview.py output/atom_dens.plt
	python3 ../pltview.py ke_step output
	python3 ../pltview.py ion_step output
	python3 ../pltview.py count output

clean:
	rm -f $(STAGES) avalanche
	rm -rf output
