-include ../../config.mk

INCLUDES = -I ../../include -I ../../src

STAGES = stage_1 stage_2 stage_3 stage_4 stage_5

.PHONY: telescope $(STAGES) clean

# command to compile telescope
#telescope: telescope.cpp
#	$(ACPP) $(COMMON_FLAGS) $(INCLUDES) -o telescope telescope.cpp $(LIBS) $(LDFLAGS)
#	mkdir -p output
#	./telescope
#	python3 ../pltview.py density_step output

stage_1: stage_1.cpp
	$(ACPP) $(COMMON_FLAGS) $(INCLUDES) -o stage_1 stage_1.cpp $(LIBS) $(LDFLAGS)
	mkdir -p output
	./stage_1
	python3 ../pltview.py density_step output

stage_2: stage_2.cpp
	$(ACPP) $(COMMON_FLAGS) $(INCLUDES) -o stage_2 stage_2.cpp $(LIBS) $(LDFLAGS)
	mkdir -p output
	./stage_2
	python3 ../pltview.py density_step output

stage_3: stage_3.cpp
	$(ACPP) $(COMMON_FLAGS) $(INCLUDES) -o stage_3 stage_3.cpp $(LIBS) $(LDFLAGS)
	mkdir -p output
	./stage_3
	python3 ../pltview.py density_step output

stage_4: stage_4.cpp
	$(ACPP) $(COMMON_FLAGS) $(INCLUDES) -o stage_4 stage_4.cpp $(LIBS) $(LDFLAGS)
	mkdir -p output
	./stage_4
	python3 ../pltview.py density_step output

stage_5: stage_5.cpp
	$(ACPP) $(COMMON_FLAGS) $(INCLUDES) -o stage_5 stage_5.cpp $(LIBS) $(LDFLAGS)
	mkdir -p output
	./stage_5
	python3 ../pltview.py density_step output

clean:
	rm -f $(STAGES) telescope
	rm -rf output
