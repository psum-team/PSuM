-include ../../config.mk

INCLUDES = -I ../../include -I ../../src

STAGES = stage_1 stage_2 stage_3 stage_4 stage_5

.PHONY: neon_light $(STAGES) clean

# command to compile neon_light
#neon_light: neon_light.cpp
#	$(ACPP) $(COMMON_FLAGS) $(INCLUDES) -o neon_light neon_light.cpp $(LIBS) $(LDFLAGS)
#	mkdir -p output
#	./neon_light
#	python3 ../pltview.py output/mask.plt

stage_1: stage_1.cpp
	$(ACPP) $(COMMON_FLAGS) $(INCLUDES) -o stage_1 stage_1.cpp $(LIBS) $(LDFLAGS)
	mkdir -p output
	./stage_1
	python3 ../pltview.py output/mask.plt

stage_2: stage_2.cpp
	$(ACPP) $(COMMON_FLAGS) $(INCLUDES) -o stage_2 stage_2.cpp $(USE_BACKENDS) $(LIBS) $(LDFLAGS)
	mkdir -p output
	./stage_2
	python3 ../pltview.py output/mask.plt
	python3 ../pltview.py output/phi.plt

stage_3: stage_3.cpp
	$(ACPP) $(COMMON_FLAGS) $(INCLUDES) -o stage_3 stage_3.cpp $(USE_BACKENDS) $(LIBS) $(LDFLAGS)
	mkdir -p output
	./stage_3
	python3 ../pltview.py output/mask.plt
	python3 ../pltview.py output/phi.plt

stage_4: stage_4.cpp
	$(ACPP) $(COMMON_FLAGS) $(INCLUDES) -o stage_4 stage_4.cpp $(USE_BACKENDS) $(LIBS) $(LDFLAGS)
	mkdir -p output
	./stage_4
	python3 ../pltview.py output/mask.plt
	python3 ../pltview.py phi_step output

stage_5: stage_5.cpp
	$(ACPP) $(COMMON_FLAGS) $(INCLUDES) -o stage_5 stage_5.cpp $(USE_BACKENDS) $(LIBS) $(LDFLAGS)
	mkdir -p output
	./stage_5
	python3 ../pltview.py output/mask.plt
	python3 ../pltview.py phi_step output

clean:
	rm -f $(STAGES) neon_light
	rm -rf output
