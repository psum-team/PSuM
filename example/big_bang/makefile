-include ../../config.mk

INCLUDES = -I ../../include -I ../../src

STAGES = stage_1 stage_2 stage_3 stage_4 stage_5 stage_6

.PHONY: big_bang $(STAGES) clean

# command to compile big_bang
#big_bang: big_bang.cpp
#	$(ACPP) $(COMMON_FLAGS) $(INCLUDES) -o bb big_bang.cpp $(LIBS) $(LDFLAGS)
#	./bb

stage_1: stage_1.cpp
	$(ACPP) $(COMMON_FLAGS) $(INCLUDES) -o stage_1 stage_1.cpp $(LIBS) $(LDFLAGS)
	./stage_1

stage_2: stage_2.cpp
	$(ACPP) $(COMMON_FLAGS) $(INCLUDES) -o stage_2 stage_2.cpp $(LIBS) $(LDFLAGS)
	mkdir -p output
	./stage_2
	python3 ../pltview.py output/initial_rho.plt

stage_3: stage_3.cpp
	$(ACPP) $(COMMON_FLAGS) $(INCLUDES) -o stage_3 stage_3.cpp $(LIBS) $(LDFLAGS)
	mkdir -p output
	./stage_3
	python3 ../pltview.py rho_step output

stage_4: stage_4.cpp
	$(ACPP) $(COMMON_FLAGS) $(INCLUDES) -o stage_4 stage_4.cpp $(USE_BACKENDS) $(LIBS) $(LDFLAGS)
	mkdir -p output
	./stage_4
	python3 ../pltview.py phi_step output

stage_5: stage_5.cpp
	$(ACPP) $(COMMON_FLAGS) $(INCLUDES) -o stage_5 stage_5.cpp $(USE_BACKENDS) $(LIBS) $(LDFLAGS)
	mkdir -p output
	./stage_5
	python3 ../pltview.py rho_step output
	python3 ../pltview.py phi_step output

stage_6: stage_6.cpp
	$(ACPP) $(COMMON_FLAGS) $(INCLUDES) -o stage_6 stage_6.cpp $(USE_BACKENDS) $(LIBS) $(LDFLAGS)
	mkdir -p output
	./stage_6
	python3 ../pltview.py rho_step output
	python3 ../pltview.py phi_step output
	python3 ../pltview.py energy output

clean:
	rm -f $(STAGES) bb
