-include ../../../config.mk

INCLUDES = -I ../../../include -I ../../../src

sheath_batch: main.cpp
	$(ACPP) $(COMMON_FLAGS) main.cpp $(LIBS) $(LDFLAGS) \
	-I ../../../include \
	-o sheath_batch
	mkdir -p output

run: sheath_batch
	./sheath_batch

clean:
	rm -f sheath_batch output/*.plt numline_xy.plt

.PHONY: run clean
