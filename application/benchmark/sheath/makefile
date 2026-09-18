-include ../../../config.mk

sheath: main.cpp
	$(ACPP) $(COMMON_FLAGS) main.cpp $(LIBS) $(LDFLAGS) \
	-I ../../../include \
	-o sheath
	mkdir -p output
	./sheath

clean:
	rm -f sheath output/*.plt numline_xy.plt

.PHONY: sheath clean
