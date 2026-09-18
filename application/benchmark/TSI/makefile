-include ../../../config.mk

tsi: main.cpp
	$(ACPP) $(COMMON_FLAGS) main.cpp $(LIBS) $(LDFLAGS) \
	-I ../../../include \
	-o tsi
	mkdir -p output
	./tsi

clean:
	rm -f tsi output/*.plt

.PHONY: tsi clean
