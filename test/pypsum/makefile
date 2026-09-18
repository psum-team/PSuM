-include ../../config.mk
PYPSUM_VENV ?= ${PROJECT_ROOT}/.pypsum_venv

INCLUDES = -I ../../include -I ../../src -I ../../third_party -I ../../../

py_test: test_mas.py
	. $(PYPSUM_VENV)/bin/activate && python3 test_mas.py
	rm *.mas

py_interop: test_interop_cpp_py.cpp
	. $(PYPSUM_VENV)/bin/activate && g++ -fopenmp -o test_interop test_interop_cpp_py.cpp -std=c++20 -O3 $(INCLUDES) && ./test_interop
	rm test_interop

# Verify the installed package actually imports and exposes the extension.
# Requires the extension to have been built (pypsum/serialization/build.sh) and
# the package installed into the venv.
py_import:
	. $(PYPSUM_VENV)/bin/activate && python3 -c "import pypsum; assert pypsum.MasFile is not None; print('pypsum import OK:', pypsum.MasFile)"

test: py_import py_test py_interop

.PHONY: py_test py_interop py_import test