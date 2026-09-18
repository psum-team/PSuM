#!/bin/bash

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)

VENV_PATH="${SCRIPT_DIR}/../../.pypsum_venv"

echo "Target VENV path: $VENV_PATH"

if [ ! -d "$VENV_PATH" ] || [ ! -f "$VENV_PATH/bin/activate" ]; then
    echo "Creating virtual environment..."
    rm -rf "$VENV_PATH"
    if ! python3 -m venv "$VENV_PATH" 2>/dev/null || [ ! -f "$VENV_PATH/bin/activate" ]; then
        rm -rf "$VENV_PATH"
        echo "python3 -m venv unavailable (missing python3-venv / ensurepip). Trying virtualenv..."
        if command -v virtualenv >/dev/null 2>&1; then
            virtualenv "$VENV_PATH" || { echo "Build Failed: cannot create virtual environment"; exit 1; }
        else
            echo "Build Failed: no usable venv. Install one of:" >&2
            echo "  sudo apt install python3-venv   # or:  pip install --user virtualenv" >&2
            exit 1
        fi
    fi
fi

source "$VENV_PATH/bin/activate"

echo "Installing dependencies..."
pip install --upgrade pip
pip install pybind11 numpy

PY_SUFFIX=$(python3-config --extension-suffix)
PY_INCLUDES=$(python3 -m pybind11 --includes)
PY_INCLUDES=${PY_INCLUDES//\'/}

BUILD_BIN_DIR="${SCRIPT_DIR}/../bin"
mkdir -p "$BUILD_BIN_DIR"

echo "Compiling mas_file_py${PY_SUFFIX}..."

if g++ -O3 -Wall -shared -std=c++20 -fPIC \
    ${PY_INCLUDES} \
    "${SCRIPT_DIR}/mas_file_py.cpp" \
    -o "${BUILD_BIN_DIR}/mas_file_py${PY_SUFFIX}" \
    -lreadline \
    -I"${SCRIPT_DIR}/" \
    -I"${SCRIPT_DIR}/../../"; then
    echo "Compiled: ${BUILD_BIN_DIR}/mas_file_py${PY_SUFFIX}"
else
    echo "Build Failed: g++ compilation error"
    exit 1
fi

if ! pip install -e "${SCRIPT_DIR}/../"; then
    echo "[warn] pip editable install failed (non-fatal: the extension .so is already built)."
    echo "       'import pypsum' then requires manual sys.path setup; see pypsum/load_python_module*.md."
fi

if [ -f "${BUILD_BIN_DIR}/mas_file_py${PY_SUFFIX}" ]; then
    echo "Success: ${BUILD_BIN_DIR}/mas_file_py${PY_SUFFIX}"
else
    echo "Build Failed: extension module not found"
    exit 1
fi