#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "${SCRIPT_DIR}/env_scan.sh"
source "${SCRIPT_DIR}/env_load.sh"

echo "=== Building field solver backends ==="
cd "${SCRIPT_DIR}/src/field_solver"
make all

echo ""
echo "=== Testing field solver backends ==="
cd "${SCRIPT_DIR}/test/field_solver"
make test

if [ -z "${PSUM_SKIP_PYTHON:-}" ]; then
    # Keep an existing .pypsum_venv by default (it is reused, not wiped).
    # Set PSUM_FORCE_VENV=1 to rebuild it from scratch.
    if [ -n "${PSUM_FORCE_VENV:-}" ]; then
        rm -rf "${SCRIPT_DIR}/.pypsum_venv"
    fi
    echo ""
    echo "=== Building pypsum serialization module ==="
    cd "${SCRIPT_DIR}/pypsum/serialization"
    bash build.sh

    echo ""
    echo "=== Testing pypsum module ==="
    cd "${SCRIPT_DIR}/test/pypsum"
    make py_interop
else
    echo ""
    echo "=== Skipping Python module (PSUM_SKIP_PYTHON is set) ==="
fi

echo ""
echo "=== Build complete ==="