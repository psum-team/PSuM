#!/usr/bin/env bash
set -e  # exit when failure.
set -x  # view output for each operation.
if [ "$(id -u)" -ne 0 ]; then
  sudo apt-get install libreadline-dev
fi
g++ -o masker masker.cpp -O2 -std=c++20 -lreadline
set +x
echo "Compilation succeeded."
bin_dir="$(pwd)/bin"
mkdir -p "$bin_dir"
mv masker "$bin_dir/"
echo "install succeeded: $bin_dir/masker"
