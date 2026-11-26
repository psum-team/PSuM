#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")"
src=$PWD
root=$(cd ../../../.. && pwd)
mkdir -p ../build
out=$(mktemp -d "$src/../build/maxwell-XXXXXX")
echo "Results: $out"
export ACPP_VISIBILITY_MASK=cuda
for name in run probe; do
    "${ACPP:-acpp}" --acpp-targets=generic -O3 -std=c++20 -I"$root" -I"$root/include" -I"${EIGEN_INCLUDE:-/usr/include/eigen3}" "$name.cpp" -o "$out/$name"
done
for seed in {301..306}; do
    mkdir -p "$out/$seed/snapshots"
    cd "$out/$seed"
    "$out/run" --init "$seed"
    sha256sum group*.bin >inputs.sha256
    "$out/run" "$((10000+seed))" >trajectory.log 2>&1
    "$out/probe" "$((20000+seed))" >probe.log 2>&1
    python3 "$src/analyze.py" >analysis.txt
done
cd "$out"
python3 "$src/summarize.py"
python3 "$src/sensitivity.py"
