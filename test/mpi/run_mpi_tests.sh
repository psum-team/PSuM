#!/usr/bin/env bash
# Independent multi-process MPI test path.
# Not part of the CPU-only test/runCheck.sh (single-process host tests); invoke it
# from an MPI environment or CI. Any MPI-specific runtime variables (e.g.
# HWLOC_COMPONENTS in headless environments) must be exported by the caller.
#
# Usage: bash test/mpi/run_mpi_tests.sh
#   MAKE may be overridden; the launcher and its arguments are separate:
#     MPIRUN=mpirun MPIRUN_ARGS="--oversubscribe" bash test/mpi/run_mpi_tests.sh
#   Each invocation is bounded by RUN_TIMEOUT seconds (default 120) so a single
#   hung rank cannot stall the whole suite:
#     RUN_TIMEOUT=300 bash test/mpi/run_mpi_tests.sh
set -u
cd "$(dirname "$0")" || exit 1

MAKE=${MAKE:-make}
MPIRUN=${MPIRUN:-mpirun}
MPIRUN_ARGS=${MPIRUN_ARGS:-}
RUN_NP_LIST=${RUN_NP_LIST:-"2 3 4"}
RUN_TIMEOUT=${RUN_TIMEOUT:-120}

# launcher stays one element (its path may contain spaces); MPIRUN_ARGS is split
# on whitespace into plain flags only (no shell syntax, no eval).
MPIRUN_ARR=("$MPIRUN")
if [ -n "$MPIRUN_ARGS" ]; then
    read -r -a MPIRUN_EXTRA <<< "$MPIRUN_ARGS"
    MPIRUN_ARR+=("${MPIRUN_EXTRA[@]}")
fi

fail=0
run_one() {   # $1 = np, $2 = test binary
    local np="$1"
    local tgt="$2"
    local log="run_${tgt}_np${np}.log"
    echo "--- ${tgt} -np ${np} (timeout ${RUN_TIMEOUT}s) ---"
    if timeout "$RUN_TIMEOUT" "${MPIRUN_ARR[@]}" -np "$np" "./$tgt" > "$log" 2>&1; then
        tail -n 2 "$log"
    else
        local rc=$?
        if [ "$rc" -eq 124 ]; then
            echo "FAILED: ${tgt} -np ${np} timed out after ${RUN_TIMEOUT}s (see ${log})"
        else
            echo "FAILED: ${tgt} -np ${np} (rc=${rc}, see ${log})"
        fi
        cat "$log"
        fail=1
    fi
}

"$MAKE" all > /dev/null || { echo "build failed"; exit 1; }

# bidirectional wrapper test requires exactly 2 ranks
run_one 2 test_comm_wrapper

# edge cases and distributed vector: rank-count agnostic, run 2/3/4
for np in $RUN_NP_LIST; do
    run_one "$np" test_mpi_edges
    run_one "$np" test_distributedvec
done

rm -f run_test_*.log
if [ "$fail" -eq 0 ]; then
    echo "MPI TESTS ALL PASS"
    exit 0
else
    echo "MPI TESTS FAILED"
    exit 1
fi
