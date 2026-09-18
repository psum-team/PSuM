#!/bin/bash
COMPILERS_REGEX='^\s*(g\+\+|gcc|clang\+\+|clang|acpp)'

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
cd "${1:-$SCRIPT_DIR}" || exit 1
SKIP_DIRS=${SKIP_DIRS:-""}
TIMEOUT=${TIMEOUT:-1}
JOBS=${JOBS:-8}

if [ -n "${RUNCHECK_SKIP_FILE:-}" ] && [ -z "${RUNCHECK_SKIP_TARGETS:-}" ]; then
    _f="$SCRIPT_DIR/${RUNCHECK_SKIP_FILE}"
    if [ -f "$_f" ]; then
        RUNCHECK_SKIP_TARGETS=$(grep -vE '^[[:space:]]*(#|$)' "$_f" | paste -sd, -)
    else
        echo "ERROR: RUNCHECK_SKIP_FILE='$RUNCHECK_SKIP_FILE' not found in $SCRIPT_DIR" >&2
        exit 1
    fi
fi

total=0
passed=0
failed=0
run_failed=0
timeouts=0
skipped=0
delegated=0
uncategorized=0
# Per-target logs live in results_dir. Set RUNCHECK_KEEP_LOGS=<dir> to keep them
# under that directory instead of a wiped temp dir (used by CI evidence runs).
if [ -n "${RUNCHECK_KEEP_LOGS:-}" ]; then
    results_dir="${RUNCHECK_KEEP_LOGS}"
    mkdir -p "$results_dir"
else
    results_dir=$(mktemp -d)
    trap "rm -rf $results_dir" EXIT
fi

base_dir=$(pwd)
echo "=== Run Check (timeout=${TIMEOUT}s, jobs=${JOBS}) ==="
if [ -n "${RUNCHECK_KEEP_LOGS:-}" ]; then echo "Per-target logs: $results_dir"; fi
echo ""

test_dirs=$(find . -name makefile -exec dirname {} \; | sed 's|^\./||' | sort)

while read dir; do
    if [ -z "$dir" ]; then
        continue
    fi
    if echo "$SKIP_DIRS" | grep -q "$dir"; then
        echo "⏭️  Skipping $dir"
        continue
    fi
    cd "$base_dir/$dir"
    echo "✨ Testing in directory: $dir/"
    # Aggregate targets are declared by each makefile itself via AGGREGATE_TARGETS;
    # runCheck excludes them along with the built-in non-test targets.
    aggregate=$(grep -E '^AGGREGATE_TARGETS[[:space:]]*[:?+]?=' makefile 2>/dev/null | head -1 | sed 's/^[^=]*=[[:space:]]*//')
    exclude="^(test_cov|py_test|all|clean)$( [ -n "$aggregate" ] && printf "|%s" "$aggregate" )"
    targets=$(grep -E '^[a-zA-Z0-9_-]+:' makefile 2>/dev/null | sed 's/:.*//' | grep -v -E "$exclude")
    key=$(echo "$dir" | tr '/' '_')

    # RUNCHECK_SKIP_TARGETS prefix: with a directory argument the entries are
    # relative to the invocation path; without it, relative to this test root.
    match_dir="$dir"
    if [ -n "${1:-}" ]; then match_dir="${1#./}"; fi

    # parallel compile phase
    for target in $targets; do
        if [ -n "${RUNCHECK_SKIP_TARGETS:-}" ]; then
            case ",${RUNCHECK_SKIP_TARGETS}," in
                *",${match_dir}/${target},"*)
                    echo "  Skipping $target (long/CI-unsuitable)"
                    skipped=$((skipped + 1))
                    continue;;
            esac
        fi
        command=$(make -B -n "$target" 2>&1 | sed ':a;/\\$/{N;s/\\\n/ /;ba}' | head -1)
        if echo "$command" | grep -qE "$COMPILERS_REGEX"; then
            output_file=$(echo "$command" | grep -oE '\-o\s+\S+' | head -1 | sed 's/-o //')
            if [ -n "$output_file" ]; then
                while [ $(jobs -rp | wc -l) -ge $JOBS ]; do
                    sleep 0.1
                done
                echo "  [$target] compiling..."
                total=$((total + 1))
                (
                    if eval "$command" >"$results_dir/${key}__${target}.compile.log" 2>&1; then
                        echo "ok" > "$results_dir/${key}__${target}.compile"
                    else
                        echo "fail" > "$results_dir/${key}__${target}.compile"
                    fi
                ) &
            else
                echo "  [$target] uncategorized (compiler line without -o)"
                uncategorized=$((uncategorized + 1))
            fi
        elif echo "$command" | grep -qE '(^|[ /])(mpicc|mpicxx|mpirun)([[:space:]]|$)'; then
            echo "  [$target] delegated (MPI job)"
            delegated=$((delegated + 1))
        elif echo "$command" | grep -qE '(^|[ /])(python3?|pytest)([[:space:]]|$)|^[[:space:]]*\.[[:space:]]'; then
            echo "  [$target] delegated (Python gate)"
            delegated=$((delegated + 1))
        else
            echo "  [$target] uncategorized"
            uncategorized=$((uncategorized + 1))
        fi
    done
    wait

    # serial run phase
    for target in $targets; do
        compile_result="$results_dir/${key}__${target}.compile"
        if [ -f "$compile_result" ]; then
            if grep -q "fail" "$compile_result"; then
                echo "  [$target] ❌ compile failed"
                [ -s "$results_dir/${key}__${target}.compile.log" ] && tail -n 20 "$results_dir/${key}__${target}.compile.log"
                failed=$((failed + 1))
            else
                command=$(make -B -n "$target" 2>&1 | sed ':a;/\\$/{N;s/\\\n/ /;ba}' | head -1)
                output_file=$(echo "$command" | grep -oE '\-o\s+\S+' | head -1 | sed 's/-o //')
                if [ -f "$output_file" ]; then
                    run_output=$(timeout $TIMEOUT ./$output_file 2>&1)
                    exit_code=$?
                    if [ $exit_code -eq 0 ]; then
                        echo "  [$target] ✅ run ok"
                        passed=$((passed + 1))
                    elif [ $exit_code -eq 124 ]; then
                        if [ "${RUNCHECK_STRICT:-0}" = "1" ]; then
                            echo "  [$target] ⏱️  timeout (FAIL: strict mode)"
                            run_failed=$((run_failed + 1))
                        else
                            echo "  [$target] ⏱️  timeout (smoke only)"
                            timeouts=$((timeouts + 1))
                        fi
                    else
                        if [ $exit_code -eq 139 ]; then
                            fail_reason="SEGFAULT"
                        elif [ $exit_code -eq 136 ]; then
                            fail_reason="SIGFPE"
                        elif [ $exit_code -eq 134 ]; then
                            fail_reason="SIGABRT"
                        elif echo "$run_output" | grep -qi "terminate\|exception\|throw"; then
                            fail_reason="exception"
                        else
                            fail_reason="exit($exit_code)"
                        fi
                        echo "  [$target] ❌ $fail_reason"
                        [ -n "$run_output" ] && echo "$run_output" | tail -n 20
                        run_failed=$((run_failed + 1))
                    fi
                    rm -f "$output_file"
                else
                    echo "  [$target] ❌ output binary missing after compile: $output_file"
                    failed=$((failed + 1))
                fi
            fi
            rm -f "$compile_result"
        fi
    done
    echo "  ✅ Done in $dir/"
done <<< "$test_dirs"

echo ""
echo "================================"
echo "Total: $total, Passed: $passed, Timeout (smoke): $timeouts, Compile Failed: $failed, Run Failed: $run_failed, Skipped (long/CI-unsuitable): $skipped, Delegated (MPI/Python): $delegated, Uncategorized: $uncategorized"
if [ "$total" -eq 0 ]; then
    echo "❌ No tests were found/executed"
    exit 1
fi
if [ "$uncategorized" -gt 0 ]; then
    echo "❌ $uncategorized target(s) could not be classified (see above)"
    exit 1
fi
if [ $((failed + run_failed)) -eq 0 ]; then
    if [ "$timeouts" -eq 0 ]; then
        echo "✅ All tests passed!"
    else
        echo "⚠️  No failures, but $timeouts test(s) hit the ${TIMEOUT}s timeout (smoke only;"
        echo "   raise TIMEOUT or set RUNCHECK_STRICT=1 to enforce full runs)."
    fi
    exit 0
fi
echo "❌ Some tests failed"
exit 1
