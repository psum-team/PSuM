#!/bin/bash

# PSuM Dependency Detection Tool (Scan + Env Generator)

# ---------- 0. Parse arguments ----------
SEARCH_DEPTH=3

while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--depth)
            SEARCH_DEPTH="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# ---------- 0. Output file ----------
ENV_FILE="env_load.sh"
> "$ENV_FILE"

echo "# Auto-generated environment loader" >> "$ENV_FILE"
echo "# Generated at: $(date)" >> "$ENV_FILE"
echo "" >> "$ENV_FILE"

# Generate search pattern based on depth
case $SEARCH_DEPTH in
    0) SEARCH_PATTERN="." ;;
    1) SEARCH_PATTERN="{.,*}" ;;
    2) SEARCH_PATTERN="{.,*,*/*}" ;;
    3) SEARCH_PATTERN="{.,*,*/*,*/*/*}" ;;
    4) SEARCH_PATTERN="{.,*,*/*,*/*/*,*/*/*/*}" ;;
    *) SEARCH_PATTERN="{.,*,*/*,*/*/*,*/*/*/*}" ;;
esac

# --- 1. Expand search root directories ---
SEARCH_ROOTS=()

for p in $(echo "$PATH:$LD_LIBRARY_PATH" | tr ':' '\n' | grep -E '/(bin|lib|lib64|include)(/|$)'); do
    dir="${p%/bin*}"
    dir="${dir%/lib*}"
    dir="${dir%/include*}"
    [ -d "$dir" ] && SEARCH_ROOTS+=("$dir")
done

while IFS= read -r dir; do
    SEARCH_ROOTS+=("${dir%/}")
done < <(ls -d ~/*/ 2>/dev/null)

SEARCH_ROOTS+=(/usr /opt /usr/local /data/apps /share /public)

echo "Searching for dependencies in:"
for root in "${SEARCH_ROOTS[@]}"; do
    echo "  $root"
done
echo "Depth: $SEARCH_DEPTH"
echo ""

# ---------- 2. Detection rules ----------
RULES=(
    "LLVM|llvm-config|bin|."
    "AdaptiveCpp|libacpp-rt.so|lib|."
    "Eigen|Eigen/src/Core/EigenBase.h|eigen3,include,."
    "CUDA|nvcc|bin|."
    "BLAS|{*openblas*.so,*libblas*.so}|lib,lib64|."
    "UMFPACK|*umfpack*.so|lib,lib64|."
)

declare -A NAME_PATHS
declare -A NAME_CPATHS
declare -A NAME_LPATHS

EXCLUDE_PATTERNS=(
    "*/wsl/driver/*"
    "*/usr/lib/wsl"
    "/mnt/[a-z]/*"
)

echo "================================================"
echo "PSuM Dependency Detection Tool"
echo "------------------------------------------------"

for rule in "${RULES[@]}"; do
    IFS='|' read -r NAME KEY SUBDIRS ALT <<< "$rule"
    SUB_GLOB="{$SUBDIRS,$ALT}"
    FOUND=false

    for root in "${SEARCH_ROOTS[@]}"; do
        if [ ! -d "$root" ]; then continue; fi
        SKIP=false
        for pattern in "${EXCLUDE_PATTERNS[@]}"; do
            if [[ "$root" == $pattern ]]; then
                SKIP=true
                break
            fi
        done
        [ "$SKIP" = true ] && continue
        CMD="ls -d ${root}/${SEARCH_PATTERN}/${SUB_GLOB}/${KEY} 2>/dev/null | head -n 1"
        RESULT=$(eval "$CMD")

        if [ -n "$RESULT" ]; then
            REAL_FILE=$(realpath "$RESULT")

            KEY_DEPTH=1
            TMP="${KEY%/}"
            while [[ "$TMP" == */* ]]; do
                TMP="${TMP#*/}"
                ((KEY_DEPTH++))
            done

            BASE_DIR="$REAL_FILE"
            for ((i=0;i<KEY_DEPTH;i++)); do
                BASE_DIR=$(dirname "$BASE_DIR")
            done

            echo "✅ [Found $NAME]: $REAL_FILE"

            # bin
            if [[ "$BASE_DIR" == */bin ]]; then
                NAME_PATHS["$NAME"]+="$BASE_DIR "

                ROOT_DIR=$(dirname "$BASE_DIR")
                for lib in lib lib64; do
                    [ -d "$ROOT_DIR/$lib" ] && NAME_LPATHS["$NAME"]+="$ROOT_DIR/$lib "
                done
                [ -d "$ROOT_DIR/include" ] && NAME_CPATHS["$NAME"]+="$ROOT_DIR/include "

            # lib
            elif [[ "$BASE_DIR" == */lib* ]]; then
                NAME_LPATHS["$NAME"]+="$BASE_DIR "

                for inc in "include" "../include" "../../include"; do
                    INC_PATH=$(realpath "$BASE_DIR/$inc" 2>/dev/null)
                    [ -d "$INC_PATH" ] && NAME_CPATHS["$NAME"]+="$INC_PATH "
                done
                ROOT_DIR=$(dirname "$BASE_DIR")
                [ -d "$ROOT_DIR/bin" ] && NAME_PATHS["$NAME"]+="$ROOT_DIR/bin "

            # include
            elif [[ "$BASE_DIR" == *include* ]]; then
                NAME_CPATHS["$NAME"]+="$BASE_DIR "

                ROOT_DIR=$(dirname "$BASE_DIR")
                [ -d "$ROOT_DIR/bin" ] && NAME_PATHS["$NAME"]+="$ROOT_DIR/bin "
                for lib in lib lib64; do
                    [ -d "$ROOT_DIR/$lib" ] && NAME_LPATHS["$NAME"]+="$ROOT_DIR/$lib "
                done
            else
                # For Eigen the KEY file lives at <eigen3|include>/Eigen/src/...,
                # so the header root (<eigen3> or <include>) is BASE_DIR itself;
                # its parent would make <Eigen/Core> unresolvable via CPATH.
                if [[ "$NAME" == "Eigen" ]]; then
                    NAME_CPATHS["$NAME"]+="$BASE_DIR "
                else
                    ROOT_DIR=$(dirname "$BASE_DIR")
                    NAME_CPATHS["$NAME"]+="$ROOT_DIR"
                fi
            fi

            FOUND=true; break
        fi
    done
    [ "$FOUND" = false ] && echo "❌ [Not Found $NAME]"
done

# ---------- 3. Generate environment loader ----------
echo "------------------------------------------------"
echo "Generating $ENV_FILE ..."
echo "" >> "$ENV_FILE"

for rule in "${RULES[@]}"; do
    IFS='|' read -r NAME _ <<< "$rule"

    P=$(echo "${NAME_PATHS[$NAME]}" | tr ' ' '\n' | sort -u)
    C=$(echo "${NAME_CPATHS[$NAME]}" | tr ' ' '\n' | sort -u)
    L=$(echo "${NAME_LPATHS[$NAME]}" | tr ' ' '\n' | sort -u)

    [ -z "$P$C$L" ] && continue

    echo "# --------------------------------------------------" >> "$ENV_FILE"
    echo "# $NAME" >> "$ENV_FILE"
    echo "# --------------------------------------------------" >> "$ENV_FILE"

    for p in $P; do
        echo "export PATH=$p:\$PATH" >> "$ENV_FILE"
    done

    for c in $C; do
        echo "export CPATH=$c:\$CPATH" >> "$ENV_FILE"
    done

    for l in $L; do
        echo "export LIBRARY_PATH=$l:\$LIBRARY_PATH" >> "$ENV_FILE"
        echo "export LD_LIBRARY_PATH=$l:\$LD_LIBRARY_PATH" >> "$ENV_FILE"
    done

    echo "" >> "$ENV_FILE"
done

chmod +x "$ENV_FILE"

echo "✅ env_load.sh generated successfully."
echo "try 'source env_load.sh' to activate the environment."
echo "================================================"