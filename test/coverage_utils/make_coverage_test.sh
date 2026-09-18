#!/bin/bash
# build_coverage_test
# 用法: ./build_coverage_test src1.cpp src2.cpp ...
# 生成 coverage_report

set -e

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 source_files..."
    exit 1
fi

SRC_FILES=("$@")
TMP_DIR=$(mktemp -d coverage_tmp.XXXX)
PROFINFO_LIST=()

mkdir -p coverage_test
mkdir -p coverage_test/coverage_report

# 1. 编译每个源文件并运行
for SRC in "${SRC_FILES[@]}"; do
    BASE=$(basename "$SRC" .cpp)
    EXE="$TMP_DIR/$BASE.exe"
    PROFRAW="$TMP_DIR/$BASE.profraw"
    PROFDATA="$TMP_DIR/$BASE.profdata"
    INFO="$TMP_DIR/$BASE.info"

    echo "Compiling $SRC ..."
    acpp --acpp-platform=cpu --acpp-targets=omp \
        -g -O0 -fprofile-instr-generate -fcoverage-mapping \
        -fPIC -std=c++20 "$SRC" -o "$EXE"

    echo "Running $EXE ..."
    LLVM_PROFILE_FILE="$PROFRAW" "$EXE"
    # 2. 导出 lcov info 格式
    echo "Exporting lcov info for $SRC ..."
    llvm-profdata-16 merge -sparse "$PROFRAW" -o "$PROFDATA"
    
    llvm-cov-16 export "$EXE" \
        -instr-profile="$PROFDATA" \
        -format=lcov > "$INFO"

    PROFINFO_LIST+=("$INFO")
done

# 3. 合并所有 .info 文件
MERGED_INFO="$TMP_DIR/merged.info"
echo "Merging lcov info files ..."
FIRST=1
for INFO in "${PROFINFO_LIST[@]}"; do
    if [ $FIRST -eq 1 ]; then
        cp "$INFO" "$MERGED_INFO"
        FIRST=0
    else
        TMP_MERGED="$TMP_DIR/tmp_merged.info"
        lcov -a "$MERGED_INFO" -a "$INFO" -o "$TMP_MERGED"
        mv "$TMP_MERGED" "$MERGED_INFO"
    fi
done

# 4. 生成 HTML 报告
lcov --extract "$MERGED_INFO" "*/*psum*/src/*" --output-file coverage_test/filtered.info
genhtml coverage_test/filtered.info --output-directory coverage_test/coverage_report

# 5. 清理中间文件
rm -rf "$TMP_DIR"
rm -rf coverage_test/filtered.info

echo "Coverage report generated in coverage_test/coverage_report/"
