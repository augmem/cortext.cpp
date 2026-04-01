#!/usr/bin/env bash
# Copyright 2024 sqlite-objstore
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build-coverage"
COVERAGE_DIR="${BUILD_DIR}/coverage"

for tool in llvm-profdata llvm-cov; do
  if ! command -v "${tool}" >/dev/null 2>&1; then
    echo "error: ${tool} is required but was not found on PATH" >&2
    exit 1
  fi
done

cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE=Debug \
  -DOBJSTORE_ENABLE_COVERAGE=ON
cmake --build "${BUILD_DIR}"

rm -rf "${COVERAGE_DIR}"
mkdir -p "${COVERAGE_DIR}"

export LLVM_PROFILE_FILE="${COVERAGE_DIR}/objstore-%p.profraw"
ctest --test-dir "${BUILD_DIR}"

PROFRAW_FILES=("${COVERAGE_DIR}"/objstore-*.profraw)
if [ ! -e "${PROFRAW_FILES[0]}" ]; then
  echo "error: no .profraw files were generated" >&2
  exit 1
fi

llvm-profdata merge -sparse "${COVERAGE_DIR}"/objstore-*.profraw \
  -o "${COVERAGE_DIR}/objstore.profdata"

REPORT_HTML_DIR="${COVERAGE_DIR}/html"
rm -rf "${REPORT_HTML_DIR}"
mkdir -p "${REPORT_HTML_DIR}"

BINARIES=(
  "${BUILD_DIR}/tests/objstore_tests"
  "${BUILD_DIR}/tests/objstore_wasm_matrix"
)

llvm-cov show "${BINARIES[@]}" \
  -instr-profile "${COVERAGE_DIR}/objstore.profdata" \
  -format=html \
  -output-dir "${REPORT_HTML_DIR}" \
  -ignore-filename-regex='/usr/.*/' \
  -ignore-filename-regex='third_party' \
  -ignore-filename-regex='unity'

llvm-cov report "${BINARIES[@]}" \
  -instr-profile "${COVERAGE_DIR}/objstore.profdata" \
  -ignore-filename-regex='unity' \
  -ignore-filename-regex='/usr/.*/' \
  -ignore-filename-regex='third_party' \
  > "${COVERAGE_DIR}/summary.txt"

echo "Coverage summary written to ${COVERAGE_DIR}/summary.txt"
echo "HTML report available under ${REPORT_HTML_DIR}/index.html"

