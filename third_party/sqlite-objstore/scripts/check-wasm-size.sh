#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

gtimeout 120s cmake --preset wasm-release >/dev/null
gtimeout 300s cmake --build --preset wasm-release

wasm_test="${ROOT}/build/wasm-release/tests/objstore_tests.wasm"
wasm_lib="${ROOT}/build/wasm-release/src/libobjstore.a"

if [[ ! -f "${wasm_test}" ]]; then
  echo "error: ${wasm_test} not found" >&2
  exit 1
fi

if [[ ! -f "${wasm_lib}" ]]; then
  echo "error: ${wasm_lib} not found" >&2
  exit 1
fi

size_cmd=(stat -f%z)
if ! stat -f%z "${wasm_test}" >/dev/null 2>&1; then
  size_cmd=(stat -c%s)
fi

test_bytes="$("${size_cmd[@]}" "${wasm_test}")"
lib_bytes="$("${size_cmd[@]}" "${wasm_lib}")"

echo "objstore_tests.wasm size: ${test_bytes} bytes"
echo "libobjstore.a size: ${lib_bytes} bytes"

limit=$((400 * 1024))
if (( test_bytes > limit )); then
  echo "warning: test harness exceeds ${limit} bytes" >&2
fi

