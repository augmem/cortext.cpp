#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT}/build/wasm-release"
DIST_DIR="${ROOT}/dist/wasm"
BUNDLE_ID="${OBJSTORE_BUNDLE_ID:-$(git -C "${ROOT}" rev-parse --short HEAD 2>/dev/null || printf 'dev')}"
BUNDLE_ROOT="${DIST_DIR}/${BUNDLE_ID}"

# Ensure the WASM target is built first so artifacts exist
"${ROOT}/scripts/build-wasm.sh" "$@"

MODULE_SRC="${BUILD_DIR}/tests/objstore_wasm_matrix.wasm"
TEST_SRC="${BUILD_DIR}/tests/objstore_tests.wasm"
LIB_SRC="${BUILD_DIR}/src/libobjstore.a"

if [[ ! -f "${MODULE_SRC}" ]]; then
  echo "error: ${MODULE_SRC} not found; ensure objstore_wasm_matrix target exists" >&2
  exit 1
fi
if [[ ! -f "${LIB_SRC}" ]]; then
  echo "error: ${LIB_SRC} not found; run scripts/build-wasm.sh first" >&2
  exit 1
fi

rm -rf "${BUNDLE_ROOT}"
mkdir -p "${BUNDLE_ROOT}/wasi" "${BUNDLE_ROOT}/sql/matrix" "${BUNDLE_ROOT}/js"

cp "${MODULE_SRC}" "${BUNDLE_ROOT}/wasi/objstore_wasm_matrix.wasm"
if [[ -f "${TEST_SRC}" ]]; then
  cp "${TEST_SRC}" "${BUNDLE_ROOT}/wasi/objstore_tests.wasm"
fi
cp "${LIB_SRC}" "${BUNDLE_ROOT}/libobjstore.a"

cp "${ROOT}/tests/wasm/sql/matrix"/*.sql "${BUNDLE_ROOT}/sql/matrix/"
cp "${ROOT}/tests/wasm/sql/matrix/matrix.json" "${BUNDLE_ROOT}/matrix.json"

# Reuse the existing OPFS worker harness scripts so browser runs can attach easily.
cp "${ROOT}/tests/opfs_worker.js" "${BUNDLE_ROOT}/js/opfs_worker.js"
cp "${ROOT}/tests/opfs_harness.mjs" "${BUNDLE_ROOT}/js/opfs_harness.mjs"

# Record build metadata
cat <<META > "${BUNDLE_ROOT}/bundle.json"
{
  "id": "${BUNDLE_ID}",
  "generated_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "commit": "$(git -C "${ROOT}" rev-parse HEAD 2>/dev/null || printf 'unknown')"
}
META

# Generate deterministic SHA256 checksums for CI validation
checksum_tool=""
if command -v shasum >/dev/null 2>&1; then
  checksum_tool="shasum -a 256"
elif command -v sha256sum >/dev/null 2>&1; then
  checksum_tool="sha256sum"
else
  echo "error: neither shasum nor sha256sum is available" >&2
  exit 1
fi
(
  cd "${BUNDLE_ROOT}" >/dev/null
  find . -type f ! -name 'checksums.txt' -print0 | sort -z |
    xargs -0 ${checksum_tool} > checksums.txt
)

echo "Bundle created at ${BUNDLE_ROOT}"
