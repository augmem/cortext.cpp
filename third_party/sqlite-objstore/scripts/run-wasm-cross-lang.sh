#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TIMEOUT_BIN="gtimeout"
if ! command -v "${TIMEOUT_BIN}" >/dev/null 2>&1; then
  if command -v timeout >/dev/null 2>&1; then
    TIMEOUT_BIN="timeout"
  else
    echo "error: gtimeout/timeout not found" >&2
    exit 1
  fi
fi

BUNDLE_ID="${OBJSTORE_BUNDLE_ID:-cross-lang}"
BUNDLE_DIR="${ROOT}/dist/wasm/${BUNDLE_ID}"
MODULE_PATH="${BUNDLE_DIR}/wasi/objstore_wasm_matrix.wasm"

OBJSTORE_BUNDLE_ID="${BUNDLE_ID}" "${ROOT}/scripts/package-wasm-bundle.sh"

run_step() {
  local name="$1"
  shift
  echo "==> ${name}"
  "${TIMEOUT_BIN}" 600s "$@"
}

run_step "python harness" python3 "${ROOT}/tests/wasm/python/harness.py" \
  --bundle "${BUNDLE_DIR}" --module "${MODULE_PATH}"

run_step "go harness" bash -c \
  "cd '${ROOT}/tests/wasm/go' && go run . --bundle '${BUNDLE_DIR}' --module '${MODULE_PATH}'"

run_step "rust harness" bash -c \
  "cd '${ROOT}/tests/wasm/rust' && cargo run --locked --release -- --bundle '${BUNDLE_DIR}' --module '${MODULE_PATH}'"

run_step "node harness" node "${ROOT}/tests/wasm/js/node_harness.mjs" \
  --bundle "${BUNDLE_DIR}" --module "${MODULE_PATH}"

echo "All WASM cross-language harnesses passed"
