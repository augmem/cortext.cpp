#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

gtimeout 120s cmake --preset wasm-release >/dev/null
gtimeout 300s cmake --build --preset wasm-release "$@"

