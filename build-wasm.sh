#!/bin/bash
# Build script for the browser WebAssembly bundle using Emscripten.

set -e
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Check if emscripten is available
if ! command -v emcc &> /dev/null; then
    echo "Error: emcc (Emscripten) not found. Please install Emscripten first."
    echo "See: https://emscripten.org/docs/getting_started/downloads.html"
    exit 1
fi

cmake --preset wasm "$@"
cmake --build --preset wasm

echo "WebAssembly build complete!"
echo "Output files are in: ${ROOT_DIR}/build-wasm/dist/wasm/"
ls -la "${ROOT_DIR}/build-wasm/dist/wasm/"*.wasm "${ROOT_DIR}/build-wasm/dist/wasm/"*.js 2>/dev/null \
  || echo "No .wasm/.js files found - check build output"
