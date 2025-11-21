#!/bin/bash
# Build script for WebAssembly version using Emscripten

set -e

# Check if emscripten is available
if ! command -v emcc &> /dev/null; then
    echo "Error: emcc (Emscripten) not found. Please install Emscripten first."
    echo "See: https://emscripten.org/docs/getting_started/downloads.html"
    exit 1
fi

# Create build directory
BUILD_DIR="build-wasm"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Configure with Emscripten toolchain
emcmake cmake .. \
    -DCMAKE_TOOLCHAIN_FILE=../cmake/EmscriptenToolchain.cmake \
    -DBUILD_TESTING=OFF \
    -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build . --config Release

echo "WebAssembly build complete!"
echo "Output files are in: $BUILD_DIR/"
ls -la *.wasm *.js 2>/dev/null || echo "No .wasm/.js files found - check build output"
