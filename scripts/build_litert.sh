#!/bin/bash
# Build LiteRT-LM library with Bazel
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LITERT_DIR="$SCRIPT_DIR/../third_party/litert-lm"
OUTPUT_DIR="$SCRIPT_DIR/../third_party/litert-lm-build"

if [ ! -d "$LITERT_DIR" ]; then
  echo "Error: LiteRT-LM not found at $LITERT_DIR"
  echo "Run: git submodule update --init --recursive"
  exit 1
fi

cd "$LITERT_DIR"

# Detect platform and set config
if [[ "$(uname)" == "Darwin" ]]; then
  if [[ "$(uname -m)" == "arm64" ]]; then
    CONFIG="macos_arm64"
  else
    CONFIG="macos_x86_64"
  fi
elif [[ "$(uname)" == "Linux" ]]; then
  if [[ "$(uname -m)" == "aarch64" ]]; then
    CONFIG="linux_arm64"
  else
    CONFIG="linux_x86_64"
  fi
else
  echo "Error: Unsupported platform $(uname)"
  exit 1
fi

echo "========================================"
echo "Building LiteRT-LM for $CONFIG"
echo "========================================"
echo ""

# Check for bazel/bazelisk
if command -v bazelisk &> /dev/null; then
  BAZEL_CMD="bazelisk"
elif command -v bazel &> /dev/null; then
  BAZEL_CMD="bazel"
else
  echo "Error: Bazel not found. Install with:"
  echo "  brew install bazelisk"
  echo "or download from https://github.com/bazelbuild/bazelisk/releases"
  exit 1
fi

echo "Using: $BAZEL_CMD (version: $($BAZEL_CMD --version))"
echo ""

# Build the shared library (bundles all dependencies)
echo "Building //runtime/engine:litert_lm_shared..."
$BAZEL_CMD build //runtime/engine:litert_lm_shared \
  --config=$CONFIG \
  --compilation_mode=opt \
  --verbose_failures

echo ""
echo "Build successful!"
echo ""

# Create output directory
rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/lib" "$OUTPUT_DIR/include"

# Copy the shared library
echo "Copying shared library to $OUTPUT_DIR/lib..."
cp bazel-bin/runtime/engine/liblitert_lm_shared.dylib "$OUTPUT_DIR/lib/" 2>/dev/null || true
cp bazel-bin/runtime/engine/liblitert_lm_shared.so "$OUTPUT_DIR/lib/" 2>/dev/null || true

# Fix install name for macOS (Bazel bakes in a relative path)
if [[ "$(uname)" == "Darwin" ]] && [[ -f "$OUTPUT_DIR/lib/liblitert_lm_shared.dylib" ]]; then
  echo "Fixing dylib install name..."
  chmod +w "$OUTPUT_DIR/lib/liblitert_lm_shared.dylib"
  install_name_tool -id "@rpath/liblitert_lm_shared.dylib" "$OUTPUT_DIR/lib/liblitert_lm_shared.dylib"
fi

# Copy prebuilt runtime libraries (needed for GPU/accelerator support)
echo "Copying prebuilt runtime libraries..."
cp prebuilt/$CONFIG/*.dylib "$OUTPUT_DIR/lib/" 2>/dev/null || true
cp prebuilt/$CONFIG/*.so "$OUTPUT_DIR/lib/" 2>/dev/null || true

# Copy headers
echo "Copying headers to $OUTPUT_DIR/include..."
mkdir -p "$OUTPUT_DIR/include/engine"
mkdir -p "$OUTPUT_DIR/include/core"
mkdir -p "$OUTPUT_DIR/include/conversation"
mkdir -p "$OUTPUT_DIR/include/c"

cp runtime/engine/*.h "$OUTPUT_DIR/include/engine/" 2>/dev/null || true
cp runtime/core/*.h "$OUTPUT_DIR/include/core/" 2>/dev/null || true
cp runtime/conversation/*.h "$OUTPUT_DIR/include/conversation/" 2>/dev/null || true
cp c/engine.h "$OUTPUT_DIR/include/c/" 2>/dev/null || true

echo ""
echo "========================================"
echo "Build complete!"
echo "Output directory: $OUTPUT_DIR"
echo ""
echo "Contents:"
ls -la "$OUTPUT_DIR/lib/"
echo ""
echo "To use with CMake, configure with:"
echo "  cmake -DBENCHMARK_ENABLE_LITERT=ON ..."
echo "========================================"
