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

# Append our custom BUILD target for C++ shared library (if not already added)
# Build file is kept in project root (not in the submodule) so it's version-controlled
CORTEXT_BUILD="$SCRIPT_DIR/../litert-lm-BUILD.bazel"
ROOT_BUILD="$LITERT_DIR/BUILD"
if [[ -f "$CORTEXT_BUILD" ]] && ! grep -q "liblitert_lm_cpp" "$ROOT_BUILD" 2>/dev/null; then
  echo "Adding C++ shared library target to BUILD file..."
  cat "$CORTEXT_BUILD" >> "$ROOT_BUILD"
fi

# Build the shared library (C API), static library, and C++ shared library
echo "Building //runtime/engine:litert_lm_shared, //runtime/engine:litert_lm_lib, and //:liblitert_lm_cpp.so..."
$BAZEL_CMD build //runtime/engine:litert_lm_shared //runtime/engine:litert_lm_lib //:liblitert_lm_cpp.so \
  --config=$CONFIG \
  --compilation_mode=opt \
  --verbose_failures

echo ""
echo "Build successful!"
echo ""

# Create output directory
rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/lib" "$OUTPUT_DIR/include"

# Copy the shared library (C API)
echo "Copying shared library to $OUTPUT_DIR/lib..."
cp bazel-bin/runtime/engine/liblitert_lm_shared.dylib "$OUTPUT_DIR/lib/" 2>/dev/null || true
cp bazel-bin/runtime/engine/liblitert_lm_shared.so "$OUTPUT_DIR/lib/" 2>/dev/null || true

# Copy the C++ shared library (exports all symbols)
echo "Copying C++ shared library to $OUTPUT_DIR/lib..."
cp bazel-bin/liblitert_lm_cpp.so "$OUTPUT_DIR/lib/" 2>/dev/null || true
# On macOS, Bazel still creates a .so file even though convention is .dylib
if [[ "$(uname)" == "Darwin" ]] && [[ -f "bazel-bin/liblitert_lm_cpp.so" ]]; then
  cp bazel-bin/liblitert_lm_cpp.so "$OUTPUT_DIR/lib/liblitert_lm_cpp.dylib"
  chmod +w "$OUTPUT_DIR/lib/liblitert_lm_cpp.dylib"
  install_name_tool -id "@rpath/liblitert_lm_cpp.dylib" "$OUTPUT_DIR/lib/liblitert_lm_cpp.dylib"
fi

# Copy the static library (C++ API)
echo "Copying static library (C++ API) to $OUTPUT_DIR/lib..."
cp bazel-bin/runtime/engine/liblitert_lm_lib.a "$OUTPUT_DIR/lib/" 2>/dev/null || true
cp bazel-bin/runtime/engine/liblitert_lm_lib.pic.a "$OUTPUT_DIR/lib/" 2>/dev/null || true

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
mkdir -p "$OUTPUT_DIR/include/runtime"

cp runtime/engine/*.h "$OUTPUT_DIR/include/engine/" 2>/dev/null || true
cp runtime/core/*.h "$OUTPUT_DIR/include/core/" 2>/dev/null || true
cp runtime/conversation/*.h "$OUTPUT_DIR/include/conversation/" 2>/dev/null || true
cp c/engine.h "$OUTPUT_DIR/include/c/" 2>/dev/null || true

# Copy full runtime directory structure (for runtime/ includes)
cp -r runtime/* "$OUTPUT_DIR/include/runtime/" 2>/dev/null || true

# Find Bazel external directory (contains abseil, litert, etc.)
BAZEL_EXTERNAL=""
if [[ -L "$LITERT_DIR/bazel-litert-lm" ]]; then
  EXECROOT=$(readlink "$LITERT_DIR/bazel-litert-lm")
  # Go up from execroot/litert_lm to the cache root, then into external
  BAZEL_EXTERNAL="${EXECROOT%/execroot/*}/external"
fi

# Copy external dependencies fetched by Bazel (abseil, litert)
if [[ -d "$BAZEL_EXTERNAL" ]]; then
  echo "Copying Bazel external dependencies..."

  # Copy LiteRT headers (provides litert/cc/*.h)
  if [[ -d "$BAZEL_EXTERNAL/litert" ]]; then
    echo "  Copying LiteRT headers..."
    mkdir -p "$OUTPUT_DIR/include/litert"
    cp -r "$BAZEL_EXTERNAL/litert/litert"/* "$OUTPUT_DIR/include/litert/" 2>/dev/null || true
  fi

  # Copy Abseil headers (provides absl/*.h)
  if [[ -d "$BAZEL_EXTERNAL/com_google_absl" ]]; then
    echo "  Copying Abseil headers..."
    mkdir -p "$OUTPUT_DIR/include/absl"
    cp -r "$BAZEL_EXTERNAL/com_google_absl/absl"/* "$OUTPUT_DIR/include/absl/" 2>/dev/null || true
  fi

  # Copy Protobuf headers (provides google/protobuf/*.h)
  if [[ -d "$BAZEL_EXTERNAL/com_google_protobuf" ]]; then
    echo "  Copying Protobuf headers..."
    mkdir -p "$OUTPUT_DIR/include/google"
    cp -r "$BAZEL_EXTERNAL/com_google_protobuf/src/google"/* "$OUTPUT_DIR/include/google/" 2>/dev/null || true
  fi
else
  echo "Warning: Bazel external directory not found. C++ API headers may be incomplete."
  echo "  The C API (c/engine.h) should still work."
fi

# Copy generated headers from bazel-bin (e.g., build_config.h, protobuf headers)
BAZEL_BIN="$LITERT_DIR/bazel-bin"
if [[ -d "$BAZEL_BIN/external/litert/litert" ]]; then
  echo "Copying Bazel-generated LiteRT headers..."
  # Merge generated headers with source headers
  cp -r "$BAZEL_BIN/external/litert/litert"/* "$OUTPUT_DIR/include/litert/" 2>/dev/null || true
fi

# Copy generated protobuf headers and runtime files
if [[ -d "$BAZEL_BIN/runtime" ]]; then
  echo "Copying Bazel-generated runtime headers (protobuf, etc.)..."
  # Copy proto generated files
  if [[ -d "$BAZEL_BIN/runtime/proto" ]]; then
    mkdir -p "$OUTPUT_DIR/include/runtime/proto"
    cp "$BAZEL_BIN/runtime/proto"/*.pb.h "$OUTPUT_DIR/include/runtime/proto/" 2>/dev/null || true
  fi
  # Copy any other generated runtime files
  find "$BAZEL_BIN/runtime" -name "*.h" -exec cp {} "$OUTPUT_DIR/include/runtime/" \; 2>/dev/null || true
fi

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
