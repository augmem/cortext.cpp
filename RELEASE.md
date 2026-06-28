# Cortext 1.0 Release Checklist

This checklist is the minimum evidence for a v1 tag.

## Version Surfaces

- `CMakeLists.txt` project version.
- `bindings/python/pyproject.toml`.
- `bindings/javascript/package.json`.
- `bindings/dart/pubspec.yaml`.
- `build.zig`.

`cortext_version()` and every FFI package should report the same version.

## Native Build Gate

Install a C++20 compiler and CMake. SQLite is built from the bundled
`third_party/sqlite` source tree.

```bash
cmake -S . -B build/release -DCMAKE_BUILD_TYPE=Release -DCORTEXT_DISABLE_OPENTELEMETRY=ON
cmake --build build/release -j
./build/release/tests/cortext_tests '~[aist]' --reporter compact
```

Run the AIST-tagged tests separately on a host with the model installed.

```bash
./build/release/tests/cortext_tests '[aist]' --reporter compact
```

## FFI Gate

```bash
cmake --preset ffi-release-node
cmake --build --preset ffi-release-node --target cortext cortext_node -j
(cd bindings/go && go test)
(cd bindings/dart && dart pub get && dart analyze && dart test)
python3 -m py_compile bindings/python/cortext/__init__.py
node --check bindings/javascript/index.js bindings/wasm/cortext.js examples/web/main.js
node -e "const c=require('./bindings/javascript'); if (c.version() !== '1.0.0') process.exit(1)"
```

## Browser WebAssembly Gate

Install or source Emscripten, then build:

```bash
./build-wasm.sh
test -s build-wasm/dist/wasm/cortext.js
test -s build-wasm/dist/wasm/cortext.wasm
```

For a browser demo, serve the repository root and open `examples/web/`.

```bash
python3 -m http.server 8000
```

The demo can load `AIST-87M_q8_0.gguf` from a file picker, or the model can be
preloaded into `/models` at build time:

```bash
./build-wasm.sh -DCORTEXT_WASM_PRELOAD_MODELS_DIR="$PWD/models/AIST-87M-GGUF"
```

## CI Gate

The GitHub workflow must pass with optional OpenTelemetry disabled for
deterministic model-free CI:

- Ubuntu native release build and non-AIST tests.
- Arch Linux native release build and non-AIST tests.
- Browser WebAssembly bundle build.
- Zig host and Linux cross-build smoke checks.

## Documentation Gate

When algorithm behavior or experiment conclusions change, update
`docs/paper/sections/` and regenerate:

```bash
QUARTO_DISABLE_GIT=1 QUARTO_DISABLE_GITHUB=1 quarto render docs/paper
```

For packaging/API changes, update the root README and all affected binding
READMEs before tagging.
