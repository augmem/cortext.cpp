# Cortext WebAssembly

The browser build emits an ES module plus a `.wasm` payload:

```bash
./build-wasm.sh
```

Outputs:

```text
build-wasm/dist/wasm/cortext.js
build-wasm/dist/wasm/cortext.wasm
```

The generated module exports the C ABI from `include/cortext/capi.h` and
`malloc/free`. `bindings/wasm/cortext.js` is a small convenience wrapper for
JavaScript callers.

## Model Assets

Cortext still requires the AIST GGUF embedding model. For browser demos you can
either load it at runtime into Emscripten's filesystem:

```js
await runtime.writeModelFile(file);
const handle = runtime.createContext({ dbPath: ":memory:", modelsDir: "/models" });
```

or embed a model directory into the bundle at build time:

```bash
./build-wasm.sh -DCORTEXT_WASM_PRELOAD_MODELS_DIR="$PWD/models/AIST-87M-GGUF"
```

Build-time preloading writes the directory to `/models` inside the virtual
filesystem.

## Demo

After building the bundle, serve the repository root:

```bash
python3 -m http.server 8000
```

Then open:

```text
http://localhost:8000/examples/web/
```
