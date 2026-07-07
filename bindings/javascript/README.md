# @augmem/cortext

The JavaScript package is a Node-API addon with TypeScript declarations. The
browser WebAssembly wrapper lives separately in `bindings/wasm`. Release
packages include bundled N-API addons for:

- `linux-x64`
- `linux-arm64`
- `darwin-x64`
- `darwin-arm64`
- `win32-x64`
- `win32-arm64`

## Build

From `bindings/javascript`:

```bash
npm run build
```

That builds:

- `build/ffi-release/libcortext.*`
- `build/ffi-release/bindings/javascript/cortext.node`

Set `CORTEXT_NODE_ADDON_PATH` to load a specific `.node` file.

To build the npm tarball with bundled native addons:

```bash
npm run build:package
```

Set `ZIG=/path/to/zig` or pass `--zig /path/to/zig` when Zig is not on `PATH`.

## Use

```js
const { Cortext, version } = require("@augmem/cortext");

const engine = new Cortext(undefined, ":memory:", "models");
console.log(version());

const ctx = engine.processText("Bailey likes tennis balls.", "chat/main");
console.log(ctx.should_interrupt);

const embedding = engine.embedText("embed without storing");
console.log(embedding.length);

engine.consolidate();
engine.flush();
engine.reset();
```

## API

- `new Cortext(config?, dbPath?, modelsDir?)`
- `processText`, `processAudio`, `processImage`
- `processTextJson`, `processAudioJson`, `processImageJson`
- `embedText`, `embedAudio`, `embedImage`
- `embedTextJson`, `embedAudioJson`, `embedImageJson`
- `consolidate`, `consolidateJson`, `flush`, `reset`
- `version`, `lastError`

TypeScript users should import `CortextConfig`, `CortextContext`, and
`CortextEmbedding` from `index.d.ts`. Audio input is `Float32Array` containing
16 kHz mono PCM. Image input is `Uint8Array` with explicit dimensions and
channel count.

For browser demos, build the Emscripten bundle with `./build-wasm.sh` and serve
`examples/web/` from the repository root.
