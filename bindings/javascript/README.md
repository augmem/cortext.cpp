# cortext-js

Build the Node addon from the repository root with:

```bash
cd bindings/javascript
npm run build
```

That produces:

- `build/ffi-release/libcortext.*`
- `build/ffi-release/bindings/javascript/cortext.node`

Then load the package directly from the repository:

```bash
node -e "const { version } = require('./bindings/javascript'); console.log(version())"
```

Set `CORTEXT_NODE_ADDON_PATH` if you want to point at a different `.node` file.

The core shared library can also be built with Zig for FFI consumers:

```bash
npm run build:zig
```

That produces `zig-out/lib/libcortext.*`. The Node addon itself still uses the
CMake build path in this phase.
