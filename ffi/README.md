# FFI glue (engine-owned)

This directory holds **optional** foreign-function glue compiled with the
engine. It is not a language package.

| Path | Role |
| --- | --- |
| `node/addon.cpp` | N-API addon source for optional `cortext_node` / `cortext.node` builds |

Language packages are published from sibling repositories:

- [cortext.py](https://github.com/augmem/cortext.py)
- [cortext.ts](https://github.com/augmem/cortext.ts)
- [cortext.go](https://github.com/augmem/cortext.go)
- [cortext.dart](https://github.com/augmem/cortext.dart)
- [cortext.wasm](https://github.com/augmem/cortext.wasm)

Enable the Node addon via CMake (`CORTEXT_BUILD_NODE_BINDINGS`) or Zig
(`-Dnode-addon=true`). On Windows, CMake searches the Node prefix for
`node.lib` (including `x64/` and `lib/`); set `-DCORTEXT_NODE_LIBRARY=...` to
override the import-library path when Node is installed elsewhere.
