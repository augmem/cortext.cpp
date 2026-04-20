# Bindings

The repository now ships thin wrappers over the C ABI for:

- `bindings/python`: Python via `ctypes`
- `bindings/go`: Go via `cgo`
- `bindings/javascript`: Node.js plus TypeScript declarations
- `bindings/dart`: Dart via `dart:ffi`

All four assume the shared library is built with the repository-local FFI preset:

```bash
cmake --preset ffi-release
cmake --build --preset ffi-release --target cortext
```

For the Node addon, use the Node-enabled variant:

```bash
cmake --preset ffi-release-node
cmake --build --preset ffi-release-node --target cortext_node
```
