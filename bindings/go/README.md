# cortext (Go)

Build the native shared library first:

```bash
zig build -Dshared=true -Dllama=false

# or the legacy CMake path:
cmake --preset ffi-release
cmake --build --preset ffi-release --target cortext
```

Then the package can link against `zig-out/lib/libcortext.*` or
`build/ffi-release/libcortext.*` directly:

```bash
cd bindings/go
go test .
```

The package uses the JSON-returning C ABI so callers can either consume raw JSON or unmarshal into their own structs.
