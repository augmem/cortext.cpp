# cortext (Go)

Build the native shared library first:

```bash
cmake --preset ffi-release
cmake --build --preset ffi-release --target cortext
```

Then the package can link against `build/ffi-release/libcortext.*` directly:

```bash
cd bindings/go
go test .
```

The package uses the JSON-returning C ABI so callers can either consume raw JSON or unmarshal into their own structs.
