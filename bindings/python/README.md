# cortext (Python)

This package loads the native Cortext shared library built from the repository root.

Build the native library first:

```bash
cmake --preset ffi-release
cmake --build --preset ffi-release --target cortext
```

Then use the package from the repo root:

```bash
PYTHONPATH=bindings/python python3 -c "import cortext; print(cortext.version())"
```

Set `CORTEXT_LIBRARY_PATH` if you want to point at a non-default shared library path.

