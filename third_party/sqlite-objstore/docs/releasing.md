# Releasing

Use this checklist before cutting a public `0.1.x` prerelease.

## Native release gate

```sh
cmake --preset full-debug
cmake --build --preset full-debug
ctest --preset full-debug

cmake --preset full-asan
cmake --build --preset full-asan
ctest --preset full-asan

cmake --preset full-release
cmake --build --preset full-release
cmake --install build/full-release --prefix "$PWD/.install-smoke"
sh scripts/verify-install.sh "$PWD/.install-smoke"
```

## Preconditions

- Update `CHANGELOG.md` for the release.
- Confirm the documented savepoint semantics in `docs/transactions.md` still match runtime behavior.
- Ensure the install prefix contains only consumer-facing artifacts (`objstore`, bundled SQLite/BLAKE3 headers when enabled, CMake package files) and no Unity test artifacts.
- Re-run any examples you plan to mention in the release notes.
- On macOS with Homebrew LLVM available, verify the fuzz target still builds and starts:
  `cmake --preset fuzz-debug && cmake --build --preset fuzz-debug --target objstore_sql_surface_fuzzer && build/fuzz-debug/tests/objstore_sql_surface_fuzzer -runs=1`
- If the release notes mention WASM/WASI parity, run `scripts/run-wasm-cross-lang.sh` separately and record the result; that path is not part of the default native CI matrix.
- If you ship the native file backend for long-lived deployments, document the metadata query you expect operators to use with `objstore_example_orphan_sweep`.

## Tagging

- Create a GitHub prerelease tag after the native release gate passes on Linux
  and macOS CI.
- Attach release notes that call out supported platforms, savepoint support,
  the current performance envelope, and any non-blocking WASM/WASI caveats.
