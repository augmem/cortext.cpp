# WASM & OPFS Backends

The WASM builds rely on the portable filesystem backend that powers both the VFS (WASI) and OPFS (browser) implementations. This document explains how to build, test, and manually exercise those targets. For the broader architectural context (content-addressed IDs, streaming invariants, transaction semantics) see `docs/architecture.md`.

## Prerequisites

| Target | Requirement     | Notes                                                        |
| ------ | --------------- | ------------------------------------------------------------ |
| WASI   | `WASI_SDK_PATH` | Point to a wasi-sdk installation (e.g., `/opt/wasi-sdk`).    |
| OPFS   | `EMSDK`         | Path to the Emscripten SDK root (`emsdk` install directory). |
| Tests  | `wasmtime`      | Used to execute `objstore_tests.wasm` under WASI.            |

All helper scripts wrap their invocations with `gtimeout` as required by the workspace policy.

## Building for WASI (VFS backend)

```sh
scripts/run-wasi-tests.sh
```

The script configures the `wasi-release` preset, builds the library, and runs the Unity suite via `wasmtime`. CTest automatically wraps the `objstore_tests.wasm` binary whenever `CMAKE_SYSTEM_NAME` equals `WASI`.

Key characteristics:

* Default storage root: `/data/objstore`.
* Enabled backends: VFS + SQLite fallback (native backend disabled).
* `tests/backend_vfs_test.c` mirrors the native CRUD tests, including an 8 MiB streaming roundtrip to verify bounded memory usage.

## Building for Browsers / OPFS

```sh
scripts/build-wasm.sh        # configure + build
scripts/check-wasm-size.sh   # report libobjstore.a + objstore_tests.wasm size
```

The OPFS build uses the Emscripten toolchain preset (`wasm-release`). The portable backend automatically defaults to `/opfs/objstore` and skips compilation unless the worker runs inside a Web Worker with `FileSystemSyncAccessHandle` support.

The size script prints the current `.wasm` and archive sizes so we can keep tracking the `<100 KB` goal for production bundles.

## Packaging the WASM Bundle

```sh
scripts/package-wasm-bundle.sh   # builds wasm-release and assembles dist/wasm/<id>
```

The packaging script drives the `wasm-release` preset, then copies the shared fixtures and shims into `dist/wasm/<bundle-id>/`:

* `bundle.json` + `checksums.txt` document provenance and SHA256s.
* `matrix.json` plus `sql/matrix/*.sql` encode the CRUD/streaming/transaction matrix.
* `wasi/objstore_wasm_matrix.wasm` is the standalone WASI binary that embeds SQLite + objstore and reports JSON assertion results.
* `libobjstore.a` and `wasi/objstore_tests.wasm` are included for embedders who need the static library or Unity regression harness.
* `js/opfs_worker.js` and `js/opfs_harness.mjs` are copied verbatim so browser workers can be launched without hunting through the repo.

Every consumer (Python, Go, Rust, Node, browser) mounts the bundle at `/bundle`, verifies `checksums.txt`, and streams the same fixtures so regressions are reproducible across languages.

## Cross-Language Harness & CI

```sh
scripts/run-wasm-cross-lang.sh   # packages bundle + runs Python/Go/Rust/Node harnesses
```

The orchestration script builds a fresh bundle (or reuses `OBJSTORE_BUNDLE_ID` when provided) and runs each harness under a `600s` timeout:

| Harness | Runtime                                               | Notes                                                             |
| ------- | ----------------------------------------------------- | ----------------------------------------------------------------- |
| Python  | `wasmtime` Python bindings (`pip install wasmtime`)   | writes stdout/stderr to temp files so JSON summaries are captured |
| Go      | `wazero` (Go 1.24.5)                                  | uses `WithDirMount` to expose `/bundle` inside WASI               |
| Rust    | `cargo run --release` with `wasmtime`/`wasmtime-wasi` | honors `--module`/`--bundle` flags                                |
| Node.js | Built-in `node:wasi` (Node 20+)                       | captures WASI stdout via temp descriptors                         |

Minimum tool versions: Python 3.10+, Go 1.24.5 (`.tool-versions`), Rust 1.79+ (`cargo`), and Node 20+ (`NODE_OPTIONS=--experimental-wasi-unstable-preview1` when required). Install `wasmtime` for Python via `pip install wasmtime`.

Each harness loads `matrix.json`, runs every fixture, and prints `[lang] fixture-id: ok|failed`. Failures bubble up to `scripts/run-wasm-cross-lang.sh`, which aborts immediately.

Run a single harness by pointing it at a prepared bundle:

```sh
python3 tests/wasm/python/harness.py --bundle dist/wasm/<id>
go run ./tests/wasm/go --bundle dist/wasm/<id>
cargo run --release --manifest-path tests/wasm/rust/Cargo.toml -- --bundle dist/wasm/<id>
node tests/wasm/js/node_harness.mjs --bundle dist/wasm/<id>
```

Set `-DOBJSTORE_ENABLE_WASM_CROSS_LANG=ON` when configuring CMake to register this script with CTest (`ctest -R wasm_cross_language`). CI pipelines can then enforce the WASM matrix before tagging releases.
All harness invocations run under `gtimeout`/`timeout` per workspace policy so hung WASM executions cannot block CI.

## Manual OPFS Harness

The `tests/opfs_worker.js` and `tests/opfs_harness.mjs` files provide a minimal Web Worker harness:

1. Ensure the SQLite WASM bundle is available (e.g., `sqlite3.js` from the official SQLite WASM distribution) and exposes `sqlite3InitModule`.

2. Host the `tests/` directory with any static server (e.g., `npx http-server tests/`).

3. Load an HTML stub that imports `opfs_harness.mjs`:

   ```html
   <script type="module" src="./opfs_harness.mjs"></script>
   ```

4. The harness loads `opfs_worker.js` (shipped in `dist/wasm/<id>/js`), posts an `init` message (include `extensionPath` pointing to the compiled `objstore.wasm`), and then performs a CRUD round-trip. Results are logged via `postMessage`.

Use the harness as a template for automated Playwright runs or manual smoke tests inside Chrome/Edge/Safari workers.

## Default Storage Roots

| Backend | Default root when unset                              | Notes                                                         |
| ------- | ---------------------------------------------------- | ------------------------------------------------------------- |
| VFS     | `/data/objstore` (WASI) / `objstore` (other targets) | Uses host filesystem via WASI VFS.                            |
| OPFS    | `/opfs/objstore`                                     | Requires Web Worker context and `FileSystemSyncAccessHandle`. |

To override the root, set `objstore_config.storage_root` or the `PRAGMA objstore_storage_root` before calling `objstore_register()`.
