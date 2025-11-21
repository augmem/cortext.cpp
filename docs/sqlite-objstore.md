# sqlite-objstore

`sqlite-objstore` is a SQLite extension that exposes an `objstore(id, data)` virtual table with pluggable backends (native filesystem, OPFS, VFS, SQLite). The architecture, invariants, and roadmap now live in [`docs/architecture.md`](../../sqlite-objstore/docs/architecture.md); transactional nuances, tuning advice, and platform notes live in the docs listed below.

## v1 Feature Checklist

* Virtual table `objstore(id BLOB PRIMARY KEY, data BLOB NOT NULL)` with streaming BLAKE3 hashing.
* Scalar helpers: `objstore_put`, `objstore_put_with_id`, `objstore_get`, `objstore_delete`, `objstore_exists`.
* Backends: native file store, SQLite fallback, OPFS (WASM), and VFS (WASI/embedded).
* Commit/rollback hooks that keep blob storage in sync with SQLite transactions.
* Indexed `rowid` lookups: `rowid = ?` filters hit backend-provided indexes (file backend `rowidx/` tree or SQLite's `objstore_rowidx` table) so `SELECT * FROM objstore WHERE rowid = ?` is O(1).
* WASM/WASI presets plus optional SQL matrix fixtures and cross-language harnesses.
* Example programs and docs covering metadata schemas, cache patterns, and transaction semantics.

Release notes live in [`CHANGELOG.md`](../../sqlite-objstore/CHANGELOG.md).

## Documentation Map

* [`docs/architecture.md`](../../sqlite-objstore/docs/architecture.md) – global invariants, backend descriptions, and platform targets.
* [`docs/getting-started.md`](../../sqlite-objstore/docs/getting-started.md) – configure/build/test walkthroughs.
* [`docs/transactions.md`](../../sqlite-objstore/docs/transactions.md) – commit hooks, snapshot behaviour, and savepoint limitations.
* [`docs/metadata-patterns.md`](../../sqlite-objstore/docs/metadata-patterns.md) – sample schemas that reference `objstore(id)`.
* [`docs/performance.md`](../../sqlite-objstore/docs/performance.md) & [`docs/perf/file-backend.md`](../../sqlite-objstore/docs/perf/file-backend.md) – benchmark harnesses plus acceptance targets.
* [`docs/tuning.md`](../../sqlite-objstore/docs/tuning.md) – chunk size, sync modes, and size-hint guidance for the native backend.
* [`docs/wasm.md`](../../sqlite-objstore/docs/wasm.md) – WASI/OPFS builds, bundle layout, and harnesses.

## Requirements

* CMake **3.28+**
* A C17/C++20 toolchain (Clang/GCC/MSVC)
* Ninja (recommended; provided automatically in CI)
* `sqlite3` development headers (part of Xcode SDK/Homebrew on macOS, `libsqlite3-dev` on Linux)

All builds are **out-of-source**. `CMakePresets.json` pins the binary tree to `build/<preset>` and configures sane defaults (PIC, warnings, sanitizers).

## Configure, Build, Test

See [`docs/getting-started.md`](../../sqlite-objstore/docs/getting-started.md) for a narrated walkthrough that covers building, installing, and embedding the extension.

`CMakePresets.json` exposes the deployment variants described in `docs/architecture.md`. Each preset keeps builds out-of-source under `build/<preset>` and selects the appropriate backends + artifact types:

| Preset             | Variant   | Details                                                                                                                                   |
| ------------------ | --------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| `full-debug`       | Full      | All backends (file, SQLite, OPFS, VFS), builds both shared + static libraries, enables benchmarks/tests with Debug flags                  |
| `full-release`     | Full      | Same feature set as `full-debug` with Release optimizations; use for packaging installers                                                 |
| `full-asan`        | Full      | Debug build with AddressSanitizer + UndefinedBehaviorSanitizer enabled for crash-safety sweeps                                            |
| `wasi-release`     | WASM/VFS  | Uses the WASI toolchain, enables the VFS backend only, builds a static library for use with `wasmtime`-driven tests                       |
| `wasm-release`     | WASM/OPFS | Targets browsers/Workers via Emscripten, enables OPFS + VFS backends, emits a static `.a` suitable for bundling                           |
| `embedded-release` | Embedded  | Release build tuned for constrained devices: file + SQLite + VFS backends, static-only artifacts, benchmarks/tests disabled to save flash |
| `minimal-release`  | Minimal   | SQLite-backend-only footprint (<50KB target), static library only, benchmarks/tests disabled                                              |

Typical native dev loop:

```sh
cmake --preset full-debug
cmake --build --preset full-debug
ctest --preset full-debug
```

If Doxygen is installed you can generate API reference material with:

```sh
cmake --build --preset full-debug --target objstore_doc
```

Use `full-release` for optimized binaries or `full-asan` when chasing undefined behavior.

### Installing the extension

Full builds emit both a shared loadable extension (`libobjstore.{so,dylib}`) and an optional static archive (`libobjstore.a`). Install them alongside headers and the generated CMake package config with:

```sh
cmake --build --preset full-release --target install
# or stage to a custom prefix
cmake --install build/full-release --prefix /tmp/objstore-root
```

The install tree contains:

* `lib/` — `libobjstore.{so,dylib}`, optional `libobjstore.a`, and (when SQLite is bundled) `libsqlite3_amalgamation.a`
* `include/objstore/` — public headers
* `lib/cmake/objstore/ObjstoreConfig.cmake` — provides `objstore::objstore`, `objstore::objstore_shared`, and `objstore::objstore_static`

Preset variants that disable shared builds (`embedded-release`, `minimal-release`, `wasi-release`, `wasm-release`) still install the static archive so downstream toolchains can statically link the extension.

### WASI / WASM builds

The `wasi-release` and `wasm-release` presets from the table above back the helper scripts in `scripts/`:

```sh
scripts/run-wasi-tests.sh          # builds wasi-release and executes wasmtime
scripts/build-wasm.sh              # builds wasm-release (Emscripten)
scripts/check-wasm-size.sh         # reports .wasm/.a size after build
scripts/package-wasm-bundle.sh     # assembles dist/wasm/<id> with fixtures/shims
scripts/run-wasm-cross-lang.sh     # packages + runs Python/Go/Rust/Node harnesses
```

Set `-DOBJSTORE_ENABLE_WASM_MATRIX=ON` when you want the SQL matrix fixtures folded into `ctest`. The flag defaults to `OFF` so native presets stay lean; WASI/WASM builds can opt in when the matrix suite is needed.

### Examples

The `examples/` directory ships two tiny C programs that exercise common metadata patterns:

* `objstore_example_file_metadata` stores a blob plus metadata row and fetches the payload through `objstore_get`.
* `objstore_example_cache` implements a TTL cache that persists values in `objstore` while metadata tracks keys and expiration timestamps.

They build automatically with every preset, and `ctest` runs them under the `example_*` targets. Run them ad hoc via `cmake --build --preset full-debug --target objstore_example_cache`.
The WASI build enables the VFS backend and runs Unity tests under `wasmtime`. The Emscripten/OPFS build enables both VFS and OPFS backends and is designed to be loaded inside a Web Worker. See `docs/wasm.md` for harness instructions (`tests/opfs_worker.js` + `tests/opfs_harness.mjs`).

Set `-DOBJSTORE_ENABLE_WASM_CROSS_LANG=ON` when configuring CMake to register the cross-language harness with CTest so `ctest` can gate release builds on the shared WASM matrix.

### Backend selection

Backends are built and linked via CMake options so you can tailor the extension to each platform:

* `-DOBJSTORE_BACKEND_FILE=ON` (default) builds the native sharded file backend with WAL/manifest recovery.
* `-DOBJSTORE_BACKEND_SQLITE=ON` (default) builds the portable SQLite-backed store plus the shared CRUD test suite.
* `-DOBJSTORE_BACKEND_OPFS=ON` / `-DOBJSTORE_BACKEND_VFS=ON` build the portable filesystem backend for WASM/WASI.

Artifact types follow the `OBJSTORE_BUILD_SHARED` / `OBJSTORE_BUILD_STATIC` cache variables. Full presets enable both so packaging installs a shared extension and a static archive. Embedded/minimal/WASM presets disable the shared target to minimize size or match toolchain limitations.

The Catch2-era tests were replaced by Unity-based smoke and backend suites. Running `ctest --preset full-debug` now exercises both the SQLite fallback backend (duplicated inserts, rollback semantics) and the file backend (CRUD, durability, recovery).

## SQL Usage

The extension exposes a single virtual table, `objstore(id BLOB PRIMARY KEY, data BLOB NOT NULL)`, plus scalar helpers. Typical usage pairs the virtual table with an application-defined metadata table:

```sql
CREATE VIRTUAL TABLE objstore USING objstore();

CREATE TABLE files (
    id BLOB PRIMARY KEY,
    filename TEXT NOT NULL,
    size INTEGER NOT NULL,
    created_at INTEGER NOT NULL
);

WITH new_obj AS (
    SELECT objstore_put(readfile('photo.jpg')) AS id,
           length(readfile('photo.jpg')) AS sz
)
INSERT INTO files (id, filename, size, created_at)
    SELECT id, 'photo.jpg', sz, unixepoch() FROM new_obj;

SELECT filename, objstore_get(id) FROM files WHERE filename = 'photo.jpg';
```

Scalar helpers map directly to the backend abstraction:

* `objstore_put(data)` hashes the payload (BLAKE3) and returns the new id.
* `objstore_put_with_id(id, data)` stores data under an explicit 32-byte id.
* `objstore_get(id)` returns the stored blob.
* `objstore_delete(id)` removes the object and returns 1 when the row existed.
* `objstore_exists(id)` returns 1/0.

BLAKE3 replaced the legacy SHA256 pipeline in Phase 5.11. Existing databases that stored SHA-derived IDs must be rehydrated (re-ingest payloads) before upgrading; see `docs/perf/blake3-migration.md` for guidance.

All writes participate in SQLite transactions via commit hooks, so inserting rows in `objstore` and referencing metadata tables can live in the same `BEGIN … COMMIT` block. **Important:** savepoints are best-effort in v1—after any objstore write, `ROLLBACK TO name` aborts the entire SQLite transaction instead of unwinding just the inner frame. Structure transactions so that objstore writes live outside savepoint-heavy sections (or restart the transaction) to avoid surprises. See `docs/transactions.md` for the full ordering guarantees and limitations.

For additional schema ideas (file catalogs, TTL caches, cleanup workflows) see `docs/metadata-patterns.md`.

## Static Analysis & Tooling

Pass boolean cache variables when configuring to tighten analysis loops:

* `-DOBJSTORE_ENABLE_CLANG_TIDY=ON` runs `clang-tidy` for C/C++ targets.
* `-DOBJSTORE_ENABLE_CPPCHECK=ON` runs `cppcheck`.
* `-DOBJSTORE_ENABLE_WARNINGS_AS_ERRORS=OFF` relaxes CI defaults when experimenting with new compilers.
* `-DOBJSTORE_ENABLE_COVERAGE=ON` enables `-fprofile-instr-generate -fcoverage-mapping` so `scripts/run-coverage.sh` can emit HTML reports.

Example clang-tidy run:

```sh
cmake --preset full-debug -DOBJSTORE_ENABLE_CLANG_TIDY=ON
cmake --build --preset full-debug
```

## Performance & Crash Harnesses

`docs/performance.md` describes the dedicated micro-benchmark, stress, streaming,
and crash tests. Enable them with `-DOBJSTORE_ENABLE_PERF_TESTS=ON`, then run
`ctest -L perf` (benchmarks) or `ctest -R crash_kill_during_commit`
(`SIGKILL`-based recovery validation).

## Repository Layout

* `include/objstore/objstore.h` — public C API surface.
* `src/` — core extension sources.
* `tests/` — Unity-based smoke/regression tests, wired into `ctest`.
* `cmake/` — package config template and helper modules.
* `examples/` — small C samples demonstrating metadata tables + caching helpers, also wired into `ctest`.
* `scripts/` — helper tooling for WASI/WASM packaging.
* `docs/` — user-facing architecture, transaction, tuning, metadata, and platform guides.

## Code of Conduct

We follow the [Contributor Covenant](../../sqlite-objstore/CODE_OF_CONDUCT.md). Report concerns via
the GitHub issues tracker linked in that document.

## License

`sqlite-objstore` is released under the [MIT License](../../sqlite-objstore/LICENSE).

## Continuous Integration

`.github/workflows/ci.yml` exercises the `full-debug` and `full-asan` presets on `ubuntu-latest` plus the `full-debug` preset on `macos-latest`. CI runs `cmake`, `cmake --build`, and `ctest --preset <name>` for each axis and fails on any warning or sanitizer regression.
