# sqlite-objstore

`sqlite-objstore` is a SQLite extension that adds an `objstore(id, data)`
virtual table plus scalar helpers for storing blobs outside your application
schema. It supports multiple backends: native filesystem, OPFS, VFS, and a
SQLite-backed fallback.

The main motivation is private on-device storage: keeping large blobs local to
the device, out of the main SQLite file, while still preserving transactional
behavior and a simple SQLite-facing API.

## Highlights

* Virtual table `objstore(id BLOB PRIMARY KEY, data BLOB NOT NULL)` with streaming BLAKE3 hashing.
* Scalar helpers: `objstore_put`, `objstore_put_with_id`, `objstore_get`, `objstore_get_range`, `objstore_delete`, `objstore_exists`.
* Backends: native file store, SQLite fallback, OPFS (WASM), and VFS (WASI/embedded).
* Commit/rollback hooks that keep blob storage in sync with SQLite transactions.
* Indexed `rowid` lookups: `rowid = ?` filters hit backend-provided indexes (file backend `rowidx/` tree or SQLite's `objstore_rowidx` table) so `SELECT * FROM objstore WHERE rowid = ?` is O(1).
* WASM/WASI presets plus opt-in SQL matrix fixtures and cross-language harnesses.
* Example programs and docs covering metadata schemas, cache patterns, and transaction semantics.

Release notes live in [`CHANGELOG.md`](CHANGELOG.md).

## Status

`sqlite-objstore` is a release-gated `v0.1.x` prerelease. The code is intended
to be production-capable for the native Linux/macOS path documented here, but
it is not presented as a project with broad public production deployment
history yet.

What is exercised today:

* CI runs debug, ASan, and release-install smoke gates on Linux and macOS.
* The SQLite backend has a crash-kill harness that verifies staged writes do
  not commit after a forced process death during commit.
* The native file backend has recovery coverage for pending-commit replay,
  delete replay, manifest corruption, and stale row-index cleanup.
* WASI/WASM harnesses are available, but they are not part of the default
  native CI matrix yet.

## Documentation Map

- [`docs/architecture.md`](docs/architecture.md) – global invariants, backend descriptions, and platform targets.
- [`docs/getting-started.md`](docs/getting-started.md) – configure/build/test walkthroughs.
- [`docs/transactions.md`](docs/transactions.md) – commit hooks, snapshot behaviour, and savepoint semantics.
- [`docs/ranges.md`](docs/ranges.md) – byte-range reads (SQL + C API).
- [`docs/metadata-patterns.md`](docs/metadata-patterns.md) – sample schemas that reference `objstore(id)`.
- [`docs/performance.md`](docs/performance.md) & [`docs/perf/file-backend.md`](docs/perf/file-backend.md) – benchmark harnesses plus acceptance targets.
- [`docs/tuning.md`](docs/tuning.md) – chunk size, sync modes, and size-hint guidance for the native backend.
- [`docs/wasm.md`](docs/wasm.md) – WASI/OPFS builds, bundle layout, and harnesses.
- [`docs/releasing.md`](docs/releasing.md) – native prerelease checklist and install-smoke workflow.

## Requirements

* CMake **3.28+**
* A C17/C++20 toolchain (Clang/GCC for the release-gated native path)
* Ninja (recommended; provided automatically in CI)
* `sqlite3` development headers (part of Xcode SDK/Homebrew on macOS, `libsqlite3-dev` on Linux)

All builds are **out-of-source**. `CMakePresets.json` pins the binary tree to `build/<preset>` and configures sane defaults (PIC, warnings, sanitizers).

## Configure, Build, Test

See [`docs/getting-started.md`](docs/getting-started.md) for the full build,
install, and embedding flow.

`CMakePresets.json` exposes the deployment variants described in `docs/architecture.md`. Each preset keeps builds out-of-source under `build/<preset>` and selects the appropriate backends + artifact types:

| Preset             | Variant   | Details                                                                                                                                   |
| ------------------ | --------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| `full-debug`       | Full      | All backends (file, SQLite, OPFS, VFS), builds both shared + static libraries, enables benchmarks/tests with Debug flags                  |
| `full-release`     | Full      | Same feature set as `full-debug` with Release optimizations; use for packaging installers                                                 |
| `full-asan`        | Full      | Debug build with AddressSanitizer + UndefinedBehaviorSanitizer enabled for crash-safety sweeps                                            |
| `fuzz-debug`       | Hardening | macOS/Homebrew LLVM build that enables ASan/UBSan plus the libFuzzer target for the SQL/blob/range surface                               |
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
* `lib/cmake/objstore/objstoreConfig.cmake` — provides `objstore::objstore`, `objstore::objstore_shared`, and `objstore::objstore_static`

Preset variants that disable shared builds (`embedded-release`, `minimal-release`, `wasi-release`, `wasm-release`) still install the static archive so downstream toolchains can statically link the extension.

Smoke-check the installed package with:

```sh
cmake --install build/full-release --prefix "$PWD/.install-smoke"
sh scripts/verify-install.sh "$PWD/.install-smoke"
```

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

The `examples/` directory includes two small C programs that show common
metadata patterns:

* `objstore_example_file_metadata` stores a blob plus metadata row and fetches the payload through `objstore_get`.
* `objstore_example_cache` implements a TTL cache that persists values in `objstore` while metadata tracks keys and expiration timestamps.

They build automatically with every preset, and `ctest` runs them under the
`example_*` targets. You can also run them directly with
`cmake --build --preset full-debug --target objstore_example_cache`.

The WASI build enables the VFS backend and runs Unity tests under `wasmtime`.
The Emscripten/OPFS build enables both VFS and OPFS backends and is meant to be
loaded inside a Web Worker. See `docs/wasm.md` for the harness setup
(`tests/opfs_worker.js` + `tests/opfs_harness.mjs`).

Set `-DOBJSTORE_ENABLE_WASM_CROSS_LANG=ON` when configuring CMake to register
the cross-language harness with CTest. The harness is intended for dedicated
WASM validation jobs or pre-release checks, not the default native CI loop.

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
* `objstore_get_range(id, 'bytes=START-END')` returns a byte-range slice (S3
  semantics, inclusive end; suffix ranges like `bytes=-1024`).
* `objstore_delete(id)` removes the object and returns 1 when the row existed.
* `objstore_exists(id)` returns 1/0.

BLAKE3 replaced the legacy SHA256 pipeline in Phase 5.11. Existing databases that stored SHA-derived IDs must be rehydrated (re-ingest payloads) before upgrading; see `docs/perf/blake3-migration.md` for guidance.

All writes participate in SQLite transactions via commit hooks, so `objstore`
writes and metadata-table writes can live in the same `BEGIN ... COMMIT` block.
Nested savepoints are supported too: `SAVEPOINT` opens a new internal frame,
`ROLLBACK TO name` rewinds only the objstore writes from that frame, and
`RELEASE name` folds the frame back into its parent while keeping the outer
transaction open. `docs/transactions.md` covers the ordering and visibility
rules in more detail.

## Current Tradeoffs

Some behaviors are worth knowing up front:

* Full table scans are still O(number of objects). The file backend now
  enumerates the `rowidx/` tree instead of walking payload directories
  directly, but listing every object is still linear work.
* Transaction snapshot construction is linear in the number of staged
  operations visible to the cursor. It no longer rescans the growing snapshot
  for every log entry.
* Backend-first commits can still leave orphaned blobs behind if the process
  dies after backend flush succeeds but before SQLite finishes its own commit.
  Recovery tests cover staged replay, and long-lived deployments should run the
  `objstore_example_orphan_sweep` utility with a metadata-specific live-id
  query to reclaim unreferenced payloads.

For additional schema ideas (file catalogs, TTL caches, cleanup workflows) see `docs/metadata-patterns.md`.

## Static Analysis & Tooling

Pass boolean cache variables when configuring to tighten analysis loops:

* `-DOBJSTORE_ENABLE_CLANG_TIDY=ON` runs `clang-tidy` for C/C++ targets.
* `-DOBJSTORE_ENABLE_CPPCHECK=ON` runs `cppcheck`.
* `-DOBJSTORE_ENABLE_FUZZERS=ON` builds the `objstore_sql_surface_fuzzer`
  libFuzzer target for the SQL/blob/range surface when using a Clang toolchain
  that ships the libFuzzer runtime.
* `-DOBJSTORE_ENABLE_WARNINGS_AS_ERRORS=OFF` relaxes CI defaults when experimenting with new compilers.
* `-DOBJSTORE_ENABLE_COVERAGE=ON` enables `-fprofile-instr-generate -fcoverage-mapping` so `scripts/run-coverage.sh` can emit HTML reports.

Example clang-tidy run:

```sh
cmake --preset full-debug -DOBJSTORE_ENABLE_CLANG_TIDY=ON
cmake --build --preset full-debug
```

On macOS with Homebrew LLVM installed, the dedicated fuzzing loop is:

```sh
cmake --preset fuzz-debug
cmake --build --preset fuzz-debug --target objstore_sql_surface_fuzzer
build/fuzz-debug/tests/objstore_sql_surface_fuzzer -runs=1
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

We follow the [Contributor Covenant](CODE_OF_CONDUCT.md). Report concerns via
the GitHub issues tracker linked in that document.

## License

`sqlite-objstore` is released under the [Apache License 2.0](LICENSE).
Bundled third-party components are documented in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Continuous Integration

`.github/workflows/ci.yml` runs `full-debug` and `full-asan` on
`ubuntu-latest`, plus `full-debug` on `macos-latest`. Each job runs `cmake`,
`cmake --build`, and `ctest --preset <name>`.

The workflow also stages a `full-release` install on Linux and macOS, then
configures a small downstream consumer against the installed CMake package to
catch packaging regressions.
