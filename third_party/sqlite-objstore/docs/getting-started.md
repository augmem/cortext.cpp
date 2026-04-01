# Getting Started

This guide walks through building `sqlite-objstore`, running the test suite, and
embedding the virtual table inside a C application.

## 1. Build & Test

```sh
git clone https://github.com/.../sqlite-objstore.git
cd sqlite-objstore

cmake --preset full-debug
cmake --build --preset full-debug
ctest --preset full-debug
```

The `full-debug` preset enables every backend, builds both shared + static
libraries, and runs the unit tests plus the two example binaries. Use
`full-release` for an optimized build or `full-asan` when tracking down memory
bugs.

## 2. Install

Install headers, libraries, and CMake package files with:

```sh
cmake --build --preset full-release --target install
# or
cmake --install build/full-release --prefix /your/staging/prefix
```

The install tree contains:

- `include/objstore/*.h` — public headers (`objstore/objstore.h` is the primary
  entry point).
- `lib/libobjstore.{so,dylib}` — shared library (disabled on WASM builds).
- `lib/libobjstore.a` — static archive (enable/disable via
  `OBJSTORE_BUILD_STATIC`).
- `lib/cmake/objstore/ObjstoreConfig.cmake` — importable targets
  (`objstore::objstore`, `objstore::objstore_static`).

## 3. Embed in C

`sqlite-objstore` registers itself as a virtual table module. A typical flow:

```c
#include <objstore/objstore.h>
#include <sqlite3.h>

sqlite3 *db = NULL;
sqlite3_open(":memory:", &db);

objstore_register(db, NULL);                   // wires up the module + scalars
sqlite3_exec(db, "CREATE VIRTUAL TABLE objstore USING objstore();", NULL, NULL, NULL);

// Write a blob + metadata
sqlite3_exec(
    db,
    "WITH new_file AS (SELECT objstore_put(x'01020304') AS id)"
    "INSERT INTO files(id) SELECT id FROM new_file;",
    NULL, NULL, NULL);
```

Once registered, applications can use both the virtual table and the scalar
functions (`objstore_put`, `objstore_get`, …) in regular SQL. The example
programs under `examples/` demonstrate metadata catalogs and TTL caches built on
top of the virtual table.

## 4. Explore the Patterns

- `docs/metadata-patterns.md` covers canonical schemas and cleanup strategies.
- `docs/transactions.md` describes the commit hooks, savepoint caveats, and the
  backend-first commit ordering.
- `README.md` documents presets, WASM builds, and additional tooling knobs.

