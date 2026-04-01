# Architecture & Invariants

`sqlite-objstore` is a SQLite extension that turns the database into a
content-addressed object store. The core extension exposes a single virtual
table, `objstore(id BLOB PRIMARY KEY, data BLOB NOT NULL)`, and pushes all
metadata responsibilities to normal SQLite tables so applications stay in
control of schemas, indexes, and access patterns.

This document replaces the historical `design.md`/`plans/` bundle and captures
the durable facts contributors and integrators need before diving into code.

## Executive Summary

- **Portable storage** – a single extension binary works across native hosts,
  WASI runtimes, WebAssembly/OPFS workers, and embedded targets.
- **Content-addressed** – BLAKE3-derived 32-byte IDs give deduplication and
  integrity guarantees by default; callers can still supply explicit IDs for
  mutable workflows.
- **Streaming I/O** – every read/write path is chunked (default 64 KiB) so
  storing multi‑GiB objects never requires matching RAM and works inside WASM
  limits.
- **Transactional** – backend transactions commit **before** SQLite finishes,
  keeping SQL metadata and blob bytes in sync. Rollbacks drop staged payloads,
  and cursors honour snapshot semantics.
- **Pluggable backends** – native sharded filesystem backend, portable OPFS/VFS
  backend, and SQLite-backed storage cover every deployment tier.

## Global Invariants

1. **Schema** – the virtual table only exposes `id` and `data`. All metadata
   lives in first-class SQLite tables that reference `objstore(id)` via foreign
   keys. See `docs/metadata-patterns.md` for idioms.
2. **Hashing** – unless the caller supplies an explicit key, `objstore_put`
   computes `BLAKE3(data)` and uses it as the immutable ID. Inserting identical
   bytes becomes a no-op; mismatched bytes raise `SQLITE_CONSTRAINT`.
3. **Chunking** – helpers default to 64 KiB reads/writes
   (`OBJSTORE_DEFAULT_CHUNK_SIZE`). Backends must stream data and are not allowed
   to buffer entire payloads.
4. **Transactions** – writes stream into backend-owned staging areas. Commit
   hooks call `commit_staged()` **before** SQLite finalizes the transaction.
   Rollbacks drop staged files/rows, keeping storage consistent. Savepoints are
   conservative: `ROLLBACK TO` behaves like `ROLLBACK` once an objstore write
   has occurred. Details live in `docs/transactions.md`.
5. **Snapshot isolation** – every connection owns an `objstore_txn_log`. Cursors
   capture the log sequence at `xFilter` so mid-scan mutations stay invisible
   until a new cursor opens. Scalar helpers always consult the latest log, so
   SQL inside the same transaction sees staged results immediately.
6. **Rowid mapping** – the leading eight bytes of every ID are treated as a
   big-endian integer. `rowid = ?` constraints use backend-provided
   `lookup_id_by_rowid()` hooks (native + SQLite backends) or fall back to
   scans. Helper functions live in `include/objstore/backend.h`.
7. **Resource bounds** – staging buffers, manifest builders, and SQLite backend
   queues all stay bounded (single chunk buffer per writer, 16 MiB queue cap,
   etc.) to preserve embedded/WASM friendliness.

## System Architecture

```
┌────────────────────────┐
│ Application / SQLite   │
│  (SQL, scalar calls)   │
└─────────────┬──────────┘
              │ virtual table API
              ▼
┌────────────────────────┐
│ sqlite-objstore        │
│  • Module & scalars    │
│  • Object manager      │
│  • Txn log + snapshots │
│  • Backend abstraction │
└─────────────┬──────────┘
              │ backend vtable (put/get/scan/txn hooks)
  ┌───────────┼───────────────┬──────────────┐
  ▼           ▼               ▼              ▼
File backend  OPFS/VFS backend SQLite backend (table fallback) ...
```

- **Module & scalars** expose SQL entry points (virtual table operations,
  `objstore_put`, `objstore_get`, etc.) and share a reference-counted
  `objstore_connection` per `sqlite3*` handle.
- **Object manager** handles validation, streaming, and hashing. It converts
  SQLite BLOB arguments into `objstore_stream_reader` structs (with `size_hint`)
  so backends can preallocate space.
- **Transaction log** records pending PUT/DELETE metadata with monotonically
  increasing sequence numbers. Cursors turn the log into read-only snapshots,
  enabling consistent scans even while writes occur.
- **Backend abstraction** standardizes environment lifetimes, staged writers,
  transaction hooks, cursors, and optional accelerators like
  `lookup_id_by_rowid`.

## Backend Summaries

### Native File Backend

- Directory layout:
  ```
  <root>/objects/<first-2-hex>/<full-id>.dat
  <root>/.staging/active/<txn-id>/{put,delete,manifest.log}
  <root>/.staging/commit/<txn-id>/
  <root>/rowidx/<first-2-rowid-hex>/<rowid-hex>/<full-id>
  ```
- Writes stream into `.staging/active/<txn>/put/<id>.dat` while manifest entries
  accumulate in memory. `commit_staged()` fsyncs `manifest.log`, atomically
  renames the directory into `.staging/commit/<txn>`, replays the manifest into
  `objects/` + `rowidx/`, and removes the commit directory once replay
  succeeds.
- Deletes drop committed payloads if present and remove rowidx files.
- Crash recovery clears stray `.staging/active` directories, replays remaining
  `.staging/commit` manifests, and tolerates partially replayed objects.
- Config knobs (PRAGMA or `objstore_config`): `storage_root`, `shard_width`,
  `sync_mode` (`full|metadata|off`), and `chunk_size`.

### Portable OPFS / VFS Backend

- Shared implementation (`backend_portable`) used by:
  - **OPFS** builds (Emscripten, browser Web Workers). Default root
    `/opfs/objstore`, requires `FileSystemSyncAccessHandle`.
  - **VFS/WASI** builds (`wasi-release` preset). Default root `/data/objstore`
    (or `objstore` outside WASI).
- Reuses the same staging/manifest layout as the native backend but only
  depends on ANSI C/stdio calls, making it viable in constrained environments.
- WASM bundles include SQL fixtures and a WASI matrix harness
  (`docs/wasm.md` covers build + packaging steps).

### SQLite Backend (Fallback)

- Stores objects inside tables in the host database:
  ```sql
  CREATE TABLE objstore_data(
      id   BLOB PRIMARY KEY,
      data BLOB NOT NULL
  );

  CREATE TABLE objstore_rowidx(
      rowid_prefix BLOB NOT NULL,
      id           BLOB NOT NULL,
      PRIMARY KEY(rowid_prefix, id)
  );
  ```
- Mutations queue up in memory during SQL statement execution to avoid
  re-entrancy. The queue enforces a strict byte cap and is flushed inside the
  commit hook.
- Read helpers consult the queue first so in-flight writes are visible to later
  statements in the same transaction.

## Platform Targets & Presets

| Preset             | Backends           | Primary Use                                              |
| ------------------ | ------------------ | -------------------------------------------------------- |
| `full-*`           | File + OPFS + VFS + SQLite | Desktop/server builds, shared + static artifacts. |
| `full-asan`        | Same as `full-debug` | Sanitizer sweeps (Address/Undefined).                |
| `embedded-release` | File + VFS + SQLite (static only) | Flash/SoC deployments with tight RAM budgets. |
| `minimal-release`  | SQLite backend only | Ultra-small static lib (<50 KiB).                       |
| `wasi-release`     | VFS + SQLite       | WASI/CLI runtimes driven by `wasmtime`.                  |
| `wasm-release`     | OPFS + VFS         | Browser/Web Worker bundles, `<100 KB` goal.              |

All presets live in `CMakePresets.json`, and `docs/getting-started.md`
walks through the typical `cmake --preset full-debug` workflow.

## Document Map

- `docs/getting-started.md` – build/install/embedding walkthrough.
- `docs/metadata-patterns.md` – how to model ownership and metadata.
- `docs/transactions.md` – commit ordering, rollback behaviour, snapshot rules,
  and orphan handling.
- `docs/performance.md` & `docs/perf/file-backend.md` – benchmark harnesses,
  acceptance targets, and sample results.
- `docs/tuning.md` – chunk size, sync-mode, and size-hint recommendations.
- `docs/wasm.md` – WASM/WASI builds, OPFS requirements, and matrix harness.

Treat this file + the docs above as the public source of truth going forward.

