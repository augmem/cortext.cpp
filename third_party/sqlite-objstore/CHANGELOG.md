# Changelog

All notable changes will be documented in this file. Public releases follow
semver conventions today, but `0.x` still signals that APIs and operational
guidance may tighten before `1.0`.

## Unreleased

- Fixed native range-read error handling so unsatisfied reads surface as
  `SQLITE_RANGE` consistently across SQLite, file, and VFS backends. This keeps
  `objstore_get_range(...)` behavior aligned across implementations.
- Removed default stderr logging during successful backend initialization.
- Tightened packaging so install exports only consumer-facing objstore targets
  and no longer leaks Unity test artifacts into the install prefix.
- Added an install-smoke workflow and downstream CMake consumer check for
  `full-release` builds on Linux and macOS.
- Added a backend-level `lookup_id_by_rowid` hook plus planner support so `rowid = ?`
  queries no longer trigger full scans. The file backend now maintains a
  persistent `rowidx/` directory tree and the SQLite backend mirrors every
  object into an `objstore_rowidx(rowid_prefix, id)` table.
- Documented the rowid contract (rowid = first 8 bytes of the ID) and updated
  the transaction docs to reflect savepoint-aware rollback support:
  `ROLLBACK TO name` now rewinds only the objstore writes staged inside that
  savepoint frame while keeping the outer transaction open.
- Reworked `objstore_txn_snapshot_build()` so snapshot construction stays
  linear in the number of visible staged operations instead of degrading
  quadratically on large transactions.
- Switched file-backend full scans to enumerate the persistent `rowidx/` tree
  instead of recursively walking payload directories, while pruning stale
  row-index entries discovered during scan setup.

## v0.1.0 – Deployment & Docs Prep

- Added build presets for Full/WASM/Embedded/Minimal variants plus shared/static
  install targets.
- Introduced optional WASM SQL matrix toggle and two runnable examples
  (`objstore_example_file_metadata`, `objstore_example_cache`) wired into `ctest`.
- Authored the initial documentation set:
  - `docs/metadata-patterns.md` (catalog and cache schemas)
  - `docs/transactions.md` (backend-first commit semantics)
  - `docs/getting-started.md` (build/install/embed walkthrough)
- Polished public headers with Doxygen comments and added the optional
  `objstore_doc` CMake target for generating reference docs.
