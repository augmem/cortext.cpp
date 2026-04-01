# Changelog

All notable changes will be documented in this file. Version numbers follow
semver once v1.0 ships; earlier tags track milestones in the v1 roadmap.

# Changelog

All notable changes will be documented in this file. Version numbers follow
semver once v1.0 ships; earlier tags track milestones in the v1 roadmap.

## Unreleased

- Added a backend-level `lookup_id_by_rowid` hook plus planner support so `rowid = ?`
  queries no longer trigger full scans. The file backend now maintains a
  persistent `rowidx/` directory tree and the SQLite backend mirrors every
  object into an `objstore_rowidx(rowid_prefix, id)` table.
- Documented the rowid contract (rowid = first 8 bytes of the ID) and emphasized
  the v1 savepoint limitation (any `ROLLBACK TO` after an objstore write aborts
  the entire transaction) across the design doc, README, and `docs/transactions.md`.

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

