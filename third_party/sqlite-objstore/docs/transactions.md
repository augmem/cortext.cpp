# Transaction Semantics

`sqlite-objstore` coordinates its backends with SQLite transactions so SQL code
can treat `objstore` like any other table. This document summarizes the current
behavior, the guarantees provided by the commit hooks, and the known
limitations.

## Backend-First Commits

Every connection registers commit/rollback hooks via `sqlite3_commit_hook()` and
`sqlite3_rollback_hook()`. Writes stream into backend-managed staging areas until
SQLite asks to commit:

1. SQL statements enqueue PUT/DELETE metadata into an in-memory log and stream
   payload bytes into backend staging.
2. When `COMMIT` occurs, the commit hook runs *before* SQLite finalizes the
   transaction. The backend promotes staged objects to durable storage and
   flushes its write-ahead log.
3. Only after the backend succeeds does SQLite complete the commit and release
   database locks.
4. If the backend fails, the hook returns non-zero which aborts the SQLite
   commit. SQLite then calls the rollback hook, which drops all staged payloads.

This ordering guarantees that SQLite metadata rows and blob bytes become visible
atomically. It also means backend failures never leave half-written metadata in
the database—`COMMIT` simply fails.

## Rollbacks, Savepoints, and Snapshots

Rollback behavior is symmetric:

- `ROLLBACK` runs the rollback hook, which deletes staged objects and clears the
  per-connection log so previously inserted IDs are invisible.
- Reads inside the same transaction always see pending writes because the log
  is consulted before touching the backend.

Savepoints currently behave conservatively:

- `SAVEPOINT` increments an internal depth counter so objstore knows how many
  frames exist.
- `ROLLBACK TO` discards all staged work for the entire transaction (partial
  rewinds are not supported in v1).
- `RELEASE` is a no-op for the backend because data was already staged.

In other words, nested savepoints work, but rolling back to an inner savepoint
aborts the whole transaction instead of unwinding to an intermediate state.

**Best practice:** wrap objstore writes in their own top-level transactions or
only issue `ROLLBACK TO` before any objstore calls. Once a write has touched the
object store, treat `ROLLBACK TO` as equivalent to `ROLLBACK` because the backend
must discard every staged byte to stay crash-safe.

### Snapshot visibility

Every connection owns an `objstore_txn_log` that records PUT/DELETE metadata in
monotonic order. When a virtual-table cursor starts (`xFilter`) it captures the
current log sequence and replays only entries ≤ that snapshot. This yields
familiar SQL semantics:

- Existing cursors continue scanning the snapshot they started with, even if new
  writes happen mid-scan.
- Subsequent cursors (or scalar helpers) see whatever is staged at the moment
  they begin because they consult the live log first and only hit the backend
  when no pending entry exists.
- Point lookups (`WHERE id = ?` or `rowid = ?`) follow the same rule, so in-row
  metadata joins always see the latest data whereas long-running scans stay
  stable.

## Orphaned Objects

Because the backend commits before SQLite, a backend crash after the commit hook
returns success but before SQLite finishes can leave “orphaned” payloads (blobs
without metadata rows). These orphans are safe—the backend is crash consistent—
but they consume storage. Long-term deployments should run periodic janitors
that scan `objstore(id)` for references that no longer exist in metadata tables.

## Example

```sql
BEGIN;
WITH new_photo AS (
    SELECT objstore_put(readfile('photo.jpg')) AS id,
           length(readfile('photo.jpg')) AS size
)
INSERT INTO files(id, filename, size, created_at)
SELECT id, 'photo.jpg', size, strftime('%s','now') FROM new_photo;
SELECT objstore_get(id) FROM files WHERE filename = 'photo.jpg'; -- sees staged bytes
COMMIT; -- backend flushes first, SQLite commits second
```

If any statement fails or `ROLLBACK` is issued, the `files` row disappears and
the staged payload is dropped.

## References

- `docs/metadata-patterns.md` demonstrates how metadata tables reference
  `objstore(id)` inside transactions.
- `docs/architecture.md` expands on global invariants (BLAKE3 IDs, streaming
  buffers, staging layout) and backend-specific recovery behaviour.

