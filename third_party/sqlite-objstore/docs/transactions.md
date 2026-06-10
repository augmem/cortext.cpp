# Transaction Semantics

`sqlite-objstore` coordinates its backends with SQLite transactions so SQL code
can treat `objstore` like any other table. This document summarizes the current
behavior and the guarantees provided by the commit hooks.

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

Savepoints follow SQLite's nested transaction model:

- `SAVEPOINT` pushes a new frame in both the transaction log and the active
  backend transaction.
- `ROLLBACK TO name` rewinds only the objstore writes staged after that
  savepoint, restoring the visible state from the parent frame while keeping
  the outer transaction open.
- `RELEASE name` merges the inner frame into its parent. The staged bytes stay
  pending until the outermost `COMMIT` or `ROLLBACK`.

In practice, that means you can mix metadata updates, nested savepoints, and
partial retries without restarting the whole transaction.

### Snapshot visibility

Every connection owns an `objstore_txn_log` that records PUT/DELETE metadata in
monotonic order. When a virtual-table cursor starts (`xFilter`) it captures the
current log sequence and replays only entries ≤ that snapshot. This gives you
the usual SQL behavior:

- Existing cursors continue scanning the snapshot they started with, even if new
  writes happen mid-scan.
- Subsequent cursors (or scalar helpers) see whatever is staged at the moment
  they begin because they consult the live log first and only hit the backend
  when no pending entry exists.
- Point lookups (`WHERE id = ?` or `rowid = ?`) follow the same rule, so in-row
  metadata joins always see the latest data whereas long-running scans stay
  stable.

Snapshot construction is linear in the number of staged operations visible to
the cursor. Large transactions still cost proportionally more to snapshot, but
the builder no longer performs nested rescans as the log grows.

## Orphaned Objects

Because the backend commits before SQLite, a crash after the commit hook returns
success but before SQLite finishes can leave orphaned payloads behind: blobs
with no metadata rows pointing at them. They are safe, but they still take up
space. In long-lived deployments, plan to run a periodic cleanup job. The
`objstore_example_orphan_sweep` utility can scan a file-backend storage root and
delete payloads that are missing from a metadata-specific live-id query.

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
SELECT objstore_get_range(id, 'bytes=0-1023') FROM files WHERE filename = 'photo.jpg';
COMMIT; -- backend flushes first, SQLite commits second
```

If any statement fails or `ROLLBACK` is issued, the `files` row disappears and
the staged payload is dropped.

Savepoints behave as expected too:

```sql
BEGIN;
INSERT INTO objstore(id, data) VALUES(?1, ?2);
SAVEPOINT s1;
DELETE FROM objstore WHERE id = ?1;
ROLLBACK TO s1; -- the object becomes visible again
COMMIT;
```

## References

- `docs/metadata-patterns.md` demonstrates how metadata tables reference
  `objstore(id)` inside transactions.
- `docs/architecture.md` expands on global invariants (BLAKE3 IDs, streaming
  buffers, staging layout) and backend-specific recovery behaviour.
