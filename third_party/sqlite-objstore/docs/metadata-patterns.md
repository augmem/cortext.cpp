# Metadata Patterns

`sqlite-objstore` only ships the `objstore(id, data)` virtual table. Applications
model ownership, metadata, and lifetimes in their own tables that reference the
immutable object IDs. This guide captures the two most common patterns and
highlights the SQL that keeps metadata in sync with the object store. For the
full list of invariants (BLAKE3 hashing, streaming chunk sizes, transaction
ordering) see `docs/architecture.md`.

## 1. File Catalogs

Applications that treat the object store as a content-addressed archive
typically maintain a lightweight catalog table:

```sql
CREATE VIRTUAL TABLE objstore USING objstore();

CREATE TABLE files (
    id        BLOB PRIMARY KEY REFERENCES objstore(id),
    filename  TEXT NOT NULL,
    size      INTEGER NOT NULL,
    created_at INTEGER NOT NULL
);
```

Writing a file stores the payload in `objstore`, then records metadata:

```sql
WITH new_file AS (
    SELECT objstore_put(?1) AS id, length(?1) AS size
)
INSERT INTO files (id, filename, size, created_at)
SELECT id, ?2, size, strftime('%s','now') FROM new_file;
```

Reading the metadata and payload keeps object data in `objstore`:

```sql
SELECT filename, size, objstore_get(id)
FROM files
WHERE filename = ?1;
```

Deleting a row removes both metadata and the payload:

```sql
WITH removed AS (
    DELETE FROM files WHERE filename = ?1 RETURNING id
)
DELETE FROM objstore WHERE id IN (SELECT id FROM removed);
```

> The `objstore_example_file_metadata` program under `examples/` demonstrates
> the full pattern end-to-end.

## 2. TTL Cache Entries

`objstore` also excels at storing large cache values without bloating SQLite
pages. The metadata table tracks keys and expiration:

```sql
CREATE TABLE cache_entries (
    cache_key  TEXT PRIMARY KEY,
    obj_id     BLOB NOT NULL REFERENCES objstore(id),
    expires_at INTEGER NOT NULL
);
```

Writing a cache entry:

```sql
WITH new_value AS (SELECT objstore_put(?2) AS id)
INSERT INTO cache_entries(cache_key, obj_id, expires_at)
SELECT ?1, id, strftime('%s','now') + ?3 FROM new_value;
```

Reading with automatic expiration:

```sql
SELECT objstore_get(obj_id)
FROM cache_entries
WHERE cache_key = ?1
  AND expires_at >= strftime('%s','now');
```

The `objstore_example_cache` binary shows this pattern, including updates and
cache misses.

## Cleanup & Cascades

Because blobs live outside SQLite pages, metadata tables own lifecycle
management. Common approaches:

- Foreign keys referencing `objstore(id)` plus `ON DELETE CASCADE` ensure
  deleting metadata also removes the payload.
- Periodic jobs can reclaim orphaned objects by scanning `objstore` ids and
  cross-checking metadata tables. `objstore_example_orphan_sweep` automates the
  file-backend sweep once you provide a query that returns every live object id
  from your metadata schema.
- For caches, `DELETE FROM cache_entries WHERE expires_at < strftime('%s','now');`
  releases unused payload IDs before vacuuming `objstore`.

Example sweep command:

```sh
cmake --build --preset full-release --target objstore_example_orphan_sweep
build/full-release/examples/objstore_example_orphan_sweep \
  --db /var/lib/myapp/app.sqlite3 \
  --storage-root /var/lib/myapp/objstore \
  --live-query "SELECT file_id FROM files UNION SELECT obj_id FROM cache_entries"
```

Designing schemas this way keeps objstore small and makes it clear which table
owns each object.
