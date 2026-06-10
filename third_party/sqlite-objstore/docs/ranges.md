# Byte Range Reads

`sqlite-objstore` supports S3-style byte range reads at both the C API and SQL
helper layers. Ranges are **single range only** (no multi-range).

## SQL helper

```
SELECT objstore_get_range(id, 'bytes=0-99') FROM files;
```

Accepted forms (whitespace ignored):

- `bytes=START-END` (inclusive end)
- `bytes=START-` (to end of object)
- `bytes=-SUFFIX` (last N bytes)

Behavior:

- Unsatisfiable ranges return `NULL`.
- Malformed range strings raise `SQLITE_RANGE`.
- Missing objects return `NULL` (same as `objstore_get`).

## C API

Use `objstore_object_read_stream_range` or `objstore_object_read_blob_range`
with `objstore_range_spec`:

```c
objstore_range_spec range = { .has_start = 1, .has_end = 1, .start = 0, .end = 99 };
rc = objstore_object_read_stream_range(conn, txn, &id, (sqlite3_uint64)-1,
                                       &range, &writer);
```

Semantics match S3:

- Inclusive end when both start/end set.
- `has_start` only == `start-`
- `has_end` only == suffix `-end`

Unsatisfiable ranges return `SQLITE_RANGE`.
