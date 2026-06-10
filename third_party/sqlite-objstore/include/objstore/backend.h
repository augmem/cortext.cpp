#ifndef OBJSTORE_BACKEND_H
#define OBJSTORE_BACKEND_H

#ifdef __cplusplus
extern "C"
{
#endif

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include <sqlite3.h>

#include "objstore/objstore.h"

#define OBJSTORE_ID_SIZE 32
#define OBJSTORE_MIN_CHUNK_SIZE 1024u
#define OBJSTORE_MAX_CHUNK_SIZE (64u * 1024u)
#define OBJSTORE_DEFAULT_CHUNK_SIZE 65536u
#define OBJSTORE_ROWID_PREFIX_SIZE 8u
#define OBJSTORE_ROWID_HEX_CHARS (OBJSTORE_ROWID_PREFIX_SIZE * 2u)

  /**
   * Backend API Contract (all functions return SQLite result codes):
   * Section: Lifecycle & Transactions
   *   - open_env: Creates an env tied to the sqlite3* connection. Returns
   *     SQLITE_OK or an error (SQLITE_IOERR*, SQLITE_NOMEM, etc.).
   *   - begin_txn: Mirrors SQLite xBegin; returns SQLITE_BUSY if another txn
   * is active. Transactions must be idempotent: rollback after begin must
   * never leak partially written objects.
   *   - commit_txn: Must durably flush backend state (fsync/log replay) before
   *     returning SQLITE_OK. Any failure must map to SQLITE_IOERR*,
   * SQLITE_FULL, etc., and caller will abort the SQLite commit.
   *   - rollback_txn: Best-effort cleanup; must swallow SQLITE_NOTFOUND so
   * that callers can invoke it during error unwinding.
   * Section: Staging
   *   - staged_write_begin: Allocates backend-owned staging state tied to the
   *     active transaction and returns an opaque writer handle.
   *   - staged_write_push: Streams bytes from SQLite into the staging handle.
   *     This is invoked repeatedly as the object manager hashes content. It
   *     must honor chunk-size guarantees and return SQLITE_OK / SQLITE_IOERR*
   *     the same way backend->put previously behaved.
   *   - staged_write_finalize: Seals the staging handle once hashing
   * completes. The object manager passes the finalized objstore_id (or NULL to
   * discard the payload on error). After finalize the handle is invalid and
   * backend must ensure the payload participates in commit/rollback.
   *   - commit_staged: Atomically promotes staged payloads (rename
   * directories, replay manifests) before SQLite commits. Returns SQLite-style
   * errors so SQLite can abort its own commit when promotion fails.
   *   - rollback_staged: Drops staged payloads without touching committed
   *     storage. Must tolerate multiple invocations and partial cleanup.
   *
   * Section: Object I/O
   *   - put: Streams data via objstore_stream_reader; duplicate IDs must
   * compare bytes and either return SQLITE_OK (same content) or
   * SQLITE_CONSTRAINT. Readers signal EOF via SQLITE_DONE.
   *   - get: Streams bytes via objstore_stream_writer; return SQLITE_NOTFOUND
   *     when object is missing, SQLITE_IOERR_* on storage failures.
   *   - get_range: Optional. Streams a byte range into the writer (offset +
   *     length resolved by caller). Return SQLITE_NOTFOUND when missing.
   *   - get_size: Returns payload size in bytes for id; return SQLITE_NOTFOUND
   *     when object is missing.
   *   - delete_fn: Idempotent delete; return SQLITE_OK even if object was
   * absent.
   *
   * Section: Cursors
   *   - scan_open/next/close expose ID-only iteration. scan_next must copy
   * exactly OBJSTORE_ID_SIZE bytes into out_id or return SQLITE_DONE at EOF.
   *
   * Common Error Mapping
   *   - SQLITE_MISUSE: invalid arguments / state bugs
   *   - SQLITE_BUSY: backend-level contention
   *   - SQLITE_IOERR_*: filesystem/WAL failures
   *   - SQLITE_FULL / SQLITE_NOMEM: resource exhaustion
   *   - SQLITE_NOTFOUND: missing object or cursor exhausted
   */

  typedef struct objstore_backend_env objstore_backend_env;
  typedef struct objstore_backend_txn objstore_backend_txn;
  typedef struct objstore_backend_cursor objstore_backend_cursor;
  typedef struct objstore_backend_staged_writer objstore_backend_staged_writer;

  typedef struct objstore_id
  {
    uint8_t bytes[OBJSTORE_ID_SIZE];
  } objstore_id;

  static inline sqlite3_int64
  objstore_rowid_from_id (const objstore_id *id)
  {
    if (id == NULL)
      {
        return 0;
      }
    sqlite3_uint64 value = 0;
    for (size_t i = 0; i < OBJSTORE_ROWID_PREFIX_SIZE; ++i)
      {
        value = (value << 8) | (sqlite3_uint64)id->bytes[i];
      }
    return (sqlite3_int64)value;
  }

  static inline void
  objstore_rowid_prefix_from_id (const objstore_id *id,
                                 unsigned char out_prefix[OBJSTORE_ROWID_PREFIX_SIZE])
  {
    if (id == NULL || out_prefix == NULL)
      {
        return;
      }
    memcpy (out_prefix, id->bytes, OBJSTORE_ROWID_PREFIX_SIZE);
  }

  static inline void
  objstore_rowid_to_bytes (sqlite3_int64 rowid,
                           unsigned char out_prefix[OBJSTORE_ROWID_PREFIX_SIZE])
  {
    if (out_prefix == NULL)
      {
        return;
      }
    sqlite3_uint64 value = (sqlite3_uint64)rowid;
    for (size_t i = 0; i < OBJSTORE_ROWID_PREFIX_SIZE; ++i)
      {
        const size_t index = OBJSTORE_ROWID_PREFIX_SIZE - 1u - i;
        out_prefix[index] = (unsigned char)(value & 0xFFu);
        value >>= 8;
      }
  }

  typedef int (*objstore_stream_pull_fn) (void *ctx, void *buffer,
                                          size_t capacity, size_t *nread);
  typedef int (*objstore_stream_push_fn) (void *ctx, const void *buffer,
                                          size_t nread);

  typedef struct objstore_stream_reader
  {
    void *ctx;
    objstore_stream_pull_fn pull; /* Must return SQLITE_OK or SQLITE_DONE. */
    sqlite3_int64 size_hint; /* -1 when unknown; otherwise bytes remaining. */
  } objstore_stream_reader;

  typedef struct objstore_stream_writer
  {
    void *ctx;
    objstore_stream_push_fn push; /* Return SQLITE_OK or an error to abort. */
  } objstore_stream_writer;

  /**
   * Backend implementations must return canonical SQLite result codes:
   *   - SQLITE_OK on success.
   *   - SQLITE_NOTFOUND when an object does not exist (get/delete/scan_next).
   *   - SQLITE_CONSTRAINT when inserting an existing ID with different bytes.
   *   - SQLITE_IOERR[_*], SQLITE_NOMEM, SQLITE_BUSY, etc. for system failures.
   *   - SQLITE_MISUSE for programmer/sequence errors (NULL pointers, double
   *     commits, calling APIs out of order, etc.).
   *
   * Streaming callbacks must honor objstore_effective_chunk_size() caps and
   * avoid reading/writing past the requested byte counts.  Backends are
   * responsible for durability: commit_txn must report an error if any WAL or
   * fsync step fails so SQLite can abort its own commit.
   */
  typedef struct objstore_backend
  {
    objstore_backend_kind kind;
    const char *name;

    /**
     * Creates or reuses a backend environment that outlives individual
     * transactions. Must initialize chunk sizes from config and return
     * SQLITE_OK on success. May return SQLITE_IOERR/SQLITE_NOMEM/etc.
     */
    int (*open_env) (sqlite3 *db, const objstore_config *config,
                     objstore_backend_env **out_env);

    /**
     * Releases resources allocated by open_env. Must tolerate NULL and never
     * throw.
     */
    void (*close_env) (objstore_backend_env *env);

    /**
     * Starts a backend transaction aligned with SQLite's xBegin. Must return
     * SQLITE_OK and populate out_txn, or propagate SQLite-style errors.
     */
    int (*begin_txn) (objstore_backend_env *env,
                      objstore_backend_txn **out_txn);

    /**
     * Completes a backend transaction. Must flush WAL/object data before
     * returning SQLITE_OK so SQLite can safely commit. Propagate IO/locking
     * failures as SQLite codes.
     */
    int (*commit_txn) (objstore_backend_txn *txn);

    /**
     * Rolls back an in-flight transaction. Must delete any uncommitted objects
     * and tolerate NULL txn pointers.
     */
    void (*rollback_txn) (objstore_backend_txn *txn);

    /**
     * Optional nested savepoint hooks. Backends that support partial
     * SAVEPOINT rollback must mirror SQLite's push/release/rollback behavior
     * for staged state without aborting the outer transaction.
     */
    int (*savepoint_begin) (objstore_backend_txn *txn);
    int (*savepoint_release) (objstore_backend_txn *txn);
    int (*savepoint_rollback) (objstore_backend_txn *txn);

    /**
     * Begins a staged write. Returns an opaque writer handle tied to txn. The
     * handle remains valid until staged_write_finalize is called.
     */
    int (*staged_write_begin) (objstore_backend_txn *txn,
                               objstore_backend_staged_writer **out_writer);

    /**
     * Pushes bytes into an active staged write. Backends must consume exactly
     * nread bytes (when nread>0) or return an error. Partial writes must be
     * surfaced as SQLITE_IOERR_* or SQLITE_FULL.
     */
    int (*staged_write_push) (objstore_backend_staged_writer *writer,
                              const void *buffer, size_t nread);

    /**
     * Finalizes a staged write. Passing a non-NULL id seals the payload under
     * that objstore_id; passing NULL discards any buffered bytes and releases
     * resources. After this call the writer handle must not be reused.
     */
    int (*staged_write_finalize) (objstore_backend_staged_writer *writer,
                                  const objstore_id *id);

    /**
     * Optional hint invoked before streaming to inform the backend about the
     * expected payload size. Backends may ignore the request (NULL pointer)
     * but those that support it can preallocate disk space or enforce limits.
     */
    int (*staged_write_set_size_hint) (objstore_backend_staged_writer *writer,
                                       sqlite3_int64 size_hint);

    /**
     * Promotes staged payloads into committed storage. Must execute before
     * SQLite commits and report errors so the SQL transaction can be aborted.
     */
    int (*commit_staged) (objstore_backend_txn *txn);

    /**
     * Erases staged payloads without touching committed data. Invoked on
     * rollback/savepoint unwind and must tolerate NULL txn pointers.
     */
    void (*rollback_staged) (objstore_backend_txn *txn);

    /**
     * Writes an object identified by id. Duplicate IDs must verify the stored
     * payload and return SQLITE_CONSTRAINT when bytes differ. Returns
     * SQLITE_OK on success, SQLITE_IOERR/SQLITE_NOMEM/etc. on failure.
     */
    int (*put) (objstore_backend_txn *txn, const objstore_id *id,
                const objstore_stream_reader *reader);

    /**
     * Streams an object's payload into writer. Returns SQLITE_NOTFOUND when no
     * object exists for id.
     */
    int (*get) (objstore_backend_txn *txn, const objstore_id *id,
                const objstore_stream_writer *writer);

    /**
     * Optional range read: streams a byte range starting at offset, spanning
     * length bytes. Returns SQLITE_NOTFOUND when object is missing.
     */
    int (*get_range) (objstore_backend_txn *txn, const objstore_id *id,
                      sqlite3_uint64 offset, sqlite3_uint64 length,
                      const objstore_stream_writer *writer);

    /**
     * Returns the stored payload size in bytes. Returns SQLITE_NOTFOUND when
     * object is missing.
     */
    int (*get_size) (objstore_backend_txn *txn, const objstore_id *id,
                     sqlite3_int64 *out_size);

    /**
     * Removes the object identified by id. Missing rows should return
     * SQLITE_NOTFOUND so callers can translate to SQL-level errors.
     */
    int (*delete_fn) (objstore_backend_txn *txn, const objstore_id *id);

    /**
     * Checks whether an object exists without touching payload bytes. Returns
     * SQLITE_OK when the object is present, SQLITE_NOTFOUND otherwise.
     */
    int (*exists) (objstore_backend_txn *txn, const objstore_id *id);

    /**
     * Optional acceleration hook for mapping a rowid (the big-endian encoding of
     * the first 8 bytes of objstore_id) back to the full objstore_id. Backends
     * that do not supply this hook fall back to cursor scans. Implementations
     * must return SQLITE_OK and populate out_id when a matching object exists,
     * SQLITE_NOTFOUND when no object matches, or a SQLite error code on failure.
     */
    int (*lookup_id_by_rowid) (objstore_backend_txn *txn,
                               sqlite3_int64 rowid, objstore_id *out_id);

    /**
     * Creates a cursor for ID-only scans. Must allocate cursor state and
     * return SQLITE_OK or propagate allocation/storage failures.
     */
    int (*scan_open) (objstore_backend_txn *txn,
                      objstore_backend_cursor **out_cursor);

    /**
     * Advances the cursor. Returns SQLITE_OK and fills out_id when another ID
     * is available, SQLITE_DONE at EOF, or a SQLite error code on failure.
     */
    int (*scan_next) (objstore_backend_cursor *cursor, objstore_id *out_id);

    /**
     * Releases cursor resources. Must tolerate NULL.
     */
    void (*scan_close) (objstore_backend_cursor *cursor);
  } objstore_backend;

  const objstore_backend *
  objstore_backend_by_kind (objstore_backend_kind kind);

  /**
   * Selects the backend based on the requested kind (or AUTO) and returns
   * SQLITE_OK plus the resolved backend. SQLITE_NOTFOUND indicates that the
   * build does not provide a backend for the request.
   */
  int objstore_backend_resolve (const objstore_config *config,
                                const objstore_backend **out_backend,
                                objstore_backend_kind *out_kind);

  /**
   * Clamps config->chunk_size_bytes into the supported range (1 KiB .. 64 KiB)
   * and falls back to OBJSTORE_DEFAULT_CHUNK_SIZE when unset.
   */
  size_t objstore_effective_chunk_size (const objstore_config *config);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* OBJSTORE_BACKEND_H */
