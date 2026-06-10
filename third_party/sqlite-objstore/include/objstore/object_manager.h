#ifndef OBJSTORE_OBJECT_MANAGER_H
#define OBJSTORE_OBJECT_MANAGER_H

#ifdef __cplusplus
extern "C"
{
#endif

#include <sqlite3.h>

#include "objstore/backend.h"

  typedef struct objstore_connection objstore_connection;

  /**
   * S3-style byte range spec:
   * - has_start + has_end => "start-end" (inclusive)
   * - has_start only      => "start-"
   * - has_end only        => "-suffix_length"
   */
  typedef struct objstore_range_spec
  {
    int has_start;
    int has_end;
    sqlite3_uint64 start;
    sqlite3_uint64 end;
  } objstore_range_spec;

  /**
   * Validates that a SQLite value is a non-NULL BLOB suitable for storage.
   * Returns SQLITE_OK on success or SQLITE_MISMATCH for unsupported types.
   */
  int objstore_object_validate_data_value (sqlite3_value *value);

  /**
   * Streams a SQLite BLOB value into the txn log, computes its BLAKE3 digest,
   * and returns the derived objstore_id.
   */
  int objstore_object_put_value (objstore_connection *conn,
                                 sqlite3_value *value, objstore_id *out_id);

  /**
   * Same as objstore_object_put_value but enforces that the caller supplied a
   * specific 32-byte id.
   */
  int objstore_object_put_value_with_id (objstore_connection *conn,
                                         const objstore_id *id,
                                         sqlite3_value *value);

  /**
   * Streams data from an arbitrary reader, computing the content-addressed id.
   */
  int objstore_object_put_reader (objstore_connection *conn,
                                  const objstore_stream_reader *reader,
                                  objstore_id *out_id);

  /**
   * Streams data from an arbitrary reader using a caller-provided id.
   */
  int
  objstore_object_put_reader_with_id (objstore_connection *conn,
                                      const objstore_id *id,
                                      const objstore_stream_reader *reader);

  /**
   * Enqueues a DELETE entry for the given id.
   */
  int objstore_object_delete (objstore_connection *conn,
                              const objstore_id *id);

  /**
   * Checks whether an object exists, combining txn-log state with backend
   * visibility. out_present is set to 1/0 on success.
   */
  int objstore_object_exists (objstore_connection *conn, const objstore_id *id,
                              int *out_present);

  /**
   * Streams an object payload into the provided writer while honoring the
   * caller's snapshot (sequence_limit).
   */
  int objstore_object_read_stream (objstore_connection *conn,
                                   objstore_backend_txn *txn,
                                   const objstore_id *id,
                                   sqlite3_uint64 sequence_limit,
                                   const objstore_stream_writer *writer);

  /**
   * Streams a range from an object payload into the provided writer.
   * Range spec follows S3 semantics (inclusive end, suffix ranges).
   */
  int objstore_object_read_stream_range (objstore_connection *conn,
                                         objstore_backend_txn *txn,
                                         const objstore_id *id,
                                         sqlite3_uint64 sequence_limit,
                                         const objstore_range_spec *range,
                                         const objstore_stream_writer *writer);

  /**
   * Convenience wrapper that materializes an entire blob owned by SQLite.
   */
  int objstore_object_read_blob (sqlite3_context *ctx,
                                 objstore_connection *conn,
                                 objstore_backend_txn *txn,
                                 const objstore_id *id,
                                 sqlite3_uint64 sequence_limit);

  /**
   * Convenience wrapper that materializes a range of a blob owned by SQLite.
   */
  int objstore_object_read_blob_range (sqlite3_context *ctx,
                                       objstore_connection *conn,
                                       objstore_backend_txn *txn,
                                       const objstore_id *id,
                                       sqlite3_uint64 sequence_limit,
                                       const objstore_range_spec *range);

  /**
   * Re-reads an object's bytes, recomputes BLAKE3, and reports whether the
   * stored payload matches the provided id. out_valid receives 1/0.
   */
  int objstore_object_verify (objstore_connection *conn, const objstore_id *id,
                              int *out_valid);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* OBJSTORE_OBJECT_MANAGER_H */
