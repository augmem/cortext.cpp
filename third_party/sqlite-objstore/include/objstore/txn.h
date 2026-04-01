#ifndef OBJSTORE_TXN_H
#define OBJSTORE_TXN_H

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>

#include <sqlite3.h>

#include "objstore/backend.h"

  /**
   * Per-connection transaction log that tracks staged PUT/DELETE operations so
   * cursors and scalar functions observe uncommitted changes.
   */
  typedef struct objstore_txn_log objstore_txn_log;

  /**
   * Read-only snapshot of the txn log used when serving cursors opened inside a
   * transaction. Snapshots freeze the log at a given sequence number.
   */
  typedef struct objstore_txn_snapshot objstore_txn_snapshot;

  typedef enum objstore_txn_entry_kind
  {
    OBJSTORE_TXN_ENTRY_PUT = 0,
    OBJSTORE_TXN_ENTRY_DELETE = 1,
  } objstore_txn_entry_kind;

  typedef struct objstore_txn_entry
  {
    objstore_txn_entry_kind kind;
    objstore_id id;
    sqlite3_int64 payload_size;
    sqlite3_uint64 sequence;
  } objstore_txn_entry;

  typedef enum objstore_txn_lookup_state
  {
    OBJSTORE_TXN_LOOKUP_NONE = 0,
    OBJSTORE_TXN_LOOKUP_PUT = 1,
    OBJSTORE_TXN_LOOKUP_DELETE = 2,
  } objstore_txn_lookup_state;

  /** Allocates a new txn log bound to the sqlite3* connection. */
  int objstore_txn_log_create (sqlite3 *db, const objstore_config *config,
                               objstore_txn_log **out_log);

  /** Releases all resources owned by a txn log. */
  void objstore_txn_log_destroy (objstore_txn_log *log);

  /** Drops all pending entries but keeps the log allocated. */
  void objstore_txn_log_clear (objstore_txn_log *log);

  /** Returns true when no staged operations exist. */
  bool objstore_txn_log_is_empty (const objstore_txn_log *log);

  /** Number of PUT/DELETE entries appended so far. */
  sqlite3_uint64 objstore_txn_log_operation_count (const objstore_txn_log *log);

  /** Appends a PUT entry plus payload size metadata. */
  int objstore_txn_log_append_put (objstore_txn_log *log,
                                   const objstore_id *id,
                                   sqlite3_int64 payload_size);

  /** Appends a DELETE entry. */
  int objstore_txn_log_append_delete (objstore_txn_log *log,
                                      const objstore_id *id);

  /** Returns the visible state for id at or before sequence_limit. */
  objstore_txn_lookup_state
  objstore_txn_log_state_for_id (const objstore_txn_log *log,
                                 const objstore_id *id,
                                 sqlite3_uint64 sequence_limit);

  /** Maps a virtual table rowid back to the staged objstore_id. */
  int objstore_txn_log_lookup_rowid (const objstore_txn_log *log,
                                     sqlite3_int64 rowid, objstore_id *out_id);

  /** Savepoint helpers (BEGIN/RELEASE/ROLLBACK). */
  int objstore_txn_log_begin_frame (objstore_txn_log *log);
  int objstore_txn_log_release_frame (objstore_txn_log *log);
  int objstore_txn_log_rollback_frame (objstore_txn_log *log);

  /** Random access helpers for cursor snapshots. */
  size_t objstore_txn_log_entry_count (const objstore_txn_log *log);
  const objstore_txn_entry *
  objstore_txn_log_entry_at (const objstore_txn_log *log, size_t index);

  int objstore_txn_log_drop_last (objstore_txn_log *log);

  /** Builds a snapshot that only exposes entries <= sequence_limit. */
  objstore_txn_snapshot *
  objstore_txn_snapshot_build (const objstore_txn_log *log,
                               sqlite3_uint64 sequence_limit);
  void objstore_txn_snapshot_destroy (objstore_txn_snapshot *snapshot);
  size_t objstore_txn_snapshot_count (const objstore_txn_snapshot *snapshot);
  const objstore_txn_entry *
  objstore_txn_snapshot_entry (const objstore_txn_snapshot *snapshot,
                               size_t index);
  bool objstore_txn_snapshot_contains (const objstore_txn_snapshot *snapshot,
                                       const objstore_id *id);

#ifdef __cplusplus
}
#endif

#endif /* OBJSTORE_TXN_H */
