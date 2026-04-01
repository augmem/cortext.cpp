#ifndef OBJSTORE_BACKEND_PORTABLE_H
#define OBJSTORE_BACKEND_PORTABLE_H

#include "objstore/backend.h"

typedef enum objstore_portable_kind
{
  OBJSTORE_PORTABLE_VFS = 0,
  OBJSTORE_PORTABLE_OPFS = 1,
} objstore_portable_kind;

int objstore_portable_open_env (objstore_portable_kind kind, sqlite3 *db,
                                const objstore_config *config,
                                objstore_backend_env **out_env);
void objstore_portable_close_env (objstore_backend_env *env);
int objstore_portable_begin_txn (objstore_backend_env *env,
                                 objstore_backend_txn **out_txn);
int objstore_portable_commit_txn (objstore_backend_txn *txn);
void objstore_portable_rollback_txn (objstore_backend_txn *txn);
int
objstore_portable_staged_write_begin (objstore_backend_txn *txn,
                                      objstore_backend_staged_writer **out);
int
objstore_portable_staged_write_push (objstore_backend_staged_writer *writer,
                                     const void *buffer, size_t nread);
int objstore_portable_staged_write_finalize (
    objstore_backend_staged_writer *writer, const objstore_id *id);
int objstore_portable_staged_write_set_hint (
    objstore_backend_staged_writer *writer, sqlite3_int64 size_hint);
int objstore_portable_commit_staged (objstore_backend_txn *txn);
void objstore_portable_rollback_staged (objstore_backend_txn *txn);
int objstore_portable_put (objstore_backend_txn *txn, const objstore_id *id,
                           const objstore_stream_reader *reader);
int objstore_portable_get (objstore_backend_txn *txn, const objstore_id *id,
                           const objstore_stream_writer *writer);
int objstore_portable_delete (objstore_backend_txn *txn,
                              const objstore_id *id);
int objstore_portable_exists (objstore_backend_txn *txn,
                              const objstore_id *id);
int objstore_portable_scan_open (objstore_backend_txn *txn,
                                 objstore_backend_cursor **out_cursor);
int objstore_portable_scan_next (objstore_backend_cursor *cursor,
                                 objstore_id *out_id);
void objstore_portable_scan_close (objstore_backend_cursor *cursor);

#endif /* OBJSTORE_BACKEND_PORTABLE_H */
