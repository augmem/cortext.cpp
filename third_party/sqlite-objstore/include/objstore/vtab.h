#ifndef OBJSTORE_VTAB_H
#define OBJSTORE_VTAB_H

#ifdef __cplusplus
extern "C"
{
#endif

#include <sqlite3.h>
#include <stddef.h>

#include "objstore/backend.h"
#include "objstore/txn.h"

  typedef enum objstore_txn_state
  {
    OBJSTORE_TXN_NONE = 0,
    OBJSTORE_TXN_READ = 1,
    OBJSTORE_TXN_WRITE = 2,
  } objstore_txn_state;

  typedef enum objstore_error_reason
  {
    OBJSTORE_ERROR_NONE = 0,
    OBJSTORE_ERROR_SAVEPOINT = 1,
  } objstore_error_reason;

  struct objstore_cursor;

  /**
   * Per-connection state shared by every cursor/scalar registered against a
   * sqlite3* handle. Holds backend pointers, txn logs, and staging metadata.
   */
  typedef struct objstore_connection
  {
    sqlite3 *db;
    objstore_config config;
    char *storage_root;
    const objstore_backend *backend;
    objstore_backend_env *env;
    objstore_backend_txn *write_txn;
    objstore_txn_log *txn_log;
    objstore_txn_state txn_state;
    int read_txn_depth;
    int savepoint_depth;
    objstore_error_reason last_error;
    size_t chunk_size;
    int refcount;
  } objstore_connection;

  /**
   * Allocates a ref-counted connection object plus backend environment. The
   * returned structure outlives individual SQL statements until unref'd.
   */
  int objstore_connection_create (sqlite3 *db, const objstore_backend *backend,
                                  const objstore_config *config,
                                  size_t chunk_size,
                                  objstore_connection **out_conn);

  /** Releases all resources owned by the connection. */
  void objstore_connection_destroy (objstore_connection *conn);

  /** Reference counting helpers used by sqlite3_module and scalars. */
  void objstore_connection_ref (objstore_connection *conn);
  void objstore_connection_unref (objstore_connection *conn);

  /** Registers the virtual table module + scalar functions with SQLite. */
  int objstore_module_register (sqlite3 *db, objstore_connection *conn);

  /** Ensures a write transaction is open and returns the backend handle. */
  int objstore_connection_begin_write (objstore_connection *conn,
                                       objstore_backend_txn **out_txn);

  /** Begins a read transaction (used by cursors + objstore_get). */
  int objstore_begin_read_txn (objstore_connection *conn,
                               objstore_backend_txn **out_txn);

  /** Ends a read transaction, translating rc into the caller's result code. */
  int objstore_end_read_txn (objstore_connection *conn,
                             objstore_backend_txn *txn, int rc);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* OBJSTORE_VTAB_H */
