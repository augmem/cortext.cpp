#include "objstore/vtab.h"

#include <assert.h>
#include <ctype.h>
#include <errno.h>
#include <limits.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "objstore/backend.h"
#include "objstore/object_manager.h"

#define OBJSTORE_MODULE_NAME "objstore"

enum
{
  OBJSTORE_COLUMN_ID = 0,
  OBJSTORE_COLUMN_DATA = 1,
  OBJSTORE_COLUMN_COUNT = 2,
};

static const uint8_t kObjstoreDefaultShardWidth = 2u;
static const uint8_t kObjstoreMaxShardWidth = OBJSTORE_ID_SIZE * 2u;

enum
{
  OBJSTORE_PLAN_FULL_SCAN = 0,
  OBJSTORE_PLAN_POINT_LOOKUP = 1,
  OBJSTORE_PLAN_ROWID_LOOKUP = 2,
};

typedef struct objstore_vtab
{
  sqlite3_vtab base;
  objstore_connection *conn;
} objstore_vtab;

typedef struct objstore_cursor
{
  sqlite3_vtab_cursor base;
  objstore_connection *conn;
  objstore_backend_txn *read_txn;
  objstore_backend_cursor *scan;
  objstore_txn_snapshot *snapshot;
  size_t snapshot_index;
  sqlite3_uint64 snapshot_seq;
  objstore_id current_id;
  sqlite3_int64 rowid;
  bool at_eof;
  bool is_lookup;
  bool lookup_by_rowid;
  bool needs_advance;
} objstore_cursor;

static int objstore_connection_copy_config (objstore_connection *conn,
                                            const objstore_config *config);
static int objstore_connection_install_hooks (objstore_connection *conn);
static int objstore_connection_ensure_env (objstore_connection *conn);
static int objstore_commit_hook (void *ctx);
static void objstore_rollback_hook (void *ctx);
static int objstore_authorizer (void *ctx, int action, const char *param1,
                                const char *param2, const char *db_name,
                                const char *trigger);
static void objstore_connection_unref_cb (void *ctx);

static int objstore_value_to_id (sqlite3_value *value, objstore_id *out);
static int objstore_lookup_id_by_rowid (objstore_connection *conn,
                                        objstore_backend_txn *txn,
                                        sqlite3_int64 rowid, objstore_id *out);
static int objstore_lookup_id_by_rowid_backend (objstore_connection *conn,
                                                sqlite3_int64 rowid,
                                                objstore_id *out);
static int objstore_resolve_delete_id (objstore_connection *conn,
                                       sqlite3_value *rowid_value,
                                       sqlite3_value *id_value,
                                       objstore_id *out);
static void objstore_sql_put (sqlite3_context *ctx, int argc,
                              sqlite3_value **argv);
static void objstore_sql_put_with_id (sqlite3_context *ctx, int argc,
                                      sqlite3_value **argv);
static void objstore_sql_get (sqlite3_context *ctx, int argc,
                              sqlite3_value **argv);
static void objstore_sql_delete (sqlite3_context *ctx, int argc,
                                 sqlite3_value **argv);
static void objstore_sql_exists (sqlite3_context *ctx, int argc,
                                 sqlite3_value **argv);
static bool
objstore_connection_has_active_txn (const objstore_connection *conn);
static void objstore_connection_on_config_change (objstore_connection *conn);
static int objstore_authorizer_handle_pragma (objstore_connection *conn,
                                              const char *name,
                                              const char *value);
static objstore_txn_lookup_state
objstore_txn_rowid_state (objstore_connection *conn,
                          sqlite3_uint64 sequence_limit, sqlite3_int64 rowid,
                          objstore_id *out_id);

/* Virtual table forward declarations */
static int objstore_vtab_connect (sqlite3 *db, void *p_aux, int argc,
                                  const char *const *argv,
                                  sqlite3_vtab **out_vtab, char **out_err);
static int objstore_vtab_create (sqlite3 *db, void *p_aux, int argc,
                                 const char *const *argv,
                                 sqlite3_vtab **out_vtab, char **out_err);
static int objstore_vtab_best_index (sqlite3_vtab *vtab,
                                     sqlite3_index_info *info);
static int objstore_vtab_disconnect (sqlite3_vtab *vtab);
static int objstore_vtab_open (sqlite3_vtab *vtab, sqlite3_vtab_cursor **out);
static int objstore_vtab_close (sqlite3_vtab_cursor *cursor);
static int objstore_vtab_filter (sqlite3_vtab_cursor *cursor, int idx_num,
                                 const char *idx_str, int argc,
                                 sqlite3_value **argv);
static int objstore_vtab_next (sqlite3_vtab_cursor *cursor);
static int objstore_vtab_eof (sqlite3_vtab_cursor *cursor);
static int objstore_vtab_column (sqlite3_vtab_cursor *cursor,
                                 sqlite3_context *ctx, int column);
static int objstore_vtab_rowid (sqlite3_vtab_cursor *cursor,
                                sqlite3_int64 *out_rowid);
static int objstore_vtab_update (sqlite3_vtab *vtab, int argc,
                                 sqlite3_value **argv,
                                 sqlite3_int64 *out_rowid);
static int objstore_vtab_rename (sqlite3_vtab *vtab, const char *new_name);

static int objstore_cursor_emit_snapshot (objstore_cursor *cursor);
static int objstore_cursor_load_next (objstore_cursor *cursor);
static int objstore_cursor_ensure_valid (objstore_cursor *cursor);

static const sqlite3_module g_objstore_module = {
  .iVersion = 3,
  .xCreate = objstore_vtab_create,
  .xConnect = objstore_vtab_connect,
  .xBestIndex = objstore_vtab_best_index,
  .xDisconnect = objstore_vtab_disconnect,
  .xDestroy = objstore_vtab_disconnect,
  .xOpen = objstore_vtab_open,
  .xClose = objstore_vtab_close,
  .xFilter = objstore_vtab_filter,
  .xNext = objstore_vtab_next,
  .xEof = objstore_vtab_eof,
  .xColumn = objstore_vtab_column,
  .xRowid = objstore_vtab_rowid,
  .xUpdate = objstore_vtab_update,
  .xBegin = 0,
  .xSync = 0,
  .xCommit = 0,
  .xRollback = 0,
  .xFindFunction = 0,
  .xRename = objstore_vtab_rename,
  .xSavepoint = 0,
  .xRelease = 0,
  .xRollbackTo = 0,
  .xShadowName = 0,
};

static void
objstore_free_string (char **target)
{
  if (target != NULL && *target != NULL)
    {
      sqlite3_free (*target);
      *target = NULL;
    }
}

static objstore_txn_lookup_state
objstore_txn_rowid_state (objstore_connection *conn,
                          sqlite3_uint64 sequence_limit, sqlite3_int64 rowid,
                          objstore_id *out_id)
{
  if (conn == NULL || conn->txn_log == NULL)
    {
      return OBJSTORE_TXN_LOOKUP_NONE;
    }
  const objstore_txn_log *log = conn->txn_log;
  const size_t count = objstore_txn_log_entry_count (log);
  for (ptrdiff_t i = (ptrdiff_t)count - 1; i >= 0; --i)
    {
      const objstore_txn_entry *entry = objstore_txn_log_entry_at (log,
                                                                   (size_t)i);
      if (entry == NULL || entry->sequence >= sequence_limit)
        {
          continue;
        }
      if (objstore_rowid_from_id (&entry->id) != rowid)
        {
          continue;
        }
      if (entry->kind == OBJSTORE_TXN_ENTRY_DELETE)
        {
          return OBJSTORE_TXN_LOOKUP_DELETE;
        }
      if (out_id != NULL)
        {
          *out_id = entry->id;
        }
      return OBJSTORE_TXN_LOOKUP_PUT;
    }
  return OBJSTORE_TXN_LOOKUP_NONE;
}

static int
objstore_connection_copy_config (objstore_connection *conn,
                                 const objstore_config *config)
{
  objstore_config local = { 0 };
  if (config != NULL)
    {
      local = *config;
    }
  local.chunk_size_bytes = conn->chunk_size;
  conn->config = local;
  if (local.storage_root != NULL)
    {
      conn->storage_root = sqlite3_mprintf ("%s", local.storage_root);
      if (conn->storage_root == NULL)
        {
          return SQLITE_NOMEM;
        }
      conn->config.storage_root = conn->storage_root;
    }
  else
    {
      conn->storage_root = NULL;
      conn->config.storage_root = NULL;
    }
  return SQLITE_OK;
}

static bool
objstore_connection_has_active_txn (const objstore_connection *conn)
{
  if (conn == NULL)
    {
      return false;
    }
  return conn->txn_state != OBJSTORE_TXN_NONE || conn->read_txn_depth > 0
         || conn->write_txn != NULL;
}

static void
objstore_connection_on_config_change (objstore_connection *conn)
{
  if (conn == NULL)
    {
      return;
    }
  conn->chunk_size = objstore_effective_chunk_size (&conn->config);
  if (conn->txn_log != NULL)
    {
      objstore_txn_log_destroy (conn->txn_log);
      conn->txn_log = NULL;
    }
  if (conn->backend != NULL && conn->env != NULL)
    {
      conn->backend->close_env (conn->env);
      conn->env = NULL;
    }
}

static int
objstore_connection_replace_string (char **slot, const char **config_slot,
                                    const char *value)
{
  char *dup = NULL;
  if (value != NULL && value[0] != '\0')
    {
      dup = sqlite3_mprintf ("%s", value);
      if (dup == NULL)
        {
          return SQLITE_NOMEM;
        }
    }
  sqlite3_free (*slot);
  *slot = dup;
  *config_slot = dup;
  return SQLITE_OK;
}

static char *
objstore_authorizer_unquote (const char *value)
{
  if (value == NULL)
    {
      return NULL;
    }
  size_t len = strlen (value);
  size_t start = 0;
  size_t end = len;
  if (len >= 2
      && ((value[0] == '\'' && value[len - 1] == '\'')
          || (value[0] == '"' && value[len - 1] == '"')))
    {
      start = 1;
      end = len - 1;
    }
  char *copy = sqlite3_malloc64 (end - start + 1);
  if (copy == NULL)
    {
      return NULL;
    }
  memcpy (copy, value + start, end - start);
  copy[end - start] = '\0';
  return copy;
}

static bool
objstore_parse_uint64 (const char *text, sqlite3_uint64 *out_value)
{
  if (text == NULL || out_value == NULL)
    {
      return false;
    }
  errno = 0;
  char *end = NULL;
  unsigned long long parsed = strtoull (text, &end, 10);
  if (errno != 0 || end == text || (end != NULL && *end != '\0'))
    {
      return false;
    }
  *out_value = (sqlite3_uint64)parsed;
  return true;
}

static int
objstore_authorizer_require_idle (objstore_connection *conn,
                                  const char *pragma_name)
{
  if (!objstore_connection_has_active_txn (conn))
    {
      return SQLITE_OK;
    }
  sqlite3_log (SQLITE_BUSY,
               "objstore pragma %s requires no active transactions",
               (pragma_name != NULL) ? pragma_name : "objstore");
  return SQLITE_BUSY;
}

static int
objstore_authorizer_handle_path_pragma (objstore_connection *conn,
                                        const char *pragma_name,
                                        const char *raw_value, char **slot,
                                        const char **config_slot)
{
  if (raw_value == NULL)
    {
      const char *current = (slot != NULL && *slot != NULL) ? *slot : NULL;
      sqlite3_log (SQLITE_OK, "%s = %s", pragma_name,
                   (current != NULL && current[0] != '\0') ? current
                                                           : "(default)");
      return SQLITE_IGNORE;
    }

  int idle_rc = objstore_authorizer_require_idle (conn, pragma_name);
  if (idle_rc != SQLITE_OK)
    {
      return SQLITE_DENY;
    }

  char *trimmed = objstore_authorizer_unquote (raw_value);
  if (trimmed == NULL)
    {
      sqlite3_log (SQLITE_NOMEM, "objstore pragma %s allocation failed",
                   pragma_name);
      return SQLITE_DENY;
    }
  const char *normalized = (trimmed[0] == '\0') ? NULL : trimmed;
  int rc = objstore_connection_replace_string (slot, config_slot, normalized);
  sqlite3_free (trimmed);
  if (rc != SQLITE_OK)
    {
      sqlite3_log (rc, "objstore pragma %s allocation failed", pragma_name);
      return SQLITE_DENY;
    }
  objstore_connection_on_config_change (conn);
  return SQLITE_IGNORE;
}

static const char *
objstore_sync_mode_to_string (objstore_sync_mode mode)
{
  switch (mode)
    {
    case OBJSTORE_SYNC_METADATA:
      return "metadata";
    case OBJSTORE_SYNC_OFF:
      return "off";
    case OBJSTORE_SYNC_FULL:
    default:
      return "full";
    }
}

static bool
objstore_parse_sync_mode (const char *text, objstore_sync_mode *out_mode)
{
  if (text == NULL || out_mode == NULL)
    {
      return false;
    }
  if (sqlite3_stricmp (text, "full") == 0)
    {
      *out_mode = OBJSTORE_SYNC_FULL;
      return true;
    }
  if (sqlite3_stricmp (text, "metadata") == 0
      || sqlite3_stricmp (text, "meta") == 0)
    {
      *out_mode = OBJSTORE_SYNC_METADATA;
      return true;
    }
  if (sqlite3_stricmp (text, "off") == 0)
    {
      *out_mode = OBJSTORE_SYNC_OFF;
      return true;
    }
  return false;
}

static int
objstore_authorizer_handle_pragma (objstore_connection *conn, const char *name,
                                   const char *value)
{
  if (conn == NULL || name == NULL)
    {
      return SQLITE_OK;
    }

  if (sqlite3_stricmp (name, "objstore_storage_root") == 0)
    {
      return objstore_authorizer_handle_path_pragma (
          conn, name, value, &conn->storage_root,
          (const char **)&conn->config.storage_root);
    }
  if (sqlite3_stricmp (name, "objstore_chunk_size") == 0)
    {
      if (value == NULL)
        {
          sqlite3_log (SQLITE_OK,
                       "objstore_chunk_size = %zu (effective %zu bytes)",
                       conn->config.chunk_size_bytes, conn->chunk_size);
          return SQLITE_IGNORE;
        }
      if (objstore_authorizer_require_idle (conn, name) != SQLITE_OK)
        {
          return SQLITE_DENY;
        }
      char *trimmed = objstore_authorizer_unquote (value);
      if (trimmed == NULL)
        {
          sqlite3_log (SQLITE_NOMEM, "objstore pragma %s allocation failed",
                       name);
          return SQLITE_DENY;
        }
      sqlite3_uint64 parsed = 0;
      bool ok = objstore_parse_uint64 (trimmed, &parsed);
      sqlite3_free (trimmed);
      if (!ok || parsed > (sqlite3_uint64)SIZE_MAX)
        {
          sqlite3_log (SQLITE_MISUSE,
                       "objstore pragma %s expects a non-negative integer "
                       "byte count",
                       name);
          return SQLITE_DENY;
        }
      conn->config.chunk_size_bytes = (size_t)parsed;
      objstore_connection_on_config_change (conn);
      return SQLITE_IGNORE;
    }
  if (sqlite3_stricmp (name, "objstore_shard_width") == 0)
    {
      if (value == NULL)
        {
          uint8_t configured = conn->config.shard_width;
          uint8_t effective
              = (configured != 0) ? configured : kObjstoreDefaultShardWidth;
          sqlite3_log (SQLITE_OK,
                       "objstore_shard_width = %u (effective %u hex digits)",
                       (unsigned)configured, (unsigned)effective);
          return SQLITE_IGNORE;
        }
      if (objstore_authorizer_require_idle (conn, name) != SQLITE_OK)
        {
          return SQLITE_DENY;
        }
      char *trimmed = objstore_authorizer_unquote (value);
      if (trimmed == NULL)
        {
          sqlite3_log (SQLITE_NOMEM, "objstore pragma %s allocation failed",
                       name);
          return SQLITE_DENY;
        }
      sqlite3_uint64 parsed = 0;
      bool ok = objstore_parse_uint64 (trimmed, &parsed);
      sqlite3_free (trimmed);
      if (!ok || parsed > kObjstoreMaxShardWidth)
        {
          sqlite3_log (
              SQLITE_MISUSE,
              "objstore pragma %s must be between 0 and %u hex digits", name,
              (unsigned)kObjstoreMaxShardWidth);
          return SQLITE_DENY;
        }
      if (parsed != 0 && (parsed % 2u) != 0u)
        {
          sqlite3_log (SQLITE_MISUSE,
                       "objstore pragma %s requires an even number of hex "
                       "digits",
                       name);
          return SQLITE_DENY;
        }
      conn->config.shard_width = (uint8_t)parsed;
      objstore_connection_on_config_change (conn);
      return SQLITE_IGNORE;
    }
  if (sqlite3_stricmp (name, "objstore_sync_mode") == 0)
    {
      if (value == NULL)
        {
          sqlite3_log (SQLITE_OK, "objstore_sync_mode = %s",
                       objstore_sync_mode_to_string (conn->config.sync_mode));
          return SQLITE_IGNORE;
        }
      if (objstore_authorizer_require_idle (conn, name) != SQLITE_OK)
        {
          return SQLITE_DENY;
        }
      char *trimmed = objstore_authorizer_unquote (value);
      if (trimmed == NULL)
        {
          sqlite3_log (SQLITE_NOMEM, "objstore pragma %s allocation failed",
                       name);
          return SQLITE_DENY;
        }
      objstore_sync_mode mode = OBJSTORE_SYNC_FULL;
      bool ok = objstore_parse_sync_mode (trimmed, &mode);
      sqlite3_free (trimmed);
      if (!ok)
        {
          sqlite3_log (SQLITE_MISUSE,
                       "objstore pragma %s expects full, metadata, or off",
                       name);
          return SQLITE_DENY;
        }
      conn->config.sync_mode = mode;
      objstore_connection_on_config_change (conn);
      return SQLITE_IGNORE;
    }
  return SQLITE_OK;
}

static int
objstore_connection_install_hooks (objstore_connection *conn)
{
  sqlite3_commit_hook (conn->db, objstore_commit_hook, conn);
  sqlite3_rollback_hook (conn->db, objstore_rollback_hook, conn);
  sqlite3_set_authorizer (conn->db, objstore_authorizer, conn);
  return SQLITE_OK;
}

int
objstore_connection_create (sqlite3 *db, const objstore_backend *backend,
                            const objstore_config *config, size_t chunk_size,
                            objstore_connection **out_conn)
{
  if (db == NULL || backend == NULL || out_conn == NULL)
    {
      return SQLITE_MISUSE;
    }

  objstore_connection *conn
      = (objstore_connection *)sqlite3_malloc (sizeof (*conn));
  if (conn == NULL)
    {
      return SQLITE_NOMEM;
    }
  memset (conn, 0, sizeof (*conn));
  conn->db = db;
  conn->backend = backend;
  conn->chunk_size = chunk_size;
  conn->refcount = 1;

  int rc = objstore_connection_copy_config (conn, config);
  if (rc != SQLITE_OK)
    {
      objstore_connection_destroy (conn);
      return rc;
    }

  rc = objstore_connection_install_hooks (conn);
  if (rc != SQLITE_OK)
    {
      objstore_connection_destroy (conn);
      return rc;
    }

  rc = objstore_connection_ensure_env (conn);
  if (rc != SQLITE_OK)
    {
      objstore_connection_destroy (conn);
      return rc;
    }

  rc = objstore_txn_log_create (db, &conn->config, &conn->txn_log);
  if (rc != SQLITE_OK)
    {
      objstore_connection_destroy (conn);
      return rc;
    }

  *out_conn = conn;
  return SQLITE_OK;
}

void
objstore_connection_destroy (objstore_connection *conn)
{
  if (conn == NULL)
    {
      return;
    }

  if (conn->db != NULL)
    {
      sqlite3_trace_v2 (conn->db, 0, NULL, NULL);
    }
  sqlite3_commit_hook (conn->db, NULL, NULL);
  sqlite3_rollback_hook (conn->db, NULL, NULL);
  sqlite3_set_authorizer (conn->db, NULL, NULL);

  if (conn->write_txn != NULL && conn->backend != NULL)
    {
      conn->backend->rollback_staged (conn->write_txn);
      conn->backend->rollback_txn (conn->write_txn);
    }
  conn->write_txn = NULL;
  conn->txn_state = OBJSTORE_TXN_NONE;

  if (conn->backend != NULL && conn->env != NULL)
    {
      conn->backend->close_env (conn->env);
      conn->env = NULL;
    }

  if (conn->txn_log != NULL)
    {
      objstore_txn_log_destroy (conn->txn_log);
      conn->txn_log = NULL;
    }

  objstore_free_string (&conn->storage_root);
  sqlite3_free (conn);
}

void
objstore_connection_ref (objstore_connection *conn)
{
  if (conn != NULL)
    {
      ++conn->refcount;
    }
}

void
objstore_connection_unref (objstore_connection *conn)
{
  if (conn == NULL)
    {
      return;
    }
  if (--conn->refcount == 0)
    {
      objstore_connection_destroy (conn);
    }
}

static void
objstore_connection_unref_cb (void *ctx)
{
  objstore_connection_unref ((objstore_connection *)ctx);
}

static int
objstore_connection_ensure_env (objstore_connection *conn)
{
  if (conn->env != NULL)
    {
      return SQLITE_OK;
    }
  int rc = conn->backend->open_env (conn->db, &conn->config, &conn->env);
  return rc;
}

int
objstore_connection_begin_write (objstore_connection *conn,
                                 objstore_backend_txn **out_txn)
{
  if (conn == NULL)
    {
      return SQLITE_MISUSE;
    }
  conn->last_error = OBJSTORE_ERROR_NONE;
  int rc = objstore_connection_ensure_env (conn);
  if (rc != SQLITE_OK)
    {
      return rc;
    }
  if (conn->txn_log == NULL)
    {
      rc = objstore_txn_log_create (conn->db, &conn->config, &conn->txn_log);
      if (rc != SQLITE_OK)
        {
          return rc;
        }
      for (int depth = 0; depth < conn->savepoint_depth; ++depth)
        {
          rc = objstore_txn_log_begin_frame (conn->txn_log);
          if (rc != SQLITE_OK)
            {
              return rc;
            }
        }
    }
  conn->txn_state = OBJSTORE_TXN_WRITE;
  if (conn->write_txn == NULL)
    {
      rc = conn->backend->begin_txn (conn->env, &conn->write_txn);
      if (rc != SQLITE_OK)
        {
          return rc;
        }
    }
  if (out_txn != NULL)
    {
      *out_txn = conn->write_txn;
    }
  return SQLITE_OK;
}

static int
objstore_commit_hook (void *ctx)
{
  objstore_connection *conn = (objstore_connection *)ctx;
  if (conn == NULL || conn->backend == NULL)
    {
      return 0;
    }
  if (objstore_txn_log_is_empty (conn->txn_log))
    {
      if (conn->write_txn != NULL)
        {
          conn->backend->rollback_staged (conn->write_txn);
          conn->backend->rollback_txn (conn->write_txn);
          conn->write_txn = NULL;
        }
      conn->txn_state = OBJSTORE_TXN_NONE;
      conn->savepoint_depth = 0;
      return 0;
    }

  sqlite3 *db = conn->db;
  if (db != NULL)
    {
      sqlite3_commit_hook (db, NULL, NULL);
      sqlite3_rollback_hook (db, NULL, NULL);
    }

  int rc = SQLITE_OK;
  if (conn->write_txn == NULL)
    {
      rc = SQLITE_MISUSE;
    }
  if (rc == SQLITE_OK)
    {
      rc = conn->backend->commit_staged (conn->write_txn);
    }
  if (rc == SQLITE_OK)
    {
      rc = conn->backend->commit_txn (conn->write_txn);
    }
  else if (conn->write_txn != NULL)
    {
      conn->backend->rollback_staged (conn->write_txn);
      conn->backend->rollback_txn (conn->write_txn);
    }
  conn->write_txn = NULL;

  if (db != NULL)
    {
      sqlite3_commit_hook (db, objstore_commit_hook, conn);
      sqlite3_rollback_hook (db, objstore_rollback_hook, conn);
    }

  if (rc != SQLITE_OK)
    {
      return 1;
    }

  conn->txn_state = OBJSTORE_TXN_NONE;
  conn->savepoint_depth = 0;
  objstore_txn_log_clear (conn->txn_log);
  return 0;
}

static void
objstore_rollback_hook (void *ctx)
{
  objstore_connection *conn = (objstore_connection *)ctx;
  if (conn == NULL || conn->backend == NULL)
    {
      return;
    }
  if (conn->write_txn != NULL)
    {
      conn->backend->rollback_staged (conn->write_txn);
      conn->backend->rollback_txn (conn->write_txn);
      conn->write_txn = NULL;
    }
  conn->txn_state = OBJSTORE_TXN_NONE;
  conn->savepoint_depth = 0;
  if (conn->txn_log != NULL)
    {
      objstore_txn_log_clear (conn->txn_log);
    }
}

static int
objstore_authorizer (void *ctx, int action, const char *param1,
                     const char *param2, const char *db_name,
                     const char *trigger)
{
  (void)db_name;
  (void)trigger;
  objstore_connection *conn = (objstore_connection *)ctx;
  if (conn == NULL)
    {
      return SQLITE_OK;
    }
  if (action == SQLITE_PRAGMA)
    {
      int pragma_rc = objstore_authorizer_handle_pragma (conn, param1, param2);
      if (pragma_rc != SQLITE_OK)
        {
          return pragma_rc;
        }
      return SQLITE_OK;
    }
  if (action == SQLITE_SAVEPOINT)
    {
      const char *op = (param1 != NULL) ? param1 : "";
      if (sqlite3_stricmp (op, "BEGIN") == 0)
        {
          if (conn->savepoint_depth < INT_MAX)
            {
              ++conn->savepoint_depth;
            }
          if (conn->txn_log != NULL)
            {
              int rc = objstore_txn_log_begin_frame (conn->txn_log);
              if (rc != SQLITE_OK)
                {
                  return SQLITE_DENY;
                }
            }
        }
      else if (sqlite3_stricmp (op, "RELEASE") == 0)
        {
          if (conn->savepoint_depth > 0)
            {
              --conn->savepoint_depth;
            }
          if (conn->txn_log != NULL)
            {
              (void)objstore_txn_log_release_frame (conn->txn_log);
            }
        }
      else if (sqlite3_stricmp (op, "ROLLBACK") == 0)
        {
          bool has_pending = (conn->txn_log != NULL)
                             && !objstore_txn_log_is_empty (conn->txn_log);
          if (has_pending)
            {
              if (conn->write_txn != NULL)
                {
                  conn->backend->rollback_staged (conn->write_txn);
                  conn->backend->rollback_txn (conn->write_txn);
                  conn->write_txn = NULL;
                }
              if (conn->txn_log != NULL)
                {
                  objstore_txn_log_clear (conn->txn_log);
                }
              conn->txn_state = OBJSTORE_TXN_NONE;
              conn->savepoint_depth = 0;
              conn->last_error = OBJSTORE_ERROR_SAVEPOINT;
              return SQLITE_DENY;
            }
          if (conn->txn_log != NULL)
            {
              (void)objstore_txn_log_rollback_frame (conn->txn_log);
            }
        }
      return SQLITE_OK;
    }
  if (action == SQLITE_TRANSACTION)
    {
      return SQLITE_OK;
    }
  return SQLITE_OK;
}

static int
objstore_value_to_id (sqlite3_value *value, objstore_id *out)
{
  if (value == NULL || out == NULL)
    {
      return SQLITE_MISUSE;
    }
  if (sqlite3_value_type (value) != SQLITE_BLOB
      || sqlite3_value_bytes (value) != OBJSTORE_ID_SIZE)
    {
      return SQLITE_MISMATCH;
    }
  const void *blob = sqlite3_value_blob (value);
  if (blob == NULL)
    {
      return SQLITE_MISMATCH;
    }
  memcpy (out->bytes, blob, OBJSTORE_ID_SIZE);
  return SQLITE_OK;
}

static int
objstore_lookup_id_by_rowid (objstore_connection *conn,
                             objstore_backend_txn *txn, sqlite3_int64 rowid,
                             objstore_id *out)
{
  if (conn == NULL || conn->backend == NULL || txn == NULL || out == NULL)
    {
      return SQLITE_MISUSE;
    }

  if (conn->backend->lookup_id_by_rowid != NULL)
    {
      return conn->backend->lookup_id_by_rowid (txn, rowid, out);
    }

  objstore_backend_cursor *cursor = NULL;
  int rc = conn->backend->scan_open (txn, &cursor);
  if (rc != SQLITE_OK)
    {
      return rc;
    }

  while (true)
    {
      objstore_id candidate;
      rc = conn->backend->scan_next (cursor, &candidate);
      if (rc == SQLITE_DONE)
        {
          rc = SQLITE_NOTFOUND;
          break;
        }
      if (rc != SQLITE_OK)
        {
          break;
        }
      if (objstore_rowid_from_id (&candidate) == rowid)
        {
          *out = candidate;
          rc = SQLITE_OK;
          break;
        }
    }

  conn->backend->scan_close (cursor);
  return rc;
}

static int
objstore_lookup_id_by_rowid_backend (objstore_connection *conn,
                                     sqlite3_int64 rowid, objstore_id *out)
{
  if (conn == NULL || out == NULL)
    {
      return SQLITE_MISUSE;
    }
  objstore_backend_txn *txn = NULL;
  int rc = objstore_begin_read_txn (conn, &txn);
  if (rc != SQLITE_OK)
    {
      return rc;
    }
  rc = objstore_lookup_id_by_rowid (conn, txn, rowid, out);
  return objstore_end_read_txn (conn, txn, rc);
}

static int
objstore_resolve_delete_id (objstore_connection *conn,
                            sqlite3_value *rowid_value,
                            sqlite3_value *id_value, objstore_id *out)
{
  if (conn == NULL || rowid_value == NULL || out == NULL)
    {
      return SQLITE_MISUSE;
    }

  if (id_value != NULL && !sqlite3_value_nochange (id_value)
      && sqlite3_value_type (id_value) == SQLITE_BLOB
      && sqlite3_value_bytes (id_value) == OBJSTORE_ID_SIZE)
    {
      return objstore_value_to_id (id_value, out);
    }

  if (sqlite3_value_type (rowid_value) != SQLITE_INTEGER)
    {
      return SQLITE_MISUSE;
    }

  sqlite3_int64 rowid = sqlite3_value_int64 (rowid_value);
  if (conn->txn_log != NULL)
    {
      int log_rc = objstore_txn_log_lookup_rowid (conn->txn_log, rowid, out);
      if (log_rc == SQLITE_OK || log_rc != SQLITE_NOTFOUND)
        {
          return log_rc;
        }
    }
  return objstore_lookup_id_by_rowid_backend (conn, rowid, out);
}

int
objstore_begin_read_txn (objstore_connection *conn,
                         objstore_backend_txn **out_txn)
{
  if (conn == NULL || out_txn == NULL)
    {
      return SQLITE_MISUSE;
    }
  int rc = objstore_connection_ensure_env (conn);
  if (rc != SQLITE_OK)
    {
      return rc;
    }
  rc = conn->backend->begin_txn (conn->env, out_txn);
  if (rc == SQLITE_OK)
    {
      if (conn->read_txn_depth < INT_MAX)
        {
          ++conn->read_txn_depth;
        }
      if (conn->txn_state == OBJSTORE_TXN_NONE)
        {
          conn->txn_state = OBJSTORE_TXN_READ;
        }
    }
  return rc;
}

int
objstore_end_read_txn (objstore_connection *conn, objstore_backend_txn *txn,
                       int rc)
{
  if (txn == NULL || conn == NULL || conn->backend == NULL)
    {
      return rc;
    }
  int result = rc;
  if (rc == SQLITE_OK || rc == SQLITE_DONE || rc == SQLITE_NOTFOUND)
    {
      int commit_rc = conn->backend->commit_txn (txn);
      if (rc == SQLITE_OK)
        {
          result = commit_rc;
        }
      else if (rc == SQLITE_NOTFOUND)
        {
          result = (commit_rc == SQLITE_OK) ? SQLITE_NOTFOUND : commit_rc;
        }
      else
        {
          result = commit_rc;
        }
    }
  else
    {
      conn->backend->rollback_txn (txn);
      result = rc;
    }
  if (conn->read_txn_depth > 0)
    {
      --conn->read_txn_depth;
    }
  if (conn->read_txn_depth == 0 && conn->txn_state == OBJSTORE_TXN_READ)
    {
      conn->txn_state = OBJSTORE_TXN_NONE;
    }
  return result;
}

static int
objstore_vtab_configure (sqlite3 *db, sqlite3_vtab **out_vtab,
                         objstore_connection *conn, char **out_err)
{
  if (sqlite3_declare_vtab (
          db, "CREATE TABLE x(id BLOB PRIMARY KEY, data BLOB NOT NULL)")
      != SQLITE_OK)
    {
      if (out_err != NULL)
        {
          *out_err = sqlite3_mprintf ("%s", sqlite3_errmsg (db));
        }
      return SQLITE_ERROR;
    }
  objstore_vtab *vtab = (objstore_vtab *)sqlite3_malloc (sizeof (*vtab));
  if (vtab == NULL)
    {
      if (out_err != NULL)
        {
          *out_err = sqlite3_mprintf ("out of memory");
        }
      return SQLITE_NOMEM;
    }
  memset (vtab, 0, sizeof (*vtab));
  vtab->conn = conn;
  *out_vtab = (sqlite3_vtab *)vtab;
  return SQLITE_OK;
}

static int
objstore_vtab_connect (sqlite3 *db, void *p_aux, int argc,
                       const char *const *argv, sqlite3_vtab **out_vtab,
                       char **out_err)
{
  (void)argc;
  (void)argv;
  objstore_connection *conn = (objstore_connection *)p_aux;
  return objstore_vtab_configure (db, out_vtab, conn, out_err);
}

static int
objstore_vtab_create (sqlite3 *db, void *p_aux, int argc,
                      const char *const *argv, sqlite3_vtab **out_vtab,
                      char **out_err)
{
  return objstore_vtab_connect (db, p_aux, argc, argv, out_vtab, out_err);
}

static int
objstore_vtab_best_index (sqlite3_vtab *vtab, sqlite3_index_info *info)
{
  objstore_vtab *ovtab = (objstore_vtab *)vtab;
  const bool backend_supports_rowid
      = (ovtab != NULL && ovtab->conn != NULL
         && ovtab->conn->backend != NULL
         && ovtab->conn->backend->lookup_id_by_rowid != NULL);
  int best = -1;
  int rowid_constraint = -1;
  for (int i = 0; i < info->nConstraint; ++i)
    {
      const struct sqlite3_index_constraint *c = &info->aConstraint[i];
      if (!c->usable)
        {
          continue;
        }
      if (c->iColumn == OBJSTORE_COLUMN_DATA)
        {
          vtab->zErrMsg = sqlite3_mprintf (
              "objstore data column cannot be filtered directly");
          return SQLITE_CONSTRAINT;
        }
      if (c->iColumn == OBJSTORE_COLUMN_ID
          && c->op != SQLITE_INDEX_CONSTRAINT_EQ)
        {
          vtab->zErrMsg
              = sqlite3_mprintf ("objstore id only supports equality filters");
          return SQLITE_CONSTRAINT;
        }
      if (c->iColumn == OBJSTORE_COLUMN_ID
          && c->op == SQLITE_INDEX_CONSTRAINT_EQ)
        {
          best = i;
          break;
        }
      if (c->iColumn == -1 && c->op == SQLITE_INDEX_CONSTRAINT_EQ
          && backend_supports_rowid)
        {
          rowid_constraint = i;
        }
    }
  if (best >= 0)
    {
      info->idxNum = OBJSTORE_PLAN_POINT_LOOKUP;
      info->estimatedRows = 1;
      info->estimatedCost = 1.0;
      info->aConstraintUsage[best].argvIndex = 1;
      info->aConstraintUsage[best].omit = 0;
      info->idxFlags |= SQLITE_INDEX_SCAN_UNIQUE;
    }
  else if (rowid_constraint >= 0)
    {
      info->idxNum = OBJSTORE_PLAN_ROWID_LOOKUP;
      info->estimatedRows = 1;
      info->estimatedCost = 1.0;
      info->aConstraintUsage[rowid_constraint].argvIndex = 1;
      info->aConstraintUsage[rowid_constraint].omit = 1;
      info->idxFlags |= SQLITE_INDEX_SCAN_UNIQUE;
    }
  else
    {
      info->idxNum = OBJSTORE_PLAN_FULL_SCAN;
      info->estimatedRows = 1000;
      info->estimatedCost = 1000.0;
    }
  return SQLITE_OK;
}

static int
objstore_vtab_disconnect (sqlite3_vtab *vtab)
{
  objstore_vtab *ovtab = (objstore_vtab *)vtab;
  sqlite3_free (ovtab);
  return SQLITE_OK;
}

static int
objstore_vtab_open (sqlite3_vtab *vtab, sqlite3_vtab_cursor **out)
{
  objstore_vtab *ovtab = (objstore_vtab *)vtab;
  objstore_cursor *cursor
      = (objstore_cursor *)sqlite3_malloc (sizeof (*cursor));
  if (cursor == NULL)
    {
      return SQLITE_NOMEM;
    }
  memset (cursor, 0, sizeof (*cursor));
  cursor->conn = ovtab->conn;
  cursor->snapshot = NULL;
  cursor->snapshot_index = 0;
  cursor->snapshot_seq = 0;
  cursor->needs_advance = true;

  int rc = objstore_begin_read_txn (cursor->conn, &cursor->read_txn);
  if (rc != SQLITE_OK)
    {
      sqlite3_free (cursor);
      return rc;
    }
  *out = (sqlite3_vtab_cursor *)cursor;
  return SQLITE_OK;
}

static int
objstore_vtab_close (sqlite3_vtab_cursor *cursor_base)
{
  objstore_cursor *cursor = (objstore_cursor *)cursor_base;
  if (cursor->scan != NULL && cursor->conn != NULL
      && cursor->conn->backend != NULL)
    {
      cursor->conn->backend->scan_close (cursor->scan);
      cursor->scan = NULL;
    }
  int rc = objstore_end_read_txn (cursor->conn, cursor->read_txn, SQLITE_OK);
  cursor->read_txn = NULL;
  if (cursor->snapshot != NULL)
    {
      objstore_txn_snapshot_destroy (cursor->snapshot);
      cursor->snapshot = NULL;
    }
  sqlite3_free (cursor);
  return rc;
}

static int
objstore_cursor_emit_snapshot (objstore_cursor *cursor)
{
  if (cursor->snapshot == NULL)
    {
      return SQLITE_DONE;
    }
  size_t count = objstore_txn_snapshot_count (cursor->snapshot);
  while (cursor->snapshot_index < count)
    {
      const objstore_txn_entry *entry = objstore_txn_snapshot_entry (
          cursor->snapshot, cursor->snapshot_index++);
      if (entry == NULL)
        {
          return SQLITE_CORRUPT;
        }
      cursor->current_id = entry->id;
      cursor->rowid = objstore_rowid_from_id (&entry->id);
      cursor->at_eof = false;
      return SQLITE_OK;
    }
  return SQLITE_DONE;
}

static int
objstore_cursor_load_next (objstore_cursor *cursor)
{
  while (cursor->scan != NULL)
    {
      int rc = cursor->conn->backend->scan_next (cursor->scan,
                                                 &cursor->current_id);
      if (rc == SQLITE_DONE)
        {
          cursor->conn->backend->scan_close (cursor->scan);
          cursor->scan = NULL;
          break;
        }
      if (rc != SQLITE_OK)
        {
          return rc;
        }
      objstore_txn_lookup_state state = objstore_txn_log_state_for_id (
          cursor->conn != NULL ? cursor->conn->txn_log : NULL,
          &cursor->current_id, cursor->snapshot_seq);
      if (state != OBJSTORE_TXN_LOOKUP_NONE)
        {
          continue;
        }
      cursor->rowid = objstore_rowid_from_id (&cursor->current_id);
      cursor->at_eof = false;
      return SQLITE_OK;
    }

  int snapshot_rc = objstore_cursor_emit_snapshot (cursor);
  if (snapshot_rc == SQLITE_OK)
    {
      return SQLITE_OK;
    }
  if (snapshot_rc != SQLITE_DONE)
    {
      return snapshot_rc;
    }
  cursor->at_eof = true;
  return SQLITE_OK;
}

static int
objstore_cursor_ensure_valid (objstore_cursor *cursor)
{
  if (cursor == NULL)
    {
      return SQLITE_MISUSE;
    }

  if (cursor->needs_advance)
    {
      cursor->needs_advance = false;
      int advance_rc = objstore_cursor_load_next (cursor);
      if (advance_rc != SQLITE_OK)
        {
          return advance_rc;
        }
    }

  return SQLITE_OK;
}

static int
objstore_vtab_filter (sqlite3_vtab_cursor *cursor_base, int idx_num,
                      const char *idx_str, int argc, sqlite3_value **argv)
{
  (void)idx_str;
  objstore_cursor *cursor = (objstore_cursor *)cursor_base;
  if (cursor->scan != NULL)
    {
      cursor->conn->backend->scan_close (cursor->scan);
      cursor->scan = NULL;
    }
  if (cursor->snapshot != NULL)
    {
      objstore_txn_snapshot_destroy (cursor->snapshot);
      cursor->snapshot = NULL;
    }
  cursor->snapshot_index = 0;
  cursor->snapshot_seq
      = (cursor->conn != NULL && cursor->conn->txn_log != NULL)
            ? objstore_txn_log_operation_count (cursor->conn->txn_log)
            : 0;
  cursor->at_eof = false;
  cursor->is_lookup = (idx_num == OBJSTORE_PLAN_POINT_LOOKUP);
  cursor->lookup_by_rowid = (idx_num == OBJSTORE_PLAN_ROWID_LOOKUP);

  if (cursor->is_lookup)
    {
      if (argc != 1)
        {
          return SQLITE_MISUSE;
        }
      int rc = objstore_value_to_id (argv[0], &cursor->current_id);
      if (rc != SQLITE_OK)
        {
          return rc;
        }
      objstore_txn_lookup_state state = objstore_txn_log_state_for_id (
          cursor->conn != NULL ? cursor->conn->txn_log : NULL,
          &cursor->current_id, cursor->snapshot_seq);
      if (state == OBJSTORE_TXN_LOOKUP_PUT)
        {
          cursor->rowid = objstore_rowid_from_id (&cursor->current_id);
          cursor->at_eof = false;
          cursor->needs_advance = false;
          return SQLITE_OK;
        }
      if (state == OBJSTORE_TXN_LOOKUP_DELETE)
        {
          cursor->at_eof = true;
          cursor->needs_advance = false;
          return SQLITE_OK;
        }
      rc = cursor->conn->backend->exists (cursor->read_txn,
                                          &cursor->current_id);
      if (rc == SQLITE_NOTFOUND)
        {
          cursor->at_eof = true;
          return SQLITE_OK;
        }
      if (rc != SQLITE_OK)
        {
          return rc;
        }
      cursor->rowid = objstore_rowid_from_id (&cursor->current_id);
      cursor->needs_advance = false;
      return SQLITE_OK;
    }

  if (cursor->lookup_by_rowid)
    {
      if (argc != 1)
        {
          return SQLITE_MISUSE;
        }
      sqlite3_value *rowid_value = argv[0];
      if (sqlite3_value_type (rowid_value) != SQLITE_INTEGER)
        {
          return SQLITE_MISMATCH;
        }
      const sqlite3_int64 lookup_rowid = sqlite3_value_int64 (rowid_value);
      objstore_id txn_id = { 0 };
      objstore_txn_lookup_state txn_state = objstore_txn_rowid_state (
          cursor->conn, cursor->snapshot_seq, lookup_rowid, &txn_id);
      if (txn_state == OBJSTORE_TXN_LOOKUP_PUT)
        {
          cursor->current_id = txn_id;
          cursor->rowid = lookup_rowid;
          cursor->needs_advance = false;
          cursor->at_eof = false;
          return SQLITE_OK;
        }
      if (txn_state == OBJSTORE_TXN_LOOKUP_DELETE)
        {
          cursor->needs_advance = false;
          cursor->at_eof = true;
          return SQLITE_OK;
        }
      objstore_id backend_id = { 0 };
      int lookup_rc = objstore_lookup_id_by_rowid (
          cursor->conn, cursor->read_txn, lookup_rowid, &backend_id);
      if (lookup_rc == SQLITE_NOTFOUND)
        {
          cursor->needs_advance = false;
          cursor->at_eof = true;
          return SQLITE_OK;
        }
      if (lookup_rc != SQLITE_OK)
        {
          return lookup_rc;
        }
      cursor->current_id = backend_id;
      cursor->rowid = lookup_rowid;
      cursor->needs_advance = false;
      cursor->at_eof = false;
      return SQLITE_OK;
    }

  (void)argc;
  (void)argv;
  int rc = cursor->conn->backend->scan_open (cursor->read_txn, &cursor->scan);
  if (rc != SQLITE_OK)
    {
      return rc;
    }
  if (cursor->conn->txn_log != NULL
      && !objstore_txn_log_is_empty (cursor->conn->txn_log))
    {
      cursor->snapshot = objstore_txn_snapshot_build (cursor->conn->txn_log,
                                                      cursor->snapshot_seq);
      if (cursor->snapshot == NULL)
        {
          return SQLITE_NOMEM;
        }
      cursor->snapshot_index = 0;
      if (objstore_txn_snapshot_count (cursor->snapshot) == 0)
        {
          objstore_txn_snapshot_destroy (cursor->snapshot);
          cursor->snapshot = NULL;
        }
    }
  cursor->needs_advance = true;
  cursor->at_eof = false;
  return SQLITE_OK;
}

static int
objstore_vtab_next (sqlite3_vtab_cursor *cursor_base)
{
  objstore_cursor *cursor = (objstore_cursor *)cursor_base;
  if (cursor->is_lookup || cursor->lookup_by_rowid)
    {
      cursor->at_eof = true;
      return SQLITE_OK;
    }
  cursor->needs_advance = true;
  cursor->at_eof = false;
  return SQLITE_OK;
}

static int
objstore_vtab_eof (sqlite3_vtab_cursor *cursor_base)
{
  objstore_cursor *cursor = (objstore_cursor *)cursor_base;
  int rc = objstore_cursor_ensure_valid (cursor);
  if (rc != SQLITE_OK)
    {
      return 1;
    }
  return cursor->at_eof ? 1 : 0;
}

static int
objstore_vtab_column (sqlite3_vtab_cursor *cursor_base, sqlite3_context *ctx,
                      int column)
{
  objstore_cursor *cursor = (objstore_cursor *)cursor_base;
  int rc = objstore_cursor_ensure_valid (cursor);
  if (rc != SQLITE_OK)
    {
      sqlite3_result_error_code (ctx, rc);
      return rc;
    }
  if (cursor->at_eof)
    {
      sqlite3_result_null (ctx);
      return SQLITE_DONE;
    }
  if (column == OBJSTORE_COLUMN_ID)
    {
      sqlite3_result_blob (ctx, cursor->current_id.bytes, OBJSTORE_ID_SIZE,
                           SQLITE_TRANSIENT);
      return SQLITE_OK;
    }
  if (column == OBJSTORE_COLUMN_DATA)
    {
      return objstore_object_read_blob (ctx, cursor->conn, cursor->read_txn,
                                        &cursor->current_id,
                                        cursor->snapshot_seq);
    }
  sqlite3_result_null (ctx);
  return SQLITE_OK;
}

static int
objstore_vtab_rowid (sqlite3_vtab_cursor *cursor_base,
                     sqlite3_int64 *out_rowid)
{
  objstore_cursor *cursor = (objstore_cursor *)cursor_base;
  int rc = objstore_cursor_ensure_valid (cursor);
  if (rc != SQLITE_OK)
    {
      return rc;
    }
  if (cursor->at_eof)
    {
      return SQLITE_DONE;
    }
  *out_rowid = cursor->rowid;
  return SQLITE_OK;
}

static int
objstore_vtab_update (sqlite3_vtab *p_vtab, int argc, sqlite3_value **argv,
                      sqlite3_int64 *out_rowid)
{
  objstore_vtab *vtab = (objstore_vtab *)p_vtab;
  objstore_connection *conn = vtab->conn;

  if (argc < 1)
    {
      return SQLITE_MISUSE;
    }

  sqlite3_value *old_rowid = argv[0];
  sqlite3_value *new_rowid = (argc > 1) ? argv[1] : NULL;
  sqlite3_value *id_value = (argc > 2) ? argv[2] : NULL;
  sqlite3_value *data_value = (argc > 3) ? argv[3] : NULL;

  const bool has_old_rowid = sqlite3_value_type (old_rowid) != SQLITE_NULL;
  const bool has_new_rowid
      = (new_rowid != NULL && sqlite3_value_type (new_rowid) != SQLITE_NULL);
  const bool is_delete = (argc == 1) || (!has_new_rowid && has_old_rowid);
  const bool is_insert = !has_old_rowid;
  const bool is_update = !is_delete && !is_insert;
  int rc = SQLITE_OK;

  if (is_update)
    {
      p_vtab->zErrMsg = sqlite3_mprintf (
          "objstore rows are immutable; delete then insert");
      return SQLITE_ERROR;
    }

  if (is_delete)
    {
      objstore_id id;
      rc = objstore_resolve_delete_id (conn, old_rowid, id_value, &id);
      if (rc == SQLITE_NOTFOUND)
        {
          return SQLITE_OK;
        }
      if (rc != SQLITE_OK)
        {
          return rc;
        }
      return objstore_object_delete (conn, &id);
    }

  rc = objstore_object_validate_data_value (data_value);
  if (rc != SQLITE_OK)
    {
      return rc;
    }

  objstore_id id;
  bool id_provided = sqlite3_value_type (id_value) == SQLITE_BLOB
                     && sqlite3_value_bytes (id_value) == OBJSTORE_ID_SIZE;
  if (id_provided)
    {
      rc = objstore_value_to_id (id_value, &id);
      if (rc != SQLITE_OK)
        {
          return rc;
        }
    }

  if (id_provided)
    {
      rc = objstore_object_put_value_with_id (conn, &id, data_value);
    }
  else
    {
      rc = objstore_object_put_value (conn, data_value, &id);
    }
  if (rc != SQLITE_OK)
    {
      return rc;
    }
  if (out_rowid != NULL)
    {
      *out_rowid = objstore_rowid_from_id (&id);
    }
  return SQLITE_OK;
}

static int
objstore_vtab_rename (sqlite3_vtab *vtab, const char *new_name)
{
  (void)vtab;
  (void)new_name;
  return SQLITE_OK;
}

static objstore_connection *
objstore_context_get_connection (sqlite3_context *ctx)
{
  return (objstore_connection *)sqlite3_user_data (ctx);
}

static void
objstore_sql_put (sqlite3_context *ctx, int argc, sqlite3_value **argv)
{
  objstore_connection *conn = objstore_context_get_connection (ctx);
  if (conn == NULL || argv == NULL || argc != 1)
    {
      sqlite3_result_error_code (ctx, SQLITE_MISUSE);
      return;
    }
  objstore_id id;
  int rc = objstore_object_put_value (conn, argv[0], &id);
  if (rc != SQLITE_OK)
    {
      sqlite3_result_error_code (ctx, rc);
      return;
    }
  sqlite3_result_blob (ctx, id.bytes, OBJSTORE_ID_SIZE, SQLITE_TRANSIENT);
}

static void
objstore_sql_put_with_id (sqlite3_context *ctx, int argc, sqlite3_value **argv)
{
  objstore_connection *conn = objstore_context_get_connection (ctx);
  if (conn == NULL || argv == NULL || argc != 2)
    {
      sqlite3_result_error_code (ctx, SQLITE_MISUSE);
      return;
    }
  objstore_id id;
  int rc = objstore_value_to_id (argv[0], &id);
  if (rc != SQLITE_OK)
    {
      sqlite3_result_error_code (ctx, rc);
      return;
    }
  rc = objstore_object_put_value_with_id (conn, &id, argv[1]);
  if (rc != SQLITE_OK)
    {
      sqlite3_result_error_code (ctx, rc);
      return;
    }
  sqlite3_result_blob (ctx, id.bytes, OBJSTORE_ID_SIZE, SQLITE_TRANSIENT);
}

static void
objstore_sql_get (sqlite3_context *ctx, int argc, sqlite3_value **argv)
{
  objstore_connection *conn = objstore_context_get_connection (ctx);
  if (conn == NULL || argv == NULL || argc != 1)
    {
      sqlite3_result_error_code (ctx, SQLITE_MISUSE);
      return;
    }
  objstore_id id;
  int rc = objstore_value_to_id (argv[0], &id);
  if (rc != SQLITE_OK)
    {
      sqlite3_result_error_code (ctx, rc);
      return;
    }
  objstore_backend_txn *txn = NULL;
  rc = objstore_begin_read_txn (conn, &txn);
  if (rc != SQLITE_OK)
    {
      sqlite3_result_error_code (ctx, rc);
      return;
    }
  rc = objstore_object_read_blob (ctx, conn, txn, &id, (sqlite3_uint64)-1);
  rc = objstore_end_read_txn (conn, txn, rc == SQLITE_OK ? SQLITE_OK : rc);
  if (rc != SQLITE_OK)
    {
      sqlite3_result_error_code (ctx, rc);
    }
}

static void
objstore_sql_delete (sqlite3_context *ctx, int argc, sqlite3_value **argv)
{
  objstore_connection *conn = objstore_context_get_connection (ctx);
  if (conn == NULL || argv == NULL || argc != 1)
    {
      sqlite3_result_error_code (ctx, SQLITE_MISUSE);
      return;
    }
  objstore_id id;
  int rc = objstore_value_to_id (argv[0], &id);
  if (rc != SQLITE_OK)
    {
      sqlite3_result_error_code (ctx, rc);
      return;
    }
  int present = 0;
  rc = objstore_object_exists (conn, &id, &present);
  if (rc != SQLITE_OK)
    {
      sqlite3_result_error_code (ctx, rc);
      return;
    }
  if (!present)
    {
      sqlite3_result_int (ctx, 0);
      return;
    }
  rc = objstore_object_delete (conn, &id);
  if (rc != SQLITE_OK)
    {
      sqlite3_result_error_code (ctx, rc);
      return;
    }
  sqlite3_result_int (ctx, 1);
}

static void
objstore_sql_exists (sqlite3_context *ctx, int argc, sqlite3_value **argv)
{
  objstore_connection *conn = objstore_context_get_connection (ctx);
  if (conn == NULL || argv == NULL || argc != 1)
    {
      sqlite3_result_error_code (ctx, SQLITE_MISUSE);
      return;
    }
  objstore_id id;
  int rc = objstore_value_to_id (argv[0], &id);
  if (rc != SQLITE_OK)
    {
      sqlite3_result_error_code (ctx, rc);
      return;
    }
  int present = 0;
  rc = objstore_object_exists (conn, &id, &present);
  if (rc != SQLITE_OK)
    {
      sqlite3_result_error_code (ctx, rc);
      return;
    }
  sqlite3_result_int (ctx, present);
}

typedef struct objstore_scalar_spec
{
  const char *name;
  int argc;
  void (*fn) (sqlite3_context *, int, sqlite3_value **);
} objstore_scalar_spec;

int
objstore_module_register (sqlite3 *db, objstore_connection *conn)
{
  if (db == NULL || conn == NULL)
    {
      return SQLITE_MISUSE;
    }

  static const objstore_scalar_spec kScalars[] = {
    { "objstore_put", 1, objstore_sql_put },
    { "objstore_put_with_id", 2, objstore_sql_put_with_id },
    { "objstore_get", 1, objstore_sql_get },
    { "objstore_delete", 1, objstore_sql_delete },
    { "objstore_exists", 1, objstore_sql_exists },
  };
  bool scalar_registered[sizeof (kScalars) / sizeof (kScalars[0])] = { false };

  int rc = SQLITE_OK;
  for (size_t i = 0; i < sizeof (kScalars) / sizeof (kScalars[0]); ++i)
    {
      objstore_connection_ref (conn);
      rc = sqlite3_create_function_v2 (db, kScalars[i].name, kScalars[i].argc,
                                       SQLITE_UTF8 | SQLITE_DETERMINISTIC,
                                       conn, kScalars[i].fn, NULL, NULL,
                                       objstore_connection_unref_cb);
      if (rc != SQLITE_OK)
        {
          objstore_connection_unref (conn);
          goto fail;
        }
      scalar_registered[i] = true;
    }

  objstore_connection_ref (conn);
  rc = sqlite3_create_module_v2 (db, OBJSTORE_MODULE_NAME, &g_objstore_module,
                                 conn, objstore_connection_unref_cb);
  if (rc != SQLITE_OK)
    {
      objstore_connection_unref (conn);
      goto fail;
    }

  objstore_connection_unref (conn); /* drop creation reference */
  return SQLITE_OK;

fail:
  for (size_t i = 0; i < sizeof (kScalars) / sizeof (kScalars[0]); ++i)
    {
      if (scalar_registered[i])
        {
          sqlite3_create_function_v2 (db, kScalars[i].name, kScalars[i].argc,
                                      SQLITE_UTF8 | SQLITE_DETERMINISTIC, NULL,
                                      NULL, NULL, NULL, NULL);
        }
    }
  objstore_connection_unref (conn);
  return rc;
}
