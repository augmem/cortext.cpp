#include <stdint.h>
#include <string.h>

#include <sqlite3.h>

#include "objstore/objstore.h"

static sqlite3 *g_db = NULL;
static sqlite3_stmt *g_put_stmt = NULL;
static sqlite3_stmt *g_range_stmt = NULL;

static int
objstore_fuzzer_init (void)
{
  if (g_db != NULL)
    {
      return SQLITE_OK;
    }
  int rc = sqlite3_open (":memory:", &g_db);
  if (rc != SQLITE_OK)
    {
      return rc;
    }
  objstore_config cfg = {
    .backend = OBJSTORE_BACKEND_SQLITE,
    .storage_root = NULL,
    .chunk_size_bytes = 0,
    .shard_width = 0,
    .sync_mode = OBJSTORE_SYNC_FULL,
    .reserved_flags = 0,
  };
  rc = objstore_register (g_db, &cfg);
  if (rc != SQLITE_OK)
    {
      return rc;
    }
  rc = sqlite3_exec (g_db, "CREATE VIRTUAL TABLE objstore USING objstore();",
                     NULL, NULL, NULL);
  if (rc != SQLITE_OK)
    {
      return rc;
    }
  rc = sqlite3_prepare_v2 (g_db, "SELECT objstore_put(?1);", -1, &g_put_stmt,
                           NULL);
  if (rc != SQLITE_OK)
    {
      return rc;
    }
  return sqlite3_prepare_v2 (g_db, "SELECT objstore_get_range(?1, ?2);", -1,
                             &g_range_stmt, NULL);
}

int
LLVMFuzzerInitialize (int *argc, char ***argv)
{
  (void)argc;
  (void)argv;
  return objstore_fuzzer_init () == SQLITE_OK ? 0 : 1;
}

int
LLVMFuzzerTestOneInput (const uint8_t *data, size_t size)
{
  if (objstore_fuzzer_init () != SQLITE_OK)
    {
      return 0;
    }

  const size_t remaining = (size > 0u) ? (size - 1u) : 0u;
  const size_t range_len
      = (size > 0u) ? ((size_t)data[0] % (remaining + 1u)) : 0u;
  const unsigned char *range_bytes = (size > 0u) ? (data + 1u) : data;
  const unsigned char *payload = range_bytes + range_len;
  const size_t payload_len = remaining - range_len;

  sqlite3_exec (g_db, "BEGIN;", NULL, NULL, NULL);

  sqlite3_reset (g_put_stmt);
  sqlite3_clear_bindings (g_put_stmt);
  sqlite3_bind_blob (g_put_stmt, 1, payload, (int)payload_len, SQLITE_TRANSIENT);
  if (sqlite3_step (g_put_stmt) == SQLITE_ROW)
    {
      const void *id_blob = sqlite3_column_blob (g_put_stmt, 0);
      const int id_len = sqlite3_column_bytes (g_put_stmt, 0);
      if (id_blob != NULL && id_len == 32)
        {
          unsigned char id[32];
          memcpy (id, id_blob, sizeof (id));

          sqlite3_reset (g_range_stmt);
          sqlite3_clear_bindings (g_range_stmt);
          sqlite3_bind_blob (g_range_stmt, 1, id, (int)sizeof (id),
                             SQLITE_TRANSIENT);
          sqlite3_bind_text (g_range_stmt, 2, (const char *)range_bytes,
                             (int)range_len, SQLITE_TRANSIENT);
          (void)sqlite3_step (g_range_stmt);
          sqlite3_reset (g_range_stmt);
          sqlite3_clear_bindings (g_range_stmt);
        }
    }
  sqlite3_reset (g_put_stmt);
  sqlite3_clear_bindings (g_put_stmt);

  sqlite3_exec (g_db, "ROLLBACK;", NULL, NULL, NULL);
  return 0;
}
