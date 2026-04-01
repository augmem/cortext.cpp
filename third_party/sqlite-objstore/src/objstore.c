#include "objstore/objstore.h"
#include "objstore/backend.h"
#include "objstore/vtab.h"

static int
objstore_validate_config (const objstore_config *config)
{
  if (config == NULL)
    {
      return SQLITE_OK;
    }
  if (config->reserved_flags != 0U)
    {
      return SQLITE_MISUSE;
    }
  const uint8_t max_shard_width = (uint8_t)(OBJSTORE_ID_SIZE * 2);
  if (config->shard_width > max_shard_width)
    {
      return SQLITE_MISUSE;
    }
  if (config->shard_width != 0U && (config->shard_width % 2U) != 0U)
    {
      return SQLITE_MISUSE;
    }
  if (config->sync_mode < OBJSTORE_SYNC_FULL
      || config->sync_mode > OBJSTORE_SYNC_OFF)
    {
      return SQLITE_MISUSE;
    }
  switch (config->backend)
    {
    case OBJSTORE_BACKEND_AUTO:
    case OBJSTORE_BACKEND_FILE:
    case OBJSTORE_BACKEND_SQLITE:
    case OBJSTORE_BACKEND_OPFS:
    case OBJSTORE_BACKEND_VFS:
      break;
    default:
      return SQLITE_MISUSE;
    }
  return SQLITE_OK;
}

int
objstore_register (sqlite3 *db, const objstore_config *config)
{
  if (db == NULL)
    {
      return SQLITE_MISUSE;
    }

  int rc = objstore_validate_config (config);
  if (rc != SQLITE_OK)
    {
      return rc;
    }

  const objstore_backend *backend = NULL;
  objstore_backend_kind resolved_kind = OBJSTORE_BACKEND_AUTO;
  rc = objstore_backend_resolve (config, &backend, &resolved_kind);
  if (rc != SQLITE_OK)
    {
      return rc;
    }

  size_t chunk_size = objstore_effective_chunk_size (config);

  objstore_connection *conn = NULL;
  rc = objstore_connection_create (db, backend, config, chunk_size, &conn);
  if (rc != SQLITE_OK)
    {
      return rc;
    }

  rc = objstore_module_register (db, conn);
  if (rc != SQLITE_OK)
    {
      objstore_connection_destroy (conn);
      return rc;
    }

  (void)resolved_kind;
  return SQLITE_OK;
}

const char *
objstore_version (void)
{
  return OBJSTORE_VERSION_STRING;
}
