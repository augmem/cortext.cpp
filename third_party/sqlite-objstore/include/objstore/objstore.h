#ifndef OBJSTORE_OBJSTORE_H
#define OBJSTORE_OBJSTORE_H

#ifdef __cplusplus
extern "C"
{
#endif

#include <stddef.h>
#include <stdint.h>

#include <sqlite3.h>

/**
 * Public version tuple for build-system consumers.
 */
#define OBJSTORE_VERSION_MAJOR 0
#define OBJSTORE_VERSION_MINOR 1
#define OBJSTORE_VERSION_PATCH 0

#define OBJSTORE_VERSION_STRING "0.1.0"

  /**
   * Supported backend kinds (native file, OPFS, SQLite fallback, etc.).
   * Keep this in sync with the matrix described in docs/architecture.md.
   */
  typedef enum objstore_backend_kind
  {
    OBJSTORE_BACKEND_AUTO = 0,
    OBJSTORE_BACKEND_FILE,
    OBJSTORE_BACKEND_SQLITE,
    OBJSTORE_BACKEND_OPFS,
    OBJSTORE_BACKEND_VFS
  } objstore_backend_kind;

  typedef enum objstore_sync_mode
  {
    OBJSTORE_SYNC_FULL = 0,
    OBJSTORE_SYNC_METADATA = 1,
    OBJSTORE_SYNC_OFF = 2,
  } objstore_sync_mode;

  /**
   * Basic bootstrap configuration for the extension.
   * Fields map directly to the invariants defined in docs/architecture.md.
   */
  typedef struct objstore_config
  {
    objstore_backend_kind backend;
    /** Root directory or URI for backend-specific state (e.g., objects/).
     *  File backend expects sharded layout (objects/aa/<sha>.dat) and places
     *  its `.staging/` scratch hierarchy under this root automatically.
     */
    const char *storage_root;
    /** Max bytes to buffer per streaming chunk; 0 selects the default. */
    size_t chunk_size_bytes;
    /** Number of leading hex digits used for sharding (0 -> default). */
    uint8_t shard_width;
    /** Fsync strategy for native backends. */
    objstore_sync_mode sync_mode;
    /** Reserved for future flags; must be zero for v1. */
    uint32_t reserved_flags;
  } objstore_config;

  /**
   * Registers the objstore virtual table and scalar APIs with the provided
   * SQLite handle. The call is idempotent; safe to invoke once per sqlite3*
   * connection.
   */
  int objstore_register (sqlite3 *db, const objstore_config *config);

  /** Returns a semver-style version string (e.g., 0.1.0). */
  const char *objstore_version (void);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* OBJSTORE_OBJSTORE_H */
