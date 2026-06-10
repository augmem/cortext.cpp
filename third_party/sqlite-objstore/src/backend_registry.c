#include "objstore/backend.h"

#include <stdbool.h>
#include <stddef.h>

#ifndef OBJSTORE_HAVE_BACKEND_FILE
#define OBJSTORE_HAVE_BACKEND_FILE 1
#endif

#ifndef OBJSTORE_HAVE_BACKEND_SQLITE
#define OBJSTORE_HAVE_BACKEND_SQLITE 1
#endif

#ifndef OBJSTORE_HAVE_BACKEND_OPFS
#define OBJSTORE_HAVE_BACKEND_OPFS 1
#endif

#ifndef OBJSTORE_HAVE_BACKEND_VFS
#define OBJSTORE_HAVE_BACKEND_VFS 1
#endif

#if OBJSTORE_HAVE_BACKEND_FILE
extern const objstore_backend objstore_backend_file;
#endif

#if OBJSTORE_HAVE_BACKEND_SQLITE
extern const objstore_backend objstore_backend_sqlite;
#endif

#if OBJSTORE_HAVE_BACKEND_OPFS
extern const objstore_backend objstore_backend_opfs;
#endif

#if OBJSTORE_HAVE_BACKEND_VFS
extern const objstore_backend objstore_backend_vfs;
#endif

static const objstore_backend *const g_backends[] = {
#if OBJSTORE_HAVE_BACKEND_FILE
  &objstore_backend_file,
#endif
#if OBJSTORE_HAVE_BACKEND_SQLITE
  &objstore_backend_sqlite,
#endif
#if OBJSTORE_HAVE_BACKEND_OPFS
  &objstore_backend_opfs,
#endif
#if OBJSTORE_HAVE_BACKEND_VFS
  &objstore_backend_vfs,
#endif
};

static size_t
objstore_backend_count (void)
{
  return sizeof (g_backends) / sizeof (g_backends[0]);
}

const objstore_backend *
objstore_backend_by_kind (objstore_backend_kind kind)
{
  for (size_t i = 0; i < objstore_backend_count (); ++i)
    {
      if (g_backends[i]->kind == kind)
        {
          return g_backends[i];
        }
    }
  return NULL;
}

static const objstore_backend *
objstore_backend_preferred (void)
{
#if defined(__EMSCRIPTEN__)
  const objstore_backend_kind order[] = {
    OBJSTORE_BACKEND_OPFS,
    OBJSTORE_BACKEND_VFS,
    OBJSTORE_BACKEND_SQLITE,
    OBJSTORE_BACKEND_FILE,
  };
#elif defined(__wasi__)
  const objstore_backend_kind order[] = {
    OBJSTORE_BACKEND_VFS,
    OBJSTORE_BACKEND_SQLITE,
    OBJSTORE_BACKEND_FILE,
    OBJSTORE_BACKEND_OPFS,
  };
#else
  const objstore_backend_kind order[] = {
    OBJSTORE_BACKEND_FILE,
    OBJSTORE_BACKEND_SQLITE,
    OBJSTORE_BACKEND_OPFS,
    OBJSTORE_BACKEND_VFS,
  };
#endif
  for (size_t i = 0; i < sizeof (order) / sizeof (order[0]); ++i)
    {
      const objstore_backend *backend = objstore_backend_by_kind (order[i]);
      if (backend != NULL)
        {
          return backend;
        }
    }
  return NULL;
}

static int
objstore_backend_validate (const objstore_backend *backend)
{
  if (backend == NULL)
    {
      return SQLITE_NOTFOUND;
    }
  if (backend->staged_write_begin == NULL || backend->staged_write_push == NULL
      || backend->staged_write_finalize == NULL
      || backend->commit_staged == NULL || backend->rollback_staged == NULL)
    {
      const char *name = backend->name != NULL ? backend->name : "(unknown)";
      sqlite3_log (SQLITE_MISUSE,
                   "objstore backend %s missing staging hook implementation",
                   name);
      return SQLITE_MISUSE;
    }
  if (backend->put == NULL || backend->get == NULL || backend->get_size == NULL
      || backend->delete_fn == NULL || backend->exists == NULL)
    {
      const char *name = backend->name != NULL ? backend->name : "(unknown)";
      sqlite3_log (SQLITE_MISUSE,
                   "objstore backend %s missing object io implementation",
                   name);
      return SQLITE_MISUSE;
    }
  return SQLITE_OK;
}

int
objstore_backend_resolve (const objstore_config *config,
                          const objstore_backend **out_backend,
                          objstore_backend_kind *out_kind)
{
  if (out_backend == NULL)
    {
      return SQLITE_MISUSE;
    }

  objstore_backend_kind requested = OBJSTORE_BACKEND_AUTO;
  if (config != NULL && config->backend != OBJSTORE_BACKEND_AUTO)
    {
      requested = config->backend;
    }

  const objstore_backend *backend = NULL;
  if (requested == OBJSTORE_BACKEND_AUTO)
    {
      backend = objstore_backend_preferred ();
    }
  else
    {
      backend = objstore_backend_by_kind (requested);
    }

  if (backend == NULL)
    {
      return SQLITE_NOTFOUND;
    }

  int validate_rc = objstore_backend_validate (backend);
  if (validate_rc != SQLITE_OK)
    {
      return validate_rc;
    }

  *out_backend = backend;
  if (out_kind != NULL)
    {
      *out_kind = backend->kind;
    }

  return SQLITE_OK;
}

size_t
objstore_effective_chunk_size (const objstore_config *config)
{
  size_t chunk_size = OBJSTORE_DEFAULT_CHUNK_SIZE;
  if (config != NULL && config->chunk_size_bytes > 0)
    {
      chunk_size = config->chunk_size_bytes;
    }
  if (chunk_size < OBJSTORE_MIN_CHUNK_SIZE)
    {
      chunk_size = OBJSTORE_MIN_CHUNK_SIZE;
    }
  if (chunk_size > OBJSTORE_MAX_CHUNK_SIZE)
    {
      chunk_size = OBJSTORE_MAX_CHUNK_SIZE;
    }
  return chunk_size;
}
