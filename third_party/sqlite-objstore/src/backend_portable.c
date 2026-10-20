#include "backend_portable.h"

#if defined(_WIN32)
#include "win_dirent.h"
#else
#include <dirent.h>
#endif
#include <errno.h>
#include <limits.h>
#include <stddef.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#if defined(_WIN32)
#include "win_posix.h"
#else
#include <unistd.h>
#endif

#if defined(__EMSCRIPTEN__)
#include <emscripten.h>
#endif

#include "backend_fs_common.h"

#define PORTABLE_DEFAULT_SHARD_WIDTH 2u

typedef struct objstore_portable_traits
{
  const char *default_root;
  bool requires_opfs;
} objstore_portable_traits;

#if defined(__wasi__)
#define OBJSTORE_VFS_DEFAULT_ROOT "/data/objstore"
#else
#define OBJSTORE_VFS_DEFAULT_ROOT "objstore"
#endif

#define OBJSTORE_OPFS_DEFAULT_ROOT "/opfs/objstore"

static const objstore_portable_traits kPortableTraits[] = {
  [OBJSTORE_PORTABLE_VFS] = { OBJSTORE_VFS_DEFAULT_ROOT, false },
  [OBJSTORE_PORTABLE_OPFS] = { OBJSTORE_OPFS_DEFAULT_ROOT, true },
};

typedef struct objstore_portable_backend_env
{
  sqlite3 *db;
  size_t chunk_size;
  size_t shard_width;
  objstore_sync_mode sync_mode;
  char *storage_root;
  char *objects_root;
  char *wal_root;
  char *wal_active_root;
  char *wal_commit_root;
  const objstore_portable_traits *traits;
} objstore_portable_backend_env;

typedef struct objstore_portable_backend_txn
{
  objstore_portable_backend_env *env;
  char txn_id[33];
  char *active_dir;
  sqlite3_str *manifest;
  struct objstore_portable_savepoint_frame *frames;
  size_t frame_count;
  size_t frame_capacity;
  bool manifest_written;
  bool staging_promoted;
} objstore_portable_backend_txn;

typedef struct objstore_portable_savepoint_entry
{
  objstore_id id;
  bool had_pending;
  bool had_delete_marker;
} objstore_portable_savepoint_entry;

typedef struct objstore_portable_savepoint_frame
{
  size_t manifest_length;
  objstore_portable_savepoint_entry *entries;
  size_t entry_count;
  size_t entry_capacity;
} objstore_portable_savepoint_frame;

typedef struct objstore_portable_staged_writer
{
  objstore_portable_backend_txn *txn;
  FILE *file;
  char *temp_path;
  sqlite3_int64 expected_size;
  sqlite3_int64 bytes_written;
} objstore_portable_staged_writer;

typedef struct objstore_portable_cursor
{
  objstore_portable_backend_env *env;
  objstore_id *ids;
  size_t count;
  size_t index;
} objstore_portable_cursor;

static void portable_savepoint_frames_clear (
    objstore_portable_backend_txn *txn);

static inline objstore_portable_backend_env *
portable_env_from (objstore_backend_env *env)
{
  return (objstore_portable_backend_env *)env;
}

static inline objstore_portable_backend_txn *
portable_txn_from (objstore_backend_txn *txn)
{
  return (objstore_portable_backend_txn *)txn;
}

static inline objstore_portable_staged_writer *
portable_writer_from (objstore_backend_staged_writer *writer)
{
  return (objstore_portable_staged_writer *)writer;
}

static inline objstore_portable_cursor *
portable_cursor_from (objstore_backend_cursor *cursor)
{
  return (objstore_portable_cursor *)cursor;
}

static int
portable_compare_ids (const void *lhs, const void *rhs)
{
  return memcmp (lhs, rhs, sizeof (objstore_id));
}

#if defined(__EMSCRIPTEN__)
EM_JS (int, objstore_portable_mount_opfs, (), {
  if (self.__objstore_opfs_ready)
    {
      return 0;
    }
  try
    {
      if (!FS.analyzePath ("/opfs").exists)
        {
          FS.mkdir ("/opfs");
        }
      FS.mount (FS.filesystems.OPFS, {}, "/opfs");
      self.__objstore_opfs_ready = true;
      return 0;
    }
  catch (e)
    {
      return 1;
    }
});
#endif

static const objstore_portable_traits *
portable_traits_for_kind (objstore_portable_kind kind)
{
  if (kind < 0
      || kind >= (int)(sizeof (kPortableTraits) / sizeof (kPortableTraits[0])))
    {
      return NULL;
    }
  return &kPortableTraits[kind];
}

static void
portable_env_free (objstore_portable_backend_env *env)
{
  if (env == NULL)
    {
      return;
    }
  sqlite3_free (env->storage_root);
  sqlite3_free (env->objects_root);
  sqlite3_free (env->wal_root);
  sqlite3_free (env->wal_active_root);
  sqlite3_free (env->wal_commit_root);
  sqlite3_free (env);
}

static void
portable_txn_reset_active (objstore_portable_backend_txn *txn, bool drop_tree)
{
  if (txn == NULL)
    {
      return;
    }
  if (txn->manifest != NULL)
    {
      char *value = sqlite3_str_finish (txn->manifest);
      sqlite3_free (value);
      txn->manifest = NULL;
    }
  txn->manifest_written = false;
  if (drop_tree && txn->active_dir != NULL)
    {
      objstore_fs_remove_tree (txn->active_dir);
    }
  sqlite3_free (txn->active_dir);
  txn->active_dir = NULL;
}

static void
portable_txn_free (objstore_portable_backend_txn *txn)
{
  if (txn == NULL)
    {
      return;
    }
  portable_savepoint_frames_clear (txn);
  portable_txn_reset_active (txn, true);
  sqlite3_free (txn);
}

static char *
portable_active_subdir (objstore_portable_backend_txn *txn, const char *subdir)
{
  if (txn == NULL || subdir == NULL)
    {
      return NULL;
    }
  return objstore_fs_path_join (txn->active_dir, subdir);
}

static char *
portable_put_path (objstore_portable_backend_txn *txn, const char *hex_id)
{
  char *dir = portable_active_subdir (txn, OBJSTORE_PUT_DIR);
  if (dir == NULL)
    {
      return NULL;
    }
  char *name = sqlite3_mprintf ("%s%s", hex_id, OBJSTORE_FILE_SUFFIX);
  if (name == NULL)
    {
      sqlite3_free (dir);
      return NULL;
    }
  char *path = objstore_fs_path_join (dir, name);
  sqlite3_free (dir);
  sqlite3_free (name);
  return path;
}

static char *
portable_delete_marker (objstore_portable_backend_txn *txn, const char *hex_id)
{
  char *dir = portable_active_subdir (txn, OBJSTORE_DELETE_DIR);
  if (dir == NULL)
    {
      return NULL;
    }
  char *path = objstore_fs_path_join (dir, hex_id);
  sqlite3_free (dir);
  return path;
}

static void
portable_savepoint_frame_dispose (objstore_portable_savepoint_frame *frame)
{
  if (frame == NULL)
    {
      return;
    }
  sqlite3_free (frame->entries);
  frame->entries = NULL;
  frame->entry_count = 0;
  frame->entry_capacity = 0;
  frame->manifest_length = 0;
}

static void
portable_savepoint_frames_clear (objstore_portable_backend_txn *txn)
{
  if (txn == NULL)
    {
      return;
    }
  for (size_t i = 0; i < txn->frame_count; ++i)
    {
      portable_savepoint_frame_dispose (&txn->frames[i]);
    }
  sqlite3_free (txn->frames);
  txn->frames = NULL;
  txn->frame_count = 0;
  txn->frame_capacity = 0;
}

static int
portable_savepoint_frames_ensure_capacity (objstore_portable_backend_txn *txn)
{
  if (txn == NULL)
    {
      return SQLITE_MISUSE;
    }
  if (txn->frame_count < txn->frame_capacity)
    {
      return SQLITE_OK;
    }
  size_t new_capacity
      = (txn->frame_capacity == 0) ? 4u : txn->frame_capacity * 2u;
  objstore_portable_savepoint_frame *resized = sqlite3_realloc64 (
      txn->frames, new_capacity * sizeof (*txn->frames));
  if (resized == NULL)
    {
      return SQLITE_NOMEM;
    }
  txn->frames = resized;
  memset (&txn->frames[txn->frame_capacity], 0,
          (new_capacity - txn->frame_capacity) * sizeof (*txn->frames));
  txn->frame_capacity = new_capacity;
  return SQLITE_OK;
}

static int
portable_savepoint_entries_ensure_capacity (
    objstore_portable_savepoint_frame *frame)
{
  if (frame == NULL)
    {
      return SQLITE_MISUSE;
    }
  if (frame->entry_count < frame->entry_capacity)
    {
      return SQLITE_OK;
    }
  size_t new_capacity
      = (frame->entry_capacity == 0) ? 4u : frame->entry_capacity * 2u;
  objstore_portable_savepoint_entry *resized = sqlite3_realloc64 (
      frame->entries, new_capacity * sizeof (*frame->entries));
  if (resized == NULL)
    {
      return SQLITE_NOMEM;
    }
  frame->entries = resized;
  frame->entry_capacity = new_capacity;
  return SQLITE_OK;
}

static ptrdiff_t
portable_savepoint_frame_find_entry (
    const objstore_portable_savepoint_frame *frame, const objstore_id *id)
{
  if (frame == NULL || id == NULL)
    {
      return -1;
    }
  for (size_t i = 0; i < frame->entry_count; ++i)
    {
      if (memcmp (frame->entries[i].id.bytes, id->bytes, OBJSTORE_ID_SIZE) == 0)
        {
          return (ptrdiff_t)i;
        }
    }
  return -1;
}

static int
portable_manifest_reset_prefix (objstore_portable_backend_txn *txn,
                                size_t prefix_length)
{
  if (txn == NULL || txn->manifest == NULL)
    {
      return SQLITE_MISUSE;
    }
  const char *current = sqlite3_str_value (txn->manifest);
  const size_t current_length = (size_t)sqlite3_str_length (txn->manifest);
  if (prefix_length > current_length)
    {
      return SQLITE_CORRUPT;
    }

  sqlite3_str *replacement = sqlite3_str_new (NULL);
  if (replacement == NULL)
    {
      return SQLITE_NOMEM;
    }
  if (prefix_length > 0 && current != NULL)
    {
      sqlite3_str_append (replacement, current, (int)prefix_length);
    }
  int rc = sqlite3_str_errcode (replacement);
  if (rc != SQLITE_OK)
    {
      char *value = sqlite3_str_finish (replacement);
      sqlite3_free (value);
      return rc;
    }

  char *value = sqlite3_str_finish (txn->manifest);
  sqlite3_free (value);
  txn->manifest = replacement;
  txn->manifest_written = false;
  return SQLITE_OK;
}

static int
portable_savepoint_capture_state (objstore_portable_backend_txn *txn,
                                  const objstore_id *id)
{
  if (txn == NULL || id == NULL)
    {
      return SQLITE_MISUSE;
    }
  if (txn->frame_count == 0)
    {
      return SQLITE_OK;
    }
  objstore_portable_savepoint_frame *frame
      = &txn->frames[txn->frame_count - 1];
  if (portable_savepoint_frame_find_entry (frame, id) >= 0)
    {
      return SQLITE_OK;
    }
  int rc = portable_savepoint_entries_ensure_capacity (frame);
  if (rc != SQLITE_OK)
    {
      return rc;
    }

  char hex[OBJSTORE_ID_SIZE * 2 + 1];
  objstore_fs_id_to_hex (id, hex);
  char *pending = portable_put_path (txn, hex);
  char *marker = portable_delete_marker (txn, hex);
  if (pending == NULL || marker == NULL)
    {
      sqlite3_free (pending);
      sqlite3_free (marker);
      return SQLITE_NOMEM;
    }

  objstore_portable_savepoint_entry *entry = &frame->entries[frame->entry_count++];
  memset (entry, 0, sizeof (*entry));
  entry->id = *id;
  entry->had_pending = objstore_fs_path_exists (pending);
  entry->had_delete_marker = objstore_fs_path_exists (marker);
  sqlite3_free (pending);
  sqlite3_free (marker);
  return SQLITE_OK;
}

static int
portable_savepoint_restore_entry (
    objstore_portable_backend_txn *txn,
    const objstore_portable_savepoint_entry *entry)
{
  if (txn == NULL || entry == NULL)
    {
      return SQLITE_MISUSE;
    }

  char hex[OBJSTORE_ID_SIZE * 2 + 1];
  objstore_fs_id_to_hex (&entry->id, hex);
  char *pending = portable_put_path (txn, hex);
  char *marker = portable_delete_marker (txn, hex);
  if (pending == NULL || marker == NULL)
    {
      sqlite3_free (pending);
      sqlite3_free (marker);
      return SQLITE_NOMEM;
    }

  if (!entry->had_pending && unlink (pending) != 0 && errno != ENOENT)
    {
      sqlite3_free (pending);
      sqlite3_free (marker);
      return SQLITE_IOERR_DELETE;
    }
  if (entry->had_delete_marker)
    {
      FILE *file = fopen (marker, "wb");
      if (file == NULL)
        {
          sqlite3_free (pending);
          sqlite3_free (marker);
          return SQLITE_IOERR;
        }
      fclose (file);
    }
  else if (unlink (marker) != 0 && errno != ENOENT)
    {
      sqlite3_free (pending);
      sqlite3_free (marker);
      return SQLITE_IOERR_DELETE;
    }

  sqlite3_free (pending);
  sqlite3_free (marker);
  return SQLITE_OK;
}

static int
portable_stat_size (const char *path, sqlite3_int64 *out_size)
{
  if (path == NULL || out_size == NULL)
    {
      return SQLITE_MISUSE;
    }
  struct stat st;
  if (stat (path, &st) != 0)
    {
      return (errno == ENOENT) ? SQLITE_NOTFOUND : SQLITE_IOERR;
    }
  if (!S_ISREG (st.st_mode))
    {
      return SQLITE_NOTFOUND;
    }
  if (st.st_size < 0)
    {
      return SQLITE_IOERR;
    }
  *out_size = (sqlite3_int64)st.st_size;
  return SQLITE_OK;
}

static int
portable_manifest_append (objstore_portable_backend_txn *txn, const char *op,
                          const char *hex_id)
{
  if (txn == NULL || txn->manifest == NULL || op == NULL || hex_id == NULL)
    {
      return SQLITE_MISUSE;
    }
  sqlite3_str_appendf (txn->manifest, "%s %s\n", op, hex_id);
  return sqlite3_str_errcode (txn->manifest);
}

static int
portable_manifest_flush (objstore_portable_backend_txn *txn)
{
  if (txn == NULL || txn->manifest == NULL || txn->manifest_written)
    {
      return txn == NULL ? SQLITE_MISUSE : SQLITE_OK;
    }
  char *manifest_path
      = objstore_fs_path_join (txn->active_dir, OBJSTORE_MANIFEST_FILE);
  if (manifest_path == NULL)
    {
      return SQLITE_NOMEM;
    }
  FILE *file = fopen (manifest_path, "wb");
  sqlite3_free (manifest_path);
  if (file == NULL)
    {
      return SQLITE_IOERR;
    }
  const char *data = sqlite3_str_value (txn->manifest);
  size_t length
      = data != NULL ? (size_t)sqlite3_str_length (txn->manifest) : 0u;
  if (length > 0 && fwrite (data, 1, length, file) != length)
    {
      int err = errno;
      fclose (file);
      return (err == ENOSPC || err == EFBIG) ? SQLITE_FULL
                                             : SQLITE_IOERR_WRITE;
    }
  if (fflush (file) != 0)
    {
      int err = errno;
      fclose (file);
      return (err == ENOSPC || err == EFBIG) ? SQLITE_FULL
                                             : SQLITE_IOERR_FSYNC;
    }
  fclose (file);
  txn->manifest_written = true;
  return SQLITE_OK;
}

static int portable_apply_commit_dir (objstore_portable_backend_env *env,
                                      const char *commit_dir);

static int
portable_promote_staged (objstore_portable_backend_txn *txn)
{
  if (txn == NULL)
    {
      return SQLITE_MISUSE;
    }
  if (txn->staging_promoted)
    {
      return SQLITE_OK;
    }
  int rc = portable_manifest_flush (txn);
  if (rc != SQLITE_OK)
    {
      return rc;
    }
  char *commit_dir
      = objstore_fs_path_join (txn->env->wal_commit_root, txn->txn_id);
  if (commit_dir == NULL)
    {
      return SQLITE_NOMEM;
    }
  rc = objstore_fs_rename (txn->active_dir, commit_dir);
  if (rc == SQLITE_OK)
    {
      sqlite3_free (txn->active_dir);
      txn->active_dir = commit_dir;
      commit_dir = NULL;
      rc = portable_apply_commit_dir (txn->env, txn->active_dir);
      if (rc == SQLITE_OK)
        {
          txn->staging_promoted = true;
          sqlite3_free (txn->active_dir);
          txn->active_dir = NULL;
        }
    }
  sqlite3_free (commit_dir);
  return rc;
}

static int
portable_stream_file (FILE *file, const objstore_stream_writer *writer,
                      size_t chunk)
{
  if (file == NULL || writer == NULL || writer->push == NULL)
    {
      return SQLITE_MISUSE;
    }
  unsigned char *buffer = sqlite3_malloc64 (chunk);
  if (buffer == NULL)
    {
      return SQLITE_NOMEM;
    }
  int rc = SQLITE_OK;
  while (rc == SQLITE_OK)
    {
      size_t nread = fread (buffer, 1, chunk, file);
      if (nread > 0)
        {
          rc = writer->push (writer->ctx, buffer, nread);
        }
      else
        {
          if (ferror (file))
            {
              rc = SQLITE_IOERR_READ;
            }
          break;
        }
    }
  sqlite3_free (buffer);
  return rc;
}

static int
portable_seek (FILE *file, sqlite3_uint64 offset)
{
#if defined(_WIN32)
  return _fseeki64 (file, (long long)offset, SEEK_SET);
#else
  if (offset > (sqlite3_uint64)LONG_MAX)
    {
      return -1;
    }
  return fseek (file, (long)offset, SEEK_SET);
#endif
}

static int
portable_stream_file_range (FILE *file, const objstore_stream_writer *writer,
                            size_t chunk, sqlite3_uint64 offset,
                            sqlite3_uint64 length)
{
  if (file == NULL || writer == NULL || writer->push == NULL)
    {
      return SQLITE_MISUSE;
    }
  if (length == 0)
    {
      return SQLITE_OK;
    }
  if (offset > (sqlite3_uint64)LONG_MAX)
    {
      return SQLITE_RANGE;
    }
  if (portable_seek (file, offset) != 0)
    {
      return SQLITE_IOERR;
    }
  unsigned char *buffer = sqlite3_malloc64 (chunk);
  if (buffer == NULL)
    {
      return SQLITE_NOMEM;
    }
  sqlite3_uint64 remaining = length;
  int rc = SQLITE_OK;
  while (remaining > 0)
    {
      size_t to_read = (remaining < (sqlite3_uint64)chunk)
                           ? (size_t)remaining
                           : chunk;
      size_t nread = fread (buffer, 1, to_read, file);
      if (nread != to_read)
        {
          rc = ferror (file) ? SQLITE_IOERR_READ : SQLITE_RANGE;
          break;
        }
      rc = writer->push (writer->ctx, buffer, nread);
      if (rc != SQLITE_OK)
        {
          break;
        }
      remaining -= (sqlite3_uint64)nread;
    }
  sqlite3_free (buffer);
  return rc;
}

static int
portable_apply_put (objstore_portable_backend_env *env, const char *commit_dir,
                    const char *hex_id)
{
  char *payload_dir = objstore_fs_path_join (commit_dir, OBJSTORE_PUT_DIR);
  if (payload_dir == NULL)
    {
      return SQLITE_NOMEM;
    }
  char *filename = sqlite3_mprintf ("%s%s", hex_id, OBJSTORE_FILE_SUFFIX);
  if (filename == NULL)
    {
      sqlite3_free (payload_dir);
      return SQLITE_NOMEM;
    }
  char *source = objstore_fs_path_join (payload_dir, filename);
  sqlite3_free (payload_dir);
  sqlite3_free (filename);
  if (source == NULL)
    {
      return SQLITE_NOMEM;
    }

  char *dest_dir = objstore_fs_object_directory (env->objects_root,
                                                 env->shard_width, hex_id);
  if (dest_dir == NULL)
    {
      sqlite3_free (source);
      return SQLITE_NOMEM;
    }
  int rc = objstore_fs_mkdirs (dest_dir);
  if (rc != SQLITE_OK)
    {
      sqlite3_free (dest_dir);
      sqlite3_free (source);
      return rc;
    }
  char *dest_name
      = objstore_fs_object_path (env->objects_root, env->shard_width, hex_id);
  if (dest_name == NULL)
    {
      sqlite3_free (dest_dir);
      sqlite3_free (source);
      return SQLITE_NOMEM;
    }
  if (unlink (dest_name) != 0 && errno != ENOENT)
    {
      sqlite3_free (dest_dir);
      sqlite3_free (source);
      sqlite3_free (dest_name);
      return SQLITE_IOERR;
    }
  if (rename (source, dest_name) != 0)
    {
      int err = errno;
      sqlite3_free (dest_dir);
      sqlite3_free (source);
      sqlite3_free (dest_name);
      if (err == ENOENT && objstore_fs_path_exists (dest_name))
        {
          return SQLITE_OK;
        }
      return SQLITE_IOERR;
    }
  sqlite3_free (dest_dir);
  sqlite3_free (source);
  sqlite3_free (dest_name);
  return SQLITE_OK;
}

static int
portable_apply_delete (objstore_portable_backend_env *env, const char *hex_id)
{
  char *path
      = objstore_fs_object_path (env->objects_root, env->shard_width, hex_id);
  if (path == NULL)
    {
      return SQLITE_NOMEM;
    }
  if (unlink (path) != 0 && errno != ENOENT)
    {
      sqlite3_free (path);
      return SQLITE_IOERR;
    }
  sqlite3_free (path);
  return SQLITE_OK;
}

static int
portable_apply_commit_dir (objstore_portable_backend_env *env,
                           const char *commit_dir)
{
  char *manifest_path
      = objstore_fs_path_join (commit_dir, OBJSTORE_MANIFEST_FILE);
  if (manifest_path == NULL)
    {
      return SQLITE_NOMEM;
    }
  FILE *manifest = fopen (manifest_path, "rb");
  sqlite3_free (manifest_path);
  if (manifest == NULL)
    {
      return SQLITE_IOERR;
    }
  char line[256];
  int rc = SQLITE_OK;
  while (fgets (line, sizeof (line), manifest) != NULL && rc == SQLITE_OK)
    {
      char op[4] = { 0 };
      char hex[OBJSTORE_ID_SIZE * 2 + 1] = { 0 };
      if (sscanf (line, "%3s %64s", op, hex) != 2)
        {
          rc = SQLITE_CORRUPT;
          break;
        }
      if (strcmp (op, "PUT") == 0)
        {
          rc = portable_apply_put (env, commit_dir, hex);
        }
      else if (strcmp (op, "DEL") == 0)
        {
          rc = portable_apply_delete (env, hex);
        }
      else
        {
          rc = SQLITE_CORRUPT;
        }
    }
  if (rc == SQLITE_OK && ferror (manifest))
    {
      rc = SQLITE_IOERR_READ;
    }
  fclose (manifest);
  if (rc != SQLITE_OK)
    {
      return rc;
    }
  return objstore_fs_remove_tree (commit_dir);
}

static int
portable_cleanup_directory (const char *path)
{
  if (path == NULL)
    {
      return SQLITE_MISUSE;
    }
  DIR *dir = opendir (path);
  if (dir == NULL)
    {
      return (errno == ENOENT) ? objstore_fs_mkdirs (path) : SQLITE_IOERR;
    }
  struct dirent *entry = NULL;
  while ((entry = readdir (dir)) != NULL)
    {
      if (strcmp (entry->d_name, ".") == 0
          || strcmp (entry->d_name, "..") == 0)
        {
          continue;
        }
      char *child = objstore_fs_path_join (path, entry->d_name);
      if (child == NULL)
        {
          closedir (dir);
          return SQLITE_NOMEM;
        }
      int rc = objstore_fs_remove_tree (child);
      sqlite3_free (child);
      if (rc != SQLITE_OK)
        {
          closedir (dir);
          return rc;
        }
    }
  closedir (dir);
  return SQLITE_OK;
}

static int
portable_recover (objstore_portable_backend_env *env)
{
  int rc = portable_cleanup_directory (env->wal_active_root);
  if (rc != SQLITE_OK)
    {
      return rc;
    }
  DIR *dir = opendir (env->wal_commit_root);
  if (dir == NULL)
    {
      return (errno == ENOENT) ? SQLITE_OK : SQLITE_IOERR;
    }
  struct dirent *entry = NULL;
  while ((entry = readdir (dir)) != NULL)
    {
      if (strcmp (entry->d_name, ".") == 0
          || strcmp (entry->d_name, "..") == 0)
        {
          continue;
        }
      char *commit_dir
          = objstore_fs_path_join (env->wal_commit_root, entry->d_name);
      if (commit_dir == NULL)
        {
          closedir (dir);
          return SQLITE_NOMEM;
        }
      rc = portable_apply_commit_dir (env, commit_dir);
      sqlite3_free (commit_dir);
      if (rc != SQLITE_OK)
        {
          closedir (dir);
          return rc;
        }
    }
  closedir (dir);
  return SQLITE_OK;
}

int
objstore_portable_open_env (objstore_portable_kind kind, sqlite3 *db,
                            const objstore_config *config,
                            objstore_backend_env **out_env)
{
  if (out_env == NULL)
    {
      return SQLITE_MISUSE;
    }
  const objstore_portable_traits *traits = portable_traits_for_kind (kind);
  if (traits == NULL)
    {
      return SQLITE_NOTFOUND;
    }
#if !defined(__EMSCRIPTEN__)
  if (traits->requires_opfs)
    {
      sqlite3_log (SQLITE_NOTFOUND,
                   "objstore opfs backend requested but unavailable on host");
      return SQLITE_NOTFOUND;
    }
#else
  if (traits->requires_opfs)
    {
      if (objstore_portable_mount_opfs () != 0)
        {
          sqlite3_log (
              SQLITE_IOERR,
              "objstore failed to mount OPFS (worker context required)");
          return SQLITE_IOERR;
        }
    }
#endif

  objstore_portable_backend_env *env = sqlite3_malloc (sizeof (*env));
  if (env == NULL)
    {
      return SQLITE_NOMEM;
    }
  memset (env, 0, sizeof (*env));
  env->db = db;
  env->chunk_size = objstore_effective_chunk_size (config);
  env->shard_width = objstore_fs_effective_shard_width (
      config, PORTABLE_DEFAULT_SHARD_WIDTH);
  env->sync_mode = (config != NULL) ? config->sync_mode : OBJSTORE_SYNC_FULL;
  env->traits = traits;

  const char *root = (config != NULL && config->storage_root != NULL
                      && config->storage_root[0] != '\0')
                         ? config->storage_root
                         : traits->default_root;
  env->storage_root = sqlite3_mprintf ("%s", root);
  if (env->storage_root == NULL)
    {
      portable_env_free (env);
      return SQLITE_NOMEM;
    }
  env->objects_root = objstore_fs_path_join (env->storage_root, "objects");
  env->wal_root = objstore_fs_path_join (env->storage_root, "wal");
  env->wal_active_root = objstore_fs_path_join (env->wal_root, "active");
  env->wal_commit_root = objstore_fs_path_join (env->wal_root, "commit");
  if (env->objects_root == NULL || env->wal_root == NULL
      || env->wal_active_root == NULL || env->wal_commit_root == NULL)
    {
      portable_env_free (env);
      return SQLITE_NOMEM;
    }

  int rc = objstore_fs_mkdirs (env->objects_root);
  if (rc == SQLITE_OK)
    rc = objstore_fs_mkdirs (env->wal_root);
  if (rc == SQLITE_OK)
    rc = objstore_fs_mkdirs (env->wal_active_root);
  if (rc == SQLITE_OK)
    rc = objstore_fs_mkdirs (env->wal_commit_root);
  if (rc != SQLITE_OK)
    {
      portable_env_free (env);
      return rc;
    }

  rc = portable_recover (env);
  if (rc != SQLITE_OK)
    {
      portable_env_free (env);
      return rc;
    }

  *out_env = (objstore_backend_env *)env;
  return SQLITE_OK;
}

void
objstore_portable_close_env (objstore_backend_env *env)
{
  portable_env_free (portable_env_from (env));
}

static int
portable_writer_flush (objstore_portable_staged_writer *writer)
{
  if (writer == NULL || writer->file == NULL)
    {
      return SQLITE_OK;
    }
  if (fflush (writer->file) != 0)
    {
      return SQLITE_IOERR_FSYNC;
    }
  return SQLITE_OK;
}

static void
portable_writer_dispose (objstore_portable_staged_writer *writer,
                         bool unlink_temp)
{
  if (writer == NULL)
    {
      return;
    }
  if (writer->file != NULL)
    {
      fclose (writer->file);
      writer->file = NULL;
    }
  if (unlink_temp && writer->temp_path != NULL)
    {
      unlink (writer->temp_path);
    }
  sqlite3_free (writer->temp_path);
  sqlite3_free (writer);
}

int
objstore_portable_begin_txn (objstore_backend_env *env,
                             objstore_backend_txn **out_txn)
{
  if (env == NULL || out_txn == NULL)
    {
      return SQLITE_MISUSE;
    }
  objstore_portable_backend_env *portable_env = portable_env_from (env);
  objstore_portable_backend_txn *txn = sqlite3_malloc (sizeof (*txn));
  if (txn == NULL)
    {
      return SQLITE_NOMEM;
    }
  memset (txn, 0, sizeof (*txn));
  txn->env = portable_env;
  objstore_fs_random_hex (txn->txn_id, sizeof (txn->txn_id));
  txn->active_dir
      = objstore_fs_path_join (portable_env->wal_active_root, txn->txn_id);
  if (txn->active_dir == NULL)
    {
      portable_txn_free (txn);
      return SQLITE_NOMEM;
    }
  int rc = objstore_fs_mkdirs (txn->active_dir);
  if (rc == SQLITE_OK)
    {
      char *put_dir = portable_active_subdir (txn, OBJSTORE_PUT_DIR);
      char *del_dir = portable_active_subdir (txn, OBJSTORE_DELETE_DIR);
      if (put_dir == NULL || del_dir == NULL)
        {
          sqlite3_free (put_dir);
          sqlite3_free (del_dir);
          portable_txn_free (txn);
          return SQLITE_NOMEM;
        }
      rc = objstore_fs_mkdirs (put_dir);
      if (rc == SQLITE_OK)
        {
          rc = objstore_fs_mkdirs (del_dir);
        }
      sqlite3_free (put_dir);
      sqlite3_free (del_dir);
    }
  if (rc != SQLITE_OK)
    {
      portable_txn_free (txn);
      return rc;
    }
  txn->manifest = sqlite3_str_new (NULL);
  if (txn->manifest == NULL)
    {
      portable_txn_free (txn);
      return SQLITE_NOMEM;
    }
  txn->manifest_written = false;
  txn->staging_promoted = false;
  *out_txn = (objstore_backend_txn *)txn;
  return SQLITE_OK;
}

int
objstore_portable_commit_txn (objstore_backend_txn *txn)
{
  if (txn == NULL)
    {
      return SQLITE_MISUSE;
    }
  objstore_portable_backend_txn *portable_txn = portable_txn_from (txn);
  int rc = portable_promote_staged (portable_txn);
  if (rc != SQLITE_OK)
    {
      portable_txn_free (portable_txn);
      return rc;
    }
  portable_txn_free (portable_txn);
  return SQLITE_OK;
}

void
objstore_portable_rollback_txn (objstore_backend_txn *txn)
{
  if (txn == NULL)
    {
      return;
    }
  objstore_portable_backend_txn *portable_txn = portable_txn_from (txn);
  objstore_portable_rollback_staged (txn);
  portable_txn_free (portable_txn);
}

int
objstore_portable_savepoint_begin (objstore_backend_txn *txn)
{
  if (txn == NULL)
    {
      return SQLITE_MISUSE;
    }
  objstore_portable_backend_txn *portable_txn = portable_txn_from (txn);
  int rc = portable_savepoint_frames_ensure_capacity (portable_txn);
  if (rc != SQLITE_OK)
    {
      return rc;
    }
  objstore_portable_savepoint_frame *frame
      = &portable_txn->frames[portable_txn->frame_count];
  portable_savepoint_frame_dispose (frame);
  frame->manifest_length
      = (portable_txn->manifest != NULL)
            ? (size_t)sqlite3_str_length (portable_txn->manifest)
            : 0u;
  ++portable_txn->frame_count;
  return SQLITE_OK;
}

int
objstore_portable_savepoint_release (objstore_backend_txn *txn)
{
  if (txn == NULL)
    {
      return SQLITE_MISUSE;
    }
  objstore_portable_backend_txn *portable_txn = portable_txn_from (txn);
  if (portable_txn->frame_count == 0)
    {
      return SQLITE_OK;
    }
  objstore_portable_savepoint_frame *child
      = &portable_txn->frames[portable_txn->frame_count - 1];
  if (portable_txn->frame_count > 1)
    {
      objstore_portable_savepoint_frame *parent
          = &portable_txn->frames[portable_txn->frame_count - 2];
      for (size_t i = 0; i < child->entry_count; ++i)
        {
          if (portable_savepoint_frame_find_entry (parent,
                                                   &child->entries[i].id)
              >= 0)
            {
              continue;
            }
          int rc = portable_savepoint_entries_ensure_capacity (parent);
          if (rc != SQLITE_OK)
            {
              return rc;
            }
          parent->entries[parent->entry_count++] = child->entries[i];
        }
    }
  portable_savepoint_frame_dispose (child);
  --portable_txn->frame_count;
  return SQLITE_OK;
}

int
objstore_portable_savepoint_rollback (objstore_backend_txn *txn)
{
  if (txn == NULL)
    {
      return SQLITE_MISUSE;
    }
  objstore_portable_backend_txn *portable_txn = portable_txn_from (txn);
  if (portable_txn->frame_count == 0)
    {
      return SQLITE_OK;
    }
  objstore_portable_savepoint_frame *frame
      = &portable_txn->frames[portable_txn->frame_count - 1];
  for (size_t i = frame->entry_count; i > 0; --i)
    {
      int rc = portable_savepoint_restore_entry (portable_txn,
                                                 &frame->entries[i - 1]);
      if (rc != SQLITE_OK)
        {
          return rc;
        }
    }
  return portable_manifest_reset_prefix (portable_txn, frame->manifest_length);
}

int
objstore_portable_staged_write_begin (objstore_backend_txn *txn,
                                      objstore_backend_staged_writer **out)
{
  if (txn == NULL || out == NULL)
    {
      return SQLITE_MISUSE;
    }
  objstore_portable_backend_txn *portable_txn = portable_txn_from (txn);
  objstore_portable_staged_writer *writer = sqlite3_malloc (sizeof (*writer));
  if (writer == NULL)
    {
      return SQLITE_NOMEM;
    }
  memset (writer, 0, sizeof (*writer));
  writer->txn = portable_txn;
  writer->expected_size = -1;
  char *put_dir = portable_active_subdir (portable_txn, OBJSTORE_PUT_DIR);
  if (put_dir == NULL)
    {
      sqlite3_free (writer);
      return SQLITE_NOMEM;
    }
  char random_hex[33];
  objstore_fs_random_hex (random_hex, sizeof (random_hex));
  char *temp_name
      = sqlite3_mprintf (".tmp_%s%s", random_hex, OBJSTORE_FILE_SUFFIX);
  if (temp_name == NULL)
    {
      sqlite3_free (put_dir);
      sqlite3_free (writer);
      return SQLITE_NOMEM;
    }
  writer->temp_path = objstore_fs_path_join (put_dir, temp_name);
  sqlite3_free (put_dir);
  sqlite3_free (temp_name);
  if (writer->temp_path == NULL)
    {
      sqlite3_free (writer);
      return SQLITE_NOMEM;
    }
  writer->file = fopen (writer->temp_path, "wb");
  if (writer->file == NULL)
    {
      portable_writer_dispose (writer, true);
      return SQLITE_IOERR;
    }
  *out = (objstore_backend_staged_writer *)writer;
  return SQLITE_OK;
}

int
objstore_portable_staged_write_push (objstore_backend_staged_writer *writer,
                                     const void *buffer, size_t nread)
{
  objstore_portable_staged_writer *portable_writer
      = portable_writer_from (writer);
  if (portable_writer == NULL || portable_writer->file == NULL)
    {
      return SQLITE_MISUSE;
    }
  if (nread == 0)
    {
      return SQLITE_OK;
    }
  if (fwrite (buffer, 1, nread, portable_writer->file) != nread)
    {
      return SQLITE_IOERR_WRITE;
    }
  portable_writer->bytes_written += (sqlite3_int64)nread;
  return SQLITE_OK;
}

int
objstore_portable_staged_write_finalize (
    objstore_backend_staged_writer *writer, const objstore_id *id)
{
  objstore_portable_staged_writer *portable_writer
      = portable_writer_from (writer);
  if (portable_writer == NULL)
    {
      return SQLITE_MISUSE;
    }
  int rc = portable_writer_flush (portable_writer);
  if (rc != SQLITE_OK)
    {
      portable_writer_dispose (portable_writer, true);
      return rc;
    }
  if (id == NULL)
    {
      portable_writer_dispose (portable_writer, true);
      return SQLITE_OK;
    }
  char hex[OBJSTORE_ID_SIZE * 2 + 1];
  objstore_fs_id_to_hex (id, hex);
  rc = portable_savepoint_capture_state (portable_writer->txn, id);
  if (rc != SQLITE_OK)
    {
      portable_writer_dispose (portable_writer, true);
      return rc;
    }
  char *dest = portable_put_path (portable_writer->txn, hex);
  if (dest == NULL)
    {
      portable_writer_dispose (portable_writer, true);
      return SQLITE_NOMEM;
    }
  if (rename (portable_writer->temp_path, dest) != 0)
    {
      int err = errno;
      sqlite3_free (dest);
      portable_writer_dispose (portable_writer, true);
      return (err == ENOSPC || err == EFBIG) ? SQLITE_FULL : SQLITE_IOERR;
    }
  sqlite3_free (dest);
  objstore_portable_backend_txn *txn = portable_writer->txn;
  portable_writer_dispose (portable_writer, false);
  return portable_manifest_append (txn, "PUT", hex);
}

int
objstore_portable_staged_write_set_hint (
    objstore_backend_staged_writer *writer, sqlite3_int64 size_hint)
{
  objstore_portable_staged_writer *portable_writer
      = portable_writer_from (writer);
  if (portable_writer == NULL)
    {
      return SQLITE_MISUSE;
    }
  portable_writer->expected_size = size_hint;
  return SQLITE_OK;
}

int
objstore_portable_commit_staged (objstore_backend_txn *txn)
{
  return portable_promote_staged (portable_txn_from (txn));
}

void
objstore_portable_rollback_staged (objstore_backend_txn *txn)
{
  objstore_portable_backend_txn *portable_txn = portable_txn_from (txn);
  if (portable_txn == NULL || portable_txn->staging_promoted)
    {
      return;
    }
  portable_savepoint_frames_clear (portable_txn);
  portable_txn_reset_active (portable_txn, true);
}

static int
portable_stream_pending (objstore_portable_backend_txn *txn,
                         const char *hex_id,
                         const objstore_stream_writer *writer)
{
  char *pending = portable_put_path (txn, hex_id);
  if (pending == NULL)
    {
      return SQLITE_NOMEM;
    }
  FILE *file = fopen (pending, "rb");
  sqlite3_free (pending);
  if (file == NULL)
    {
      return SQLITE_NOTFOUND;
    }
  int rc = portable_stream_file (file, writer, txn->env->chunk_size);
  fclose (file);
  return rc;
}

static int
portable_get_committed (objstore_portable_backend_env *env, const char *hex_id,
                        const objstore_stream_writer *writer)
{
  char *path
      = objstore_fs_object_path (env->objects_root, env->shard_width, hex_id);
  if (path == NULL)
    {
      return SQLITE_NOMEM;
    }
  FILE *file = fopen (path, "rb");
  sqlite3_free (path);
  if (file == NULL)
    {
      return (errno == ENOENT) ? SQLITE_NOTFOUND : SQLITE_IOERR;
    }
  int rc = portable_stream_file (file, writer, env->chunk_size);
  fclose (file);
  return rc;
}

static int
portable_get_size_pending (objstore_portable_backend_txn *txn,
                           const char *hex_id, sqlite3_int64 *out_size)
{
  if (txn == NULL || txn->active_dir == NULL)
    {
      return SQLITE_NOTFOUND;
    }
  char *pending = portable_put_path (txn, hex_id);
  if (pending == NULL)
    {
      return SQLITE_NOMEM;
    }
  int rc = portable_stat_size (pending, out_size);
  sqlite3_free (pending);
  return rc;
}

static int
portable_get_size_committed (objstore_portable_backend_env *env,
                             const char *hex_id, sqlite3_int64 *out_size)
{
  char *path
      = objstore_fs_object_path (env->objects_root, env->shard_width, hex_id);
  if (path == NULL)
    {
      return SQLITE_NOMEM;
    }
  int rc = portable_stat_size (path, out_size);
  sqlite3_free (path);
  return rc;
}

int
objstore_portable_get_size (objstore_backend_txn *txn, const objstore_id *id,
                            sqlite3_int64 *out_size)
{
  if (txn == NULL || id == NULL || out_size == NULL)
    {
      return SQLITE_MISUSE;
    }
  objstore_portable_backend_txn *portable_txn = portable_txn_from (txn);
  char hex[OBJSTORE_ID_SIZE * 2 + 1];
  objstore_fs_id_to_hex (id, hex);

  char *marker = portable_delete_marker (portable_txn, hex);
  if (marker == NULL)
    {
      return SQLITE_NOMEM;
    }
  if (objstore_fs_path_exists (marker))
    {
      sqlite3_free (marker);
      return SQLITE_NOTFOUND;
    }
  sqlite3_free (marker);

  int rc = portable_get_size_pending (portable_txn, hex, out_size);
  if (rc != SQLITE_NOTFOUND)
    {
      return rc;
    }
  return portable_get_size_committed (portable_txn->env, hex, out_size);
}

static int
portable_get_range_committed (objstore_portable_backend_env *env,
                              const char *hex_id, sqlite3_uint64 offset,
                              sqlite3_uint64 length,
                              const objstore_stream_writer *writer)
{
  char *path
      = objstore_fs_object_path (env->objects_root, env->shard_width, hex_id);
  if (path == NULL)
    {
      return SQLITE_NOMEM;
    }
  FILE *file = fopen (path, "rb");
  sqlite3_free (path);
  if (file == NULL)
    {
      return (errno == ENOENT) ? SQLITE_NOTFOUND : SQLITE_IOERR;
    }
  int rc = portable_stream_file_range (file, writer, env->chunk_size, offset,
                                       length);
  fclose (file);
  return rc;
}

int
objstore_portable_get_range (objstore_backend_txn *txn, const objstore_id *id,
                             sqlite3_uint64 offset, sqlite3_uint64 length,
                             const objstore_stream_writer *writer)
{
  if (txn == NULL || id == NULL || writer == NULL || writer->push == NULL)
    {
      return SQLITE_MISUSE;
    }
  if (length == 0)
    {
      return SQLITE_OK;
    }
  if (offset > UINT64_MAX - length)
    {
      return SQLITE_RANGE;
    }
  objstore_portable_backend_txn *portable_txn = portable_txn_from (txn);
  char hex[OBJSTORE_ID_SIZE * 2 + 1];
  objstore_fs_id_to_hex (id, hex);

  char *marker = portable_delete_marker (portable_txn, hex);
  if (marker == NULL)
    {
      return SQLITE_NOMEM;
    }
  if (objstore_fs_path_exists (marker))
    {
      sqlite3_free (marker);
      return SQLITE_NOTFOUND;
    }
  sqlite3_free (marker);

  if (portable_txn->active_dir != NULL)
    {
      char *pending = portable_put_path (portable_txn, hex);
      if (pending == NULL)
        {
          return SQLITE_NOMEM;
        }
      if (objstore_fs_path_exists (pending))
        {
          FILE *file = fopen (pending, "rb");
          sqlite3_free (pending);
          if (file == NULL)
            {
              return SQLITE_IOERR;
            }
          int rc = portable_stream_file_range (file, writer,
                                               portable_txn->env->chunk_size,
                                               offset, length);
          fclose (file);
          return rc;
        }
      sqlite3_free (pending);
    }

  return portable_get_range_committed (portable_txn->env, hex, offset, length,
                                       writer);
}

int
objstore_portable_put (objstore_backend_txn *txn, const objstore_id *id,
                       const objstore_stream_reader *reader)
{
  if (txn == NULL || id == NULL || reader == NULL || reader->pull == NULL)
    {
      return SQLITE_MISUSE;
    }
  objstore_portable_backend_txn *portable_txn = portable_txn_from (txn);
  int rc = portable_savepoint_capture_state (portable_txn, id);
  if (rc != SQLITE_OK)
    {
      return rc;
    }
  objstore_backend_staged_writer *writer = NULL;
  rc = objstore_portable_staged_write_begin (txn, &writer);
  if (rc != SQLITE_OK)
    {
      return rc;
    }
  unsigned char *buffer = sqlite3_malloc64 (portable_txn->env->chunk_size);
  if (buffer == NULL)
    {
      objstore_portable_staged_write_finalize (writer, NULL);
      return SQLITE_NOMEM;
    }
  while (true)
    {
      size_t chunk = 0;
      rc = reader->pull (reader->ctx, buffer, portable_txn->env->chunk_size,
                         &chunk);
      if (rc == SQLITE_DONE)
        {
          rc = SQLITE_OK;
          break;
        }
      if (rc != SQLITE_OK)
        {
          break;
        }
      if (chunk == 0)
        {
          continue;
        }
      rc = objstore_portable_staged_write_push (writer, buffer, chunk);
      if (rc != SQLITE_OK)
        {
          break;
        }
    }
  sqlite3_free (buffer);
  if (rc != SQLITE_OK)
    {
      objstore_portable_staged_write_finalize (writer, NULL);
      return rc;
    }
  rc = objstore_portable_staged_write_finalize (writer, id);
  if (rc != SQLITE_OK)
    {
      return rc;
    }
  return SQLITE_OK;
}

int
objstore_portable_get (objstore_backend_txn *txn, const objstore_id *id,
                       const objstore_stream_writer *writer)
{
  if (txn == NULL || id == NULL || writer == NULL || writer->push == NULL)
    {
      return SQLITE_MISUSE;
    }
  objstore_portable_backend_txn *portable_txn = portable_txn_from (txn);
  char hex[OBJSTORE_ID_SIZE * 2 + 1];
  objstore_fs_id_to_hex (id, hex);
  char *marker = portable_delete_marker (portable_txn, hex);
  if (marker == NULL)
    {
      return SQLITE_NOMEM;
    }
  if (objstore_fs_path_exists (marker))
    {
      sqlite3_free (marker);
      return SQLITE_NOTFOUND;
    }
  sqlite3_free (marker);
  int rc = portable_stream_pending (portable_txn, hex, writer);
  if (rc != SQLITE_NOTFOUND)
    {
      return rc;
    }
  return portable_get_committed (portable_txn->env, hex, writer);
}

int
objstore_portable_delete (objstore_backend_txn *txn, const objstore_id *id)
{
  if (txn == NULL || id == NULL)
    {
      return SQLITE_MISUSE;
    }
  objstore_portable_backend_txn *portable_txn = portable_txn_from (txn);
  int rc = portable_savepoint_capture_state (portable_txn, id);
  if (rc != SQLITE_OK)
    {
      return rc;
    }
  char hex[OBJSTORE_ID_SIZE * 2 + 1];
  objstore_fs_id_to_hex (id, hex);
  char *marker = portable_delete_marker (portable_txn, hex);
  if (marker == NULL)
    {
      return SQLITE_NOMEM;
    }
  FILE *file = fopen (marker, "wb");
  if (file != NULL)
    {
      fclose (file);
    }
  sqlite3_free (marker);
  return portable_manifest_append (portable_txn, "DEL", hex);
}

int
objstore_portable_exists (objstore_backend_txn *txn, const objstore_id *id)
{
  if (txn == NULL || id == NULL)
    {
      return SQLITE_MISUSE;
    }
  objstore_portable_backend_txn *portable_txn = portable_txn_from (txn);
  char hex[OBJSTORE_ID_SIZE * 2 + 1];
  objstore_fs_id_to_hex (id, hex);
  char *marker = portable_delete_marker (portable_txn, hex);
  if (marker == NULL)
    {
      return SQLITE_NOMEM;
    }
  bool has_marker = objstore_fs_path_exists (marker);
  sqlite3_free (marker);
  if (has_marker)
    {
      return SQLITE_NOTFOUND;
    }
  char *pending = portable_put_path (portable_txn, hex);
  if (pending == NULL)
    {
      return SQLITE_NOMEM;
    }
  bool has_pending = objstore_fs_path_exists (pending);
  sqlite3_free (pending);
  if (has_pending)
    {
      return SQLITE_OK;
    }
  char *committed = objstore_fs_object_path (
      portable_txn->env->objects_root, portable_txn->env->shard_width, hex);
  if (committed == NULL)
    {
      return SQLITE_NOMEM;
    }
  bool has_committed = objstore_fs_path_exists (committed);
  sqlite3_free (committed);
  return has_committed ? SQLITE_OK : SQLITE_NOTFOUND;
}

static int
portable_collect_ids_from_dir (const char *path, objstore_id **ids,
                               size_t *count, size_t *capacity)
{
  DIR *dir = opendir (path);
  if (dir == NULL)
    {
      return (errno == ENOENT) ? SQLITE_OK : SQLITE_IOERR;
    }
  const size_t suffix_len = strlen (OBJSTORE_FILE_SUFFIX);
  int rc = SQLITE_OK;
  struct dirent *entry = NULL;
  while ((entry = readdir (dir)) != NULL && rc == SQLITE_OK)
    {
      if (strcmp (entry->d_name, ".") == 0
          || strcmp (entry->d_name, "..") == 0)
        {
          continue;
        }
      char *child = objstore_fs_path_join (path, entry->d_name);
      if (child == NULL)
        {
          rc = SQLITE_NOMEM;
          break;
        }
      struct stat st;
      if (stat (child, &st) != 0)
        {
          sqlite3_free (child);
          continue;
        }
      if (S_ISDIR (st.st_mode))
        {
          rc = portable_collect_ids_from_dir (child, ids, count, capacity);
          sqlite3_free (child);
          continue;
        }
      if (!S_ISREG (st.st_mode))
        {
          sqlite3_free (child);
          continue;
        }
      size_t len = strlen (entry->d_name);
      if (len <= suffix_len
          || strcmp (entry->d_name + len - suffix_len, OBJSTORE_FILE_SUFFIX)
                 != 0)
        {
          sqlite3_free (child);
          continue;
        }
      size_t hex_len = len - suffix_len;
      if (hex_len != OBJSTORE_ID_SIZE * 2)
        {
          sqlite3_free (child);
          continue;
        }
      if (*capacity == *count)
        {
          size_t new_capacity = (*capacity == 0) ? 16 : (*capacity * 2);
          objstore_id *resized
              = sqlite3_realloc64 (*ids, new_capacity * sizeof (objstore_id));
          if (resized == NULL)
            {
              sqlite3_free (child);
              rc = SQLITE_NOMEM;
              break;
            }
          *ids = resized;
          *capacity = new_capacity;
        }
      char hex[OBJSTORE_ID_SIZE * 2 + 1] = { 0 };
      memcpy (hex, entry->d_name, hex_len);
      if (objstore_fs_hex_to_id (hex, &(*ids)[*count]) == SQLITE_OK)
        {
          ++(*count);
        }
      sqlite3_free (child);
    }
  closedir (dir);
  return rc;
}

static int
portable_collect_ids (objstore_portable_backend_env *env, objstore_id **ids,
                      size_t *count)
{
  objstore_id *buffer = NULL;
  size_t total = 0;
  size_t capacity = 0;
  int rc = portable_collect_ids_from_dir (env->objects_root, &buffer, &total,
                                          &capacity);
  if (rc != SQLITE_OK)
    {
      sqlite3_free (buffer);
      return rc;
    }
  if (total > 1)
    {
      qsort (buffer, total, sizeof (objstore_id), portable_compare_ids);
    }
  *ids = buffer;
  *count = total;
  return SQLITE_OK;
}

int
objstore_portable_scan_open (objstore_backend_txn *txn,
                             objstore_backend_cursor **out_cursor)
{
  if (txn == NULL || out_cursor == NULL)
    {
      return SQLITE_MISUSE;
    }
  objstore_portable_backend_txn *portable_txn = portable_txn_from (txn);
  objstore_id *ids = NULL;
  size_t count = 0;
  int rc = portable_collect_ids (portable_txn->env, &ids, &count);
  if (rc != SQLITE_OK)
    {
      return rc;
    }
  objstore_portable_cursor *cursor = sqlite3_malloc (sizeof (*cursor));
  if (cursor == NULL)
    {
      sqlite3_free (ids);
      return SQLITE_NOMEM;
    }
  cursor->env = portable_txn->env;
  cursor->ids = ids;
  cursor->count = count;
  cursor->index = 0;
  *out_cursor = (objstore_backend_cursor *)cursor;
  return SQLITE_OK;
}

int
objstore_portable_scan_next (objstore_backend_cursor *cursor,
                             objstore_id *out_id)
{
  if (cursor == NULL || out_id == NULL)
    {
      return SQLITE_MISUSE;
    }
  objstore_portable_cursor *portable_cursor = portable_cursor_from (cursor);
  if (portable_cursor->index >= portable_cursor->count)
    {
      return SQLITE_DONE;
    }
  *out_id = portable_cursor->ids[portable_cursor->index++];
  return SQLITE_OK;
}

void
objstore_portable_scan_close (objstore_backend_cursor *cursor)
{
  if (cursor == NULL)
    {
      return;
    }
  objstore_portable_cursor *portable_cursor = portable_cursor_from (cursor);
  sqlite3_free (portable_cursor->ids);
  sqlite3_free (portable_cursor);
}
