#if !defined(_WIN32)
/* Apple: _POSIX_C_SOURCE alone hides BSD flock() from <sys/file.h>. */
#if defined(__APPLE__)
#ifndef _DARWIN_C_SOURCE
#define _DARWIN_C_SOURCE 1
#endif
#else
#define _POSIX_C_SOURCE 200809L
#endif
#endif

#include "objstore/backend.h"

#include "backend_fs_common.h"

#if defined(_WIN32)
#include "win_dirent.h"
#else
#include <dirent.h>
#endif
#include <errno.h>
#include <fcntl.h>
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
#include <sys/file.h>
#endif
#if defined(__linux__)
extern int fallocate (int fd, int mode, off_t offset, off_t len);
#endif
#if defined(_WIN32)
#include <malloc.h>
#endif

#ifndef OBJSTORE_PATH_SEPARATOR
#define OBJSTORE_PATH_SEPARATOR '/'
#endif

#define FILE_BACKEND_DEFAULT_SHARD_WIDTH 2u
#define FILE_BACKEND_IO_ALIGNMENT 4096u
#define FILE_BACKEND_ROWIDX_DIR "rowidx"
#define FILE_BACKEND_ROWIDX_MARKER ".ready"
#define FILE_BACKEND_ROWIDX_SHARD_CHARS 2u

#define kObjstoreIdHexChars (OBJSTORE_ID_SIZE * 2u)

typedef struct file_backend_env
{
  sqlite3 *db;
  size_t chunk_size;
  char *objects_root;
  char *rowidx_root;
  char *staging_root;
  char *staging_active_root;
  char *staging_commit_root;
  size_t shard_width; /* number of hex chars used for sharding */
  objstore_sync_mode sync_mode;
  int lock_fd; /* shared flock held while this env is open */
} file_backend_env;

typedef struct file_backend_txn
{
  file_backend_env *env;
  char txn_id[33];
  char *active_dir;
  sqlite3_str *manifest_buffer;
  struct file_backend_savepoint_frame *frames;
  size_t frame_count;
  size_t frame_capacity;
  unsigned char *io_buffer;
  size_t io_buffer_size;
  bool staging_promoted;
  bool manifest_written;
} file_backend_txn;

typedef struct file_backend_savepoint_entry
{
  objstore_id id;
  bool had_pending;
  bool had_delete_marker;
} file_backend_savepoint_entry;

typedef struct file_backend_savepoint_frame
{
  size_t manifest_length;
  file_backend_savepoint_entry *entries;
  size_t entry_count;
  size_t entry_capacity;
} file_backend_savepoint_frame;

typedef struct file_backend_cursor
{
  file_backend_env *env;
  objstore_id *ids;
  size_t count;
  size_t index;
} file_backend_cursor;

typedef struct file_backend_staged_writer
{
  file_backend_txn *txn;
  int fd;
  char *temp_path;
  sqlite3_int64 expected_size;
  sqlite3_int64 bytes_written;
  bool prealloc_done;
} file_backend_staged_writer;

static int file_backend_collect_ids (file_backend_env *env,
                                     objstore_id **out_ids,
                                     size_t *out_count);
static int file_backend_collect_object_ids (file_backend_env *env,
                                            objstore_id **out_ids,
                                            size_t *out_count);

static bool
file_backend_force_disk_full (void)
{
  const char *flag = getenv ("OBJSTORE_FORCE_DISK_FULL");
  if (flag != NULL && flag[0] != '\0')
    {
#if defined(_WIN32)
      _putenv_s ("OBJSTORE_FORCE_DISK_FULL", "");
#else
      unsetenv ("OBJSTORE_FORCE_DISK_FULL");
#endif
      return true;
    }
  return false;
}
static int objstore_fsync_fd (int fd);
static void *file_backend_alloc_aligned (size_t size);
static void file_backend_free_aligned (void *ptr);
static bool file_backend_path_exists (const char *path);

static inline file_backend_txn *
file_txn_from (objstore_backend_txn *txn)
{
  return (file_backend_txn *)txn;
}

static inline file_backend_cursor *
file_cursor_from (objstore_backend_cursor *cursor)
{
  return (file_backend_cursor *)cursor;
}

static inline file_backend_staged_writer *
file_writer_from (objstore_backend_staged_writer *writer)
{
  return (file_backend_staged_writer *)writer;
}

static size_t
file_backend_effective_shard_width (const objstore_config *config)
{
  return objstore_fs_effective_shard_width (config,
                                            FILE_BACKEND_DEFAULT_SHARD_WIDTH);
}

static bool
file_backend_should_fsync_payload (const file_backend_env *env)
{
  return env != NULL && env->sync_mode == OBJSTORE_SYNC_FULL;
}

static bool
file_backend_should_fsync_manifest (const file_backend_env *env)
{
  if (env == NULL)
    {
      return true;
    }
  return env->sync_mode == OBJSTORE_SYNC_FULL
         || env->sync_mode == OBJSTORE_SYNC_METADATA;
}

static int
file_backend_flush_file (FILE *file, bool do_fsync)
{
  if (file == NULL)
    {
      return SQLITE_OK;
    }
  if (fflush (file) != 0)
    {
      return SQLITE_IOERR_FSYNC;
    }
  if (!do_fsync)
    {
      return SQLITE_OK;
    }
  return objstore_fsync_fd (fileno (file));
}

static int
file_backend_flush_fd (int fd, bool do_fsync)
{
  if (fd < 0)
    {
      return SQLITE_OK;
    }
  if (!do_fsync)
    {
      return SQLITE_OK;
    }
  return objstore_fsync_fd (fd);
}

static int
file_backend_write_all (int fd, const void *buffer, size_t nbytes)
{
  const unsigned char *cursor = (const unsigned char *)buffer;
  size_t remaining = nbytes;
  while (remaining > 0)
    {
      int wrote = (int)write (fd, cursor, remaining);
      if (wrote < 0)
        {
          if (errno == EINTR)
            {
              continue;
            }
          if (errno == ENOSPC || errno == EFBIG)
            {
              return SQLITE_FULL;
            }
          return SQLITE_IOERR_WRITE;
        }
      if (wrote == 0)
        {
          return SQLITE_IOERR_WRITE;
        }
      cursor += (size_t)wrote;
      remaining -= (size_t)wrote;
    }
  return SQLITE_OK;
}

static int
file_backend_preallocate_fd (int fd, sqlite3_int64 size)
{
  if (fd < 0 || size <= 0)
    {
      return SQLITE_OK;
    }
#if defined(__linux__)
  if (fallocate (fd, 0, 0, (off_t)size) == 0)
    {
      return SQLITE_OK;
    }
  if (errno != ENOSYS && errno != EOPNOTSUPP && errno != EPERM
      && errno != EINVAL)
    {
      return (errno == ENOSPC) ? SQLITE_FULL : SQLITE_IOERR;
    }
    /* fall back to ftruncate when unsupported */
#endif
  if (ftruncate (fd, (off_t)size) == 0)
    {
      return SQLITE_OK;
    }
  return (errno == ENOSPC) ? SQLITE_FULL : SQLITE_IOERR;
}

static void *
file_backend_alloc_aligned (size_t size)
{
#if defined(_WIN32)
  return (unsigned char *)_aligned_malloc (size, FILE_BACKEND_IO_ALIGNMENT);
#else
  void *ptr = NULL;
  if (posix_memalign (&ptr, FILE_BACKEND_IO_ALIGNMENT, size) != 0)
    {
      return NULL;
    }
  return ptr;
#endif
}

static void
file_backend_free_aligned (void *ptr)
{
#if defined(_WIN32)
  if (ptr != NULL)
    {
      _aligned_free (ptr);
    }
#else
  free (ptr);
#endif
}

static int
objstore_fsync_fd (int fd)
{
  if (fd < 0)
    {
      return SQLITE_OK;
    }
#if defined(_WIN32)
  if (_commit (fd) != 0)
    {
      return SQLITE_IOERR_FSYNC;
    }
#else
  if (fsync (fd) != 0)
    {
      return SQLITE_IOERR_FSYNC;
    }
#endif
  return SQLITE_OK;
}

static void
file_backend_manifest_dispose (file_backend_txn *txn)
{
  if (txn == NULL || txn->manifest_buffer == NULL)
    {
      return;
    }
  char *value = sqlite3_str_finish (txn->manifest_buffer);
  txn->manifest_buffer = NULL;
  if (value != NULL)
    {
      sqlite3_free (value);
    }
  txn->manifest_written = false;
}

static int
file_backend_manifest_flush (file_backend_txn *txn)
{
  if (txn == NULL)
    {
      return SQLITE_MISUSE;
    }
  if (txn->manifest_written)
    {
      return SQLITE_OK;
    }
  if (txn->manifest_buffer == NULL)
    {
      return SQLITE_MISUSE;
    }

  char *manifest_path
      = objstore_fs_path_join (txn->active_dir, OBJSTORE_MANIFEST_FILE);
  if (manifest_path == NULL)
    {
      return SQLITE_NOMEM;
    }
  FILE *manifest = fopen (manifest_path, "wb");
  sqlite3_free (manifest_path);
  if (manifest == NULL)
    {
      return SQLITE_IOERR;
    }

  const char *data = sqlite3_str_value (txn->manifest_buffer);
  const size_t length = (data != NULL)
                            ? (size_t)sqlite3_str_length (txn->manifest_buffer)
                            : 0u;
  if (length > 0 && fwrite (data, 1, length, manifest) != length)
    {
      int err = errno;
      fclose (manifest);
      if (err == ENOSPC || err == EFBIG)
        {
          return SQLITE_FULL;
        }
      return SQLITE_IOERR_WRITE;
    }

  int rc = file_backend_flush_file (
      manifest, file_backend_should_fsync_manifest (txn->env));
  fclose (manifest);
  if (rc != SQLITE_OK)
    {
      return rc;
    }

  txn->manifest_written = true;
  return SQLITE_OK;
}

static int
file_backend_manifest_append (file_backend_txn *txn, const char *op,
                              const char *hex_id)
{
  if (txn == NULL || txn->manifest_buffer == NULL)
    {
      return SQLITE_MISUSE;
    }
  sqlite3_str_appendf (txn->manifest_buffer, "%s %s\n", op, hex_id);
  return sqlite3_str_errcode (txn->manifest_buffer);
}

static char *
file_backend_put_path (file_backend_txn *txn, const char *hex_id)
{
  char *put_dir = objstore_fs_path_join (txn->active_dir, OBJSTORE_PUT_DIR);
  if (put_dir == NULL)
    {
      return NULL;
    }
  char *filename = sqlite3_mprintf ("%s%s", hex_id, OBJSTORE_FILE_SUFFIX);
  if (filename == NULL)
    {
      sqlite3_free (put_dir);
      return NULL;
    }
  char *path = objstore_fs_path_join (put_dir, filename);
  sqlite3_free (put_dir);
  sqlite3_free (filename);
  return path;
}

static char *
file_backend_delete_marker (file_backend_txn *txn, const char *hex_id)
{
  char *delete_dir
      = objstore_fs_path_join (txn->active_dir, OBJSTORE_DELETE_DIR);
  if (delete_dir == NULL)
    {
      return NULL;
    }
  char *path = objstore_fs_path_join (delete_dir, hex_id);
  sqlite3_free (delete_dir);
  return path;
}

static void
file_backend_savepoint_frame_dispose (file_backend_savepoint_frame *frame)
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
file_backend_savepoint_frames_clear (file_backend_txn *txn)
{
  if (txn == NULL)
    {
      return;
    }
  for (size_t i = 0; i < txn->frame_count; ++i)
    {
      file_backend_savepoint_frame_dispose (&txn->frames[i]);
    }
  sqlite3_free (txn->frames);
  txn->frames = NULL;
  txn->frame_count = 0;
  txn->frame_capacity = 0;
}

static int
file_backend_savepoint_frames_ensure_capacity (file_backend_txn *txn)
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
  file_backend_savepoint_frame *resized = sqlite3_realloc64 (
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
file_backend_savepoint_entries_ensure_capacity (
    file_backend_savepoint_frame *frame)
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
  file_backend_savepoint_entry *resized = sqlite3_realloc64 (
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
file_backend_savepoint_frame_find_entry (const file_backend_savepoint_frame *frame,
                                         const objstore_id *id)
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
file_backend_manifest_reset_prefix (file_backend_txn *txn, size_t prefix_length)
{
  if (txn == NULL || txn->manifest_buffer == NULL)
    {
      return SQLITE_MISUSE;
    }
  const char *current = sqlite3_str_value (txn->manifest_buffer);
  const size_t current_length
      = (size_t)sqlite3_str_length (txn->manifest_buffer);
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

  file_backend_manifest_dispose (txn);
  txn->manifest_buffer = replacement;
  txn->manifest_written = false;
  return SQLITE_OK;
}

static int
file_backend_savepoint_capture_state (file_backend_txn *txn,
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
  file_backend_savepoint_frame *frame = &txn->frames[txn->frame_count - 1];
  if (file_backend_savepoint_frame_find_entry (frame, id) >= 0)
    {
      return SQLITE_OK;
    }
  int rc = file_backend_savepoint_entries_ensure_capacity (frame);
  if (rc != SQLITE_OK)
    {
      return rc;
    }

  char hex_id[kObjstoreIdHexChars + 1];
  objstore_fs_id_to_hex (id, hex_id);
  char *pending_path = file_backend_put_path (txn, hex_id);
  char *marker_path = file_backend_delete_marker (txn, hex_id);
  if (pending_path == NULL || marker_path == NULL)
    {
      sqlite3_free (pending_path);
      sqlite3_free (marker_path);
      return SQLITE_NOMEM;
    }

  file_backend_savepoint_entry *entry = &frame->entries[frame->entry_count++];
  memset (entry, 0, sizeof (*entry));
  entry->id = *id;
  entry->had_pending = file_backend_path_exists (pending_path);
  entry->had_delete_marker = file_backend_path_exists (marker_path);
  sqlite3_free (pending_path);
  sqlite3_free (marker_path);
  return SQLITE_OK;
}

static int
file_backend_savepoint_restore_entry (file_backend_txn *txn,
                                      const file_backend_savepoint_entry *entry)
{
  if (txn == NULL || entry == NULL)
    {
      return SQLITE_MISUSE;
    }

  char hex_id[kObjstoreIdHexChars + 1];
  objstore_fs_id_to_hex (&entry->id, hex_id);
  char *pending_path = file_backend_put_path (txn, hex_id);
  char *marker_path = file_backend_delete_marker (txn, hex_id);
  if (pending_path == NULL || marker_path == NULL)
    {
      sqlite3_free (pending_path);
      sqlite3_free (marker_path);
      return SQLITE_NOMEM;
    }

  if (!entry->had_pending && unlink (pending_path) != 0 && errno != ENOENT)
    {
      sqlite3_free (pending_path);
      sqlite3_free (marker_path);
      return SQLITE_IOERR_DELETE;
    }
  if (entry->had_delete_marker)
    {
      FILE *marker = fopen (marker_path, "wb");
      if (marker == NULL)
        {
          sqlite3_free (pending_path);
          sqlite3_free (marker_path);
          return SQLITE_IOERR;
        }
      fclose (marker);
    }
  else if (unlink (marker_path) != 0 && errno != ENOENT)
    {
      sqlite3_free (pending_path);
      sqlite3_free (marker_path);
      return SQLITE_IOERR_DELETE;
    }

  sqlite3_free (pending_path);
  sqlite3_free (marker_path);
  return SQLITE_OK;
}

static char *
file_backend_committed_object_path (file_backend_env *env, const char *hex_id)
{
  if (env == NULL || hex_id == NULL)
    {
      return NULL;
    }
  return objstore_fs_object_path (env->objects_root, env->shard_width, hex_id);
}

static int
file_backend_stream_file (FILE *file, const objstore_stream_writer *writer,
                          size_t chunk_size)
{
  if (file == NULL || writer == NULL || writer->push == NULL)
    {
      return SQLITE_MISUSE;
    }
  unsigned char *buffer = (unsigned char *)sqlite3_malloc64 (chunk_size);
  if (buffer == NULL)
    {
      return SQLITE_NOMEM;
    }
  int rc = SQLITE_OK;
  size_t nread = 0;
  while ((nread = fread (buffer, 1, chunk_size, file)) > 0)
    {
      rc = writer->push (writer->ctx, buffer, nread);
      if (rc != SQLITE_OK)
        {
          break;
        }
    }
  if (rc == SQLITE_OK && ferror (file))
    {
      rc = SQLITE_IOERR_READ;
    }
  sqlite3_free (buffer);
  return rc;
}

static int
file_backend_seek (FILE *file, sqlite3_uint64 offset)
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
file_backend_stream_file_range (FILE *file, const objstore_stream_writer *writer,
                                size_t chunk_size, sqlite3_uint64 offset,
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
  if (chunk_size == 0)
    {
      chunk_size = OBJSTORE_DEFAULT_CHUNK_SIZE;
    }
  if (file_backend_seek (file, offset) != 0)
    {
      return SQLITE_IOERR;
    }
  unsigned char *buffer = (unsigned char *)sqlite3_malloc64 (chunk_size);
  if (buffer == NULL)
    {
      return SQLITE_NOMEM;
    }
  sqlite3_uint64 remaining = length;
  int rc = SQLITE_OK;
  while (remaining > 0)
    {
      size_t to_read = (remaining < (sqlite3_uint64)chunk_size)
                           ? (size_t)remaining
                           : chunk_size;
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

static bool
file_backend_path_exists (const char *path)
{
  struct stat st;
  return path != NULL && stat (path, &st) == 0;
}

static int
file_backend_stat_size (const char *path, sqlite3_int64 *out_size)
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

static void
file_backend_rowidx_bytes_to_hex (const unsigned char *bytes, size_t length,
                                  char *hex_out)
{
  static const char kDigits[] = "0123456789abcdef";
  if (bytes == NULL || hex_out == NULL)
    {
      return;
    }
  for (size_t i = 0; i < length; ++i)
    {
      hex_out[i * 2] = kDigits[(bytes[i] >> 4) & 0x0F];
      hex_out[i * 2 + 1] = kDigits[bytes[i] & 0x0F];
    }
  hex_out[length * 2] = '\0';
}

static char *
file_backend_rowidx_marker_path (const file_backend_env *env)
{
  if (env == NULL || env->rowidx_root == NULL)
    {
      return NULL;
    }
  return objstore_fs_path_join (env->rowidx_root, FILE_BACKEND_ROWIDX_MARKER);
}

static char *
file_backend_rowidx_row_dir (file_backend_env *env,
                             const unsigned char
                                 rowid_bytes[OBJSTORE_ROWID_PREFIX_SIZE],
                             bool ensure_dirs)
{
  if (env == NULL || env->rowidx_root == NULL || rowid_bytes == NULL)
    {
      return NULL;
    }
  char rowid_hex[OBJSTORE_ROWID_HEX_CHARS + 1];
  file_backend_rowidx_bytes_to_hex (rowid_bytes, OBJSTORE_ROWID_PREFIX_SIZE,
                                    rowid_hex);
  char shard_name[FILE_BACKEND_ROWIDX_SHARD_CHARS + 1];
  for (size_t i = 0; i < FILE_BACKEND_ROWIDX_SHARD_CHARS; ++i)
    {
      shard_name[i] = rowid_hex[i];
    }
  shard_name[FILE_BACKEND_ROWIDX_SHARD_CHARS] = '\0';
  char *shard_dir = objstore_fs_path_join (env->rowidx_root, shard_name);
  if (shard_dir == NULL)
    {
      return NULL;
    }
  if (ensure_dirs)
    {
      int shard_rc = objstore_fs_mkdirs (shard_dir);
      if (shard_rc != SQLITE_OK)
        {
          sqlite3_free (shard_dir);
          return NULL;
        }
    }
  char *row_dir = objstore_fs_path_join (shard_dir, rowid_hex);
  sqlite3_free (shard_dir);
  if (row_dir == NULL)
    {
      return NULL;
    }
  if (ensure_dirs)
    {
      int row_rc = objstore_fs_mkdirs (row_dir);
      if (row_rc != SQLITE_OK)
        {
          sqlite3_free (row_dir);
          return NULL;
        }
    }
  return row_dir;
}

static int
file_backend_rowidx_entry_path (file_backend_env *env, const objstore_id *id,
                                bool ensure_dirs, char **out_row_dir,
                                char **out_entry_path)
{
  if (env == NULL || id == NULL || out_entry_path == NULL)
    {
      return SQLITE_MISUSE;
    }
  unsigned char row_bytes[OBJSTORE_ROWID_PREFIX_SIZE];
  objstore_rowid_prefix_from_id (id, row_bytes);
  char *row_dir = file_backend_rowidx_row_dir (env, row_bytes, ensure_dirs);
  if (row_dir == NULL)
    {
      return SQLITE_IOERR;
    }
  char hex_id[kObjstoreIdHexChars + 1];
  objstore_fs_id_to_hex (id, hex_id);
  char *entry_path = objstore_fs_path_join (row_dir, hex_id);
  if (entry_path == NULL)
    {
      sqlite3_free (row_dir);
      return SQLITE_NOMEM;
    }
  if (out_row_dir != NULL)
    {
      *out_row_dir = row_dir;
    }
  else
    {
      sqlite3_free (row_dir);
    }
  *out_entry_path = entry_path;
  return SQLITE_OK;
}

static int
file_backend_rowidx_add_entry (file_backend_env *env, const objstore_id *id)
{
  char *row_dir = NULL;
  char *entry_path = NULL;
  int rc = file_backend_rowidx_entry_path (env, id, true, &row_dir,
                                           &entry_path);
  if (rc != SQLITE_OK)
    {
      return rc;
    }
  int fd = open (entry_path,
                 O_WRONLY | O_CREAT | O_TRUNC
#if defined(O_CLOEXEC)
                 | O_CLOEXEC
#endif
                 ,
                 0666);
  if (fd < 0)
    {
      rc = (errno == ENOSPC || errno == EFBIG) ? SQLITE_FULL : SQLITE_IOERR;
    }
  else
    {
      if (close (fd) != 0)
        {
          rc = SQLITE_IOERR;
        }
      else
        {
          rc = SQLITE_OK;
        }
    }
  sqlite3_free (row_dir);
  sqlite3_free (entry_path);
  return rc;
}

static int
file_backend_rowidx_remove_entry (file_backend_env *env, const objstore_id *id)
{
  char *row_dir = NULL;
  char *entry_path = NULL;
  int rc = file_backend_rowidx_entry_path (env, id, false, &row_dir,
                                           &entry_path);
  if (rc != SQLITE_OK)
    {
      return rc;
    }
  if (unlink (entry_path) != 0 && errno != ENOENT)
    {
      rc = SQLITE_IOERR;
    }
  else
    {
      rc = SQLITE_OK;
    }
  sqlite3_free (row_dir);
  sqlite3_free (entry_path);
  return rc;
}

static bool
file_backend_object_exists (file_backend_env *env, const objstore_id *id)
{
  if (env == NULL || id == NULL)
    {
      return false;
    }
  char hex_id[kObjstoreIdHexChars + 1];
  objstore_fs_id_to_hex (id, hex_id);
  char *path = file_backend_committed_object_path (env, hex_id);
  if (path == NULL)
    {
      return false;
    }
  bool exists = file_backend_path_exists (path);
  sqlite3_free (path);
  return exists;
}

static int file_backend_rowidx_rebuild (file_backend_env *env);

static int
file_backend_rowidx_bootstrap (file_backend_env *env)
{
  if (env == NULL || env->rowidx_root == NULL)
    {
      return SQLITE_MISUSE;
    }
  int rc = objstore_fs_mkdirs (env->rowidx_root);
  if (rc != SQLITE_OK)
    {
      return rc;
    }
  char *marker_path = file_backend_rowidx_marker_path (env);
  if (marker_path == NULL)
    {
      return SQLITE_NOMEM;
    }
  const bool ready = objstore_fs_path_exists (marker_path);
  sqlite3_free (marker_path);
  if (ready)
    {
      return SQLITE_OK;
    }
  return file_backend_rowidx_rebuild (env);
}

static int
file_backend_rowidx_rebuild (file_backend_env *env)
{
  if (env == NULL || env->rowidx_root == NULL)
    {
      return SQLITE_MISUSE;
    }
  int rc = objstore_fs_remove_tree (env->rowidx_root);
  if (rc != SQLITE_OK)
    {
      return rc;
    }
  rc = objstore_fs_mkdirs (env->rowidx_root);
  if (rc != SQLITE_OK)
    {
      return rc;
    }
  objstore_id *ids = NULL;
  size_t count = 0;
  rc = file_backend_collect_object_ids (env, &ids, &count);
  if (rc != SQLITE_OK)
    {
      sqlite3_free (ids);
      return rc;
    }
  for (size_t i = 0; i < count; ++i)
    {
      rc = file_backend_rowidx_add_entry (env, &ids[i]);
      if (rc != SQLITE_OK)
        {
          sqlite3_free (ids);
          return rc;
        }
    }
  sqlite3_free (ids);
  char *marker_path = file_backend_rowidx_marker_path (env);
  if (marker_path == NULL)
    {
      return SQLITE_NOMEM;
    }
  FILE *marker = fopen (marker_path, "wb");
  if (marker == NULL)
    {
      sqlite3_free (marker_path);
      return SQLITE_IOERR;
    }
  static const char kReadyMarker[] = "v1";
  const size_t ready_len = sizeof (kReadyMarker) - 1u;
  if (fwrite (kReadyMarker, 1u, ready_len, marker) != ready_len)
    {
      fclose (marker);
      sqlite3_free (marker_path);
      return SQLITE_IOERR_WRITE;
    }
  fclose (marker);
  sqlite3_free (marker_path);
  return SQLITE_OK;
}

static int
file_backend_lookup_id_by_rowid (objstore_backend_txn *txn, sqlite3_int64 rowid,
                                 objstore_id *out)
{
  if (txn == NULL || out == NULL)
    {
      return SQLITE_MISUSE;
    }
  file_backend_txn *file_txn = file_txn_from (txn);
  unsigned char row_bytes[OBJSTORE_ROWID_PREFIX_SIZE];
  objstore_rowid_to_bytes (rowid, row_bytes);
  char *row_dir = file_backend_rowidx_row_dir (file_txn->env, row_bytes, false);
  if (row_dir == NULL)
    {
      return SQLITE_NOMEM;
    }
  DIR *dir = opendir (row_dir);
  if (dir == NULL)
    {
      int err = errno;
      sqlite3_free (row_dir);
      if (err == ENOENT)
        {
          return SQLITE_NOTFOUND;
        }
      return SQLITE_IOERR;
    }
  struct dirent *entry = NULL;
  int rc = SQLITE_NOTFOUND;
  while ((entry = readdir (dir)) != NULL)
    {
      if (strcmp (entry->d_name, ".") == 0
          || strcmp (entry->d_name, "..") == 0)
        {
          continue;
        }
      if ((size_t)strlen (entry->d_name) != kObjstoreIdHexChars)
        {
          continue;
        }
      objstore_id candidate = { 0 };
      if (objstore_fs_hex_to_id (entry->d_name, &candidate) != SQLITE_OK)
        {
          continue;
        }
      if (objstore_rowid_from_id (&candidate) != rowid)
        {
          (void)file_backend_rowidx_remove_entry (file_txn->env, &candidate);
          continue;
        }
      if (!file_backend_object_exists (file_txn->env, &candidate))
        {
          (void)file_backend_rowidx_remove_entry (file_txn->env, &candidate);
          continue;
        }
      *out = candidate;
      rc = SQLITE_OK;
      break;
    }
  closedir (dir);
  sqlite3_free (row_dir);
  return rc;
}


static int
file_backend_stream_pending (file_backend_txn *txn, const char *hex_id,
                             const objstore_stream_writer *writer)
{
  if (txn == NULL || txn->active_dir == NULL)
    {
      return SQLITE_NOTFOUND;
    }
  char *pending_path = file_backend_put_path (txn, hex_id);
  if (pending_path == NULL)
    {
      return SQLITE_NOMEM;
    }
  if (!file_backend_path_exists (pending_path))
    {
      sqlite3_free (pending_path);
      return SQLITE_NOTFOUND;
    }
  FILE *file = fopen (pending_path, "rb");
  sqlite3_free (pending_path);
  if (file == NULL)
    {
      return SQLITE_IOERR;
    }
  int rc = file_backend_stream_file (file, writer, txn->env->chunk_size);
  fclose (file);
  return rc;
}

static int
file_backend_cleanup_directory (const char *root)
{
  DIR *dir = opendir (root);
  if (dir == NULL)
    {
      return (errno == ENOENT) ? SQLITE_OK : SQLITE_IOERR;
    }
  struct dirent *entry = NULL;
  int rc = SQLITE_OK;
  while ((entry = readdir (dir)) != NULL)
    {
      if (strcmp (entry->d_name, ".") == 0
          || strcmp (entry->d_name, "..") == 0)
        {
          continue;
        }
      char *child = objstore_fs_path_join (root, entry->d_name);
      if (child == NULL)
        {
          rc = SQLITE_NOMEM;
          break;
        }
      rc = objstore_fs_remove_tree (child);
      sqlite3_free (child);
      if (rc != SQLITE_OK)
        {
          break;
        }
    }
  closedir (dir);
  return rc;
}

static int
file_backend_apply_put (file_backend_env *env, const char *commit_dir,
                        const char *hex_id)
{
  objstore_id object_id = { 0 };
  if (objstore_fs_hex_to_id (hex_id, &object_id) != SQLITE_OK)
    {
      return SQLITE_CORRUPT;
    }

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

  char *dest_filename = sqlite3_mprintf ("%s%s", hex_id, OBJSTORE_FILE_SUFFIX);
  if (dest_filename == NULL)
    {
      sqlite3_free (dest_dir);
      sqlite3_free (source);
      return SQLITE_NOMEM;
    }
  char *dest_name = objstore_fs_path_join (dest_dir, dest_filename);
  sqlite3_free (dest_filename);
  if (dest_name == NULL)
    {
      sqlite3_free (dest_dir);
      sqlite3_free (source);
      return SQLITE_NOMEM;
    }
  struct stat source_stat;
  bool source_present = false;
  if (stat (source, &source_stat) == 0)
    {
      source_present = S_ISREG (source_stat.st_mode);
    }
  else if (errno != ENOENT)
    {
      sqlite3_free (dest_dir);
      sqlite3_free (source);
      sqlite3_free (dest_name);
      return SQLITE_IOERR;
    }

  if (!source_present)
    {
      const bool dest_exists = file_backend_path_exists (dest_name);
      sqlite3_free (dest_dir);
      sqlite3_free (source);
      sqlite3_free (dest_name);
      return dest_exists ? SQLITE_OK : SQLITE_CORRUPT;
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
      const int err = errno;
      sqlite3_free (dest_dir);
      sqlite3_free (source);
      sqlite3_free (dest_name);
      if (err == ENOENT && file_backend_path_exists (dest_name))
        {
          return SQLITE_OK;
        }
      return SQLITE_IOERR;
    }

  int rowidx_rc = file_backend_rowidx_add_entry (env, &object_id);
  sqlite3_free (dest_dir);
  sqlite3_free (source);
  sqlite3_free (dest_name);
  return rowidx_rc;
}

static int
file_backend_apply_delete (file_backend_env *env, const char *hex_id)
{
  objstore_id object_id = { 0 };
  if (objstore_fs_hex_to_id (hex_id, &object_id) != SQLITE_OK)
    {
      return SQLITE_CORRUPT;
    }

  char *dest = file_backend_committed_object_path (env, hex_id);
  if (dest == NULL)
    {
      return SQLITE_NOMEM;
    }
  if (unlink (dest) != 0 && errno != ENOENT)
    {
      sqlite3_free (dest);
      return SQLITE_IOERR;
    }
  int rc = file_backend_rowidx_remove_entry (env, &object_id);
  sqlite3_free (dest);
  return rc;
}

static int
file_backend_apply_commit_dir (file_backend_env *env, const char *commit_dir)
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
          rc = file_backend_apply_put (env, commit_dir, hex);
        }
      else if (strcmp (op, "DEL") == 0)
        {
          rc = file_backend_apply_delete (env, hex);
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
file_backend_recover (file_backend_env *env)
{
  int rc = file_backend_cleanup_directory (env->staging_active_root);
  if (rc != SQLITE_OK)
    {
      return rc;
    }

  DIR *dir = opendir (env->staging_commit_root);
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
          = objstore_fs_path_join (env->staging_commit_root, entry->d_name);
      if (commit_dir == NULL)
        {
          closedir (dir);
          return SQLITE_NOMEM;
        }
      rc = file_backend_apply_commit_dir (env, commit_dir);
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

static void file_backend_reset_active (file_backend_txn *txn, bool drop_tree);
static int file_backend_promote_staged (file_backend_txn *txn);
static void file_backend_rollback_staged (objstore_backend_txn *txn);
static int file_backend_put (objstore_backend_txn *txn, const objstore_id *id,
                             const objstore_stream_reader *reader);
static void file_backend_txn_release (file_backend_txn *txn, bool drop_active);

static void
file_backend_txn_free (file_backend_txn *txn)
{
  if (txn == NULL)
    {
      return;
    }
  file_backend_txn_release (txn, true);
  sqlite3_free (txn);
}

static void
file_backend_txn_release (file_backend_txn *txn, bool drop_active)
{
  if (txn == NULL)
    {
      return;
    }
  file_backend_savepoint_frames_clear (txn);
  file_backend_reset_active (txn, drop_active);
  if (txn->io_buffer != NULL)
    {
      file_backend_free_aligned (txn->io_buffer);
      txn->io_buffer = NULL;
    }
}

static void
file_backend_reset_active (file_backend_txn *txn, bool drop_tree)
{
  if (txn == NULL)
    {
      return;
    }
  file_backend_manifest_dispose (txn);
  if (txn->active_dir != NULL)
    {
      if (drop_tree)
        {
          objstore_fs_remove_tree (txn->active_dir);
        }
      sqlite3_free (txn->active_dir);
      txn->active_dir = NULL;
    }
}

static void
file_backend_staged_writer_cleanup (file_backend_staged_writer *writer,
                                    bool unlink_temp)
{
  if (writer == NULL)
    {
      return;
    }
  if (writer->fd >= 0)
    {
      close (writer->fd);
      writer->fd = -1;
    }
  if (unlink_temp && writer->temp_path != NULL)
    {
      unlink (writer->temp_path);
    }
  sqlite3_free (writer->temp_path);
  writer->temp_path = NULL;
  sqlite3_free (writer);
}

static int
file_backend_promote_staged (file_backend_txn *txn)
{
  if (txn == NULL)
    {
      return SQLITE_MISUSE;
    }
  if (txn->staging_promoted)
    {
      return SQLITE_OK;
    }
  if (txn->active_dir == NULL)
    {
      return SQLITE_MISUSE;
    }

  int rc = file_backend_manifest_flush (txn);
  if (rc != SQLITE_OK)
    {
      return rc;
    }

  char *commit_dir_path
      = objstore_fs_path_join (txn->env->staging_commit_root, txn->txn_id);
  if (commit_dir_path == NULL)
    {
      return SQLITE_NOMEM;
    }

  rc = objstore_fs_rename (txn->active_dir, commit_dir_path);
  if (rc == SQLITE_OK)
    {
      sqlite3_free (txn->active_dir);
      txn->active_dir = commit_dir_path;
      commit_dir_path = NULL;
      rc = file_backend_apply_commit_dir (txn->env, txn->active_dir);
      if (rc == SQLITE_OK)
        {
          txn->staging_promoted = true;
          sqlite3_free (txn->active_dir);
          txn->active_dir = NULL;
        }
    }
  sqlite3_free (commit_dir_path);
  return rc;
}

static int
file_backend_staged_write_begin (objstore_backend_txn *txn,
                                 objstore_backend_staged_writer **out_writer)
{
  if (txn == NULL || out_writer == NULL)
    {
      return SQLITE_MISUSE;
    }
  file_backend_txn *file_txn = file_txn_from (txn);
  file_backend_staged_writer *writer
      = (file_backend_staged_writer *)sqlite3_malloc (
          sizeof (file_backend_staged_writer));
  if (writer == NULL)
    {
      return SQLITE_NOMEM;
    }
  memset (writer, 0, sizeof (*writer));
  writer->txn = file_txn;
  writer->fd = -1;
  writer->expected_size = -1;
  writer->bytes_written = 0;
  writer->prealloc_done = false;

  char *put_dir
      = objstore_fs_path_join (file_txn->active_dir, OBJSTORE_PUT_DIR);
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
  char *temp_path = objstore_fs_path_join (put_dir, temp_name);
  sqlite3_free (put_dir);
  sqlite3_free (temp_name);
  if (temp_path == NULL)
    {
      sqlite3_free (writer);
      return SQLITE_NOMEM;
    }

  int flags = O_WRONLY | O_CREAT | O_TRUNC;
#if defined(O_CLOEXEC)
  flags |= O_CLOEXEC;
#endif
  int fd = open (temp_path, flags, 0666);
  if (fd < 0)
    {
      sqlite3_free (temp_path);
      sqlite3_free (writer);
      return SQLITE_IOERR;
    }
  if (file_backend_force_disk_full ())
    {
      close (fd);
      sqlite3_free (temp_path);
      sqlite3_free (writer);
      return SQLITE_FULL;
    }

  writer->fd = fd;
  writer->temp_path = temp_path;
  writer->expected_size = -1;
  writer->bytes_written = 0;
  writer->prealloc_done = false;
  *out_writer = (objstore_backend_staged_writer *)writer;
  return SQLITE_OK;
}

static int
file_backend_staged_write_set_size_hint (
    objstore_backend_staged_writer *writer_handle, sqlite3_int64 size_hint)
{
  file_backend_staged_writer *writer = file_writer_from (writer_handle);
  if (writer == NULL || writer->fd < 0)
    {
      return SQLITE_MISUSE;
    }
  writer->expected_size = size_hint;
  if (size_hint <= 0 || writer->prealloc_done)
    {
      return SQLITE_OK;
    }
  int rc = file_backend_preallocate_fd (writer->fd, size_hint);
  if (rc == SQLITE_OK)
    {
      writer->prealloc_done = true;
    }
  return rc;
}

static int
file_backend_staged_write_push (objstore_backend_staged_writer *writer_handle,
                                const void *buffer, size_t nread)
{
  if (writer_handle == NULL)
    {
      return SQLITE_MISUSE;
    }
  file_backend_staged_writer *writer = file_writer_from (writer_handle);
  if ((nread > 0 && buffer == NULL) || writer->fd < 0)
    {
      return SQLITE_MISUSE;
    }
  if (nread == 0)
    {
      return SQLITE_OK;
    }
  int rc = file_backend_write_all (writer->fd, buffer, nread);
  if (rc == SQLITE_OK)
    {
      writer->bytes_written += (sqlite3_int64)nread;
    }
  return rc;
}

static int
file_backend_staged_write_finalize (
    objstore_backend_staged_writer *writer_handle, const objstore_id *id)
{
  file_backend_staged_writer *writer = file_writer_from (writer_handle);
  if (writer == NULL)
    {
      return SQLITE_MISUSE;
    }
  if (id == NULL)
    {
      file_backend_staged_writer_cleanup (writer, true);
      return SQLITE_OK;
    }

  if (writer->expected_size > 0 && writer->bytes_written >= 0
      && writer->bytes_written < writer->expected_size)
    {
      if (ftruncate (writer->fd, (off_t)writer->bytes_written) != 0)
        {
          int truncate_err = errno;
          file_backend_staged_writer_cleanup (writer, true);
          if (truncate_err == ENOSPC)
            {
              return SQLITE_FULL;
            }
          return SQLITE_IOERR;
        }
    }

  int rc = file_backend_flush_fd (
      writer->fd, file_backend_should_fsync_payload (writer->txn->env));
  if (writer->fd >= 0)
    {
      close (writer->fd);
      writer->fd = -1;
    }
  if (rc != SQLITE_OK)
    {
      file_backend_staged_writer_cleanup (writer, true);
      return rc;
    }

  char hex_id[kObjstoreIdHexChars + 1];
  objstore_fs_id_to_hex (id, hex_id);
  rc = file_backend_savepoint_capture_state (writer->txn, id);
  if (rc != SQLITE_OK)
    {
      file_backend_staged_writer_cleanup (writer, true);
      return rc;
    }
  char *dest_path = file_backend_put_path (writer->txn, hex_id);
  if (dest_path == NULL)
    {
      file_backend_staged_writer_cleanup (writer, true);
      return SQLITE_NOMEM;
    }
  if (unlink (dest_path) != 0 && errno != ENOENT)
    {
      int err = errno;
      sqlite3_free (dest_path);
      file_backend_staged_writer_cleanup (writer, true);
      if (err == ENOSPC || err == EFBIG)
        {
          return SQLITE_FULL;
        }
      return SQLITE_IOERR;
    }
  if (rename (writer->temp_path, dest_path) != 0)
    {
      int err = errno;
      sqlite3_free (dest_path);
      file_backend_staged_writer_cleanup (writer, true);
      if (err == ENOSPC || err == EFBIG)
        {
          return SQLITE_FULL;
        }
      return SQLITE_IOERR;
    }
  sqlite3_free (dest_path);
  sqlite3_free (writer->temp_path);
  writer->temp_path = NULL;

  rc = file_backend_manifest_append (writer->txn, "PUT", hex_id);
  file_backend_staged_writer_cleanup (writer, false);
  return rc;
}

static int
file_backend_open_env (sqlite3 *db, const objstore_config *config,
                       objstore_backend_env **out_env)
{
  if (out_env == NULL)
    {
      return SQLITE_MISUSE;
    }
  file_backend_env *env = sqlite3_malloc (sizeof (*env));
  if (env == NULL)
    {
      return SQLITE_NOMEM;
    }
  env->db = db;
  env->chunk_size = objstore_effective_chunk_size (config);
  env->shard_width = file_backend_effective_shard_width (config);
  env->sync_mode = (config != NULL) ? config->sync_mode : OBJSTORE_SYNC_FULL;

  const char *objects_root
      = (config && config->storage_root && config->storage_root[0] != '\0')
            ? config->storage_root
            : "objects";
  env->objects_root = sqlite3_mprintf ("%s", objects_root);
  env->rowidx_root
      = objstore_fs_path_join (env->objects_root, FILE_BACKEND_ROWIDX_DIR);
  env->staging_root = objstore_fs_path_join (env->objects_root, ".staging");
  env->staging_active_root
      = objstore_fs_path_join (env->staging_root, "active");
  env->staging_commit_root
      = objstore_fs_path_join (env->staging_root, "commit");
  if (env->objects_root == NULL || env->rowidx_root == NULL
      || env->staging_root == NULL || env->staging_active_root == NULL
      || env->staging_commit_root == NULL)
    {
      sqlite3_free (env->objects_root);
      sqlite3_free (env->rowidx_root);
      sqlite3_free (env->staging_root);
      sqlite3_free (env->staging_active_root);
      sqlite3_free (env->staging_commit_root);
      sqlite3_free (env);
      return SQLITE_NOMEM;
    }

  env->lock_fd = -1;
  int rc = objstore_fs_mkdirs (env->objects_root);
  if (rc == SQLITE_OK)
    rc = objstore_fs_mkdirs (env->staging_root);
  if (rc == SQLITE_OK)
    rc = objstore_fs_mkdirs (env->staging_active_root);
  if (rc == SQLITE_OK)
    rc = objstore_fs_mkdirs (env->staging_commit_root);
  if (rc == SQLITE_OK)
    rc = objstore_fs_mkdirs (env->rowidx_root);
#if !defined(_WIN32)
  /* Staging recovery deletes everything under .staging/active, so it is
     only safe when no other live env shares this objects root. Each env
     holds a shared flock for its lifetime; recovery runs only if we can
     briefly take the lock exclusively. */
  bool run_recovery = true;
  if (rc == SQLITE_OK)
    {
      char *lock_path = objstore_fs_path_join (env->objects_root, ".lock");
      if (lock_path == NULL)
        {
          rc = SQLITE_NOMEM;
        }
      else
        {
          env->lock_fd = open (lock_path, O_RDWR | O_CREAT | O_CLOEXEC, 0644);
          sqlite3_free (lock_path);
          if (env->lock_fd < 0)
            {
              rc = SQLITE_IOERR;
            }
          else if (flock (env->lock_fd, LOCK_EX | LOCK_NB) == 0)
            {
              /* Sole owner for now: recovery is safe. Downgrade to shared
                 after it runs. */
            }
          else if (errno == EWOULDBLOCK)
            {
              run_recovery = false;
              if (flock (env->lock_fd, LOCK_SH) != 0)
                {
                  rc = SQLITE_IOERR;
                }
            }
          else
            {
              rc = SQLITE_IOERR;
            }
        }
    }
  if (rc == SQLITE_OK && run_recovery)
    {
      rc = file_backend_recover (env);
      if (flock (env->lock_fd, LOCK_SH) != 0 && rc == SQLITE_OK)
        {
          rc = SQLITE_IOERR;
        }
    }
#else
  if (rc == SQLITE_OK)
    rc = file_backend_recover (env);
#endif
  if (rc == SQLITE_OK)
    rc = file_backend_rowidx_bootstrap (env);
  if (rc != SQLITE_OK)
    {
      if (env->lock_fd >= 0)
        {
          close (env->lock_fd);
        }
      sqlite3_free (env->objects_root);
      sqlite3_free (env->rowidx_root);
      sqlite3_free (env->staging_root);
      sqlite3_free (env->staging_active_root);
      sqlite3_free (env->staging_commit_root);
      sqlite3_free (env);
      return rc;
    }

  *out_env = (objstore_backend_env *)env;
  return SQLITE_OK;
}

static void
file_backend_close_env (objstore_backend_env *env)
{
  if (env == NULL)
    {
      return;
    }
  file_backend_env *backend_env = (file_backend_env *)env;
  if (backend_env->lock_fd >= 0)
    {
      close (backend_env->lock_fd);
    }
  sqlite3_free (backend_env->objects_root);
  sqlite3_free (backend_env->rowidx_root);
  sqlite3_free (backend_env->staging_root);
  sqlite3_free (backend_env->staging_active_root);
  sqlite3_free (backend_env->staging_commit_root);
  sqlite3_free (backend_env);
}

static int
file_backend_begin_txn (objstore_backend_env *env,
                        objstore_backend_txn **out_txn)
{
  if (env == NULL || out_txn == NULL)
    {
      return SQLITE_MISUSE;
    }
  file_backend_env *backend_env = (file_backend_env *)env;
  file_backend_txn *txn = sqlite3_malloc (sizeof (*txn));
  if (txn == NULL)
    {
      return SQLITE_NOMEM;
    }
  memset (txn, 0, sizeof (*txn));
  txn->env = backend_env;
  txn->io_buffer_size = backend_env->chunk_size;
  objstore_fs_random_hex (txn->txn_id, sizeof (txn->txn_id));
  txn->active_dir
      = objstore_fs_path_join (backend_env->staging_active_root, txn->txn_id);
  if (txn->active_dir == NULL)
    {
      sqlite3_free (txn);
      return SQLITE_NOMEM;
    }
  int rc = objstore_fs_mkdirs (txn->active_dir);
  if (rc != SQLITE_OK)
    {
      file_backend_txn_free (txn);
      return rc;
    }
  char *put_dir = objstore_fs_path_join (txn->active_dir, OBJSTORE_PUT_DIR);
  char *delete_dir
      = objstore_fs_path_join (txn->active_dir, OBJSTORE_DELETE_DIR);
  if (put_dir == NULL || delete_dir == NULL)
    {
      sqlite3_free (put_dir);
      sqlite3_free (delete_dir);
      file_backend_txn_free (txn);
      return SQLITE_NOMEM;
    }
  rc = objstore_fs_mkdirs (put_dir);
  if (rc == SQLITE_OK)
    rc = objstore_fs_mkdirs (delete_dir);
  sqlite3_free (put_dir);
  sqlite3_free (delete_dir);
  if (rc != SQLITE_OK)
    {
      file_backend_txn_free (txn);
      return rc;
    }
  txn->manifest_buffer = sqlite3_str_new (NULL);
  if (txn->manifest_buffer == NULL)
    {
      file_backend_txn_free (txn);
      return SQLITE_NOMEM;
    }
  txn->io_buffer = file_backend_alloc_aligned (txn->io_buffer_size);
  if (txn->io_buffer == NULL)
    {
      file_backend_txn_free (txn);
      return SQLITE_NOMEM;
    }
  txn->staging_promoted = false;
  *out_txn = (objstore_backend_txn *)txn;
  return SQLITE_OK;
}

static int
file_backend_commit_txn (objstore_backend_txn *txn)
{
  if (txn == NULL)
    {
      return SQLITE_MISUSE;
    }
  file_backend_txn *file_txn = file_txn_from (txn);
  int rc = file_backend_promote_staged (file_txn);
  if (rc != SQLITE_OK)
    {
      file_backend_txn_release (file_txn, false);
      sqlite3_free (file_txn);
      return rc;
    }
  file_backend_txn_free (file_txn);
  return SQLITE_OK;
}

static void
file_backend_rollback_txn (objstore_backend_txn *txn)
{
  if (txn == NULL)
    {
      return;
    }
  file_backend_rollback_staged (txn);
  file_backend_txn_free (file_txn_from (txn));
}

static int
file_backend_commit_staged (objstore_backend_txn *txn)
{
  if (txn == NULL)
    {
      return SQLITE_MISUSE;
    }
  return file_backend_promote_staged (file_txn_from (txn));
}

static void
file_backend_rollback_staged (objstore_backend_txn *txn)
{
  if (txn == NULL)
    {
      return;
    }
  file_backend_txn *file_txn = file_txn_from (txn);
  if (file_txn->staging_promoted)
    {
      return;
    }
  file_backend_savepoint_frames_clear (file_txn);
  file_backend_reset_active (file_txn, true);
}

static int
file_backend_savepoint_begin (objstore_backend_txn *txn)
{
  if (txn == NULL)
    {
      return SQLITE_MISUSE;
    }
  file_backend_txn *file_txn = file_txn_from (txn);
  int rc = file_backend_savepoint_frames_ensure_capacity (file_txn);
  if (rc != SQLITE_OK)
    {
      return rc;
    }
  file_backend_savepoint_frame *frame = &file_txn->frames[file_txn->frame_count];
  file_backend_savepoint_frame_dispose (frame);
  frame->manifest_length = (file_txn->manifest_buffer != NULL)
                               ? (size_t)sqlite3_str_length (
                                     file_txn->manifest_buffer)
                               : 0u;
  ++file_txn->frame_count;
  return SQLITE_OK;
}

static int
file_backend_savepoint_release (objstore_backend_txn *txn)
{
  if (txn == NULL)
    {
      return SQLITE_MISUSE;
    }
  file_backend_txn *file_txn = file_txn_from (txn);
  if (file_txn->frame_count == 0)
    {
      return SQLITE_OK;
    }
  file_backend_savepoint_frame *child
      = &file_txn->frames[file_txn->frame_count - 1];
  if (file_txn->frame_count > 1)
    {
      file_backend_savepoint_frame *parent
          = &file_txn->frames[file_txn->frame_count - 2];
      for (size_t i = 0; i < child->entry_count; ++i)
        {
          if (file_backend_savepoint_frame_find_entry (parent,
                                                       &child->entries[i].id)
              >= 0)
            {
              continue;
            }
          int rc = file_backend_savepoint_entries_ensure_capacity (parent);
          if (rc != SQLITE_OK)
            {
              return rc;
            }
          parent->entries[parent->entry_count++] = child->entries[i];
        }
    }
  file_backend_savepoint_frame_dispose (child);
  --file_txn->frame_count;
  return SQLITE_OK;
}

static int
file_backend_savepoint_rollback (objstore_backend_txn *txn)
{
  if (txn == NULL)
    {
      return SQLITE_MISUSE;
    }
  file_backend_txn *file_txn = file_txn_from (txn);
  if (file_txn->frame_count == 0)
    {
      return SQLITE_OK;
    }
  file_backend_savepoint_frame *frame
      = &file_txn->frames[file_txn->frame_count - 1];
  for (size_t i = frame->entry_count; i > 0; --i)
    {
      int rc = file_backend_savepoint_restore_entry (file_txn,
                                                     &frame->entries[i - 1]);
      if (rc != SQLITE_OK)
        {
          return rc;
        }
    }
  return file_backend_manifest_reset_prefix (file_txn, frame->manifest_length);
}

static int
file_backend_put (objstore_backend_txn *txn, const objstore_id *id,
                  const objstore_stream_reader *reader)
{
  if (txn == NULL || id == NULL || reader == NULL || reader->pull == NULL)
    {
      return SQLITE_MISUSE;
    }
  file_backend_txn *file_txn = file_txn_from (txn);
  int rc = file_backend_savepoint_capture_state (file_txn, id);
  if (rc != SQLITE_OK)
    {
      return rc;
    }
  char hex_id[OBJSTORE_ID_SIZE * 2 + 1];
  objstore_fs_id_to_hex (id, hex_id);
  char *path = file_backend_put_path (file_txn, hex_id);
  if (path == NULL)
    {
      return SQLITE_NOMEM;
    }
  int flags = O_WRONLY | O_CREAT | O_TRUNC;
#if defined(O_CLOEXEC)
  flags |= O_CLOEXEC;
#endif
  int fd = open (path, flags, 0666);
  if (fd < 0)
    {
      sqlite3_free (path);
      return SQLITE_IOERR;
    }
  if (file_backend_force_disk_full ())
    {
      close (fd);
      sqlite3_free (path);
      return SQLITE_FULL;
    }
  rc = SQLITE_OK;
  if (reader->size_hint > 0)
    {
      rc = file_backend_preallocate_fd (fd, reader->size_hint);
      if (rc != SQLITE_OK)
        {
          close (fd);
          unlink (path);
          sqlite3_free (path);
          return rc;
        }
    }

  unsigned char *buffer = file_txn->io_buffer;
  const size_t buffer_size = (file_txn->io_buffer_size > 0)
                                 ? file_txn->io_buffer_size
                                 : file_txn->env->chunk_size;
  if (buffer == NULL || buffer_size == 0)
    {
      close (fd);
      sqlite3_free (path);
      return SQLITE_NOMEM;
    }

  for (;;)
    {
      size_t chunk = 0;
      rc = reader->pull (reader->ctx, buffer, buffer_size, &chunk);
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
      rc = file_backend_write_all (fd, buffer, chunk);
      if (rc != SQLITE_OK)
        {
          break;
        }
    }
  if (rc == SQLITE_OK)
    {
      rc = file_backend_flush_fd (
          fd, file_backend_should_fsync_payload (file_txn->env));
    }
  close (fd);
  if (rc != SQLITE_OK)
    {
      unlink (path);
      sqlite3_free (path);
      return rc;
    }

  rc = file_backend_manifest_append (file_txn, "PUT", hex_id);
  sqlite3_free (path);
  return rc;
}

static int
file_backend_get_from_committed (file_backend_env *env, const char *hex_id,
                                 const objstore_stream_writer *writer)
{
  char *path = file_backend_committed_object_path (env, hex_id);
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
  int rc = file_backend_stream_file (file, writer, env->chunk_size);
  fclose (file);
  return rc;
}

static int
file_backend_get_size_pending (file_backend_txn *txn, const char *hex_id,
                               sqlite3_int64 *out_size)
{
  if (txn == NULL || hex_id == NULL)
    {
      return SQLITE_MISUSE;
    }
  if (txn->active_dir == NULL)
    {
      return SQLITE_NOTFOUND;
    }
  char *pending_path = file_backend_put_path (txn, hex_id);
  if (pending_path == NULL)
    {
      return SQLITE_NOMEM;
    }
  if (!file_backend_path_exists (pending_path))
    {
      sqlite3_free (pending_path);
      return SQLITE_NOTFOUND;
    }
  int rc = file_backend_stat_size (pending_path, out_size);
  sqlite3_free (pending_path);
  return rc;
}

static int
file_backend_get_size_committed (file_backend_env *env, const char *hex_id,
                                 sqlite3_int64 *out_size)
{
  char *path = file_backend_committed_object_path (env, hex_id);
  if (path == NULL)
    {
      return SQLITE_NOMEM;
    }
  int rc = file_backend_stat_size (path, out_size);
  sqlite3_free (path);
  return rc;
}

static int
file_backend_get_size (objstore_backend_txn *txn, const objstore_id *id,
                       sqlite3_int64 *out_size)
{
  if (txn == NULL || id == NULL || out_size == NULL)
    {
      return SQLITE_MISUSE;
    }
  file_backend_txn *file_txn = file_txn_from (txn);
  char hex_id[OBJSTORE_ID_SIZE * 2 + 1];
  objstore_fs_id_to_hex (id, hex_id);

  char *delete_marker = file_backend_delete_marker (file_txn, hex_id);
  if (delete_marker == NULL)
    {
      return SQLITE_NOMEM;
    }
  if (file_backend_path_exists (delete_marker))
    {
      sqlite3_free (delete_marker);
      return SQLITE_NOTFOUND;
    }
  sqlite3_free (delete_marker);

  int pending_rc = file_backend_get_size_pending (file_txn, hex_id, out_size);
  if (pending_rc != SQLITE_NOTFOUND)
    {
      return pending_rc;
    }
  return file_backend_get_size_committed (file_txn->env, hex_id, out_size);
}

static int
file_backend_get_range_from_committed (file_backend_env *env, const char *hex_id,
                                       sqlite3_uint64 offset,
                                       sqlite3_uint64 length,
                                       const objstore_stream_writer *writer)
{
  char *path = file_backend_committed_object_path (env, hex_id);
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
  int rc = file_backend_stream_file_range (file, writer, env->chunk_size,
                                           offset, length);
  fclose (file);
  return rc;
}

static int
file_backend_get_range (objstore_backend_txn *txn, const objstore_id *id,
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
  file_backend_txn *file_txn = file_txn_from (txn);
  char hex_id[OBJSTORE_ID_SIZE * 2 + 1];
  objstore_fs_id_to_hex (id, hex_id);

  char *delete_marker = file_backend_delete_marker (file_txn, hex_id);
  if (delete_marker == NULL)
    {
      return SQLITE_NOMEM;
    }
  if (file_backend_path_exists (delete_marker))
    {
      sqlite3_free (delete_marker);
      return SQLITE_NOTFOUND;
    }
  sqlite3_free (delete_marker);

  if (file_txn->active_dir != NULL)
    {
      char *pending_path = file_backend_put_path (file_txn, hex_id);
      if (pending_path == NULL)
        {
          return SQLITE_NOMEM;
        }
      if (file_backend_path_exists (pending_path))
        {
          FILE *file = fopen (pending_path, "rb");
          sqlite3_free (pending_path);
          if (file == NULL)
            {
              return SQLITE_IOERR;
            }
          int rc = file_backend_stream_file_range (
              file, writer, file_txn->env->chunk_size, offset, length);
          fclose (file);
          return rc;
        }
      sqlite3_free (pending_path);
    }

  return file_backend_get_range_from_committed (file_txn->env, hex_id, offset,
                                                length, writer);
}

static int
file_backend_get (objstore_backend_txn *txn, const objstore_id *id,
                  const objstore_stream_writer *writer)
{
  if (txn == NULL || id == NULL || writer == NULL)
    {
      return SQLITE_MISUSE;
    }
  file_backend_txn *file_txn = file_txn_from (txn);
  char hex_id[OBJSTORE_ID_SIZE * 2 + 1];
  objstore_fs_id_to_hex (id, hex_id);

  char *delete_marker = file_backend_delete_marker (file_txn, hex_id);
  if (delete_marker == NULL)
    {
      return SQLITE_NOMEM;
    }
  if (file_backend_path_exists (delete_marker))
    {
      sqlite3_free (delete_marker);
      return SQLITE_NOTFOUND;
    }
  sqlite3_free (delete_marker);

  int pending_rc = file_backend_stream_pending (file_txn, hex_id, writer);
  if (pending_rc != SQLITE_NOTFOUND)
    {
      return pending_rc;
    }

  return file_backend_get_from_committed (file_txn->env, hex_id, writer);
}

static int
file_backend_exists (objstore_backend_txn *txn, const objstore_id *id)
{
  if (txn == NULL || id == NULL)
    {
      return SQLITE_MISUSE;
    }

  file_backend_txn *file_txn = file_txn_from (txn);
  char hex_id[OBJSTORE_ID_SIZE * 2 + 1];
  objstore_fs_id_to_hex (id, hex_id);

  char *marker = file_backend_delete_marker (file_txn, hex_id);
  if (marker == NULL)
    {
      return SQLITE_NOMEM;
    }
  const bool has_marker = file_backend_path_exists (marker);
  sqlite3_free (marker);
  if (has_marker)
    {
      return SQLITE_NOTFOUND;
    }

  char *pending = file_backend_put_path (file_txn, hex_id);
  if (pending == NULL)
    {
      return SQLITE_NOMEM;
    }
  const bool has_pending = file_backend_path_exists (pending);
  sqlite3_free (pending);
  if (has_pending)
    {
      return SQLITE_OK;
    }

  char *committed = file_backend_committed_object_path (file_txn->env, hex_id);
  if (committed == NULL)
    {
      return SQLITE_NOMEM;
    }
  const bool has_committed = file_backend_path_exists (committed);
  sqlite3_free (committed);
  return has_committed ? SQLITE_OK : SQLITE_NOTFOUND;
}

static int
file_backend_delete (objstore_backend_txn *txn, const objstore_id *id)
{
  if (txn == NULL || id == NULL)
    {
      return SQLITE_MISUSE;
    }
  file_backend_txn *file_txn = file_txn_from (txn);
  int rc = file_backend_savepoint_capture_state (file_txn, id);
  if (rc != SQLITE_OK)
    {
      return rc;
    }
  char hex_id[OBJSTORE_ID_SIZE * 2 + 1];
  objstore_fs_id_to_hex (id, hex_id);
  char *marker = file_backend_delete_marker (file_txn, hex_id);
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
  return file_backend_manifest_append (file_txn, "DEL", hex_id);
}

static int
compare_ids (const void *lhs, const void *rhs)
{
  return memcmp (lhs, rhs, sizeof (objstore_id));
}

static int
file_backend_collect_ids_append (objstore_id **ids, size_t *count,
                                 size_t *capacity, const objstore_id *id)
{
  if (ids == NULL || count == NULL || capacity == NULL || id == NULL)
    {
      return SQLITE_MISUSE;
    }
  if (*capacity == *count)
    {
      size_t new_capacity = (*capacity == 0) ? 16 : (*capacity * 2);
      objstore_id *resized
          = sqlite3_realloc64 (*ids, new_capacity * sizeof (objstore_id));
      if (resized == NULL)
        {
          return SQLITE_NOMEM;
        }
      *ids = resized;
      *capacity = new_capacity;
    }
  (*ids)[*count] = *id;
  ++(*count);
  return SQLITE_OK;
}

static int
file_backend_collect_object_ids_from_dir (const char *path, objstore_id **ids,
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
          rc = file_backend_collect_object_ids_from_dir (child, ids, count,
                                                         capacity);
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
      if (hex_len != kObjstoreIdHexChars)
        {
          sqlite3_free (child);
          continue;
        }
      objstore_id id = { 0 };
      char hex[OBJSTORE_ID_SIZE * 2 + 1] = { 0 };
      memcpy (hex, entry->d_name, hex_len);
      if (objstore_fs_hex_to_id (hex, &id) == SQLITE_OK)
        {
          rc = file_backend_collect_ids_append (ids, count, capacity, &id);
        }
      sqlite3_free (child);
    }
  closedir (dir);
  return rc;
}

static int
file_backend_collect_object_ids (file_backend_env *env, objstore_id **out_ids,
                                 size_t *out_count)
{
  if (env == NULL || out_ids == NULL || out_count == NULL)
    {
      return SQLITE_MISUSE;
    }
  objstore_id *ids = NULL;
  size_t count = 0;
  size_t capacity = 0;
  int rc = file_backend_collect_object_ids_from_dir (env->objects_root, &ids,
                                                     &count, &capacity);
  if (rc != SQLITE_OK)
    {
      sqlite3_free (ids);
      return rc;
    }
  if (count > 1)
    {
      qsort (ids, count, sizeof (objstore_id), compare_ids);
    }
  *out_ids = ids;
  *out_count = count;
  return SQLITE_OK;
}

static int
file_backend_collect_ids_from_row_dir (file_backend_env *env,
                                       const char *row_dir_path,
                                       objstore_id **ids, size_t *count,
                                       size_t *capacity)
{
  DIR *dir = opendir (row_dir_path);
  if (dir == NULL)
    {
      return (errno == ENOENT) ? SQLITE_OK : SQLITE_IOERR;
    }
  int rc = SQLITE_OK;
  struct dirent *entry = NULL;
  while ((entry = readdir (dir)) != NULL && rc == SQLITE_OK)
    {
      if (strcmp (entry->d_name, ".") == 0
          || strcmp (entry->d_name, "..") == 0)
        {
          continue;
        }
      if (strlen (entry->d_name) != kObjstoreIdHexChars)
        {
          continue;
        }
      objstore_id id = { 0 };
      if (objstore_fs_hex_to_id (entry->d_name, &id) != SQLITE_OK)
        {
          continue;
        }
      if (!file_backend_object_exists (env, &id))
        {
          (void)file_backend_rowidx_remove_entry (env, &id);
          continue;
        }
      rc = file_backend_collect_ids_append (ids, count, capacity, &id);
    }
  closedir (dir);
  return rc;
}

static int
file_backend_collect_ids (file_backend_env *env, objstore_id **out_ids,
                          size_t *out_count)
{
  if (env == NULL || out_ids == NULL || out_count == NULL)
    {
      return SQLITE_MISUSE;
    }
  objstore_id *ids = NULL;
  size_t count = 0;
  size_t capacity = 0;
  DIR *rowidx_root = opendir (env->rowidx_root);
  if (rowidx_root == NULL)
    {
      return (errno == ENOENT) ? SQLITE_OK : SQLITE_IOERR;
    }
  int rc = SQLITE_OK;
  struct dirent *shard_entry = NULL;
  while ((shard_entry = readdir (rowidx_root)) != NULL && rc == SQLITE_OK)
    {
      if (strlen (shard_entry->d_name) != FILE_BACKEND_ROWIDX_SHARD_CHARS)
        {
          continue;
        }
      char *shard_path = objstore_fs_path_join (env->rowidx_root,
                                                shard_entry->d_name);
      if (shard_path == NULL)
        {
          rc = SQLITE_NOMEM;
          break;
        }
      DIR *shard_dir = opendir (shard_path);
      if (shard_dir == NULL)
        {
          sqlite3_free (shard_path);
          rc = (errno == ENOENT) ? SQLITE_OK : SQLITE_IOERR;
          if (rc == SQLITE_OK)
            {
              continue;
            }
          break;
        }
      struct dirent *row_entry = NULL;
      while ((row_entry = readdir (shard_dir)) != NULL && rc == SQLITE_OK)
        {
          if (strlen (row_entry->d_name) != OBJSTORE_ROWID_HEX_CHARS)
            {
              continue;
            }
          char *row_dir_path = objstore_fs_path_join (shard_path,
                                                      row_entry->d_name);
          if (row_dir_path == NULL)
            {
              rc = SQLITE_NOMEM;
              break;
            }
          rc = file_backend_collect_ids_from_row_dir (env, row_dir_path, &ids,
                                                      &count, &capacity);
          sqlite3_free (row_dir_path);
        }
      closedir (shard_dir);
      sqlite3_free (shard_path);
    }
  closedir (rowidx_root);
  if (rc != SQLITE_OK)
    {
      sqlite3_free (ids);
      return rc;
    }

  if (count > 1)
    {
      qsort (ids, count, sizeof (objstore_id), compare_ids);
    }

  *out_ids = ids;
  *out_count = count;
  return SQLITE_OK;
}

static int
file_backend_scan_open (objstore_backend_txn *txn,
                        objstore_backend_cursor **out_cursor)
{
  if (txn == NULL || out_cursor == NULL)
    {
      return SQLITE_MISUSE;
    }
  file_backend_txn *file_txn = file_txn_from (txn);
  objstore_id *ids = NULL;
  size_t count = 0;
  int rc = file_backend_collect_ids (file_txn->env, &ids, &count);
  if (rc != SQLITE_OK)
    {
      return rc;
    }
  file_backend_cursor *cursor = sqlite3_malloc (sizeof (*cursor));
  if (cursor == NULL)
    {
      sqlite3_free (ids);
      return SQLITE_NOMEM;
    }
  cursor->env = file_txn->env;
  cursor->ids = ids;
  cursor->count = count;
  cursor->index = 0;
  *out_cursor = (objstore_backend_cursor *)cursor;
  return SQLITE_OK;
}

static int
file_backend_scan_next (objstore_backend_cursor *cursor, objstore_id *out_id)
{
  if (cursor == NULL || out_id == NULL)
    {
      return SQLITE_MISUSE;
    }
  file_backend_cursor *file_cursor = file_cursor_from (cursor);
  if (file_cursor->index >= file_cursor->count)
    {
      return SQLITE_DONE;
    }
  *out_id = file_cursor->ids[file_cursor->index++];
  return SQLITE_OK;
}

static void
file_backend_scan_close (objstore_backend_cursor *cursor)
{
  if (cursor == NULL)
    {
      return;
    }
  file_backend_cursor *file_cursor = file_cursor_from (cursor);
  sqlite3_free (file_cursor->ids);
  sqlite3_free (file_cursor);
}

const objstore_backend objstore_backend_file = {
  .kind = OBJSTORE_BACKEND_FILE,
  .name = "file",
  .open_env = file_backend_open_env,
  .close_env = file_backend_close_env,
  .begin_txn = file_backend_begin_txn,
  .commit_txn = file_backend_commit_txn,
  .rollback_txn = file_backend_rollback_txn,
  .savepoint_begin = file_backend_savepoint_begin,
  .savepoint_release = file_backend_savepoint_release,
  .savepoint_rollback = file_backend_savepoint_rollback,
  .staged_write_begin = file_backend_staged_write_begin,
  .staged_write_push = file_backend_staged_write_push,
  .staged_write_finalize = file_backend_staged_write_finalize,
  .staged_write_set_size_hint = file_backend_staged_write_set_size_hint,
  .commit_staged = file_backend_commit_staged,
  .rollback_staged = file_backend_rollback_staged,
  .put = file_backend_put,
  .get = file_backend_get,
  .get_range = file_backend_get_range,
  .get_size = file_backend_get_size,
  .delete_fn = file_backend_delete,
  .exists = file_backend_exists,
  .scan_open = file_backend_scan_open,
  .scan_next = file_backend_scan_next,
  .scan_close = file_backend_scan_close,
  .lookup_id_by_rowid = file_backend_lookup_id_by_rowid,
};
