#if !defined(_WIN32)
#define _POSIX_C_SOURCE 200809L
#endif

#include "unity.h"

#include <dirent.h>
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "test_support.h"

static const char *kPutDir = "put";
static const char *kManifestFile = "manifest.log";
static const char *kFileSuffix = ".dat";

static int
backend_file_open_memory_db (sqlite3 **out_db, const char *label)
{
  if (out_db == NULL)
    {
      return SQLITE_MISUSE;
    }
  static int counter = 0;
  char uri[128];
  const char *tag = (label != NULL) ? label : "backend";
  sqlite3_snprintf (sizeof (uri), uri,
                    "file:objstore_%s_%d?mode=memory&cache=shared", tag,
                    counter++);
  const int flags = SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE
                    | SQLITE_OPEN_URI | SQLITE_OPEN_SHAREDCACHE;
  return sqlite3_open_v2 (uri, out_db, flags, NULL);
}

typedef struct buffer_reader
{
  const uint8_t *data;
  size_t size;
  size_t offset;
} buffer_reader;

typedef struct buffer_writer
{
  uint8_t *data;
  size_t length;
  size_t capacity;
} buffer_writer;

static int
buffer_reader_pull (void *ctx, void *buffer, size_t capacity, size_t *nread)
{
  buffer_reader *reader = (buffer_reader *)ctx;
  if (reader->offset >= reader->size)
    {
      *nread = 0;
      return SQLITE_DONE;
    }
  const size_t remaining = reader->size - reader->offset;
  const size_t to_copy = remaining < capacity ? remaining : capacity;
  memcpy (buffer, reader->data + reader->offset, to_copy);
  reader->offset += to_copy;
  *nread = to_copy;
  return SQLITE_OK;
}

static int
buffer_writer_push (void *ctx, const void *buffer, size_t nread)
{
  buffer_writer *writer = (buffer_writer *)ctx;
  if (writer->length + nread > writer->capacity)
    {
      size_t new_capacity = writer->capacity == 0 ? 256 : writer->capacity;
      while (new_capacity < writer->length + nread)
        {
          new_capacity *= 2;
        }
      uint8_t *resized = (uint8_t *)realloc (writer->data, new_capacity);
      if (resized == NULL)
        {
          return SQLITE_NOMEM;
        }
      writer->data = resized;
      writer->capacity = new_capacity;
    }
  memcpy (writer->data + writer->length, buffer, nread);
  writer->length += nread;
  return SQLITE_OK;
}

static objstore_id
make_id (uint8_t seed)
{
  objstore_id id;
  for (size_t i = 0; i < OBJSTORE_ID_SIZE; ++i)
    {
      id.bytes[i] = (uint8_t)(seed + i);
    }
  return id;
}

static void
id_to_hex (const objstore_id *id, char *hex_out)
{
  static const char digits[] = "0123456789abcdef";
  for (size_t i = 0; i < OBJSTORE_ID_SIZE; ++i)
    {
      hex_out[i * 2] = digits[(id->bytes[i] >> 4) & 0xF];
      hex_out[i * 2 + 1] = digits[id->bytes[i] & 0xF];
    }
  hex_out[OBJSTORE_ID_SIZE * 2] = '\0';
}

static char *
create_temp_dir (void)
{
  for (int attempt = 0; attempt < 16; ++attempt)
    {
      unsigned char random_bytes[6] = { 0 };
      char random_hex[sizeof (random_bytes) * 2 + 1];
      sqlite3_randomness ((int)sizeof (random_bytes), random_bytes);
      for (size_t i = 0; i < sizeof (random_bytes); ++i)
        {
          sqlite3_snprintf (3, &random_hex[i * 2], "%02x", random_bytes[i]);
        }
      random_hex[sizeof (random_hex) - 1] = '\0';

      char *path
          = sqlite3_mprintf ("/tmp/objstore-file-backend-%s", random_hex);
      if (path == NULL)
        {
          return NULL;
        }
      if (mkdir (path, 0700) == 0)
        {
          return path;
        }
      sqlite3_free (path);
      if (errno != EEXIST)
        {
          return NULL;
        }
    }
  return NULL;
}

static int
remove_tree (const char *path)
{
  DIR *dir = opendir (path);
  if (dir == NULL)
    {
      return (errno == ENOENT) ? 0 : -1;
    }
  struct dirent *entry = NULL;
  while ((entry = readdir (dir)) != NULL)
    {
      if (strcmp (entry->d_name, ".") == 0
          || strcmp (entry->d_name, "..") == 0)
        {
          continue;
        }
      char *child = sqlite3_mprintf ("%s/%s", path, entry->d_name);
      if (child == NULL)
        {
          closedir (dir);
          return -1;
        }
      struct stat st;
      if (stat (child, &st) == 0 && S_ISDIR (st.st_mode))
        {
          remove_tree (child);
        }
      else
        {
          unlink (child);
        }
      sqlite3_free (child);
    }
  closedir (dir);
  return rmdir (path);
}

static void
cleanup_temp_dirs (char *base)
{
  if (base != NULL)
    {
      remove_tree (base);
      sqlite3_free (base);
    }
}

static void
ensure_dir (const char *path)
{
  if (path == NULL || path[0] == '\0')
    {
      return;
    }
  char *mutable_path = sqlite3_mprintf ("%s", path);
  if (mutable_path == NULL)
    {
      return;
    }
  for (char *p = mutable_path + 1; *p != '\0'; ++p)
    {
      if (*p == '/')
        {
          *p = '\0';
          mkdir (mutable_path, 0777);
          *p = '/';
        }
    }
  mkdir (mutable_path, 0777);
  sqlite3_free (mutable_path);
}

static void
compose_paths (char *base, char **objects_root, char **staging_root)
{
  *objects_root = sqlite3_mprintf ("%s/objects", base);
  *staging_root = (*objects_root != NULL)
                      ? sqlite3_mprintf ("%s/.staging", *objects_root)
                      : NULL;
}

static void
rowidx_entry_path (const char *objects_root, const objstore_id *id,
                   char **out_path)
{
  unsigned char prefix[OBJSTORE_ROWID_PREFIX_SIZE];
  objstore_rowid_prefix_from_id (id, prefix);
  char prefix_hex[OBJSTORE_ROWID_HEX_CHARS + 1];
  static const char digits[] = "0123456789abcdef";
  for (size_t i = 0; i < OBJSTORE_ROWID_PREFIX_SIZE; ++i)
    {
      prefix_hex[i * 2] = digits[(prefix[i] >> 4) & 0x0F];
      prefix_hex[i * 2 + 1] = digits[prefix[i] & 0x0F];
    }
  prefix_hex[OBJSTORE_ROWID_HEX_CHARS] = '\0';

  char shard_name[3] = { prefix_hex[0], prefix_hex[1], '\0' };
  char hex_id[OBJSTORE_ID_SIZE * 2 + 1];
  id_to_hex (id, hex_id);

  char *row_dir = sqlite3_mprintf ("%s/rowidx/%s/%s", objects_root, shard_name,
                                   prefix_hex);
  TEST_ASSERT_NOT_NULL (row_dir);
  ensure_dir (row_dir);
  *out_path = sqlite3_mprintf ("%s/%s", row_dir, hex_id);
  sqlite3_free (row_dir);
  TEST_ASSERT_NOT_NULL (*out_path);
}

static void
ensure_staging_layout (const char *staging_root)
{
  ensure_dir (staging_root);
  char *active_root = sqlite3_mprintf ("%s/active", staging_root);
  char *commit_root = sqlite3_mprintf ("%s/commit", staging_root);
  TEST_ASSERT_NOT_NULL (active_root);
  TEST_ASSERT_NOT_NULL (commit_root);
  ensure_dir (active_root);
  ensure_dir (commit_root);
  sqlite3_free (active_root);
  sqlite3_free (commit_root);
}

static void
init_file_backend (char **out_base, char **out_objects, char **out_staging,
                   sqlite3 **out_db, const objstore_backend **out_backend,
                   objstore_backend_env **out_env)
{
  char *base = create_temp_dir ();
  TEST_ASSERT_NOT_NULL (base);
  char *objects_root = NULL;
  char *staging_root = NULL;
  compose_paths (base, &objects_root, &staging_root);
  TEST_ASSERT_NOT_NULL (objects_root);
  TEST_ASSERT_NOT_NULL (staging_root);
  ensure_dir (objects_root);
  ensure_staging_layout (staging_root);

  sqlite3 *db = NULL;
  TEST_ASSERT_EQUAL_INT (SQLITE_OK,
                         backend_file_open_memory_db (&db, "file-put"));

  const objstore_backend *backend
      = objstore_backend_by_kind (OBJSTORE_BACKEND_FILE);
  TEST_ASSERT_NOT_NULL (backend);

  objstore_config cfg = {
    .backend = OBJSTORE_BACKEND_FILE,
    .storage_root = objects_root,
    .chunk_size_bytes = 1024,
    .reserved_flags = 0,
  };

  objstore_backend_env *env = NULL;
  TEST_ASSERT_EQUAL_INT (SQLITE_OK, backend->open_env (db, &cfg, &env));

  *out_base = base;
  *out_objects = objects_root;
  *out_staging = staging_root;
  *out_db = db;
  *out_backend = backend;
  *out_env = env;
}

static void
shutdown_file_backend (char *base, char *objects_root, char *staging_root,
                       sqlite3 *db, const objstore_backend *backend,
                       objstore_backend_env *env)
{
  if (backend != NULL && env != NULL)
    {
      backend->close_env (env);
    }
  if (db != NULL)
    {
      sqlite3_close (db);
    }
  sqlite3_free (objects_root);
  sqlite3_free (staging_root);
  cleanup_temp_dirs (base);
}

static void
file_backend_roundtrip (void)
{
  char *base = NULL;
  char *objects_root = NULL;
  char *staging_root = NULL;
  sqlite3 *db = NULL;
  const objstore_backend *backend = NULL;
  objstore_backend_env *env = NULL;
  init_file_backend (&base, &objects_root, &staging_root, &db, &backend, &env);

  objstore_id id = make_id (0x60);
  uint8_t payload[] = { 10, 11, 12, 13 };
  buffer_reader reader
      = { .data = payload, .size = sizeof (payload), .offset = 0 };
  objstore_stream_reader stream_reader
      = { .ctx = &reader,
          .pull = buffer_reader_pull,
          .size_hint = (sqlite3_int64)reader.size };

  objstore_backend_txn *txn = NULL;
  TEST_ASSERT_EQUAL_INT (SQLITE_OK, backend->begin_txn (env, &txn));
  TEST_ASSERT_EQUAL_INT (SQLITE_OK, backend->put (txn, &id, &stream_reader));
  TEST_ASSERT_EQUAL_INT (SQLITE_OK, backend->commit_txn (txn));

  buffer_writer writer = { .data = NULL, .length = 0, .capacity = 0 };
  objstore_stream_writer stream_writer
      = { .ctx = &writer, .push = buffer_writer_push };
  objstore_backend_txn *read_txn = NULL;
  TEST_ASSERT_EQUAL_INT (SQLITE_OK, backend->begin_txn (env, &read_txn));
  TEST_ASSERT_EQUAL_INT (SQLITE_OK,
                         backend->get (read_txn, &id, &stream_writer));
  TEST_ASSERT_EQUAL_INT (SQLITE_OK, backend->commit_txn (read_txn));

  TEST_ASSERT_EQUAL_UINT64 ((uint64_t)sizeof (payload),
                            (uint64_t)writer.length);
  TEST_ASSERT_EQUAL_MEMORY (payload, writer.data, sizeof (payload));
  free (writer.data);

  shutdown_file_backend (base, objects_root, staging_root, db, backend, env);
}

static void
file_backend_range_reads (void)
{
  char *base = NULL;
  char *objects_root = NULL;
  char *staging_root = NULL;
  sqlite3 *db = NULL;
  const objstore_backend *backend = NULL;
  objstore_backend_env *env = NULL;
  init_file_backend (&base, &objects_root, &staging_root, &db, &backend, &env);

  objstore_id id = make_id (0x62);
  uint8_t payload[] = { 1, 2, 3, 4, 5, 6 };
  buffer_reader reader
      = { .data = payload, .size = sizeof (payload), .offset = 0 };
  objstore_stream_reader stream_reader
      = { .ctx = &reader,
          .pull = buffer_reader_pull,
          .size_hint = (sqlite3_int64)reader.size };

  objstore_backend_txn *txn = NULL;
  TEST_ASSERT_EQUAL_INT (SQLITE_OK, backend->begin_txn (env, &txn));
  TEST_ASSERT_EQUAL_INT (SQLITE_OK, backend->put (txn, &id, &stream_reader));
  TEST_ASSERT_EQUAL_INT (SQLITE_OK, backend->commit_txn (txn));

  objstore_backend_txn *read_txn = NULL;
  TEST_ASSERT_EQUAL_INT (SQLITE_OK, backend->begin_txn (env, &read_txn));
  sqlite3_int64 size = -1;
  TEST_ASSERT_EQUAL_INT (SQLITE_OK, backend->get_size (read_txn, &id, &size));
  TEST_ASSERT_EQUAL_INT ((int)sizeof (payload), (int)size);

  buffer_writer writer = { .data = NULL, .length = 0, .capacity = 0 };
  objstore_stream_writer stream_writer
      = { .ctx = &writer, .push = buffer_writer_push };
  TEST_ASSERT_EQUAL_INT (SQLITE_OK,
                         backend->get_range (read_txn, &id, 3, 2,
                                             &stream_writer));
  TEST_ASSERT_EQUAL_UINT64 (2, (uint64_t)writer.length);
  TEST_ASSERT_EQUAL_MEMORY (payload + 3, writer.data, 2);
  free (writer.data);
  TEST_ASSERT_EQUAL_INT (SQLITE_OK, backend->commit_txn (read_txn));

  shutdown_file_backend (base, objects_root, staging_root, db, backend, env);
}

static void
file_backend_range_unsatisfied (void)
{
  char *base = NULL;
  char *objects_root = NULL;
  char *staging_root = NULL;
  sqlite3 *db = NULL;
  const objstore_backend *backend = NULL;
  objstore_backend_env *env = NULL;
  init_file_backend (&base, &objects_root, &staging_root, &db, &backend, &env);

  objstore_id id = make_id (0x63);
  uint8_t payload[] = { 1, 2, 3, 4 };
  buffer_reader reader
      = { .data = payload, .size = sizeof (payload), .offset = 0 };
  objstore_stream_reader stream_reader
      = { .ctx = &reader,
          .pull = buffer_reader_pull,
          .size_hint = (sqlite3_int64)reader.size };

  objstore_backend_txn *txn = NULL;
  TEST_ASSERT_EQUAL_INT (SQLITE_OK, backend->begin_txn (env, &txn));
  TEST_ASSERT_EQUAL_INT (SQLITE_OK, backend->put (txn, &id, &stream_reader));
  TEST_ASSERT_EQUAL_INT (SQLITE_OK, backend->commit_txn (txn));

  objstore_backend_txn *read_txn = NULL;
  TEST_ASSERT_EQUAL_INT (SQLITE_OK, backend->begin_txn (env, &read_txn));
  buffer_writer writer = { .data = NULL, .length = 0, .capacity = 0 };
  objstore_stream_writer stream_writer
      = { .ctx = &writer, .push = buffer_writer_push };
  TEST_ASSERT_EQUAL_INT (SQLITE_RANGE,
                         backend->get_range (read_txn, &id, 10, 1,
                                             &stream_writer));
  free (writer.data);
  TEST_ASSERT_EQUAL_INT (SQLITE_OK, backend->commit_txn (read_txn));

  shutdown_file_backend (base, objects_root, staging_root, db, backend, env);
}

static void
file_backend_large_object_streams (void)
{
  char *base = NULL;
  char *objects_root = NULL;
  char *staging_root = NULL;
  sqlite3 *db = NULL;
  const objstore_backend *backend = NULL;
  objstore_backend_env *env = NULL;
  init_file_backend (&base, &objects_root, &staging_root, &db, &backend, &env);

  const size_t payload_size = 2 * 1024 * 1024;
  uint8_t *payload = (uint8_t *)malloc (payload_size);
  TEST_ASSERT_NOT_NULL (payload);
  for (size_t i = 0; i < payload_size; ++i)
    {
      payload[i] = (uint8_t)(i & 0xFFu);
    }

  objstore_id id = make_id (0x64);
  buffer_reader reader
      = { .data = payload, .size = payload_size, .offset = 0 };
  objstore_stream_reader stream_reader
      = { .ctx = &reader,
          .pull = buffer_reader_pull,
          .size_hint = (sqlite3_int64)reader.size };

  objstore_backend_txn *txn = NULL;
  TEST_ASSERT_EQUAL_INT (SQLITE_OK, backend->begin_txn (env, &txn));
  TEST_ASSERT_EQUAL_INT (SQLITE_OK, backend->put (txn, &id, &stream_reader));
  TEST_ASSERT_EQUAL_INT (SQLITE_OK, backend->commit_txn (txn));

  buffer_writer writer = { .data = NULL, .length = 0, .capacity = 0 };
  objstore_stream_writer stream_writer
      = { .ctx = &writer, .push = buffer_writer_push };
  objstore_backend_txn *read_txn = NULL;
  TEST_ASSERT_EQUAL_INT (SQLITE_OK, backend->begin_txn (env, &read_txn));
  TEST_ASSERT_EQUAL_INT (SQLITE_OK,
                         backend->get (read_txn, &id, &stream_writer));
  backend->rollback_txn (read_txn);

  TEST_ASSERT_EQUAL_UINT64 ((uint64_t)payload_size, (uint64_t)writer.length);
  TEST_ASSERT_EQUAL_MEMORY (payload, writer.data, payload_size);
  free (writer.data);
  free (payload);

  shutdown_file_backend (base, objects_root, staging_root, db, backend, env);
}

static void
file_backend_delete_removes_object (void)
{
  char *base = NULL;
  char *objects_root = NULL;
  char *staging_root = NULL;
  sqlite3 *db = NULL;
  const objstore_backend *backend = NULL;
  objstore_backend_env *env = NULL;
  init_file_backend (&base, &objects_root, &staging_root, &db, &backend, &env);

  objstore_id id = make_id (0x61);
  uint8_t payload[] = { 1, 2, 3 };
  buffer_reader reader
      = { .data = payload, .size = sizeof (payload), .offset = 0 };
  objstore_stream_reader stream_reader
      = { .ctx = &reader,
          .pull = buffer_reader_pull,
          .size_hint = (sqlite3_int64)reader.size };

  objstore_backend_txn *txn = NULL;
  TEST_ASSERT_EQUAL_INT (SQLITE_OK, backend->begin_txn (env, &txn));
  TEST_ASSERT_EQUAL_INT (SQLITE_OK, backend->put (txn, &id, &stream_reader));
  TEST_ASSERT_EQUAL_INT (SQLITE_OK, backend->delete_fn (txn, &id));
  TEST_ASSERT_EQUAL_INT (SQLITE_OK, backend->commit_txn (txn));

  objstore_stream_writer stream_writer
      = { .ctx = NULL, .push = buffer_writer_push };
  objstore_backend_txn *read_txn = NULL;
  TEST_ASSERT_EQUAL_INT (SQLITE_OK, backend->begin_txn (env, &read_txn));
  TEST_ASSERT_EQUAL_INT (SQLITE_NOTFOUND,
                         backend->get (read_txn, &id, &stream_writer));
  backend->rollback_txn (read_txn);

  shutdown_file_backend (base, objects_root, staging_root, db, backend, env);
}

static void
file_backend_begin_txn_respects_permissions (void)
{
  char *base = NULL;
  char *objects_root = NULL;
  char *staging_root = NULL;
  sqlite3 *db = NULL;
  const objstore_backend *backend = NULL;
  objstore_backend_env *env = NULL;
  init_file_backend (&base, &objects_root, &staging_root, &db, &backend, &env);

  char *active_dir = sqlite3_mprintf ("%s/active", staging_root);
  TEST_ASSERT_NOT_NULL (active_dir);
  TEST_ASSERT_EQUAL_INT (0, chmod (staging_root, 0555));
  TEST_ASSERT_EQUAL_INT (0, chmod (active_dir, 0555));

  objstore_backend_txn *txn = NULL;
  TEST_ASSERT_EQUAL_INT (SQLITE_IOERR, backend->begin_txn (env, &txn));

  TEST_ASSERT_EQUAL_INT (0, chmod (staging_root, 0755));
  TEST_ASSERT_EQUAL_INT (0, chmod (active_dir, 0755));
  sqlite3_free (active_dir);
  shutdown_file_backend (base, objects_root, staging_root, db, backend, env);
}

static void
file_backend_put_handles_disk_full (void)
{
  char *base = NULL;
  char *objects_root = NULL;
  char *staging_root = NULL;
  sqlite3 *db = NULL;
  const objstore_backend *backend = NULL;
  objstore_backend_env *env = NULL;
  init_file_backend (&base, &objects_root, &staging_root, &db, &backend, &env);

  objstore_id id = make_id (0x66);
  uint8_t payload[2048];
  for (size_t i = 0; i < sizeof (payload); ++i)
    {
      payload[i] = (uint8_t)(i & 0xFFu);
    }
  buffer_reader reader
      = { .data = payload, .size = sizeof (payload), .offset = 0 };
  objstore_stream_reader stream_reader
      = { .ctx = &reader,
          .pull = buffer_reader_pull,
          .size_hint = (sqlite3_int64)reader.size };

  objstore_backend_txn *txn = NULL;
  TEST_ASSERT_EQUAL_INT (SQLITE_OK, backend->begin_txn (env, &txn));
  setenv ("OBJSTORE_FORCE_DISK_FULL", "1", 1);
  int put_rc = backend->put (txn, &id, &stream_reader);
  backend->rollback_txn (txn);
  TEST_ASSERT_EQUAL_INT (SQLITE_FULL, put_rc);

  shutdown_file_backend (base, objects_root, staging_root, db, backend, env);
}

static void
file_backend_recovers_pending_commit (void)
{
  char *base = create_temp_dir ();
  TEST_ASSERT_NOT_NULL (base);
  char *objects_root = NULL;
  char *staging_root = NULL;
  compose_paths (base, &objects_root, &staging_root);
  TEST_ASSERT_NOT_NULL (objects_root);
  TEST_ASSERT_NOT_NULL (staging_root);
  ensure_dir (objects_root);
  ensure_staging_layout (staging_root);

  char *commit_root = sqlite3_mprintf ("%s/commit", staging_root);
  char *put_dir = NULL;
  char *manifest_path = NULL;
  char *payload_path = NULL;
  TEST_ASSERT_NOT_NULL (commit_root);
  ensure_dir (commit_root);
  char *txn_dir = sqlite3_mprintf ("%s/%s", commit_root, "crash");
  TEST_ASSERT_NOT_NULL (txn_dir);
  ensure_dir (txn_dir);

  put_dir = sqlite3_mprintf ("%s/%s", txn_dir, kPutDir);
  ensure_dir (put_dir);
  manifest_path = sqlite3_mprintf ("%s/%s", txn_dir, kManifestFile);

  objstore_id id = make_id (0x62);
  char hex[OBJSTORE_ID_SIZE * 2 + 1];
  id_to_hex (&id, hex);
  payload_path = sqlite3_mprintf ("%s/%s%s", put_dir, hex, kFileSuffix);
  FILE *payload = fopen (payload_path, "wb");
  TEST_ASSERT_NOT_NULL (payload);
  uint8_t bytes[] = { 9, 9, 9 };
  fwrite (bytes, 1, sizeof (bytes), payload);
  fclose (payload);

  FILE *manifest = fopen (manifest_path, "wb");
  TEST_ASSERT_NOT_NULL (manifest);
  fprintf (manifest, "PUT %s\n", hex);
  fclose (manifest);

  sqlite3_free (put_dir);
  sqlite3_free (manifest_path);
  sqlite3_free (payload_path);

  sqlite3 *db = NULL;
  TEST_ASSERT_EQUAL_INT (SQLITE_OK,
                         backend_file_open_memory_db (&db, "file-put2"));

  const objstore_backend *backend
      = objstore_backend_by_kind (OBJSTORE_BACKEND_FILE);
  TEST_ASSERT_NOT_NULL (backend);
  objstore_config cfg = {
    .backend = OBJSTORE_BACKEND_FILE,
    .storage_root = objects_root,
    .chunk_size_bytes = 1024,
    .reserved_flags = 0,
  };

  objstore_backend_env *env = NULL;
  TEST_ASSERT_EQUAL_INT (SQLITE_OK, backend->open_env (db, &cfg, &env));

  buffer_writer writer = { .data = NULL, .length = 0, .capacity = 0 };
  objstore_stream_writer stream_writer
      = { .ctx = &writer, .push = buffer_writer_push };
  objstore_backend_txn *txn = NULL;
  TEST_ASSERT_EQUAL_INT (SQLITE_OK, backend->begin_txn (env, &txn));
  TEST_ASSERT_EQUAL_INT (SQLITE_OK, backend->get (txn, &id, &stream_writer));
  TEST_ASSERT_EQUAL_UINT64 ((uint64_t)sizeof (bytes), (uint64_t)writer.length);
  TEST_ASSERT_EQUAL_MEMORY (bytes, writer.data, sizeof (bytes));
  free (writer.data);
  backend->rollback_txn (txn);

  backend->close_env (env);
  sqlite3_close (db);
  sqlite3_free (txn_dir);
  sqlite3_free (commit_root);
  sqlite3_free (objects_root);
  sqlite3_free (staging_root);
  cleanup_temp_dirs (base);
}

static void
file_backend_recover_delete_manifest (void)
{
  char *base = create_temp_dir ();
  TEST_ASSERT_NOT_NULL (base);
  char *objects_root = NULL;
  char *staging_root = NULL;
  compose_paths (base, &objects_root, &staging_root);
  TEST_ASSERT_NOT_NULL (objects_root);
  TEST_ASSERT_NOT_NULL (staging_root);
  ensure_dir (objects_root);
  ensure_staging_layout (staging_root);
  char *commit_root = sqlite3_mprintf ("%s/commit", staging_root);
  ensure_dir (commit_root);
  sqlite3_free (commit_root);

  objstore_id id = make_id (0x65);
  char hex[OBJSTORE_ID_SIZE * 2 + 1];
  id_to_hex (&id, hex);
  char shard_dir[3] = { hex[0], hex[1], '\0' };
  char *object_dir = sqlite3_mprintf ("%s/%s", objects_root, shard_dir);
  ensure_dir (object_dir);
  char *object_path
      = sqlite3_mprintf ("%s/%s%s", object_dir, hex, kFileSuffix);
  sqlite3_free (object_dir);
  FILE *object_file = fopen (object_path, "wb");
  TEST_ASSERT_NOT_NULL (object_file);
  const uint8_t payload[] = { 0x01, 0x02, 0x03 };
  fwrite (payload, 1, sizeof (payload), object_file);
  fclose (object_file);

  char *commit_dir = sqlite3_mprintf ("%s/commit/recover_del", staging_root);
  ensure_dir (commit_dir);
  char *manifest_path = sqlite3_mprintf ("%s/%s", commit_dir, kManifestFile);
  FILE *manifest = fopen (manifest_path, "wb");
  TEST_ASSERT_NOT_NULL (manifest);
  fprintf (manifest, "DEL %s\n", hex);
  fclose (manifest);
  sqlite3_free (manifest_path);
  sqlite3_free (commit_dir);

  sqlite3 *db = NULL;
  TEST_ASSERT_EQUAL_INT (SQLITE_OK,
                         backend_file_open_memory_db (&db, "recover-del"));
  const objstore_backend *backend
      = objstore_backend_by_kind (OBJSTORE_BACKEND_FILE);
  TEST_ASSERT_NOT_NULL (backend);
  objstore_config cfg = {
    .backend = OBJSTORE_BACKEND_FILE,
    .storage_root = objects_root,
    .chunk_size_bytes = 0,
    .reserved_flags = 0,
  };
  objstore_backend_env *env = NULL;
  TEST_ASSERT_EQUAL_INT (SQLITE_OK, backend->open_env (db, &cfg, &env));
  backend->close_env (env);
  sqlite3_close (db);

  TEST_ASSERT_EQUAL_INT (-1, access (object_path, F_OK));
  sqlite3_free (object_path);
  sqlite3_free (objects_root);
  sqlite3_free (staging_root);
  cleanup_temp_dirs (base);
}

static void
file_backend_rowidx_lookup (void)
{
  char *base = NULL;
  char *objects_root = NULL;
  char *staging_root = NULL;
  sqlite3 *db = NULL;
  const objstore_backend *backend = NULL;
  objstore_backend_env *env = NULL;
  init_file_backend (&base, &objects_root, &staging_root, &db, &backend, &env);

  objstore_id id = make_id (0x41);
  uint8_t payload[64];
  for (size_t i = 0; i < sizeof (payload); ++i)
    {
      payload[i] = (uint8_t)(i & 0xFFu);
    }
  buffer_reader reader
      = { .data = payload, .size = sizeof (payload), .offset = 0 };
  objstore_stream_reader stream_reader = {
    .ctx = &reader,
    .pull = buffer_reader_pull,
    .size_hint = (sqlite3_int64)reader.size,
  };

  objstore_backend_txn *txn = NULL;
  TEST_ASSERT_EQUAL_INT (SQLITE_OK, backend->begin_txn (env, &txn));
  TEST_ASSERT_EQUAL_INT (SQLITE_OK, backend->put (txn, &id, &stream_reader));
  TEST_ASSERT_EQUAL_INT (SQLITE_OK, backend->commit_txn (txn));

  TEST_ASSERT_NOT_NULL (backend->lookup_id_by_rowid);
  sqlite3_int64 rowid = objstore_rowid_from_id (&id);
  objstore_backend_txn *read_txn = NULL;
  objstore_id resolved = { 0 };
  TEST_ASSERT_EQUAL_INT (SQLITE_OK, backend->begin_txn (env, &read_txn));
  TEST_ASSERT_EQUAL_INT (
      SQLITE_OK,
      backend->lookup_id_by_rowid (read_txn, rowid, &resolved));
  backend->rollback_txn (read_txn);
  TEST_ASSERT_EQUAL_MEMORY (&id, &resolved, sizeof (objstore_id));

  TEST_ASSERT_EQUAL_INT (SQLITE_OK, backend->begin_txn (env, &txn));
  TEST_ASSERT_EQUAL_INT (SQLITE_OK, backend->delete_fn (txn, &id));
  TEST_ASSERT_EQUAL_INT (SQLITE_OK, backend->commit_txn (txn));

  TEST_ASSERT_EQUAL_INT (SQLITE_OK, backend->begin_txn (env, &read_txn));
  TEST_ASSERT_EQUAL_INT (
      SQLITE_NOTFOUND,
      backend->lookup_id_by_rowid (read_txn, rowid, &resolved));
  backend->rollback_txn (read_txn);

  shutdown_file_backend (base, objects_root, staging_root, db, backend, env);
}

static void
file_backend_scan_uses_rowidx_and_skips_stale_entries (void)
{
  char *base = NULL;
  char *objects_root = NULL;
  char *staging_root = NULL;
  sqlite3 *db = NULL;
  const objstore_backend *backend = NULL;
  objstore_backend_env *env = NULL;
  init_file_backend (&base, &objects_root, &staging_root, &db, &backend, &env);

  objstore_id ids[3] = {
    make_id (0x31),
    make_id (0x11),
    make_id (0x21),
  };

  for (size_t i = 0; i < 3; ++i)
    {
      uint8_t payload[8] = { (uint8_t)i };
      buffer_reader reader
          = { .data = payload, .size = sizeof (payload), .offset = 0 };
      objstore_stream_reader stream_reader = {
        .ctx = &reader,
        .pull = buffer_reader_pull,
        .size_hint = (sqlite3_int64)reader.size,
      };
      objstore_backend_txn *txn = NULL;
      TEST_ASSERT_EQUAL_INT (SQLITE_OK, backend->begin_txn (env, &txn));
      TEST_ASSERT_EQUAL_INT (SQLITE_OK,
                             backend->put (txn, &ids[i], &stream_reader));
      TEST_ASSERT_EQUAL_INT (SQLITE_OK, backend->commit_txn (txn));
    }

  objstore_id stale_id = make_id (0x91);
  char *stale_entry_path = NULL;
  rowidx_entry_path (objects_root, &stale_id, &stale_entry_path);
  FILE *stale_entry = fopen (stale_entry_path, "wb");
  TEST_ASSERT_NOT_NULL (stale_entry);
  fclose (stale_entry);

  objstore_backend_txn *read_txn = NULL;
  objstore_backend_cursor *cursor = NULL;
  TEST_ASSERT_EQUAL_INT (SQLITE_OK, backend->begin_txn (env, &read_txn));
  TEST_ASSERT_EQUAL_INT (SQLITE_OK, backend->scan_open (read_txn, &cursor));

  objstore_id scanned[3];
  size_t scanned_count = 0;
  for (;;)
    {
      objstore_id id = { 0 };
      int rc = backend->scan_next (cursor, &id);
      if (rc == SQLITE_DONE)
        {
          break;
        }
      TEST_ASSERT_EQUAL_INT (SQLITE_OK, rc);
      TEST_ASSERT_TRUE (scanned_count < 3);
      scanned[scanned_count++] = id;
    }
  backend->scan_close (cursor);
  backend->rollback_txn (read_txn);

  TEST_ASSERT_EQUAL_size_t (3, scanned_count);
  TEST_ASSERT_EQUAL_MEMORY (&ids[1], &scanned[0], sizeof (objstore_id));
  TEST_ASSERT_EQUAL_MEMORY (&ids[2], &scanned[1], sizeof (objstore_id));
  TEST_ASSERT_EQUAL_MEMORY (&ids[0], &scanned[2], sizeof (objstore_id));
  TEST_ASSERT_EQUAL_INT (-1, access (stale_entry_path, F_OK));

  sqlite3_free (stale_entry_path);
  shutdown_file_backend (base, objects_root, staging_root, db, backend, env);
}

static void
file_backend_recover_handles_missing_payload (void)
{
  char *base = create_temp_dir ();
  TEST_ASSERT_NOT_NULL (base);
  char *objects_root = NULL;
  char *staging_root = NULL;
  compose_paths (base, &objects_root, &staging_root);
  TEST_ASSERT_NOT_NULL (objects_root);
  TEST_ASSERT_NOT_NULL (staging_root);
  ensure_dir (objects_root);
  ensure_staging_layout (staging_root);

  objstore_id id = make_id (0x6A);
  char hex[OBJSTORE_ID_SIZE * 2 + 1];
  id_to_hex (&id, hex);
  char shard_dir[3] = { hex[0], hex[1], '\0' };
  char *object_dir = sqlite3_mprintf ("%s/%s", objects_root, shard_dir);
  ensure_dir (object_dir);
  char *object_path
      = sqlite3_mprintf ("%s/%s%s", object_dir, hex, kFileSuffix);
  sqlite3_free (object_dir);
  FILE *object_file = fopen (object_path, "wb");
  TEST_ASSERT_NOT_NULL (object_file);
  const uint8_t payload[] = { 0x10, 0x20, 0x30, 0x40 };
  fwrite (payload, 1, sizeof (payload), object_file);
  fclose (object_file);

  char *commit_root = sqlite3_mprintf ("%s/commit", staging_root);
  ensure_dir (commit_root);
  char *txn_dir = sqlite3_mprintf ("%s/partial", commit_root);
  ensure_dir (txn_dir);
  char *manifest_path = sqlite3_mprintf ("%s/%s", txn_dir, kManifestFile);
  FILE *manifest = fopen (manifest_path, "wb");
  TEST_ASSERT_NOT_NULL (manifest);
  fprintf (manifest, "PUT %s\n", hex);
  fclose (manifest);
  sqlite3_free (manifest_path);
  sqlite3_free (commit_root);

  sqlite3 *db = NULL;
  TEST_ASSERT_EQUAL_INT (SQLITE_OK,
                         backend_file_open_memory_db (&db, "recover-miss"));
  const objstore_backend *backend
      = objstore_backend_by_kind (OBJSTORE_BACKEND_FILE);
  TEST_ASSERT_NOT_NULL (backend);
  objstore_config cfg = {
    .backend = OBJSTORE_BACKEND_FILE,
    .storage_root = objects_root,
    .chunk_size_bytes = 0,
    .reserved_flags = 0,
  };
  objstore_backend_env *env = NULL;
  TEST_ASSERT_EQUAL_INT (SQLITE_OK, backend->open_env (db, &cfg, &env));

  buffer_writer writer = { .data = NULL, .length = 0, .capacity = 0 };
  objstore_stream_writer stream_writer
      = { .ctx = &writer, .push = buffer_writer_push };
  objstore_backend_txn *txn = NULL;
  TEST_ASSERT_EQUAL_INT (SQLITE_OK, backend->begin_txn (env, &txn));
  TEST_ASSERT_EQUAL_INT (SQLITE_OK, backend->get (txn, &id, &stream_writer));
  backend->rollback_txn (txn);
  TEST_ASSERT_EQUAL_UINT64 ((uint64_t)sizeof (payload),
                            (uint64_t)writer.length);
  TEST_ASSERT_EQUAL_MEMORY (payload, writer.data, sizeof (payload));
  free (writer.data);

  backend->close_env (env);
  sqlite3_close (db);

  TEST_ASSERT_EQUAL_INT (-1, access (txn_dir, F_OK));
  TEST_ASSERT_EQUAL_INT (ENOENT, errno);

  sqlite3_free (txn_dir);
  sqlite3_free (object_path);
  sqlite3_free (objects_root);
  sqlite3_free (staging_root);
  cleanup_temp_dirs (base);
}

static void
file_backend_recover_detects_manifest_corruption (void)
{
  char *base = create_temp_dir ();
  TEST_ASSERT_NOT_NULL (base);
  char *objects_root = NULL;
  char *staging_root = NULL;
  compose_paths (base, &objects_root, &staging_root);
  TEST_ASSERT_NOT_NULL (objects_root);
  TEST_ASSERT_NOT_NULL (staging_root);
  ensure_dir (objects_root);
  ensure_staging_layout (staging_root);
  char *commit_root2 = sqlite3_mprintf ("%s/commit", staging_root);
  ensure_dir (commit_root2);
  sqlite3_free (commit_root2);

  char *commit_dir = sqlite3_mprintf ("%s/commit/bad", staging_root);
  ensure_dir (commit_dir);
  char *manifest_path = sqlite3_mprintf ("%s/%s", commit_dir, kManifestFile);
  FILE *manifest = fopen (manifest_path, "wb");
  TEST_ASSERT_NOT_NULL (manifest);
  fprintf (manifest, "???\n");
  fclose (manifest);
  sqlite3_free (manifest_path);
  sqlite3_free (commit_dir);

  sqlite3 *db = NULL;
  TEST_ASSERT_EQUAL_INT (SQLITE_OK,
                         backend_file_open_memory_db (&db, "recover-bad"));
  const objstore_backend *backend
      = objstore_backend_by_kind (OBJSTORE_BACKEND_FILE);
  TEST_ASSERT_NOT_NULL (backend);
  objstore_config cfg = {
    .backend = OBJSTORE_BACKEND_FILE,
    .storage_root = objects_root,
    .chunk_size_bytes = 0,
    .reserved_flags = 0,
  };
  objstore_backend_env *env = NULL;
  TEST_ASSERT_EQUAL_INT (SQLITE_CORRUPT, backend->open_env (db, &cfg, &env));
  sqlite3_close (db);
  sqlite3_free (objects_root);
  sqlite3_free (staging_root);
  cleanup_temp_dirs (base);
}

static void
file_backend_exists_tracks_presence (void)
{
  char *base = NULL;
  char *objects_root = NULL;
  char *staging_root = NULL;
  sqlite3 *db = NULL;
  const objstore_backend *backend = NULL;
  objstore_backend_env *env = NULL;
  init_file_backend (&base, &objects_root, &staging_root, &db, &backend, &env);

  objstore_id id = make_id (0x63);
  objstore_backend_txn *txn = NULL;
  TEST_ASSERT_EQUAL_INT (SQLITE_OK, backend->begin_txn (env, &txn));
  TEST_ASSERT_EQUAL_INT (SQLITE_NOTFOUND, backend->exists (txn, &id));
  backend->rollback_txn (txn);

  uint8_t payload[] = { 5, 4, 3, 2 };
  buffer_reader reader
      = { .data = payload, .size = sizeof (payload), .offset = 0 };
  objstore_stream_reader stream_reader
      = { .ctx = &reader,
          .pull = buffer_reader_pull,
          .size_hint = (sqlite3_int64)reader.size };
  TEST_ASSERT_EQUAL_INT (SQLITE_OK, backend->begin_txn (env, &txn));
  TEST_ASSERT_EQUAL_INT (SQLITE_OK, backend->put (txn, &id, &stream_reader));
  TEST_ASSERT_EQUAL_INT (SQLITE_OK, backend->commit_txn (txn));

  TEST_ASSERT_EQUAL_INT (SQLITE_OK, backend->begin_txn (env, &txn));
  TEST_ASSERT_EQUAL_INT (SQLITE_OK, backend->exists (txn, &id));
  backend->rollback_txn (txn);

  TEST_ASSERT_EQUAL_INT (SQLITE_OK, backend->begin_txn (env, &txn));
  TEST_ASSERT_EQUAL_INT (SQLITE_OK, backend->delete_fn (txn, &id));
  TEST_ASSERT_EQUAL_INT (SQLITE_OK, backend->commit_txn (txn));

  TEST_ASSERT_EQUAL_INT (SQLITE_OK, backend->begin_txn (env, &txn));
  TEST_ASSERT_EQUAL_INT (SQLITE_NOTFOUND, backend->exists (txn, &id));
  backend->rollback_txn (txn);

  shutdown_file_backend (base, objects_root, staging_root, db, backend, env);
}

static void
file_backend_size_hint_truncates (void)
{
  char *base = NULL;
  char *objects_root = NULL;
  char *staging_root = NULL;
  sqlite3 *db = NULL;
  const objstore_backend *backend = NULL;
  objstore_backend_env *env = NULL;
  init_file_backend (&base, &objects_root, &staging_root, &db, &backend, &env);

  objstore_backend_txn *txn = NULL;
  TEST_ASSERT_EQUAL_INT (SQLITE_OK, backend->begin_txn (env, &txn));

  objstore_backend_staged_writer *writer = NULL;
  TEST_ASSERT_EQUAL_INT (SQLITE_OK,
                         backend->staged_write_begin (txn, &writer));
  TEST_ASSERT_NOT_NULL (backend->staged_write_set_size_hint);
  TEST_ASSERT_EQUAL_INT (SQLITE_OK,
                         backend->staged_write_set_size_hint (writer, 4096));

  uint8_t payload[] = { 0xAA, 0xBB, 0xCC };
  TEST_ASSERT_EQUAL_INT (SQLITE_OK, backend->staged_write_push (
                                        writer, payload, sizeof (payload)));
  objstore_id id = make_id (0x91);
  TEST_ASSERT_EQUAL_INT (SQLITE_OK,
                         backend->staged_write_finalize (writer, &id));
  TEST_ASSERT_EQUAL_INT (SQLITE_OK, backend->commit_txn (txn));

  char hex[OBJSTORE_ID_SIZE * 2 + 1];
  id_to_hex (&id, hex);
  char shard[3] = { hex[0], hex[1], '\0' };
  char *shard_dir = sqlite3_mprintf ("%s/%s", objects_root, shard);
  TEST_ASSERT_NOT_NULL (shard_dir);
  char *object_path = sqlite3_mprintf ("%s/%s%s", shard_dir, hex, kFileSuffix);
  sqlite3_free (shard_dir);
  TEST_ASSERT_NOT_NULL (object_path);

  struct stat st;
  TEST_ASSERT_EQUAL_INT (0, stat (object_path, &st));
  sqlite3_free (object_path);
  TEST_ASSERT_EQUAL_INT ((off_t)sizeof (payload), st.st_size);

  shutdown_file_backend (base, objects_root, staging_root, db, backend, env);
}

void
backend_file_register_tests (void)
{
  objstore_tests_set_fixture (OBJSTORE_FIXTURE_SMOKE);
  RUN_TEST (file_backend_roundtrip);
  RUN_TEST (file_backend_range_reads);
  RUN_TEST (file_backend_range_unsatisfied);
  RUN_TEST (file_backend_delete_removes_object);
  RUN_TEST (file_backend_recovers_pending_commit);
  RUN_TEST (file_backend_recover_handles_missing_payload);
  RUN_TEST (file_backend_exists_tracks_presence);
  RUN_TEST (file_backend_large_object_streams);
  RUN_TEST (file_backend_begin_txn_respects_permissions);
  RUN_TEST (file_backend_put_handles_disk_full);
  RUN_TEST (file_backend_recover_delete_manifest);
  RUN_TEST (file_backend_recover_detects_manifest_corruption);
  RUN_TEST (file_backend_size_hint_truncates);
  RUN_TEST (file_backend_rowidx_lookup);
  RUN_TEST (file_backend_scan_uses_rowidx_and_skips_stale_entries);
}
