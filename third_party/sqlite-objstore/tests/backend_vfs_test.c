#if !defined(_WIN32)
#define _POSIX_C_SOURCE 200809L
#endif

#include "unity.h"

#include <sqlite3.h>

#include <dirent.h>
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "test_support.h"

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

typedef struct buffer_reader
{
  const uint8_t *data;
  size_t size;
  size_t offset;
} buffer_reader;

static int
buffer_reader_pull (void *ctx, void *buffer, size_t capacity, size_t *nread)
{
  buffer_reader *reader = (buffer_reader *)ctx;
  if (reader->offset >= reader->size)
    {
      *nread = 0;
      return SQLITE_DONE;
    }
  size_t remaining = reader->size - reader->offset;
  size_t to_copy = remaining < capacity ? remaining : capacity;
  memcpy (buffer, reader->data + reader->offset, to_copy);
  reader->offset += to_copy;
  *nread = to_copy;
  return SQLITE_OK;
}

typedef struct buffer_writer
{
  uint8_t *data;
  size_t length;
  size_t capacity;
} buffer_writer;

static int
buffer_writer_push (void *ctx, const void *buffer, size_t nread)
{
  buffer_writer *writer = (buffer_writer *)ctx;
  if (writer->length + nread > writer->capacity)
    {
      size_t new_capacity = writer->capacity == 0 ? 512 : writer->capacity;
      while (new_capacity < writer->length + nread)
        {
          new_capacity *= 2;
        }
      uint8_t *resized = sqlite3_realloc64 (writer->data, new_capacity);
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

static char *
create_temp_dir (const char *label)
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
          = sqlite3_mprintf ("/tmp/objstore-vfs-%s-%s",
                             label != NULL ? label : "", random_hex);
      TEST_ASSERT_NOT_NULL (path);
      if (mkdir (path, 0700) == 0)
        {
          return path;
        }
      sqlite3_free (path);
      TEST_ASSERT_EQUAL_INT (EEXIST, errno);
    }
  TEST_FAIL_MESSAGE ("unable to create temp directory");
  return NULL;
}

static void
remove_tree (const char *path)
{
  if (path == NULL)
    {
      return;
    }
  DIR *dir = opendir (path);
  if (dir == NULL)
    {
      rmdir (path);
      return;
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
          continue;
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
  rmdir (path);
}

static void
shutdown_vfs_backend (char *root, sqlite3 *db, const objstore_backend *backend,
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
  remove_tree (root);
  sqlite3_free (root);
}

static void
init_vfs_backend (const char *label, sqlite3 **out_db,
                  const objstore_backend **out_backend,
                  objstore_backend_env **out_env, char **out_root)
{
  char *root = create_temp_dir (label);
  sqlite3 *db = NULL;
  TEST_ASSERT_EQUAL_INT (
      SQLITE_OK,
      sqlite3_open_v2 (
          "file:vfs_backend?mode=memory&cache=shared", &db,
          SQLITE_OPEN_CREATE | SQLITE_OPEN_READWRITE | SQLITE_OPEN_URI, NULL));
  const objstore_backend *backend
      = objstore_backend_by_kind (OBJSTORE_BACKEND_VFS);
  TEST_ASSERT_NOT_NULL (backend);
  objstore_config cfg = {
    .backend = OBJSTORE_BACKEND_VFS,
    .storage_root = root,
    .chunk_size_bytes = 4096,
  };
  objstore_backend_env *env = NULL;
  TEST_ASSERT_EQUAL_INT (SQLITE_OK, backend->open_env (db, &cfg, &env));
  *out_db = db;
  *out_backend = backend;
  *out_env = env;
  *out_root = root;
}

static void
vfs_backend_roundtrip (void)
{
  sqlite3 *db = NULL;
  const objstore_backend *backend = NULL;
  objstore_backend_env *env = NULL;
  char *root = NULL;
  init_vfs_backend ("roundtrip", &db, &backend, &env, &root);

  objstore_id id = make_id (0x10);
  uint8_t payload[] = { 1, 2, 3, 4, 5 };
  buffer_reader reader = {
    .data = payload,
    .size = sizeof (payload),
    .offset = 0,
  };
  objstore_stream_reader stream_reader = {
    .ctx = &reader,
    .pull = buffer_reader_pull,
    .size_hint = (sqlite3_int64)sizeof (payload),
  };

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
  TEST_ASSERT_EQUAL_size_t (sizeof (payload), writer.length);
  TEST_ASSERT_EQUAL_MEMORY (payload, writer.data, sizeof (payload));
  sqlite3_free (writer.data);

  shutdown_vfs_backend (root, db, backend, env);
}

static void
vfs_backend_delete_and_exists (void)
{
  sqlite3 *db = NULL;
  const objstore_backend *backend = NULL;
  objstore_backend_env *env = NULL;
  char *root = NULL;
  init_vfs_backend ("delete", &db, &backend, &env, &root);

  uint8_t payload[] = { 11, 12, 13 };
  buffer_reader reader = {
    .data = payload,
    .size = sizeof (payload),
    .offset = 0,
  };
  objstore_stream_reader stream_reader = {
    .ctx = &reader,
    .pull = buffer_reader_pull,
    .size_hint = (sqlite3_int64)sizeof (payload),
  };
  objstore_id id = make_id (0x22);

  objstore_backend_txn *txn = NULL;
  TEST_ASSERT_EQUAL_INT (SQLITE_OK, backend->begin_txn (env, &txn));
  TEST_ASSERT_EQUAL_INT (SQLITE_OK, backend->put (txn, &id, &stream_reader));
  TEST_ASSERT_EQUAL_INT (SQLITE_OK, backend->commit_txn (txn));

  objstore_backend_txn *exists_txn = NULL;
  TEST_ASSERT_EQUAL_INT (SQLITE_OK, backend->begin_txn (env, &exists_txn));
  TEST_ASSERT_EQUAL_INT (SQLITE_OK, backend->delete_fn (exists_txn, &id));
  TEST_ASSERT_EQUAL_INT (SQLITE_OK, backend->commit_txn (exists_txn));

  objstore_backend_txn *check_txn = NULL;
  TEST_ASSERT_EQUAL_INT (SQLITE_OK, backend->begin_txn (env, &check_txn));
  TEST_ASSERT_EQUAL_INT (SQLITE_NOTFOUND, backend->exists (check_txn, &id));
  backend->rollback_txn (check_txn);

  shutdown_vfs_backend (root, db, backend, env);
}

static void
vfs_backend_large_object (void)
{
  sqlite3 *db = NULL;
  const objstore_backend *backend = NULL;
  objstore_backend_env *env = NULL;
  char *root = NULL;
  init_vfs_backend ("large", &db, &backend, &env, &root);

  const size_t payload_size = 8 * 1024 * 1024;
  uint8_t *payload = sqlite3_malloc64 (payload_size);
  TEST_ASSERT_NOT_NULL (payload);
  for (size_t i = 0; i < payload_size; ++i)
    {
      payload[i] = (uint8_t)(i & 0xFFu);
    }

  buffer_reader reader = {
    .data = payload,
    .size = payload_size,
    .offset = 0,
  };
  objstore_stream_reader stream_reader = {
    .ctx = &reader,
    .pull = buffer_reader_pull,
    .size_hint = (sqlite3_int64)payload_size,
  };
  objstore_id id = make_id (0x44);

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

  TEST_ASSERT_EQUAL_size_t (payload_size, writer.length);
  TEST_ASSERT_EQUAL_MEMORY (payload, writer.data, payload_size);
  sqlite3_free (writer.data);
  sqlite3_free (payload);

  shutdown_vfs_backend (root, db, backend, env);
}

static void
vfs_backend_range_reads (void)
{
  sqlite3 *db = NULL;
  const objstore_backend *backend = NULL;
  objstore_backend_env *env = NULL;
  char *root = NULL;
  init_vfs_backend ("range", &db, &backend, &env, &root);

  objstore_id id = make_id (0x12);
  uint8_t payload[] = { 5, 6, 7, 8, 9, 10 };
  buffer_reader reader = {
    .data = payload,
    .size = sizeof (payload),
    .offset = 0,
  };
  objstore_stream_reader stream_reader = {
    .ctx = &reader,
    .pull = buffer_reader_pull,
    .size_hint = (sqlite3_int64)sizeof (payload),
  };

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
                         backend->get_range (read_txn, &id, 1, 3,
                                             &stream_writer));
  TEST_ASSERT_EQUAL_size_t (3, writer.length);
  TEST_ASSERT_EQUAL_MEMORY (payload + 1, writer.data, 3);
  sqlite3_free (writer.data);
  TEST_ASSERT_EQUAL_INT (SQLITE_OK, backend->commit_txn (read_txn));

  shutdown_vfs_backend (root, db, backend, env);
}

static void
vfs_backend_range_unsatisfied (void)
{
  sqlite3 *db = NULL;
  const objstore_backend *backend = NULL;
  objstore_backend_env *env = NULL;
  char *root = NULL;
  init_vfs_backend ("range-miss", &db, &backend, &env, &root);

  objstore_id id = make_id (0x13);
  uint8_t payload[] = { 1, 2, 3 };
  buffer_reader reader = {
    .data = payload,
    .size = sizeof (payload),
    .offset = 0,
  };
  objstore_stream_reader stream_reader = {
    .ctx = &reader,
    .pull = buffer_reader_pull,
    .size_hint = (sqlite3_int64)sizeof (payload),
  };

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
                         backend->get_range (read_txn, &id, 9, 1,
                                             &stream_writer));
  sqlite3_free (writer.data);
  backend->rollback_txn (read_txn);

  shutdown_vfs_backend (root, db, backend, env);
}

void
backend_vfs_register_tests (void)
{
  objstore_tests_set_fixture (OBJSTORE_FIXTURE_SMOKE);
  RUN_TEST (vfs_backend_roundtrip);
  RUN_TEST (vfs_backend_range_reads);
  RUN_TEST (vfs_backend_range_unsatisfied);
  RUN_TEST (vfs_backend_delete_and_exists);
  RUN_TEST (vfs_backend_large_object);
}
