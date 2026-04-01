#include "unity.h"

#include <stdint.h>
#include <string.h>

#include "objstore/object_manager.h"
#include "objstore/vtab.h"
#include "test_support.h"

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
  if (reader == NULL || buffer == NULL || nread == NULL)
    {
      return SQLITE_MISUSE;
    }
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
sqlite_exec_or_fail (const char *sql)
{
  TEST_ASSERT_EQUAL_INT (SQLITE_OK, sqlite3_exec (objstore_tests_sqlite_db (),
                                                  sql, NULL, NULL, NULL));
}

static void
seed_backend_object (const objstore_id *id, const uint8_t *data, size_t size)
{
  buffer_reader reader = { .data = data, .size = size, .offset = 0 };
  objstore_stream_reader stream_reader = {
    .ctx = &reader,
    .pull = buffer_reader_pull,
    .size_hint = (sqlite3_int64)reader.size,
  };
  objstore_backend_txn *txn = NULL;
  sqlite_exec_or_fail ("BEGIN IMMEDIATE;");
  TEST_ASSERT_EQUAL_INT (SQLITE_OK,
                         objstore_tests_sqlite_backend ()->begin_txn (
                             objstore_tests_backend_env (), &txn));
  TEST_ASSERT_EQUAL_INT (SQLITE_OK, objstore_tests_sqlite_backend ()->put (
                                        txn, id, &stream_reader));
  TEST_ASSERT_EQUAL_INT (SQLITE_OK,
                         objstore_tests_sqlite_backend ()->commit_txn (txn));
  sqlite_exec_or_fail ("COMMIT;");
}

static objstore_connection *
object_manager_open_connection (void)
{
  sqlite3 *db = objstore_tests_sqlite_db ();
  TEST_ASSERT_NOT_NULL (db);
  objstore_config cfg = {
    .backend = OBJSTORE_BACKEND_SQLITE,
    .storage_root = NULL,
    .chunk_size_bytes = 0,
    .reserved_flags = 0,
  };
  size_t chunk_size = objstore_effective_chunk_size (&cfg);
  objstore_connection *conn = NULL;
  TEST_ASSERT_EQUAL_INT (SQLITE_OK, objstore_connection_create (
                                        db, objstore_tests_sqlite_backend (),
                                        &cfg, chunk_size, &conn));
  return conn;
}

static void
object_manager_close_connection (objstore_connection *conn)
{
  if (conn != NULL)
    {
      objstore_connection_destroy (conn);
    }
}

static objstore_id
compute_stream_hash (const uint8_t *data, size_t size)
{
  buffer_reader reader = { .data = data, .size = size, .offset = 0 };
  objstore_stream_reader stream_reader = {
    .ctx = &reader,
    .pull = buffer_reader_pull,
    .size_hint = (sqlite3_int64)reader.size,
  };
  objstore_connection *conn = object_manager_open_connection ();
  objstore_id id;
  TEST_ASSERT_EQUAL_INT (
      SQLITE_OK, objstore_object_put_reader (conn, &stream_reader, &id));
  object_manager_close_connection (conn);
  return id;
}

static void
test_object_manager_stream_put_same_content_stable_id (void)
{
  const uint8_t payload[] = { 0xAA, 0xBB, 0xCC, 0xDD };

  objstore_id first_id;
  {
    objstore_connection *conn = object_manager_open_connection ();
    buffer_reader reader
        = { .data = payload, .size = sizeof (payload), .offset = 0 };
    objstore_stream_reader stream_reader = {
      .ctx = &reader,
      .pull = buffer_reader_pull,
      .size_hint = (sqlite3_int64)reader.size,
    };
    TEST_ASSERT_EQUAL_INT (SQLITE_OK, objstore_object_put_reader (
                                          conn, &stream_reader, &first_id));
    object_manager_close_connection (conn);
  }

  objstore_id second_id;
  {
    objstore_connection *conn = object_manager_open_connection ();
    buffer_reader reader
        = { .data = payload, .size = sizeof (payload), .offset = 0 };
    objstore_stream_reader stream_reader = {
      .ctx = &reader,
      .pull = buffer_reader_pull,
      .size_hint = (sqlite3_int64)reader.size,
    };
    TEST_ASSERT_EQUAL_INT (SQLITE_OK, objstore_object_put_reader (
                                          conn, &stream_reader, &second_id));
    object_manager_close_connection (conn);
  }

  TEST_ASSERT_EQUAL_MEMORY (first_id.bytes, second_id.bytes, OBJSTORE_ID_SIZE);
}

static void
test_object_manager_stream_put_different_content_changes_id (void)
{
  const uint8_t payload_a[] = { 0x00, 0x01, 0x02, 0x03 };
  const uint8_t payload_b[] = { 0x00, 0x01, 0x02, 0x04 };

  objstore_connection *conn = object_manager_open_connection ();
  buffer_reader reader_a
      = { .data = payload_a, .size = sizeof (payload_a), .offset = 0 };
  buffer_reader reader_b
      = { .data = payload_b, .size = sizeof (payload_b), .offset = 0 };
  objstore_stream_reader stream_reader_a
      = { .ctx = &reader_a,
          .pull = buffer_reader_pull,
          .size_hint = (sqlite3_int64)reader_a.size };
  objstore_stream_reader stream_reader_b
      = { .ctx = &reader_b,
          .pull = buffer_reader_pull,
          .size_hint = (sqlite3_int64)reader_b.size };

  objstore_id id_a;
  objstore_id id_b;
  TEST_ASSERT_EQUAL_INT (
      SQLITE_OK, objstore_object_put_reader (conn, &stream_reader_a, &id_a));
  TEST_ASSERT_EQUAL_INT (
      SQLITE_OK, objstore_object_put_reader (conn, &stream_reader_b, &id_b));

  object_manager_close_connection (conn);
  TEST_ASSERT_NOT_EQUAL (0, memcmp (id_a.bytes, id_b.bytes, OBJSTORE_ID_SIZE));
}

static void
test_object_manager_stream_put_zero_length (void)
{
  objstore_connection *conn = object_manager_open_connection ();
  buffer_reader reader = { .data = NULL, .size = 0, .offset = 0 };
  objstore_stream_reader stream_reader = {
    .ctx = &reader,
    .pull = buffer_reader_pull,
    .size_hint = 0,
  };
  objstore_id id;
  TEST_ASSERT_EQUAL_INT (
      SQLITE_OK, objstore_object_put_reader (conn, &stream_reader, &id));
  int present = 0;
  TEST_ASSERT_EQUAL_INT (SQLITE_OK,
                         objstore_object_exists (conn, &id, &present));
  TEST_ASSERT_EQUAL_INT (1, present);
  object_manager_close_connection (conn);
}

static void
test_object_manager_put_with_id_respects_existing_backend (void)
{
  objstore_id id = make_id (0x30);
  uint8_t payload_a[16];
  uint8_t payload_b[16];
  memset (payload_a, 0x11, sizeof (payload_a));
  memset (payload_b, 0x22, sizeof (payload_b));
  seed_backend_object (&id, payload_a, sizeof (payload_a));

  objstore_connection *conn = object_manager_open_connection ();

  buffer_reader reader_same
      = { .data = payload_a, .size = sizeof (payload_a), .offset = 0 };
  objstore_stream_reader stream_same
      = { .ctx = &reader_same,
          .pull = buffer_reader_pull,
          .size_hint = (sqlite3_int64)reader_same.size };
  TEST_ASSERT_EQUAL_INT (
      SQLITE_OK, objstore_object_put_reader_with_id (conn, &id, &stream_same));

  buffer_reader reader_diff
      = { .data = payload_b, .size = sizeof (payload_b), .offset = 0 };
  objstore_stream_reader stream_diff
      = { .ctx = &reader_diff,
          .pull = buffer_reader_pull,
          .size_hint = (sqlite3_int64)reader_diff.size };
  TEST_ASSERT_EQUAL_INT (
      SQLITE_CONSTRAINT,
      objstore_object_put_reader_with_id (conn, &id, &stream_diff));

  object_manager_close_connection (conn);
}

static void
test_object_manager_verify_reports_integrity (void)
{
  uint8_t payload[32];
  for (size_t i = 0; i < sizeof (payload); ++i)
    {
      payload[i] = (uint8_t)(i + 1);
    }
  objstore_id id = compute_stream_hash (payload, sizeof (payload));
  seed_backend_object (&id, payload, sizeof (payload));
  objstore_id recomputed = compute_stream_hash (payload, sizeof (payload));
  TEST_ASSERT_EQUAL_MEMORY (id.bytes, recomputed.bytes, OBJSTORE_ID_SIZE);

  objstore_connection *conn = object_manager_open_connection ();
  int valid = 0;
  TEST_ASSERT_EQUAL_INT (SQLITE_OK,
                         objstore_object_verify (conn, &id, &valid));
  TEST_ASSERT_EQUAL_INT (1, valid);

  objstore_id missing = make_id (0x55);
  TEST_ASSERT_EQUAL_INT (SQLITE_NOTFOUND,
                         objstore_object_verify (conn, &missing, &valid));
  object_manager_close_connection (conn);
}

void
object_manager_register_tests (void)
{
  objstore_tests_set_fixture (OBJSTORE_FIXTURE_SQLITE_BACKEND);
  RUN_TEST (test_object_manager_stream_put_same_content_stable_id);
  RUN_TEST (test_object_manager_stream_put_different_content_changes_id);
  RUN_TEST (test_object_manager_stream_put_zero_length);
  RUN_TEST (test_object_manager_put_with_id_respects_existing_backend);
  RUN_TEST (test_object_manager_verify_reports_integrity);
  objstore_tests_set_fixture (OBJSTORE_FIXTURE_SMOKE);
}
