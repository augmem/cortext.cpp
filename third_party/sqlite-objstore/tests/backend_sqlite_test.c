#include "unity.h"

#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "test_support.h"

#define OBJSTORE_SQLITE_QUEUE_LIMIT_TEST_BYTES (16 * 1024 * 1024)

typedef struct buffer_reader
{
  const uint8_t *data;
  size_t size;
  size_t offset;
} buffer_reader;

typedef struct buffer_writer
{
  uint8_t *data;
  size_t capacity;
  size_t length;
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

static const objstore_backend *
current_backend (void)
{
  const objstore_backend *backend = objstore_tests_sqlite_backend ();
  TEST_ASSERT_NOT_NULL (backend);
  return backend;
}

static objstore_backend_env *
current_env (void)
{
  objstore_backend_env *env = objstore_tests_backend_env ();
  TEST_ASSERT_NOT_NULL (env);
  return env;
}

static bool
sqlite_row_exists (const objstore_id *id)
{
  sqlite3_stmt *stmt = NULL;
  TEST_ASSERT_EQUAL_INT (
      SQLITE_OK,
      sqlite3_prepare_v3 (objstore_tests_sqlite_db (),
                          "SELECT 1 FROM objstore_data WHERE id = ?1 LIMIT 1;",
                          -1, SQLITE_PREPARE_PERSISTENT, &stmt, NULL));
  TEST_ASSERT_EQUAL_INT (
      SQLITE_OK,
      sqlite3_bind_blob (stmt, 1, id->bytes, OBJSTORE_ID_SIZE, SQLITE_STATIC));
  int rc = sqlite3_step (stmt);
  sqlite3_finalize (stmt);
  return rc == SQLITE_ROW;
}

static void
sqlite_exec_or_fail (const char *sql)
{
  TEST_ASSERT_EQUAL_INT (SQLITE_OK, sqlite3_exec (objstore_tests_sqlite_db (),
                                                  sql, NULL, NULL, NULL));
}

static objstore_backend_txn *
begin_backend_txn (void)
{
  sqlite_exec_or_fail ("BEGIN IMMEDIATE");
  objstore_backend_txn *txn = NULL;
  TEST_ASSERT_EQUAL_INT (SQLITE_OK,
                         current_backend ()->begin_txn (current_env (), &txn));
  return txn;
}

static void
commit_backend_txn (objstore_backend_txn *txn)
{
  TEST_ASSERT_EQUAL_INT (SQLITE_OK, current_backend ()->commit_txn (txn));
  sqlite_exec_or_fail ("COMMIT");
}

static void
rollback_backend_txn (objstore_backend_txn *txn)
{
  current_backend ()->rollback_txn (txn);
  sqlite_exec_or_fail ("ROLLBACK");
}

static void
test_sqlite_backend_roundtrip_impl (void)
{
  uint8_t payload[] = { 1, 2, 3, 4, 5, 6, 7, 8 };
  objstore_id id = make_id (0x10);

  buffer_reader reader
      = { .data = payload, .size = sizeof (payload), .offset = 0 };
  objstore_stream_reader stream_reader = {
    .ctx = &reader,
    .pull = buffer_reader_pull,
    .size_hint = (sqlite3_int64)reader.size,
  };

  objstore_backend_txn *write_txn = begin_backend_txn ();
  TEST_ASSERT_EQUAL_INT (
      SQLITE_OK, current_backend ()->put (write_txn, &id, &stream_reader));
  commit_backend_txn (write_txn);

  buffer_writer writer = { .data = NULL, .capacity = 0, .length = 0 };
  objstore_stream_writer stream_writer = {
    .ctx = &writer,
    .push = buffer_writer_push,
  };
  objstore_backend_txn *read_txn = begin_backend_txn ();
  TEST_ASSERT_EQUAL_INT (
      SQLITE_OK, current_backend ()->get (read_txn, &id, &stream_writer));
  commit_backend_txn (read_txn);

  TEST_ASSERT_EQUAL_UINT64 ((uint64_t)sizeof (payload),
                            (uint64_t)writer.length);
  TEST_ASSERT_EQUAL_MEMORY (payload, writer.data, sizeof (payload));
  free (writer.data);
}

static void
test_sqlite_backend_duplicate_same_bytes_impl (void)
{
  uint8_t payload[16];
  memset (payload, 0xAB, sizeof (payload));
  objstore_id id = make_id (0x20);

  buffer_reader reader_a
      = { .data = payload, .size = sizeof (payload), .offset = 0 };
  objstore_stream_reader stream_reader_a
      = { .ctx = &reader_a,
          .pull = buffer_reader_pull,
          .size_hint = (sqlite3_int64)reader_a.size };
  objstore_backend_txn *txn_a = begin_backend_txn ();
  TEST_ASSERT_EQUAL_INT (
      SQLITE_OK, current_backend ()->put (txn_a, &id, &stream_reader_a));
  commit_backend_txn (txn_a);

  buffer_reader reader_b
      = { .data = payload, .size = sizeof (payload), .offset = 0 };
  objstore_stream_reader stream_reader_b
      = { .ctx = &reader_b,
          .pull = buffer_reader_pull,
          .size_hint = (sqlite3_int64)reader_b.size };
  objstore_backend_txn *txn_b = begin_backend_txn ();
  TEST_ASSERT_EQUAL_INT (
      SQLITE_OK, current_backend ()->put (txn_b, &id, &stream_reader_b));
  commit_backend_txn (txn_b);
}

static void
test_sqlite_backend_rowidx_lookup (void)
{
  uint8_t payload[] = { 0x10, 0x20, 0x30, 0x40, 0x50 };
  objstore_id id = make_id (0x55);
  buffer_reader reader
      = { .data = payload, .size = sizeof (payload), .offset = 0 };
  objstore_stream_reader stream_reader = {
    .ctx = &reader,
    .pull = buffer_reader_pull,
    .size_hint = (sqlite3_int64)reader.size,
  };
  objstore_backend_txn *txn = begin_backend_txn ();
  TEST_ASSERT_EQUAL_INT (
      SQLITE_OK, current_backend ()->put (txn, &id, &stream_reader));
  commit_backend_txn (txn);

  TEST_ASSERT_NOT_NULL (current_backend ()->lookup_id_by_rowid);
  sqlite3_int64 rowid = objstore_rowid_from_id (&id);
  objstore_backend_txn *read_txn = begin_backend_txn ();
  objstore_id resolved = { 0 };
  TEST_ASSERT_EQUAL_INT (
      SQLITE_OK,
      current_backend ()->lookup_id_by_rowid (read_txn, rowid, &resolved));
  rollback_backend_txn (read_txn);
  TEST_ASSERT_EQUAL_MEMORY (&id, &resolved, sizeof (objstore_id));

  sqlite3_stmt *stmt = NULL;
  TEST_ASSERT_EQUAL_INT (
      SQLITE_OK,
      sqlite3_prepare_v3 (objstore_tests_sqlite_db (),
                          "DELETE FROM objstore_data WHERE id = ?1;", -1,
                          SQLITE_PREPARE_PERSISTENT, &stmt, NULL));
  TEST_ASSERT_EQUAL_INT (
      SQLITE_OK,
      sqlite3_bind_blob (stmt, 1, id.bytes, OBJSTORE_ID_SIZE, SQLITE_STATIC));
  TEST_ASSERT_EQUAL_INT (SQLITE_DONE, sqlite3_step (stmt));
  sqlite3_finalize (stmt);

  read_txn = begin_backend_txn ();
  TEST_ASSERT_EQUAL_INT (
      SQLITE_NOTFOUND,
      current_backend ()->lookup_id_by_rowid (read_txn, rowid, &resolved));
  rollback_backend_txn (read_txn);

  buffer_reader reader2
      = { .data = payload, .size = sizeof (payload), .offset = 0 };
  objstore_stream_reader stream_reader2 = {
    .ctx = &reader2,
    .pull = buffer_reader_pull,
    .size_hint = (sqlite3_int64)reader2.size,
  };
  txn = begin_backend_txn ();
  TEST_ASSERT_EQUAL_INT (
      SQLITE_OK, current_backend ()->put (txn, &id, &stream_reader2));
  commit_backend_txn (txn);

  objstore_backend_txn *delete_txn = begin_backend_txn ();
  TEST_ASSERT_EQUAL_INT (SQLITE_OK,
                         current_backend ()->delete_fn (delete_txn, &id));
  commit_backend_txn (delete_txn);

  read_txn = begin_backend_txn ();
  TEST_ASSERT_EQUAL_INT (
      SQLITE_NOTFOUND,
      current_backend ()->lookup_id_by_rowid (read_txn, rowid, &resolved));
  rollback_backend_txn (read_txn);
}

static void
test_sqlite_backend_duplicate_conflict_impl (void)
{
  uint8_t payload_a[8];
  uint8_t payload_b[8];
  memset (payload_a, 0x01, sizeof (payload_a));
  memset (payload_b, 0xFF, sizeof (payload_b));
  objstore_id id = make_id (0x30);

  buffer_reader reader_a
      = { .data = payload_a, .size = sizeof (payload_a), .offset = 0 };
  objstore_stream_reader stream_reader_a
      = { .ctx = &reader_a,
          .pull = buffer_reader_pull,
          .size_hint = (sqlite3_int64)reader_a.size };
  objstore_backend_txn *txn_a = begin_backend_txn ();
  TEST_ASSERT_EQUAL_INT (
      SQLITE_OK, current_backend ()->put (txn_a, &id, &stream_reader_a));
  commit_backend_txn (txn_a);

  buffer_reader reader_b
      = { .data = payload_b, .size = sizeof (payload_b), .offset = 0 };
  objstore_stream_reader stream_reader_b
      = { .ctx = &reader_b,
          .pull = buffer_reader_pull,
          .size_hint = (sqlite3_int64)reader_b.size };
  objstore_backend_txn *txn_b = begin_backend_txn ();
  TEST_ASSERT_EQUAL_INT (SQLITE_CONSTRAINT, current_backend ()->put (
                                                txn_b, &id, &stream_reader_b));
  rollback_backend_txn (txn_b);
}

static void
test_sqlite_backend_get_missing_impl (void)
{
  objstore_id id = make_id (0x40);
  buffer_writer writer = { .data = NULL, .capacity = 0, .length = 0 };
  objstore_stream_writer stream_writer
      = { .ctx = &writer, .push = buffer_writer_push };
  objstore_backend_txn *txn = begin_backend_txn ();
  TEST_ASSERT_EQUAL_INT (SQLITE_NOTFOUND,
                         current_backend ()->get (txn, &id, &stream_writer));
  rollback_backend_txn (txn);
  free (writer.data);
}

static void
test_sqlite_backend_rollback_discards_impl (void)
{
  uint8_t payload[32];
  memset (payload, 0x77, sizeof (payload));
  objstore_id id = make_id (0x50);

  buffer_reader reader
      = { .data = payload, .size = sizeof (payload), .offset = 0 };
  objstore_stream_reader stream_reader
      = { .ctx = &reader,
          .pull = buffer_reader_pull,
          .size_hint = (sqlite3_int64)reader.size };
  objstore_backend_txn *write_txn = begin_backend_txn ();
  TEST_ASSERT_EQUAL_INT (
      SQLITE_OK, current_backend ()->put (write_txn, &id, &stream_reader));
  rollback_backend_txn (write_txn);
  TEST_ASSERT_FALSE (sqlite_row_exists (&id));

  buffer_writer writer = { .data = NULL, .capacity = 0, .length = 0 };
  objstore_stream_writer stream_writer
      = { .ctx = &writer, .push = buffer_writer_push };
  objstore_backend_txn *read_txn = begin_backend_txn ();
  TEST_ASSERT_EQUAL_INT (SQLITE_NOTFOUND, current_backend ()->get (
                                              read_txn, &id, &stream_writer));
  rollback_backend_txn (read_txn);
  free (writer.data);
}

static void
test_sqlite_backend_exists_impl (void)
{
  objstore_id id = make_id (0x70);

  objstore_backend_txn *txn = begin_backend_txn ();
  TEST_ASSERT_EQUAL_INT (SQLITE_NOTFOUND,
                         current_backend ()->exists (txn, &id));
  rollback_backend_txn (txn);

  uint8_t payload[4] = { 9, 8, 7, 6 };
  buffer_reader reader
      = { .data = payload, .size = sizeof (payload), .offset = 0 };
  objstore_stream_reader stream_reader
      = { .ctx = &reader,
          .pull = buffer_reader_pull,
          .size_hint = (sqlite3_int64)reader.size };
  objstore_backend_txn *write_txn = begin_backend_txn ();
  TEST_ASSERT_EQUAL_INT (
      SQLITE_OK, current_backend ()->put (write_txn, &id, &stream_reader));
  commit_backend_txn (write_txn);

  txn = begin_backend_txn ();
  TEST_ASSERT_EQUAL_INT (SQLITE_OK, current_backend ()->exists (txn, &id));
  rollback_backend_txn (txn);

  objstore_backend_txn *delete_txn = begin_backend_txn ();
  TEST_ASSERT_EQUAL_INT (SQLITE_OK,
                         current_backend ()->delete_fn (delete_txn, &id));
  commit_backend_txn (delete_txn);

  txn = begin_backend_txn ();
  TEST_ASSERT_EQUAL_INT (SQLITE_NOTFOUND,
                         current_backend ()->exists (txn, &id));
  rollback_backend_txn (txn);
}

static void
test_sqlite_backend_queue_visibility_impl (void)
{
  uint8_t payload[64];
  for (size_t i = 0; i < sizeof (payload); ++i)
    {
      payload[i] = (uint8_t)(0xA0 + i);
    }
  objstore_id id = make_id (0x80);

  buffer_reader reader
      = { .data = payload, .size = sizeof (payload), .offset = 0 };
  objstore_stream_reader stream_reader
      = { .ctx = &reader,
          .pull = buffer_reader_pull,
          .size_hint = (sqlite3_int64)reader.size };

  objstore_backend_txn *txn = begin_backend_txn ();
  TEST_ASSERT_EQUAL_INT (SQLITE_OK,
                         current_backend ()->put (txn, &id, &stream_reader));

  buffer_writer writer = { .data = NULL, .capacity = 0, .length = 0 };
  objstore_stream_writer stream_writer
      = { .ctx = &writer, .push = buffer_writer_push };
  TEST_ASSERT_EQUAL_INT (SQLITE_OK,
                         current_backend ()->get (txn, &id, &stream_writer));
  TEST_ASSERT_EQUAL_UINT64 ((uint64_t)sizeof (payload),
                            (uint64_t)writer.length);
  TEST_ASSERT_EQUAL_MEMORY (payload, writer.data, sizeof (payload));
  free (writer.data);

  rollback_backend_txn (txn);
  TEST_ASSERT_FALSE (sqlite_row_exists (&id));
}

static void
test_sqlite_backend_queue_delete_visibility_impl (void)
{
  uint8_t payload[32];
  memset (payload, 0xBB, sizeof (payload));
  objstore_id id = make_id (0x81);

  buffer_reader writer_reader
      = { .data = payload, .size = sizeof (payload), .offset = 0 };
  objstore_stream_reader writer_stream
      = { .ctx = &writer_reader,
          .pull = buffer_reader_pull,
          .size_hint = (sqlite3_int64)writer_reader.size };
  objstore_backend_txn *seed_txn = begin_backend_txn ();
  TEST_ASSERT_EQUAL_INT (
      SQLITE_OK, current_backend ()->put (seed_txn, &id, &writer_stream));
  commit_backend_txn (seed_txn);

  objstore_backend_txn *txn = begin_backend_txn ();
  TEST_ASSERT_EQUAL_INT (SQLITE_OK, current_backend ()->delete_fn (txn, &id));

  buffer_writer writer = { .data = NULL, .capacity = 0, .length = 0 };
  objstore_stream_writer stream_writer
      = { .ctx = &writer, .push = buffer_writer_push };
  TEST_ASSERT_EQUAL_INT (SQLITE_NOTFOUND,
                         current_backend ()->get (txn, &id, &stream_writer));
  TEST_ASSERT_EQUAL_INT (SQLITE_NOTFOUND,
                         current_backend ()->exists (txn, &id));
  free (writer.data);

  commit_backend_txn (txn);
  TEST_ASSERT_FALSE (sqlite_row_exists (&id));
}

static void
test_sqlite_backend_queue_duplicate_same_txn_impl (void)
{
  uint8_t payload[48];
  memset (payload, 0xCC, sizeof (payload));
  objstore_id id = make_id (0x82);

  buffer_reader reader_a
      = { .data = payload, .size = sizeof (payload), .offset = 0 };
  buffer_reader reader_b
      = { .data = payload, .size = sizeof (payload), .offset = 0 };
  objstore_stream_reader stream_reader_a
      = { .ctx = &reader_a,
          .pull = buffer_reader_pull,
          .size_hint = (sqlite3_int64)reader_a.size };
  objstore_stream_reader stream_reader_b
      = { .ctx = &reader_b,
          .pull = buffer_reader_pull,
          .size_hint = (sqlite3_int64)reader_b.size };

  objstore_backend_txn *txn = begin_backend_txn ();
  TEST_ASSERT_EQUAL_INT (SQLITE_OK,
                         current_backend ()->put (txn, &id, &stream_reader_a));
  TEST_ASSERT_EQUAL_INT (SQLITE_OK,
                         current_backend ()->put (txn, &id, &stream_reader_b));
  commit_backend_txn (txn);
  TEST_ASSERT_TRUE (sqlite_row_exists (&id));
}

static void
test_sqlite_backend_queue_duplicate_conflict_same_txn_impl (void)
{
  uint8_t payload_a[24];
  uint8_t payload_b[24];
  memset (payload_a, 0x11, sizeof (payload_a));
  memset (payload_b, 0x22, sizeof (payload_b));
  objstore_id id = make_id (0x83);

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

  objstore_backend_txn *txn = begin_backend_txn ();
  TEST_ASSERT_EQUAL_INT (SQLITE_OK,
                         current_backend ()->put (txn, &id, &stream_reader_a));
  TEST_ASSERT_EQUAL_INT (SQLITE_CONSTRAINT,
                         current_backend ()->put (txn, &id, &stream_reader_b));
  rollback_backend_txn (txn);
  TEST_ASSERT_FALSE (sqlite_row_exists (&id));
}

static void
test_sqlite_backend_queue_limit_impl (void)
{
  const size_t first_size = OBJSTORE_SQLITE_QUEUE_LIMIT_TEST_BYTES - 4096;
  const size_t second_size = 8192;
  objstore_id id_a = make_id (0x84);
  objstore_id id_b = make_id (0x85);

  uint8_t *payload_a = (uint8_t *)malloc (first_size);
  uint8_t *payload_b = (uint8_t *)malloc (second_size);
  TEST_ASSERT_NOT_NULL (payload_a);
  TEST_ASSERT_NOT_NULL (payload_b);
  memset (payload_a, 0x33, first_size);
  memset (payload_b, 0x44, second_size);

  buffer_reader reader_a
      = { .data = payload_a, .size = first_size, .offset = 0 };
  buffer_reader reader_b
      = { .data = payload_b, .size = second_size, .offset = 0 };
  objstore_stream_reader stream_reader_a
      = { .ctx = &reader_a,
          .pull = buffer_reader_pull,
          .size_hint = (sqlite3_int64)reader_a.size };
  objstore_stream_reader stream_reader_b
      = { .ctx = &reader_b,
          .pull = buffer_reader_pull,
          .size_hint = (sqlite3_int64)reader_b.size };

  objstore_backend_txn *txn = begin_backend_txn ();
  TEST_ASSERT_EQUAL_INT (
      SQLITE_OK, current_backend ()->put (txn, &id_a, &stream_reader_a));
  TEST_ASSERT_EQUAL_INT (
      SQLITE_FULL, current_backend ()->put (txn, &id_b, &stream_reader_b));
  rollback_backend_txn (txn);
  TEST_ASSERT_FALSE (sqlite_row_exists (&id_a));
  TEST_ASSERT_FALSE (sqlite_row_exists (&id_b));
  free (payload_a);
  free (payload_b);
}

void
backend_sqlite_register_tests (void)
{
  objstore_tests_set_fixture (OBJSTORE_FIXTURE_SQLITE_BACKEND);
  RUN_TEST (test_sqlite_backend_roundtrip_impl);
  RUN_TEST (test_sqlite_backend_duplicate_same_bytes_impl);
  RUN_TEST (test_sqlite_backend_duplicate_conflict_impl);
  RUN_TEST (test_sqlite_backend_get_missing_impl);
  RUN_TEST (test_sqlite_backend_rollback_discards_impl);
  RUN_TEST (test_sqlite_backend_exists_impl);
  RUN_TEST (test_sqlite_backend_queue_visibility_impl);
  RUN_TEST (test_sqlite_backend_queue_delete_visibility_impl);
  RUN_TEST (test_sqlite_backend_queue_duplicate_same_txn_impl);
  RUN_TEST (test_sqlite_backend_queue_duplicate_conflict_same_txn_impl);
  RUN_TEST (test_sqlite_backend_queue_limit_impl);
  RUN_TEST (test_sqlite_backend_rowidx_lookup);
  objstore_tests_set_fixture (OBJSTORE_FIXTURE_SMOKE);
}
