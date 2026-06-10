
#include "unity.h"

#include <ctype.h>
#include <dirent.h>
#include <errno.h>
#include <limits.h>
#include <sqlite3.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "objstore/backend.h"
#include "objstore/blake3.h"
#include "objstore/objstore.h"
#include "test_harness.h"
#include "test_support.h"

static void
test_assert_range_error (sqlite3 *db, sqlite3_stmt *stmt)
{
  TEST_ASSERT_NOT_NULL (db);
  TEST_ASSERT_NOT_NULL (stmt);
  const int rc = sqlite3_step (stmt);
  if (rc == SQLITE_ERROR)
    {
      TEST_ASSERT_EQUAL_INT (SQLITE_RANGE, sqlite3_extended_errcode (db));
      return;
    }
  TEST_ASSERT_EQUAL_INT (SQLITE_RANGE, rc);
}

static void
test_scalar_functions_roundtrip (void)
{
  sqlite3 *db = objstore_open_ephemeral_db ();
  const uint8_t payload[] = { 0xDE, 0xAD, 0xBE, 0xEF };
  static const uint8_t kExpectedId[OBJSTORE_ID_SIZE] = {
    0x53, 0x14, 0x7F, 0x3C, 0xE4, 0x9E, 0xD4, 0xF6, 0x0D, 0xFA, 0x5B,
    0x96, 0x54, 0xC3, 0x6B, 0xA6, 0x10, 0x3C, 0x11, 0xF5, 0x73, 0x7D,
    0xF3, 0xDA, 0xBD, 0x4C, 0xBD, 0x29, 0x6C, 0x41, 0x61, 0xBD,
  };

  sqlite3_stmt *stmt = NULL;
  TEST_ASSERT_EQUAL_INT (
      SQLITE_OK,
      sqlite3_prepare_v2 (db, "SELECT objstore_put(?1);", -1, &stmt, NULL));
  TEST_ASSERT_EQUAL_INT (SQLITE_OK, sqlite3_bind_blob (stmt, 1, payload,
                                                       (int)sizeof (payload),
                                                       SQLITE_STATIC));
  TEST_ASSERT_EQUAL_INT (SQLITE_ROW, sqlite3_step (stmt));
  TEST_ASSERT_EQUAL_INT (OBJSTORE_ID_SIZE, sqlite3_column_bytes (stmt, 0));
  const void *id_blob = sqlite3_column_blob (stmt, 0);
  TEST_ASSERT_NOT_NULL (id_blob);
  TEST_ASSERT_EQUAL_MEMORY (kExpectedId, id_blob, OBJSTORE_ID_SIZE);
  sqlite3_finalize (stmt);

  TEST_ASSERT_EQUAL_INT (
      SQLITE_OK,
      sqlite3_prepare_v2 (db, "SELECT objstore_exists(?1);", -1, &stmt, NULL));
  TEST_ASSERT_EQUAL_INT (SQLITE_OK,
                         sqlite3_bind_blob (stmt, 1, kExpectedId,
                                            OBJSTORE_ID_SIZE, SQLITE_STATIC));
  TEST_ASSERT_EQUAL_INT (SQLITE_ROW, sqlite3_step (stmt));
  TEST_ASSERT_EQUAL_INT (1, sqlite3_column_int (stmt, 0));
  sqlite3_finalize (stmt);

  TEST_ASSERT_EQUAL_INT (
      SQLITE_OK,
      sqlite3_prepare_v2 (db, "SELECT objstore_get(?1);", -1, &stmt, NULL));
  TEST_ASSERT_EQUAL_INT (SQLITE_OK,
                         sqlite3_bind_blob (stmt, 1, kExpectedId,
                                            OBJSTORE_ID_SIZE, SQLITE_STATIC));
  TEST_ASSERT_EQUAL_INT (SQLITE_ROW, sqlite3_step (stmt));
  TEST_ASSERT_EQUAL_INT ((int)sizeof (payload),
                         sqlite3_column_bytes (stmt, 0));
  TEST_ASSERT_EQUAL_MEMORY (payload, sqlite3_column_blob (stmt, 0),
                            sizeof (payload));
  sqlite3_finalize (stmt);

  TEST_ASSERT_EQUAL_INT (
      SQLITE_OK,
      sqlite3_prepare_v2 (db, "SELECT objstore_delete(?1);", -1, &stmt, NULL));
  TEST_ASSERT_EQUAL_INT (SQLITE_OK,
                         sqlite3_bind_blob (stmt, 1, kExpectedId,
                                            OBJSTORE_ID_SIZE, SQLITE_STATIC));
  TEST_ASSERT_EQUAL_INT (SQLITE_ROW, sqlite3_step (stmt));
  TEST_ASSERT_EQUAL_INT (1, sqlite3_column_int (stmt, 0));
  sqlite3_finalize (stmt);

  TEST_ASSERT_EQUAL_INT (
      SQLITE_OK,
      sqlite3_prepare_v2 (db, "SELECT objstore_exists(?1);", -1, &stmt, NULL));
  TEST_ASSERT_EQUAL_INT (SQLITE_OK,
                         sqlite3_bind_blob (stmt, 1, kExpectedId,
                                            OBJSTORE_ID_SIZE, SQLITE_STATIC));
  TEST_ASSERT_EQUAL_INT (SQLITE_ROW, sqlite3_step (stmt));
  TEST_ASSERT_EQUAL_INT (0, sqlite3_column_int (stmt, 0));
  sqlite3_finalize (stmt);

  objstore_close_ephemeral_db (db);
}

static void
test_scalar_functions_respect_transactions (void)
{
  sqlite3 *db = objstore_open_ephemeral_db ();
  const uint8_t payload[] = { 0xAB, 0xCD, 0xEF, 0x01 };
  sqlite3_stmt *stmt = NULL;

  objstore_exec_or_fail (db, "BEGIN;");
  TEST_ASSERT_EQUAL_INT (
      SQLITE_OK,
      sqlite3_prepare_v2 (db, "SELECT objstore_put(?1);", -1, &stmt, NULL));
  TEST_ASSERT_EQUAL_INT (SQLITE_OK, sqlite3_bind_blob (stmt, 1, payload,
                                                       (int)sizeof (payload),
                                                       SQLITE_TRANSIENT));
  TEST_ASSERT_EQUAL_INT (SQLITE_ROW, sqlite3_step (stmt));
  const uint8_t *id_blob = sqlite3_column_blob (stmt, 0);
  TEST_ASSERT_NOT_NULL (id_blob);
  uint8_t id_copy[OBJSTORE_ID_SIZE];
  memcpy (id_copy, id_blob, OBJSTORE_ID_SIZE);
  sqlite3_finalize (stmt);
  objstore_exec_or_fail (db, "ROLLBACK;");
  TEST_ASSERT_EQUAL_INT (0, objstore_exists_helper (db, id_copy));

  objstore_exec_or_fail (db, "BEGIN;");
  TEST_ASSERT_EQUAL_INT (
      SQLITE_OK,
      sqlite3_prepare_v2 (db, "SELECT objstore_put(?1);", -1, &stmt, NULL));
  TEST_ASSERT_EQUAL_INT (SQLITE_OK, sqlite3_bind_blob (stmt, 1, payload,
                                                       (int)sizeof (payload),
                                                       SQLITE_TRANSIENT));
  TEST_ASSERT_EQUAL_INT (SQLITE_ROW, sqlite3_step (stmt));
  sqlite3_finalize (stmt);
  objstore_exec_or_fail (db, "COMMIT;");
  TEST_ASSERT_EQUAL_INT (1, objstore_exists_helper (db, id_copy));

  objstore_close_ephemeral_db (db);
}

static void
test_objstore_put_with_id_enforces_immutability (void)
{
  sqlite3 *db = objstore_open_ephemeral_db ();
  const uint8_t payload1[] = { 0x01, 0x02 };
  const uint8_t payload2[] = { 0x03, 0x04 };
  static const uint8_t kIdBytes[OBJSTORE_ID_SIZE]
      = { 0x10, 0x10, 0x20, 0x20, 0x30, 0x30, 0x40, 0x40, 0x50, 0x50, 0x60,
          0x60, 0x70, 0x70, 0x80, 0x80, 0x90, 0x90, 0xA0, 0xA0, 0xB0, 0xB0,
          0xC0, 0xC0, 0xD0, 0xD0, 0xE0, 0xE0, 0xF0, 0xF0, 0x11, 0x11 };

  sqlite3_stmt *stmt = NULL;
  TEST_ASSERT_EQUAL_INT (
      SQLITE_OK,
      sqlite3_prepare_v2 (db, "SELECT objstore_put_with_id(?1, ?2);", -1,
                          &stmt, NULL));
  TEST_ASSERT_EQUAL_INT (
      SQLITE_OK,
      sqlite3_bind_blob (stmt, 1, kIdBytes, OBJSTORE_ID_SIZE, SQLITE_STATIC));
  TEST_ASSERT_EQUAL_INT (SQLITE_OK, sqlite3_bind_blob (stmt, 2, payload1,
                                                       (int)sizeof (payload1),
                                                       SQLITE_TRANSIENT));
  TEST_ASSERT_EQUAL_INT (SQLITE_ROW, sqlite3_step (stmt));
  sqlite3_finalize (stmt);

  TEST_ASSERT_EQUAL_INT (
      SQLITE_OK,
      sqlite3_prepare_v2 (db, "SELECT objstore_put_with_id(?1, ?2);", -1,
                          &stmt, NULL));
  TEST_ASSERT_EQUAL_INT (
      SQLITE_OK,
      sqlite3_bind_blob (stmt, 1, kIdBytes, OBJSTORE_ID_SIZE, SQLITE_STATIC));
  TEST_ASSERT_EQUAL_INT (SQLITE_OK, sqlite3_bind_blob (stmt, 2, payload2,
                                                       (int)sizeof (payload2),
                                                       SQLITE_TRANSIENT));
  TEST_ASSERT_EQUAL_INT (SQLITE_CONSTRAINT, sqlite3_step (stmt));
  TEST_ASSERT_EQUAL_INT (SQLITE_CONSTRAINT, sqlite3_extended_errcode (db));
  sqlite3_finalize (stmt);

  objstore_close_ephemeral_db (db);
}

static void
test_objstore_get_missing_returns_null (void)
{
  sqlite3 *db = objstore_open_ephemeral_db ();
  static const uint8_t kMissingId[OBJSTORE_ID_SIZE]
      = { 0xAA, 0x00, 0xBB, 0x00, 0xCC, 0x00, 0xDD, 0x00, 0xEE, 0x00, 0xFF,
          0x00, 0x11, 0x00, 0x22, 0x00, 0x33, 0x00, 0x44, 0x00, 0x55, 0x00,
          0x66, 0x00, 0x77, 0x00, 0x88, 0x00, 0x99, 0x00, 0xAA, 0x00 };

  sqlite3_stmt *stmt = NULL;
  TEST_ASSERT_EQUAL_INT (
      SQLITE_OK,
      sqlite3_prepare_v2 (db, "SELECT objstore_get(?1);", -1, &stmt, NULL));
  TEST_ASSERT_EQUAL_INT (SQLITE_OK,
                         sqlite3_bind_blob (stmt, 1, kMissingId,
                                            OBJSTORE_ID_SIZE, SQLITE_STATIC));
  TEST_ASSERT_EQUAL_INT (SQLITE_ROW, sqlite3_step (stmt));
  TEST_ASSERT_EQUAL_INT (SQLITE_NULL, sqlite3_column_type (stmt, 0));
  sqlite3_finalize (stmt);

  objstore_close_ephemeral_db (db);
}

static void
test_objstore_get_range_returns_slice (void)
{
  sqlite3 *db = objstore_open_ephemeral_db ();
  const uint8_t payload[] = { 0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16 };

  sqlite3_stmt *stmt = NULL;
  TEST_ASSERT_EQUAL_INT (
      SQLITE_OK,
      sqlite3_prepare_v2 (db, "SELECT objstore_put(?1);", -1, &stmt, NULL));
  TEST_ASSERT_EQUAL_INT (SQLITE_OK, sqlite3_bind_blob (stmt, 1, payload,
                                                       (int)sizeof (payload),
                                                       SQLITE_STATIC));
  TEST_ASSERT_EQUAL_INT (SQLITE_ROW, sqlite3_step (stmt));
  const uint8_t *id_blob = sqlite3_column_blob (stmt, 0);
  TEST_ASSERT_NOT_NULL (id_blob);
  uint8_t id_copy[OBJSTORE_ID_SIZE];
  memcpy (id_copy, id_blob, OBJSTORE_ID_SIZE);
  sqlite3_finalize (stmt);

  TEST_ASSERT_EQUAL_INT (
      SQLITE_OK,
      sqlite3_prepare_v2 (db,
                          "SELECT objstore_get_range(?1, 'bytes=2-5');", -1,
                          &stmt, NULL));
  TEST_ASSERT_EQUAL_INT (SQLITE_OK,
                         sqlite3_bind_blob (stmt, 1, id_copy, OBJSTORE_ID_SIZE,
                                            SQLITE_STATIC));
  TEST_ASSERT_EQUAL_INT (SQLITE_ROW, sqlite3_step (stmt));
  TEST_ASSERT_EQUAL_INT (4, sqlite3_column_bytes (stmt, 0));
  TEST_ASSERT_EQUAL_MEMORY (payload + 2, sqlite3_column_blob (stmt, 0), 4);
  sqlite3_finalize (stmt);

  TEST_ASSERT_EQUAL_INT (
      SQLITE_OK,
      sqlite3_prepare_v2 (db,
                          "SELECT objstore_get_range(?1, 'bytes= 1 - 3 ');",
                          -1, &stmt, NULL));
  TEST_ASSERT_EQUAL_INT (SQLITE_OK,
                         sqlite3_bind_blob (stmt, 1, id_copy, OBJSTORE_ID_SIZE,
                                            SQLITE_STATIC));
  TEST_ASSERT_EQUAL_INT (SQLITE_ROW, sqlite3_step (stmt));
  TEST_ASSERT_EQUAL_INT (3, sqlite3_column_bytes (stmt, 0));
  TEST_ASSERT_EQUAL_MEMORY (payload + 1, sqlite3_column_blob (stmt, 0), 3);
  sqlite3_finalize (stmt);

  TEST_ASSERT_EQUAL_INT (
      SQLITE_OK,
      sqlite3_prepare_v2 (db,
                          "SELECT objstore_get_range(?1, 'bytes=0-');", -1,
                          &stmt, NULL));
  TEST_ASSERT_EQUAL_INT (SQLITE_OK,
                         sqlite3_bind_blob (stmt, 1, id_copy, OBJSTORE_ID_SIZE,
                                            SQLITE_STATIC));
  TEST_ASSERT_EQUAL_INT (SQLITE_ROW, sqlite3_step (stmt));
  TEST_ASSERT_EQUAL_INT ((int)sizeof (payload), sqlite3_column_bytes (stmt, 0));
  TEST_ASSERT_EQUAL_MEMORY (payload, sqlite3_column_blob (stmt, 0),
                            sizeof (payload));
  sqlite3_finalize (stmt);

  TEST_ASSERT_EQUAL_INT (
      SQLITE_OK,
      sqlite3_prepare_v2 (db,
                          "SELECT objstore_get_range(?1, 'bytes=-3');", -1,
                          &stmt, NULL));
  TEST_ASSERT_EQUAL_INT (SQLITE_OK,
                         sqlite3_bind_blob (stmt, 1, id_copy, OBJSTORE_ID_SIZE,
                                            SQLITE_STATIC));
  TEST_ASSERT_EQUAL_INT (SQLITE_ROW, sqlite3_step (stmt));
  TEST_ASSERT_EQUAL_INT (3, sqlite3_column_bytes (stmt, 0));
  TEST_ASSERT_EQUAL_MEMORY (payload + 4, sqlite3_column_blob (stmt, 0), 3);
  sqlite3_finalize (stmt);

  TEST_ASSERT_EQUAL_INT (
      SQLITE_OK,
      sqlite3_prepare_v2 (db,
                          "SELECT objstore_get_range(?1, 'bytes=99-100');",
                          -1, &stmt, NULL));
  TEST_ASSERT_EQUAL_INT (SQLITE_OK,
                         sqlite3_bind_blob (stmt, 1, id_copy, OBJSTORE_ID_SIZE,
                                            SQLITE_STATIC));
  TEST_ASSERT_EQUAL_INT (SQLITE_ROW, sqlite3_step (stmt));
  TEST_ASSERT_EQUAL_INT (SQLITE_NULL, sqlite3_column_type (stmt, 0));
  sqlite3_finalize (stmt);

  TEST_ASSERT_EQUAL_INT (
      SQLITE_OK,
      sqlite3_prepare_v2 (db,
                          "SELECT objstore_get_range(?1, 'bytes=1-2,3-4');",
                          -1, &stmt, NULL));
  TEST_ASSERT_EQUAL_INT (SQLITE_OK,
                         sqlite3_bind_blob (stmt, 1, id_copy, OBJSTORE_ID_SIZE,
                                            SQLITE_STATIC));
  test_assert_range_error (db, stmt);
  sqlite3_finalize (stmt);

  TEST_ASSERT_EQUAL_INT (
      SQLITE_OK,
      sqlite3_prepare_v2 (db,
                          "SELECT objstore_get_range(?1, '0-1');", -1, &stmt,
                          NULL));
  TEST_ASSERT_EQUAL_INT (SQLITE_OK,
                         sqlite3_bind_blob (stmt, 1, id_copy, OBJSTORE_ID_SIZE,
                                            SQLITE_STATIC));
  TEST_ASSERT_EQUAL_INT (SQLITE_ROW, sqlite3_step (stmt));
  TEST_ASSERT_EQUAL_INT (2, sqlite3_column_bytes (stmt, 0));
  TEST_ASSERT_EQUAL_MEMORY (payload, sqlite3_column_blob (stmt, 0), 2);
  sqlite3_finalize (stmt);

  TEST_ASSERT_EQUAL_INT (
      SQLITE_OK,
      sqlite3_prepare_v2 (db,
                          "SELECT objstore_get_range(?1, 'bytes=foo');", -1,
                          &stmt, NULL));
  TEST_ASSERT_EQUAL_INT (SQLITE_OK,
                         sqlite3_bind_blob (stmt, 1, id_copy, OBJSTORE_ID_SIZE,
                                            SQLITE_STATIC));
  test_assert_range_error (db, stmt);
  sqlite3_finalize (stmt);

  static const char *const kInvalidRanges[] = {
    "bytes=",
    "bytes=-",
    "bytes=1--2",
    "bytes=1-2-3",
    "bytes=1-2,3-4",
    "bytes=abc-def",
    "bytes=1-2x",
    "bytes=+1-2",
    "bytes=1-+2",
    "bytes=18446744073709551616-1",
  };
  for (size_t i = 0; i < sizeof (kInvalidRanges) / sizeof (kInvalidRanges[0]);
       ++i)
    {
      TEST_ASSERT_EQUAL_INT (
          SQLITE_OK,
          sqlite3_prepare_v2 (db, "SELECT objstore_get_range(?1, ?2);", -1,
                              &stmt, NULL));
      TEST_ASSERT_EQUAL_INT (SQLITE_OK, sqlite3_bind_blob (
                                            stmt, 1, id_copy, OBJSTORE_ID_SIZE,
                                            SQLITE_STATIC));
      TEST_ASSERT_EQUAL_INT (SQLITE_OK, sqlite3_bind_text (
                                            stmt, 2, kInvalidRanges[i], -1,
                                            SQLITE_STATIC));
      test_assert_range_error (db, stmt);
      sqlite3_finalize (stmt);
    }

  objstore_close_ephemeral_db (db);
}

static void
test_scalar_mutations_execute_without_consuming_result (void)
{
  sqlite3 *db = objstore_open_ephemeral_db ();
  const uint8_t id[OBJSTORE_ID_SIZE] = {
    0x0A, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0A,
  };
  const uint8_t initial_payload[] = { 0xDE, 0xAD, 0xBE, 0xEF };
  const uint8_t updated_payload[] = { 0xAA, 0xBB, 0xCC, 0xDD };

  sqlite3_stmt *stmt = NULL;
  TEST_ASSERT_EQUAL_INT (
      SQLITE_OK,
      sqlite3_prepare_v2 (db, "INSERT INTO objstore(id, data) VALUES(?1, ?2);",
                          -1, &stmt, NULL));
  TEST_ASSERT_EQUAL_INT (
      SQLITE_OK,
      sqlite3_bind_blob (stmt, 1, id, OBJSTORE_ID_SIZE, SQLITE_STATIC));
  TEST_ASSERT_EQUAL_INT (SQLITE_OK,
                         sqlite3_bind_blob (stmt, 2, initial_payload,
                                            (int)sizeof (initial_payload),
                                            SQLITE_STATIC));
  TEST_ASSERT_EQUAL_INT (SQLITE_DONE, sqlite3_step (stmt));
  sqlite3_finalize (stmt);

  sqlite3_stmt *delete_stmt = NULL;
  TEST_ASSERT_EQUAL_INT (SQLITE_OK,
                         sqlite3_prepare_v2 (db, "SELECT objstore_delete(?1);",
                                             -1, &delete_stmt, NULL));
  TEST_ASSERT_EQUAL_INT (
      SQLITE_OK,
      sqlite3_bind_blob (delete_stmt, 1, id, OBJSTORE_ID_SIZE, SQLITE_STATIC));
  TEST_ASSERT_EQUAL_INT (SQLITE_ROW, sqlite3_step (delete_stmt));
  sqlite3_finalize (delete_stmt);
  TEST_ASSERT_EQUAL_INT (0, objstore_exists_helper (db, id));

  sqlite3_stmt *put_stmt = NULL;
  TEST_ASSERT_EQUAL_INT (
      SQLITE_OK,
      sqlite3_prepare_v2 (db, "SELECT objstore_put_with_id(?1, ?2);", -1,
                          &put_stmt, NULL));
  TEST_ASSERT_EQUAL_INT (
      SQLITE_OK,
      sqlite3_bind_blob (put_stmt, 1, id, OBJSTORE_ID_SIZE, SQLITE_STATIC));
  TEST_ASSERT_EQUAL_INT (SQLITE_OK,
                         sqlite3_bind_blob (put_stmt, 2, updated_payload,
                                            (int)sizeof (updated_payload),
                                            SQLITE_STATIC));
  TEST_ASSERT_EQUAL_INT (SQLITE_ROW, sqlite3_step (put_stmt));
  sqlite3_finalize (put_stmt);

  sqlite3_stmt *get_stmt = NULL;
  TEST_ASSERT_EQUAL_INT (SQLITE_OK,
                         sqlite3_prepare_v2 (db, "SELECT objstore_get(?1);",
                                             -1, &get_stmt, NULL));
  TEST_ASSERT_EQUAL_INT (
      SQLITE_OK,
      sqlite3_bind_blob (get_stmt, 1, id, OBJSTORE_ID_SIZE, SQLITE_STATIC));
  TEST_ASSERT_EQUAL_INT (SQLITE_ROW, sqlite3_step (get_stmt));
  TEST_ASSERT_EQUAL_INT ((int)sizeof (updated_payload),
                         sqlite3_column_bytes (get_stmt, 0));
  TEST_ASSERT_EQUAL_MEMORY (updated_payload, sqlite3_column_blob (get_stmt, 0),
                            sizeof (updated_payload));
  sqlite3_finalize (get_stmt);

  objstore_close_ephemeral_db (db);
}

static void
test_objstore_get_rejects_non_blob_ids (void)
{
  sqlite3 *db = objstore_open_ephemeral_db ();
  sqlite3_stmt *stmt = NULL;
  TEST_ASSERT_EQUAL_INT (
      SQLITE_OK, sqlite3_prepare_v2 (db, "SELECT objstore_get('text-id');", -1,
                                     &stmt, NULL));
  const int rc = sqlite3_step (stmt);
  if (rc == SQLITE_ERROR)
    {
      TEST_ASSERT_EQUAL_INT (SQLITE_MISMATCH, sqlite3_extended_errcode (db));
    }
  else
    {
      TEST_ASSERT_EQUAL_INT (SQLITE_MISMATCH, rc);
    }
  sqlite3_finalize (stmt);
  objstore_close_ephemeral_db (db);
}

static void
test_objstore_insert_with_short_id_fails (void)
{
  sqlite3 *db = objstore_open_ephemeral_db ();
  sqlite3_stmt *stmt = NULL;
  TEST_ASSERT_EQUAL_INT (
      SQLITE_OK, sqlite3_prepare_v2 (
                     db, "SELECT objstore_put_with_id(x'1234', zeroblob(1));",
                     -1, &stmt, NULL));
  const int rc = sqlite3_step (stmt);
  if (rc == SQLITE_ERROR)
    {
      TEST_ASSERT_EQUAL_INT (SQLITE_MISMATCH, sqlite3_extended_errcode (db));
    }
  else
    {
      TEST_ASSERT_EQUAL_INT (SQLITE_MISMATCH, rc);
    }
  sqlite3_finalize (stmt);
  objstore_close_ephemeral_db (db);
}

void
scalar_function_register_tests (void)
{
  RUN_TEST (test_scalar_functions_roundtrip);
  RUN_TEST (test_scalar_functions_respect_transactions);
  RUN_TEST (test_objstore_put_with_id_enforces_immutability);
  RUN_TEST (test_objstore_get_missing_returns_null);
  RUN_TEST (test_objstore_get_range_returns_slice);
  RUN_TEST (test_scalar_mutations_execute_without_consuming_result);
  RUN_TEST (test_objstore_get_rejects_non_blob_ids);
  RUN_TEST (test_objstore_insert_with_short_id_fails);
}
