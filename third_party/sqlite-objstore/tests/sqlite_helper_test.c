
#include "unity.h"

#include <ctype.h>
#include <dirent.h>
#include <errno.h>
#include <limits.h>
#include <stdbool.h>
#include <sqlite3.h>
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
test_sqlite_helper_nested_put_exists (void)
{
  sqlite3 *db = objstore_open_ephemeral_db ();
  sqlite3_stmt *stmt = NULL;
  TEST_ASSERT_EQUAL_INT (
      SQLITE_OK,
      sqlite3_prepare_v2 (db, "SELECT objstore_exists(objstore_put(?1));", -1,
                          &stmt, NULL));
  const uint8_t payload[] = { 0x10, 0x11, 0x12, 0x13, 0x14 };
  TEST_ASSERT_EQUAL_INT (SQLITE_OK, sqlite3_bind_blob (stmt, 1, payload,
                                                       (int)sizeof (payload),
                                                       SQLITE_STATIC));
  TEST_ASSERT_EQUAL_INT (SQLITE_ROW, sqlite3_step (stmt));
  TEST_ASSERT_EQUAL_INT (1, sqlite3_column_int (stmt, 0));
  sqlite3_finalize (stmt);
  objstore_close_ephemeral_db (db);
}

static void
test_sqlite_helper_nested_put_get (void)
{
  sqlite3 *db = objstore_open_ephemeral_db ();
  sqlite3_stmt *stmt = NULL;
  TEST_ASSERT_EQUAL_INT (
      SQLITE_OK,
      sqlite3_prepare_v2 (db, "SELECT objstore_get(objstore_put(?1));", -1,
                          &stmt, NULL));
  const uint8_t payload[] = { 0xAA, 0xBB, 0xCC, 0xDD };
  TEST_ASSERT_EQUAL_INT (SQLITE_OK, sqlite3_bind_blob (stmt, 1, payload,
                                                       (int)sizeof (payload),
                                                       SQLITE_STATIC));
  TEST_ASSERT_EQUAL_INT (SQLITE_ROW, sqlite3_step (stmt));
  const void *result = sqlite3_column_blob (stmt, 0);
  const int result_size = sqlite3_column_bytes (stmt, 0);
  TEST_ASSERT_NOT_NULL (result);
  TEST_ASSERT_EQUAL_INT ((int)sizeof (payload), result_size);
  TEST_ASSERT_EQUAL_UINT8_ARRAY (payload, result, sizeof (payload));
  sqlite3_finalize (stmt);
  objstore_close_ephemeral_db (db);
}

static void
test_sqlite_helper_bulk_put_inside_select (void)
{
  sqlite3 *db = objstore_open_ephemeral_db ();
  objstore_exec_or_fail (db, "CREATE TABLE source(data BLOB);");
  sqlite3_stmt *insert_stmt = NULL;
  TEST_ASSERT_EQUAL_INT (
      SQLITE_OK,
      sqlite3_prepare_v2 (db, "INSERT INTO source(data) VALUES (?1);", -1,
                          &insert_stmt, NULL));
  for (int i = 0; i < 4; ++i)
    {
      uint8_t payload[3] = { (uint8_t)i, (uint8_t)(i + 1), (uint8_t)(i + 2) };
      sqlite3_bind_blob (insert_stmt, 1, payload, (int)sizeof (payload),
                         SQLITE_STATIC);
      TEST_ASSERT_EQUAL_INT (SQLITE_DONE, sqlite3_step (insert_stmt));
      TEST_ASSERT_EQUAL_INT (SQLITE_OK, sqlite3_reset (insert_stmt));
      sqlite3_clear_bindings (insert_stmt);
    }
  sqlite3_finalize (insert_stmt);

  objstore_exec_or_fail (db, "CREATE TABLE sink(id BLOB PRIMARY KEY);");
  objstore_exec_or_fail (
      db, "INSERT INTO sink(id) SELECT objstore_put(data) FROM source;");

  sqlite3_stmt *count_stmt = NULL;
  TEST_ASSERT_EQUAL_INT (SQLITE_OK,
                         sqlite3_prepare_v2 (db, "SELECT COUNT(*) FROM sink;",
                                             -1, &count_stmt, NULL));
  TEST_ASSERT_EQUAL_INT (SQLITE_ROW, sqlite3_step (count_stmt));
  TEST_ASSERT_EQUAL_INT (4, sqlite3_column_int (count_stmt, 0));
  sqlite3_finalize (count_stmt);
  objstore_close_ephemeral_db (db);
}

static void
run_sqlite_helper_txn_visibility_test (bool commit_txn)
{
  sqlite3 *db = objstore_open_ephemeral_db ();
  objstore_exec_or_fail (db, "BEGIN;");
  sqlite3_stmt *put_stmt = NULL;
  TEST_ASSERT_EQUAL_INT (SQLITE_OK,
                         sqlite3_prepare_v2 (db, "SELECT objstore_put(?1);",
                                             -1, &put_stmt, NULL));
  const uint8_t payload[] = { 0x61, 0x62, 0x63, 0x64 };
  TEST_ASSERT_EQUAL_INT (SQLITE_OK, sqlite3_bind_blob (put_stmt, 1, payload,
                                                       (int)sizeof (payload),
                                                       SQLITE_STATIC));
  TEST_ASSERT_EQUAL_INT (SQLITE_ROW, sqlite3_step (put_stmt));
  const void *id_blob = sqlite3_column_blob (put_stmt, 0);
  TEST_ASSERT_NOT_NULL (id_blob);
  uint8_t id_copy[OBJSTORE_ID_SIZE];
  memcpy (id_copy, id_blob, OBJSTORE_ID_SIZE);
  sqlite3_finalize (put_stmt);

  objstore_exec_or_fail (db, commit_txn ? "COMMIT;" : "ROLLBACK;");

  sqlite3_stmt *exists_stmt = NULL;
  TEST_ASSERT_EQUAL_INT (SQLITE_OK,
                         sqlite3_prepare_v2 (db, "SELECT objstore_exists(?1);",
                                             -1, &exists_stmt, NULL));
  TEST_ASSERT_EQUAL_INT (SQLITE_OK,
                         sqlite3_bind_blob (exists_stmt, 1, id_copy,
                                            OBJSTORE_ID_SIZE, SQLITE_STATIC));
  TEST_ASSERT_EQUAL_INT (SQLITE_ROW, sqlite3_step (exists_stmt));
  const int present = sqlite3_column_int (exists_stmt, 0);
  sqlite3_finalize (exists_stmt);
  objstore_close_ephemeral_db (db);
  TEST_ASSERT_EQUAL_INT (commit_txn ? 1 : 0, present);
}

static void
test_sqlite_helper_put_commit_visibility (void)
{
  run_sqlite_helper_txn_visibility_test (true);
}

static void
test_sqlite_helper_put_rollback_visibility (void)
{
  run_sqlite_helper_txn_visibility_test (false);
}

static void
test_sqlite_helper_large_object_streaming (void)
{
  sqlite3 *db = objstore_open_ephemeral_db ();
  const size_t blob_size = 8u * 1024u * 1024u;
  uint8_t *payload = (uint8_t *)malloc (blob_size);
  TEST_ASSERT_NOT_NULL (payload);
  for (size_t i = 0; i < blob_size; ++i)
    {
      payload[i] = (uint8_t)(i & 0xFF);
    }

  sqlite3_stmt *stmt = NULL;
  TEST_ASSERT_EQUAL_INT (
      SQLITE_OK,
      sqlite3_prepare_v2 (db, "SELECT objstore_put(?1);", -1, &stmt, NULL));
  TEST_ASSERT_EQUAL_INT (
      SQLITE_OK,
      sqlite3_bind_blob64 (stmt, 1, payload, blob_size, SQLITE_STATIC));
  TEST_ASSERT_EQUAL_INT (SQLITE_ROW, sqlite3_step (stmt));
  TEST_ASSERT_EQUAL_INT (OBJSTORE_ID_SIZE, sqlite3_column_bytes (stmt, 0));
  uint8_t id_bytes[OBJSTORE_ID_SIZE];
  memcpy (id_bytes, sqlite3_column_blob (stmt, 0), OBJSTORE_ID_SIZE);
  sqlite3_finalize (stmt);

  TEST_ASSERT_EQUAL_INT (
      SQLITE_OK,
      sqlite3_prepare_v2 (db, "SELECT objstore_get(?1);", -1, &stmt, NULL));
  TEST_ASSERT_EQUAL_INT (
      SQLITE_OK,
      sqlite3_bind_blob (stmt, 1, id_bytes, OBJSTORE_ID_SIZE, SQLITE_STATIC));
  TEST_ASSERT_EQUAL_INT (SQLITE_ROW, sqlite3_step (stmt));
  TEST_ASSERT_EQUAL_INT ((int)blob_size, sqlite3_column_bytes (stmt, 0));
  TEST_ASSERT_EQUAL_MEMORY (payload, sqlite3_column_blob (stmt, 0), blob_size);
  sqlite3_finalize (stmt);

  free (payload);
  objstore_close_ephemeral_db (db);
}

void
sqlite_helper_register_tests (void)
{
  RUN_TEST (test_sqlite_helper_nested_put_exists);
  RUN_TEST (test_sqlite_helper_nested_put_get);
  RUN_TEST (test_sqlite_helper_bulk_put_inside_select);
  RUN_TEST (test_sqlite_helper_put_commit_visibility);
  RUN_TEST (test_sqlite_helper_put_rollback_visibility);
  RUN_TEST (test_sqlite_helper_large_object_streaming);
}
