
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
test_virtual_table_basic_crud (void)
{
  sqlite3 *db = objstore_open_ephemeral_db ();
  static const uint8_t kIdBytes[OBJSTORE_ID_SIZE] = {
    0xAA, 0xBB, 0xCC, 0xDD, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
    0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F, 0x10, 0x11, 0x12,
    0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1A, 0x1B, 0x1C,
  };
  const uint8_t payload[] = { 0x10, 0x20, 0x30, 0x40 };

  sqlite3_stmt *stmt = NULL;
  TEST_ASSERT_EQUAL_INT (
      SQLITE_OK,
      sqlite3_prepare_v2 (db, "INSERT INTO objstore(id, data) VALUES(?1, ?2);",
                          -1, &stmt, NULL));
  TEST_ASSERT_EQUAL_INT (
      SQLITE_OK,
      sqlite3_bind_blob (stmt, 1, kIdBytes, OBJSTORE_ID_SIZE, SQLITE_STATIC));
  TEST_ASSERT_EQUAL_INT (SQLITE_OK, sqlite3_bind_blob (stmt, 2, payload,
                                                       (int)sizeof (payload),
                                                       SQLITE_STATIC));
  TEST_ASSERT_EQUAL_INT (SQLITE_DONE, sqlite3_step (stmt));
  sqlite3_finalize (stmt);

  TEST_ASSERT_EQUAL_INT (
      SQLITE_OK,
      sqlite3_prepare_v2 (db, "SELECT data FROM objstore WHERE id = ?1;", -1,
                          &stmt, NULL));
  TEST_ASSERT_EQUAL_INT (
      SQLITE_OK,
      sqlite3_bind_blob (stmt, 1, kIdBytes, OBJSTORE_ID_SIZE, SQLITE_STATIC));
  TEST_ASSERT_EQUAL_INT (SQLITE_ROW, sqlite3_step (stmt));
  TEST_ASSERT_EQUAL_INT ((int)sizeof (payload),
                         sqlite3_column_bytes (stmt, 0));
  TEST_ASSERT_EQUAL_MEMORY (payload, sqlite3_column_blob (stmt, 0),
                            sizeof (payload));
  sqlite3_finalize (stmt);

  TEST_ASSERT_EQUAL_INT (
      SQLITE_OK, sqlite3_prepare_v2 (db, "DELETE FROM objstore WHERE id = ?1;",
                                     -1, &stmt, NULL));
  TEST_ASSERT_EQUAL_INT (
      SQLITE_OK,
      sqlite3_bind_blob (stmt, 1, kIdBytes, OBJSTORE_ID_SIZE, SQLITE_STATIC));
  TEST_ASSERT_EQUAL_INT (SQLITE_DONE, sqlite3_step (stmt));
  sqlite3_finalize (stmt);

  TEST_ASSERT_EQUAL_INT (
      SQLITE_OK,
      sqlite3_prepare_v2 (db, "SELECT objstore_exists(?1);", -1, &stmt, NULL));
  TEST_ASSERT_EQUAL_INT (
      SQLITE_OK,
      sqlite3_bind_blob (stmt, 1, kIdBytes, OBJSTORE_ID_SIZE, SQLITE_STATIC));
  TEST_ASSERT_EQUAL_INT (SQLITE_ROW, sqlite3_step (stmt));
  TEST_ASSERT_EQUAL_INT (0, sqlite3_column_int (stmt, 0));
  sqlite3_finalize (stmt);

  objstore_close_ephemeral_db (db);
}

static void
test_virtual_table_transactions (void)
{
  sqlite3 *db = objstore_open_ephemeral_db ();
  static const uint8_t kIdBytes[OBJSTORE_ID_SIZE]
      = { 0x01, 0x02, 0x03, 0x04, 0xAA, 0xBB, 0xCC, 0xDD, 0x10, 0x20, 0x30,
          0x40, 0x50, 0x60, 0x70, 0x80, 0x90, 0xA0, 0xB0, 0xC0, 0xD0, 0xE0,
          0xF0, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99 };
  const uint8_t payload[] = { 0xFE, 0xED, 0xFA, 0xCE };

  objstore_exec_or_fail (db, "BEGIN;");
  sqlite3_stmt *stmt = NULL;
  TEST_ASSERT_EQUAL_INT (
      SQLITE_OK,
      sqlite3_prepare_v2 (db, "INSERT INTO objstore(id, data) VALUES(?1, ?2);",
                          -1, &stmt, NULL));
  TEST_ASSERT_EQUAL_INT (
      SQLITE_OK,
      sqlite3_bind_blob (stmt, 1, kIdBytes, OBJSTORE_ID_SIZE, SQLITE_STATIC));
  TEST_ASSERT_EQUAL_INT (SQLITE_OK, sqlite3_bind_blob (stmt, 2, payload,
                                                       (int)sizeof (payload),
                                                       SQLITE_TRANSIENT));
  TEST_ASSERT_EQUAL_INT (SQLITE_DONE, sqlite3_step (stmt));
  sqlite3_finalize (stmt);
  objstore_exec_or_fail (db, "ROLLBACK;");
  TEST_ASSERT_EQUAL_INT (0, objstore_exists_helper (db, kIdBytes));

  objstore_exec_or_fail (db, "BEGIN;");
  TEST_ASSERT_EQUAL_INT (
      SQLITE_OK,
      sqlite3_prepare_v2 (db, "INSERT INTO objstore(id, data) VALUES(?1, ?2);",
                          -1, &stmt, NULL));
  TEST_ASSERT_EQUAL_INT (
      SQLITE_OK,
      sqlite3_bind_blob (stmt, 1, kIdBytes, OBJSTORE_ID_SIZE, SQLITE_STATIC));
  TEST_ASSERT_EQUAL_INT (SQLITE_OK, sqlite3_bind_blob (stmt, 2, payload,
                                                       (int)sizeof (payload),
                                                       SQLITE_TRANSIENT));
  TEST_ASSERT_EQUAL_INT (SQLITE_DONE, sqlite3_step (stmt));
  sqlite3_finalize (stmt);
  objstore_exec_or_fail (db, "COMMIT;");
  TEST_ASSERT_EQUAL_INT (1, objstore_exists_helper (db, kIdBytes));

  objstore_close_ephemeral_db (db);
}

static void
test_savepoint_rollback_aborts_transaction (void)
{
  sqlite3 *db = objstore_open_ephemeral_db ();
  const uint8_t payload[] = { 0x42 };
  uint8_t id_bytes[OBJSTORE_ID_SIZE] = {
    0xAA, 0xBB, 0xCC, 0xDD, 0x01, 0x02, 0x03, 0x04, 0x10, 0x20, 0x30,
    0x40, 0x50, 0x60, 0x70, 0x80, 0x90, 0xA0, 0xB0, 0xC0, 0xD0, 0xE0,
    0xF0, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99,
  };

  sqlite3_stmt *stmt = NULL;

  objstore_exec_or_fail (db, "BEGIN;");
  objstore_exec_or_fail (db, "SAVEPOINT s1;");
  TEST_ASSERT_EQUAL_INT (
      SQLITE_OK,
      sqlite3_prepare_v2 (db, "INSERT INTO objstore(id, data) VALUES(?1, ?2);",
                          -1, &stmt, NULL));
  TEST_ASSERT_EQUAL_INT (
      SQLITE_OK,
      sqlite3_bind_blob (stmt, 1, id_bytes, OBJSTORE_ID_SIZE, SQLITE_STATIC));
  TEST_ASSERT_EQUAL_INT (SQLITE_OK, sqlite3_bind_blob (stmt, 2, payload,
                                                       (int)sizeof (payload),
                                                       SQLITE_STATIC));
  TEST_ASSERT_EQUAL_INT (SQLITE_DONE, sqlite3_step (stmt));
  sqlite3_finalize (stmt);

  objstore_exec_expect_error (db, "ROLLBACK TO s1;");
  objstore_exec_or_fail (db, "ROLLBACK;");
  TEST_ASSERT_EQUAL_INT (0, objstore_exists_helper (db, id_bytes));

  objstore_exec_or_fail (db, "BEGIN;");
  objstore_exec_or_fail (db, "SAVEPOINT s2;");
  TEST_ASSERT_EQUAL_INT (
      SQLITE_OK,
      sqlite3_prepare_v2 (db, "INSERT INTO objstore(id, data) VALUES(?1, ?2);",
                          -1, &stmt, NULL));
  TEST_ASSERT_EQUAL_INT (
      SQLITE_OK,
      sqlite3_bind_blob (stmt, 1, id_bytes, OBJSTORE_ID_SIZE, SQLITE_STATIC));
  TEST_ASSERT_EQUAL_INT (SQLITE_OK, sqlite3_bind_blob (stmt, 2, payload,
                                                       (int)sizeof (payload),
                                                       SQLITE_STATIC));
  TEST_ASSERT_EQUAL_INT (SQLITE_DONE, sqlite3_step (stmt));
  sqlite3_finalize (stmt);
  objstore_exec_or_fail (db, "RELEASE s2;");
  objstore_exec_or_fail (db, "COMMIT;");
  TEST_ASSERT_EQUAL_INT (1, objstore_exists_helper (db, id_bytes));

  objstore_close_ephemeral_db (db);
}

static void
test_staging_visibility_during_transaction (void)
{
  sqlite3 *db = objstore_open_ephemeral_db ();
  static const uint8_t kIdBytes[OBJSTORE_ID_SIZE] = {
    0x01, 0x02, 0x03, 0x04, 0xAA, 0xBB, 0xCC, 0xDD, 0x10, 0x20, 0x30,
    0x40, 0x50, 0x60, 0x70, 0x80, 0x81, 0x82, 0x83, 0x84, 0x85, 0x86,
    0x87, 0x88, 0x89, 0x8A, 0x8B, 0x8C, 0x8D, 0x8E, 0x8F, 0x90,
  };
  const uint8_t payload[] = { 0x11, 0x22, 0x33, 0x44 };

  sqlite3_stmt *stmt = NULL;
  objstore_exec_or_fail (db, "BEGIN;");
  TEST_ASSERT_EQUAL_INT (
      SQLITE_OK,
      sqlite3_prepare_v2 (db, "INSERT INTO objstore(id, data) VALUES(?1, ?2);",
                          -1, &stmt, NULL));
  TEST_ASSERT_EQUAL_INT (
      SQLITE_OK,
      sqlite3_bind_blob (stmt, 1, kIdBytes, OBJSTORE_ID_SIZE, SQLITE_STATIC));
  TEST_ASSERT_EQUAL_INT (SQLITE_OK, sqlite3_bind_blob (stmt, 2, payload,
                                                       (int)sizeof (payload),
                                                       SQLITE_STATIC));
  TEST_ASSERT_EQUAL_INT (SQLITE_DONE, sqlite3_step (stmt));
  sqlite3_finalize (stmt);

  TEST_ASSERT_EQUAL_INT (1, objstore_exists_helper (db, kIdBytes));
  objstore_exec_or_fail (db, "COMMIT;");
  TEST_ASSERT_EQUAL_INT (1, objstore_exists_helper (db, kIdBytes));
  objstore_close_ephemeral_db (db);
}

static void
test_staging_rollback_discards_uncommitted_objects (void)
{
  sqlite3 *db = objstore_open_ephemeral_db ();
  static const uint8_t kIdBytes[OBJSTORE_ID_SIZE] = {
    0x10, 0x11, 0x12, 0x13, 0x21, 0x22, 0x23, 0x24, 0x31, 0x32, 0x33,
    0x34, 0x41, 0x42, 0x43, 0x44, 0xFF, 0xEE, 0xDD, 0xCC, 0xBC, 0xAD,
    0x9E, 0x8F, 0x7F, 0x6F, 0x5F, 0x4F, 0x3F, 0x2F, 0x1F, 0x0F,
  };
  const uint8_t payload[] = { 0x99, 0x88, 0x77, 0x66 };

  sqlite3_stmt *stmt = NULL;
  objstore_exec_or_fail (db, "BEGIN;");
  TEST_ASSERT_EQUAL_INT (
      SQLITE_OK,
      sqlite3_prepare_v2 (db, "INSERT INTO objstore(id, data) VALUES(?1, ?2);",
                          -1, &stmt, NULL));
  TEST_ASSERT_EQUAL_INT (
      SQLITE_OK,
      sqlite3_bind_blob (stmt, 1, kIdBytes, OBJSTORE_ID_SIZE, SQLITE_STATIC));
  TEST_ASSERT_EQUAL_INT (SQLITE_OK, sqlite3_bind_blob (stmt, 2, payload,
                                                       (int)sizeof (payload),
                                                       SQLITE_STATIC));
  TEST_ASSERT_EQUAL_INT (SQLITE_DONE, sqlite3_step (stmt));
  sqlite3_finalize (stmt);
  objstore_exec_or_fail (db, "ROLLBACK;");
  TEST_ASSERT_EQUAL_INT (0, objstore_exists_helper (db, kIdBytes));
  objstore_close_ephemeral_db (db);
}

static void
test_scan_snapshot_isolation (void)
{
  sqlite3 *db = objstore_open_ephemeral_db ();
  sqlite3_stmt *stmt = NULL;
  const uint8_t id1[OBJSTORE_ID_SIZE] = {
    0x0B, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0B,
  };
  const uint8_t id2[OBJSTORE_ID_SIZE] = {
    0x0C, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0C,
  };
  const uint8_t payload1[] = { 0xAB };
  const uint8_t payload2[] = { 0xCD };

  objstore_exec_or_fail (db, "BEGIN;");
  TEST_ASSERT_EQUAL_INT (
      SQLITE_OK,
      sqlite3_prepare_v2 (db, "INSERT INTO objstore(id, data) VALUES(?1, ?2);",
                          -1, &stmt, NULL));
  TEST_ASSERT_EQUAL_INT (
      SQLITE_OK,
      sqlite3_bind_blob (stmt, 1, id1, OBJSTORE_ID_SIZE, SQLITE_STATIC));
  TEST_ASSERT_EQUAL_INT (SQLITE_OK, sqlite3_bind_blob (stmt, 2, payload1,
                                                       (int)sizeof (payload1),
                                                       SQLITE_STATIC));
  TEST_ASSERT_EQUAL_INT (SQLITE_DONE, sqlite3_step (stmt));
  TEST_ASSERT_EQUAL_INT (SQLITE_OK, sqlite3_reset (stmt));
  TEST_ASSERT_EQUAL_INT (
      SQLITE_OK,
      sqlite3_bind_blob (stmt, 1, id2, OBJSTORE_ID_SIZE, SQLITE_STATIC));
  TEST_ASSERT_EQUAL_INT (SQLITE_OK, sqlite3_bind_blob (stmt, 2, payload2,
                                                       (int)sizeof (payload2),
                                                       SQLITE_STATIC));
  TEST_ASSERT_EQUAL_INT (SQLITE_DONE, sqlite3_step (stmt));
  sqlite3_finalize (stmt);
  objstore_exec_or_fail (db, "COMMIT;");

  objstore_exec_or_fail (db, "BEGIN;");
  sqlite3_stmt *scan = NULL;
  TEST_ASSERT_EQUAL_INT (
      SQLITE_OK,
      sqlite3_prepare_v2 (db, "SELECT id FROM objstore;", -1, &scan, NULL));

  TEST_ASSERT_EQUAL_INT (SQLITE_ROW, sqlite3_step (scan));
  TEST_ASSERT_EQUAL_MEMORY (id1, sqlite3_column_blob (scan, 0),
                            OBJSTORE_ID_SIZE);

  sqlite3_stmt *delete_stmt = NULL;
  TEST_ASSERT_EQUAL_INT (SQLITE_OK,
                         sqlite3_prepare_v2 (db, "SELECT objstore_delete(?1);",
                                             -1, &delete_stmt, NULL));
  TEST_ASSERT_EQUAL_INT (SQLITE_OK,
                         sqlite3_bind_blob (delete_stmt, 1, id2,
                                            OBJSTORE_ID_SIZE, SQLITE_STATIC));
  TEST_ASSERT_EQUAL_INT (SQLITE_ROW, sqlite3_step (delete_stmt));
  sqlite3_finalize (delete_stmt);

  TEST_ASSERT_EQUAL_INT (SQLITE_ROW, sqlite3_step (scan));
  TEST_ASSERT_EQUAL_MEMORY (id2, sqlite3_column_blob (scan, 0),
                            OBJSTORE_ID_SIZE);
  TEST_ASSERT_EQUAL_INT (SQLITE_DONE, sqlite3_step (scan));
  sqlite3_finalize (scan);
  objstore_exec_or_fail (db, "COMMIT;");

  objstore_close_ephemeral_db (db);
}

static void
test_scan_sees_post_commit_changes (void)
{
  sqlite3 *db = objstore_open_ephemeral_db ();
  sqlite3_stmt *stmt = NULL;
  const uint8_t id[OBJSTORE_ID_SIZE] = {
    0x0C, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0C,
  };
  const uint8_t payload[] = { 0xBC };

  objstore_exec_or_fail (db, "BEGIN;");
  TEST_ASSERT_EQUAL_INT (
      SQLITE_OK,
      sqlite3_prepare_v2 (db, "INSERT INTO objstore(id, data) VALUES(?1, ?2);",
                          -1, &stmt, NULL));
  TEST_ASSERT_EQUAL_INT (
      SQLITE_OK,
      sqlite3_bind_blob (stmt, 1, id, OBJSTORE_ID_SIZE, SQLITE_STATIC));
  TEST_ASSERT_EQUAL_INT (SQLITE_OK, sqlite3_bind_blob (stmt, 2, payload,
                                                       (int)sizeof (payload),
                                                       SQLITE_STATIC));
  TEST_ASSERT_EQUAL_INT (SQLITE_DONE, sqlite3_step (stmt));
  sqlite3_finalize (stmt);
  objstore_exec_or_fail (db, "COMMIT;");

  objstore_exec_or_fail (db, "BEGIN;");
  sqlite3_stmt *delete_stmt = NULL;
  TEST_ASSERT_EQUAL_INT (SQLITE_OK,
                         sqlite3_prepare_v2 (db, "SELECT objstore_delete(?1);",
                                             -1, &delete_stmt, NULL));
  TEST_ASSERT_EQUAL_INT (
      SQLITE_OK,
      sqlite3_bind_blob (delete_stmt, 1, id, OBJSTORE_ID_SIZE, SQLITE_STATIC));
  TEST_ASSERT_EQUAL_INT (SQLITE_ROW, sqlite3_step (delete_stmt));
  sqlite3_finalize (delete_stmt);
  objstore_exec_or_fail (db, "COMMIT;");

  sqlite3_stmt *scan = NULL;
  TEST_ASSERT_EQUAL_INT (
      SQLITE_OK,
      sqlite3_prepare_v2 (db, "SELECT id FROM objstore WHERE id = ?1;", -1,
                          &scan, NULL));
  TEST_ASSERT_EQUAL_INT (
      SQLITE_OK,
      sqlite3_bind_blob (scan, 1, id, OBJSTORE_ID_SIZE, SQLITE_STATIC));
  TEST_ASSERT_EQUAL_INT (SQLITE_DONE, sqlite3_step (scan));
  sqlite3_finalize (scan);

  objstore_close_ephemeral_db (db);
}

static void
test_vtab_best_index_rejects_data_constraints (void)
{
  sqlite3 *db = objstore_open_ephemeral_db ();
  sqlite3_stmt *stmt = NULL;
  int rc = sqlite3_prepare_v2 (
      db, "SELECT * FROM objstore WHERE data = zeroblob(1);", -1, &stmt, NULL);
  TEST_ASSERT_EQUAL_INT (SQLITE_ERROR, rc);
  if (stmt != NULL)
    {
      sqlite3_finalize (stmt);
    }
  objstore_close_ephemeral_db (db);
}

static void
test_vtab_best_index_rejects_non_eq_id_constraints (void)
{
  sqlite3 *db = objstore_open_ephemeral_db ();
  sqlite3_stmt *stmt = NULL;
  int rc = sqlite3_prepare_v2 (
      db, "SELECT * FROM objstore WHERE id > zeroblob(32);", -1, &stmt, NULL);
  TEST_ASSERT_EQUAL_INT (SQLITE_ERROR, rc);
  if (stmt != NULL)
    {
      sqlite3_finalize (stmt);
    }
  objstore_close_ephemeral_db (db);
}

static void
test_vtab_update_rejects_mutation (void)
{
  sqlite3 *db = objstore_open_ephemeral_db ();
  objstore_exec_or_fail (db, "INSERT INTO objstore(data) VALUES(zeroblob(1));");
  char *errmsg = NULL;
  int rc = sqlite3_exec (db, "UPDATE objstore SET data = zeroblob(2);", NULL,
                         NULL, &errmsg);
  TEST_ASSERT_EQUAL_INT (SQLITE_ERROR, rc);
  if (errmsg != NULL)
    {
      TEST_ASSERT_NOT_NULL (strstr (errmsg, "immutable"));
      sqlite3_free (errmsg);
    }
  objstore_close_ephemeral_db (db);
}

void
vtab_crud_register_tests (void)
{
  RUN_TEST (test_virtual_table_basic_crud);
  RUN_TEST (test_virtual_table_transactions);
  RUN_TEST (test_savepoint_rollback_aborts_transaction);
  RUN_TEST (test_staging_visibility_during_transaction);
  RUN_TEST (test_staging_rollback_discards_uncommitted_objects);
  RUN_TEST (test_scan_snapshot_isolation);
  RUN_TEST (test_scan_sees_post_commit_changes);
  RUN_TEST (test_vtab_best_index_rejects_data_constraints);
  RUN_TEST (test_vtab_best_index_rejects_non_eq_id_constraints);
  RUN_TEST (test_vtab_update_rejects_mutation);
}
