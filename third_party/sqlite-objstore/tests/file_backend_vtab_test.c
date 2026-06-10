
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
test_objstore_pragmas_configure_file_backend (void)
{
  char *base = objstore_create_temp_root ("objstore-file-");
  TEST_ASSERT_NOT_NULL (base);
  char *objects_root = NULL;
  char *staging_root = NULL;
  objstore_prepare_file_roots (base, &objects_root, &staging_root);

  sqlite3 *db = NULL;
  TEST_ASSERT_EQUAL_INT (SQLITE_OK,
                         objstore_open_shared_memory_db (&db, "file-pragma"));
  objstore_config cfg = {
    .backend = OBJSTORE_BACKEND_FILE,
    .storage_root = NULL,
    .chunk_size_bytes = 0,
    .reserved_flags = 0,
  };
  TEST_ASSERT_EQUAL_INT (SQLITE_OK, objstore_register (db, &cfg));

  char *pragma_sql = sqlite3_mprintf ("PRAGMA objstore_storage_root='%s';"
                                      "PRAGMA objstore_shard_width=4;"
                                      "PRAGMA objstore_sync_mode='off';",
                                      objects_root);
  TEST_ASSERT_NOT_NULL (pragma_sql);
  TEST_ASSERT_EQUAL_INT (SQLITE_OK,
                         sqlite3_exec (db, pragma_sql, NULL, NULL, NULL));
  sqlite3_free (pragma_sql);

  objstore_exec_or_fail (db, "CREATE VIRTUAL TABLE store USING objstore();");
  objstore_exec_or_fail (db, "INSERT INTO store(data) VALUES(zeroblob(4));");

  static const uint8_t kZeros[4] = { 0, 0, 0, 0 };
  objstore_blake3 hash;
  objstore_blake3_init (&hash);
  objstore_blake3_update (&hash, kZeros, sizeof (kZeros));
  objstore_id expected_id;
  objstore_blake3_final (&hash, &expected_id);
  char hex[OBJSTORE_ID_SIZE * 2 + 1];
  objstore_id_to_hex (&expected_id, hex);

  char *expected_path = objstore_build_sharded_path (objects_root, hex, 4);
  TEST_ASSERT_NOT_NULL (expected_path);
  struct stat st;
  int stat_rc = stat (expected_path, &st);
  if (stat_rc != 0)
    {
      fprintf (stderr, "expected object path %s missing (errno=%d)\n",
               expected_path, errno);
    }
  TEST_ASSERT_EQUAL_INT (0, stat_rc);
  TEST_ASSERT_TRUE (S_ISREG (st.st_mode));
  sqlite3_free (expected_path);

  objstore_exec_or_fail (db, "DROP TABLE store;");
  sqlite3_close (db);

  TEST_ASSERT_EQUAL_INT (0, objstore_remove_tree (base));
  sqlite3_free (objects_root);
  sqlite3_free (staging_root);
  sqlite3_free (base);
}

static void
test_objstore_pragmas_validate_inputs (void)
{
  char *base = objstore_create_temp_root ("objstore-validate-");
  TEST_ASSERT_NOT_NULL (base);
  char *objects_root = NULL;
  char *staging_root = NULL;
  objstore_prepare_file_roots (base, &objects_root, &staging_root);

  sqlite3 *db = objstore_open_ephemeral_db ();
  objstore_config cfg = {
    .backend = OBJSTORE_BACKEND_FILE,
    .storage_root = objects_root,
    .chunk_size_bytes = 0,
    .reserved_flags = 0,
  };
  TEST_ASSERT_EQUAL_INT (SQLITE_OK, objstore_register (db, &cfg));
  TEST_ASSERT_EQUAL_INT (
      SQLITE_OK,
      sqlite3_exec (db, "PRAGMA objstore_sync_mode;", NULL, NULL, NULL));
  TEST_ASSERT_EQUAL_INT (
      SQLITE_AUTH,
      sqlite3_exec (db, "PRAGMA objstore_shard_width=3;", NULL, NULL, NULL));
  TEST_ASSERT_EQUAL_INT (
      SQLITE_AUTH, sqlite3_exec (db, "PRAGMA objstore_sync_mode='invalid';",
                                 NULL, NULL, NULL));
  sqlite3_close (db);
  TEST_ASSERT_EQUAL_INT (0, objstore_remove_tree (base));
  sqlite3_free (objects_root);
  sqlite3_free (staging_root);
  sqlite3_free (base);
}

static void
test_objstore_runtime_reconfigures_file_backend (void)
{
  char *base = NULL;
  char *objects_root = NULL;
  char *staging_root = NULL;
  sqlite3 *db = objstore_open_file_backend_db (&base, &objects_root,
                                               &staging_root, "file-runtime");

  objstore_exec_or_fail (db, "CREATE VIRTUAL TABLE store USING objstore();");
  objstore_exec_or_fail (db, "INSERT INTO store(data) VALUES(zeroblob(3));");

  objstore_exec_or_fail (db, "PRAGMA objstore_sync_mode='metadata';");
  objstore_exec_or_fail (db, "PRAGMA objstore_chunk_size=16384;");
  objstore_exec_or_fail (db, "PRAGMA objstore_shard_width=4;");

  objstore_exec_or_fail (db, "INSERT INTO store(data) VALUES(zeroblob(5));");

  objstore_id z5;
  objstore_compute_zero_payload_id (5, &z5);
  char hex[OBJSTORE_ID_SIZE * 2 + 1];
  objstore_id_to_hex (&z5, hex);
  char *path = objstore_build_sharded_path (objects_root, hex, 4);
  TEST_ASSERT_NOT_NULL (path);
  struct stat st;
  TEST_ASSERT_EQUAL_INT (0, stat (path, &st));
  TEST_ASSERT_TRUE (S_ISREG (st.st_mode));
  sqlite3_free (path);

  objstore_exec_or_fail (db, "DROP TABLE store;");
  sqlite3_close (db);
  TEST_ASSERT_EQUAL_INT (0, objstore_remove_tree (base));
  sqlite3_free (objects_root);
  sqlite3_free (staging_root);
  sqlite3_free (base);
}

static void
test_file_backend_vtab_commit_and_rollback (void)
{
  char *base = NULL;
  char *objects_root = NULL;
  char *staging_root = NULL;
  sqlite3 *db = objstore_open_file_backend_db (
      &base, &objects_root, &staging_root, "file-vtab-commit");
  objstore_exec_or_fail (db, "CREATE VIRTUAL TABLE store USING objstore();");

  objstore_exec_or_fail (db, "BEGIN;");
  objstore_exec_or_fail (db, "INSERT INTO store(data) VALUES(zeroblob(4));");
  objstore_exec_or_fail (db, "ROLLBACK;");

  objstore_id zero_id;
  objstore_compute_zero_payload_id (4, &zero_id);
  char hex[OBJSTORE_ID_SIZE * 2 + 1];
  objstore_id_to_hex (&zero_id, hex);
  char *path = objstore_build_sharded_path (objects_root, hex, 2);
  TEST_ASSERT_NOT_NULL (path);
  struct stat st;
  TEST_ASSERT_EQUAL_INT (-1, stat (path, &st));
  TEST_ASSERT_EQUAL_INT (ENOENT, errno);

  objstore_exec_or_fail (db, "BEGIN;");
  objstore_exec_or_fail (db, "INSERT INTO store(data) VALUES(zeroblob(4));");
  objstore_exec_or_fail (db, "COMMIT;");
  TEST_ASSERT_EQUAL_INT (0, stat (path, &st));
  TEST_ASSERT_TRUE (S_ISREG (st.st_mode));
  sqlite3_free (path);

  objstore_exec_or_fail (db, "DROP TABLE store;");
  sqlite3_close (db);
  TEST_ASSERT_EQUAL_INT (0, objstore_remove_tree (base));
  sqlite3_free (objects_root);
  sqlite3_free (staging_root);
  sqlite3_free (base);
}

static void
test_file_backend_vtab_recovers_pending_commit (void)
{
  char *base = objstore_create_temp_root ("objstore-vtab-recover-");
  TEST_ASSERT_NOT_NULL (base);
  char *objects_root = NULL;
  char *staging_root = NULL;
  objstore_prepare_file_roots (base, &objects_root, &staging_root);
  char *commit_root = sqlite3_mprintf ("%s/commit", staging_root);
  objstore_ensure_dir (commit_root);
  char *txn_dir = sqlite3_mprintf ("%s/crash", commit_root);
  objstore_ensure_dir (txn_dir);
  char *put_dir = sqlite3_mprintf ("%s/%s", txn_dir, "put");
  objstore_ensure_dir (put_dir);
  objstore_id id;
  objstore_compute_zero_payload_id (3, &id);
  char hex[OBJSTORE_ID_SIZE * 2 + 1];
  objstore_id_to_hex (&id, hex);
  char *payload_path = sqlite3_mprintf ("%s/%s%s", put_dir, hex, ".dat");
  FILE *payload = fopen (payload_path, "wb");
  TEST_ASSERT_NOT_NULL (payload);
  const uint8_t bytes[] = { 0xAA, 0xBB, 0xCC };
  fwrite (bytes, 1, sizeof (bytes), payload);
  fclose (payload);
  char *manifest_path = sqlite3_mprintf ("%s/%s", txn_dir, "manifest.log");
  FILE *manifest = fopen (manifest_path, "wb");
  TEST_ASSERT_NOT_NULL (manifest);
  fprintf (manifest, "PUT %s\n", hex);
  fclose (manifest);
  sqlite3_free (manifest_path);
  sqlite3_free (payload_path);
  sqlite3_free (put_dir);
  sqlite3_free (txn_dir);
  sqlite3_free (commit_root);

  sqlite3 *db = NULL;
  TEST_ASSERT_EQUAL_INT (
      SQLITE_OK, objstore_open_shared_memory_db (&db, "file-vtab-recover"));
  objstore_config cfg = {
    .backend = OBJSTORE_BACKEND_FILE,
    .storage_root = objects_root,
    .chunk_size_bytes = 0,
    .reserved_flags = 0,
  };
  TEST_ASSERT_EQUAL_INT (SQLITE_OK, objstore_register (db, &cfg));
  objstore_exec_or_fail (db, "CREATE VIRTUAL TABLE store USING objstore();");

  sqlite3_stmt *stmt = NULL;
  TEST_ASSERT_EQUAL_INT (
      SQLITE_OK,
      sqlite3_prepare_v2 (db, "SELECT COUNT(*) FROM store;", -1, &stmt, NULL));
  TEST_ASSERT_EQUAL_INT (SQLITE_ROW, sqlite3_step (stmt));
  TEST_ASSERT_EQUAL_INT (1, sqlite3_column_int (stmt, 0));
  sqlite3_finalize (stmt);

  objstore_exec_or_fail (db, "DROP TABLE store;");
  sqlite3_close (db);
  TEST_ASSERT_EQUAL_INT (0, objstore_remove_tree (base));
  sqlite3_free (objects_root);
  sqlite3_free (staging_root);
  sqlite3_free (base);
}

static void
test_file_backend_vtab_savepoint_rollback_rewinds_inner_frame (void)
{
  char *base = NULL;
  char *objects_root = NULL;
  char *staging_root = NULL;
  sqlite3 *db = objstore_open_file_backend_db (
      &base, &objects_root, &staging_root, "file-vtab-savepoint");
  objstore_exec_or_fail (db, "CREATE VIRTUAL TABLE store USING objstore();");

  const uint8_t outer_payload[] = { 0x11 };
  const uint8_t inner_payload[] = { 0x22 };
  const uint8_t after_payload[] = { 0x33 };
  const uint8_t outer_id[OBJSTORE_ID_SIZE] = {
    0x10, 0x11, 0x12, 0x13, 0x21, 0x22, 0x23, 0x24, 0x31, 0x32, 0x33,
    0x34, 0x41, 0x42, 0x43, 0x44, 0x51, 0x52, 0x53, 0x54, 0x61, 0x62,
    0x63, 0x64, 0x71, 0x72, 0x73, 0x74, 0x81, 0x82, 0x83, 0x84,
  };
  const uint8_t inner_id[OBJSTORE_ID_SIZE] = {
    0x20, 0x21, 0x22, 0x23, 0x31, 0x32, 0x33, 0x34, 0x41, 0x42, 0x43,
    0x44, 0x51, 0x52, 0x53, 0x54, 0x61, 0x62, 0x63, 0x64, 0x71, 0x72,
    0x73, 0x74, 0x81, 0x82, 0x83, 0x84, 0x91, 0x92, 0x93, 0x94,
  };
  const uint8_t after_id[OBJSTORE_ID_SIZE] = {
    0x30, 0x31, 0x32, 0x33, 0x41, 0x42, 0x43, 0x44, 0x51, 0x52, 0x53,
    0x54, 0x61, 0x62, 0x63, 0x64, 0x71, 0x72, 0x73, 0x74, 0x81, 0x82,
    0x83, 0x84, 0x91, 0x92, 0x93, 0x94, 0xA1, 0xA2, 0xA3, 0xA4,
  };

  sqlite3_stmt *stmt = NULL;
  objstore_exec_or_fail (db, "BEGIN;");

  TEST_ASSERT_EQUAL_INT (
      SQLITE_OK,
      sqlite3_prepare_v2 (db, "INSERT INTO store(id, data) VALUES(?1, ?2);",
                          -1, &stmt, NULL));
  TEST_ASSERT_EQUAL_INT (
      SQLITE_OK,
      sqlite3_bind_blob (stmt, 1, outer_id, OBJSTORE_ID_SIZE, SQLITE_STATIC));
  TEST_ASSERT_EQUAL_INT (
      SQLITE_OK,
      sqlite3_bind_blob (stmt, 2, outer_payload, (int)sizeof (outer_payload),
                         SQLITE_STATIC));
  TEST_ASSERT_EQUAL_INT (SQLITE_DONE, sqlite3_step (stmt));
  sqlite3_finalize (stmt);

  objstore_exec_or_fail (db, "SAVEPOINT s1;");

  TEST_ASSERT_EQUAL_INT (
      SQLITE_OK,
      sqlite3_prepare_v2 (db, "INSERT INTO store(id, data) VALUES(?1, ?2);",
                          -1, &stmt, NULL));
  TEST_ASSERT_EQUAL_INT (
      SQLITE_OK,
      sqlite3_bind_blob (stmt, 1, inner_id, OBJSTORE_ID_SIZE, SQLITE_STATIC));
  TEST_ASSERT_EQUAL_INT (
      SQLITE_OK,
      sqlite3_bind_blob (stmt, 2, inner_payload, (int)sizeof (inner_payload),
                         SQLITE_STATIC));
  TEST_ASSERT_EQUAL_INT (SQLITE_DONE, sqlite3_step (stmt));
  sqlite3_finalize (stmt);

  TEST_ASSERT_EQUAL_INT (1, objstore_exists_helper (db, outer_id));
  TEST_ASSERT_EQUAL_INT (1, objstore_exists_helper (db, inner_id));

  objstore_exec_or_fail (db, "ROLLBACK TO s1;");
  TEST_ASSERT_EQUAL_INT (1, objstore_exists_helper (db, outer_id));
  TEST_ASSERT_EQUAL_INT (0, objstore_exists_helper (db, inner_id));

  TEST_ASSERT_EQUAL_INT (
      SQLITE_OK,
      sqlite3_prepare_v2 (db, "INSERT INTO store(id, data) VALUES(?1, ?2);",
                          -1, &stmt, NULL));
  TEST_ASSERT_EQUAL_INT (
      SQLITE_OK,
      sqlite3_bind_blob (stmt, 1, after_id, OBJSTORE_ID_SIZE, SQLITE_STATIC));
  TEST_ASSERT_EQUAL_INT (
      SQLITE_OK,
      sqlite3_bind_blob (stmt, 2, after_payload, (int)sizeof (after_payload),
                         SQLITE_STATIC));
  TEST_ASSERT_EQUAL_INT (SQLITE_DONE, sqlite3_step (stmt));
  sqlite3_finalize (stmt);

  objstore_exec_or_fail (db, "RELEASE s1;");
  objstore_exec_or_fail (db, "COMMIT;");

  TEST_ASSERT_EQUAL_INT (1, objstore_exists_helper (db, outer_id));
  TEST_ASSERT_EQUAL_INT (0, objstore_exists_helper (db, inner_id));
  TEST_ASSERT_EQUAL_INT (1, objstore_exists_helper (db, after_id));

  objstore_exec_or_fail (db, "DROP TABLE store;");
  sqlite3_close (db);
  TEST_ASSERT_EQUAL_INT (0, objstore_remove_tree (base));
  sqlite3_free (objects_root);
  sqlite3_free (staging_root);
  sqlite3_free (base);
}

static void
test_file_backend_vtab_savepoint_rollback_restores_outer_delete_target (void)
{
  char *base = NULL;
  char *objects_root = NULL;
  char *staging_root = NULL;
  sqlite3 *db = objstore_open_file_backend_db (
      &base, &objects_root, &staging_root, "file-vtab-savepoint-delete");
  objstore_exec_or_fail (db, "CREATE VIRTUAL TABLE store USING objstore();");

  const uint8_t payload[] = { 0x91, 0x92, 0x93 };
  const uint8_t id[OBJSTORE_ID_SIZE] = {
    0xB1, 0xB2, 0xB3, 0xB4, 0x11, 0x12, 0x13, 0x14, 0x21, 0x22, 0x23,
    0x24, 0x31, 0x32, 0x33, 0x34, 0x41, 0x42, 0x43, 0x44, 0x51, 0x52,
    0x53, 0x54, 0x61, 0x62, 0x63, 0x64, 0x71, 0x72, 0x73, 0x74,
  };

  sqlite3_stmt *stmt = NULL;
  objstore_exec_or_fail (db, "BEGIN;");

  TEST_ASSERT_EQUAL_INT (
      SQLITE_OK,
      sqlite3_prepare_v2 (db, "INSERT INTO store(id, data) VALUES(?1, ?2);",
                          -1, &stmt, NULL));
  TEST_ASSERT_EQUAL_INT (
      SQLITE_OK,
      sqlite3_bind_blob (stmt, 1, id, OBJSTORE_ID_SIZE, SQLITE_STATIC));
  TEST_ASSERT_EQUAL_INT (
      SQLITE_OK,
      sqlite3_bind_blob (stmt, 2, payload, (int)sizeof (payload),
                         SQLITE_STATIC));
  TEST_ASSERT_EQUAL_INT (SQLITE_DONE, sqlite3_step (stmt));
  sqlite3_finalize (stmt);

  objstore_exec_or_fail (db, "SAVEPOINT s1;");
  TEST_ASSERT_EQUAL_INT (
      SQLITE_OK, sqlite3_prepare_v2 (db, "DELETE FROM store WHERE id = ?1;",
                                     -1, &stmt, NULL));
  TEST_ASSERT_EQUAL_INT (
      SQLITE_OK,
      sqlite3_bind_blob (stmt, 1, id, OBJSTORE_ID_SIZE, SQLITE_STATIC));
  TEST_ASSERT_EQUAL_INT (SQLITE_DONE, sqlite3_step (stmt));
  sqlite3_finalize (stmt);

  TEST_ASSERT_EQUAL_INT (0, objstore_exists_helper (db, id));

  objstore_exec_or_fail (db, "ROLLBACK TO s1;");
  TEST_ASSERT_EQUAL_INT (1, objstore_exists_helper (db, id));

  objstore_exec_or_fail (db, "RELEASE s1;");
  objstore_exec_or_fail (db, "COMMIT;");
  TEST_ASSERT_EQUAL_INT (1, objstore_exists_helper (db, id));

  objstore_exec_or_fail (db, "DROP TABLE store;");
  sqlite3_close (db);
  TEST_ASSERT_EQUAL_INT (0, objstore_remove_tree (base));
  sqlite3_free (objects_root);
  sqlite3_free (staging_root);
  sqlite3_free (base);
}

void
file_backend_vtab_register_tests (void)
{
  RUN_TEST (test_objstore_pragmas_configure_file_backend);
  RUN_TEST (test_objstore_pragmas_validate_inputs);
  RUN_TEST (test_objstore_runtime_reconfigures_file_backend);
  RUN_TEST (test_file_backend_vtab_commit_and_rollback);
  RUN_TEST (test_file_backend_vtab_recovers_pending_commit);
  RUN_TEST (test_file_backend_vtab_savepoint_rollback_rewinds_inner_frame);
  RUN_TEST (
      test_file_backend_vtab_savepoint_rollback_restores_outer_delete_target);
}
