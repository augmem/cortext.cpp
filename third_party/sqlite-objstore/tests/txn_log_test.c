#include "unity.h"

#include <stddef.h>
#include <sqlite3.h>
#include <string.h>

#include "objstore/backend.h"
#include "objstore/txn.h"
#include "test_harness.h"
#include "test_support.h"

static objstore_id
txn_make_id (uint8_t seed)
{
  objstore_id id;
  for (size_t i = 0; i < OBJSTORE_ID_SIZE; ++i)
    {
      id.bytes[i] = (uint8_t)(seed + i);
    }
  return id;
}

static objstore_txn_log *
txn_log_create (sqlite3 **out_db)
{
  sqlite3 *db = objstore_open_ephemeral_db ();
  objstore_txn_log *log = NULL;
  objstore_config cfg = {
    .backend = OBJSTORE_BACKEND_SQLITE,
    .storage_root = NULL,
    .chunk_size_bytes = 0,
    .reserved_flags = 0,
  };
  TEST_ASSERT_EQUAL_INT (SQLITE_OK,
                         objstore_txn_log_create (db, &cfg, &log));
  if (out_db != NULL)
    {
      *out_db = db;
    }
  return log;
}

static void
test_txn_log_tracks_state_by_sequence (void)
{
  sqlite3 *db = NULL;
  objstore_txn_log *log = txn_log_create (&db);
  objstore_id id = txn_make_id (0x10);

  TEST_ASSERT_EQUAL_INT (SQLITE_OK,
                         objstore_txn_log_append_put (log, &id, 4));
  TEST_ASSERT_EQUAL_INT (OBJSTORE_TXN_LOOKUP_NONE,
                         objstore_txn_log_state_for_id (log, &id, 0));
  TEST_ASSERT_EQUAL_INT (OBJSTORE_TXN_LOOKUP_PUT,
                         objstore_txn_log_state_for_id (log, &id, 1));

  TEST_ASSERT_EQUAL_INT (SQLITE_OK,
                         objstore_txn_log_append_delete (log, &id));
  TEST_ASSERT_EQUAL_INT (OBJSTORE_TXN_LOOKUP_PUT,
                         objstore_txn_log_state_for_id (log, &id, 1));
  TEST_ASSERT_EQUAL_INT (OBJSTORE_TXN_LOOKUP_DELETE,
                         objstore_txn_log_state_for_id (log, &id, 2));

  objstore_txn_log_destroy (log);
  objstore_close_ephemeral_db (db);
}

static void
test_txn_log_snapshots_filter_by_sequence (void)
{
  sqlite3 *db = NULL;
  objstore_txn_log *log = txn_log_create (&db);
  objstore_id id1 = txn_make_id (0x01);
  objstore_id id2 = txn_make_id (0x02);

  TEST_ASSERT_EQUAL_INT (SQLITE_OK,
                         objstore_txn_log_append_put (log, &id1, 8));
  TEST_ASSERT_EQUAL_INT (SQLITE_OK,
                         objstore_txn_log_append_put (log, &id2, 16));

  objstore_txn_snapshot *snapshot = objstore_txn_snapshot_build (log, 1);
  TEST_ASSERT_NOT_NULL (snapshot);
  TEST_ASSERT_EQUAL_size_t (1, objstore_txn_snapshot_count (snapshot));
  const objstore_txn_entry *entry
      = objstore_txn_snapshot_entry (snapshot, 0);
  TEST_ASSERT_NOT_NULL (entry);
  TEST_ASSERT_EQUAL_MEMORY (id1.bytes, entry->id.bytes, OBJSTORE_ID_SIZE);
  TEST_ASSERT_TRUE (objstore_txn_snapshot_contains (snapshot, &id1));
  TEST_ASSERT_FALSE (objstore_txn_snapshot_contains (snapshot, &id2));
  objstore_txn_snapshot_destroy (snapshot);

  snapshot = objstore_txn_snapshot_build (log, 2);
  TEST_ASSERT_NOT_NULL (snapshot);
  TEST_ASSERT_EQUAL_size_t (2, objstore_txn_snapshot_count (snapshot));
  objstore_txn_snapshot_destroy (snapshot);

  objstore_txn_log_destroy (log);
  objstore_close_ephemeral_db (db);
}

static void
test_txn_log_savepoint_frames (void)
{
  sqlite3 *db = NULL;
  objstore_txn_log *log = txn_log_create (&db);
  objstore_id id = txn_make_id (0x30);

  TEST_ASSERT_EQUAL_INT (SQLITE_OK, objstore_txn_log_begin_frame (log));
  TEST_ASSERT_EQUAL_INT (SQLITE_OK,
                         objstore_txn_log_append_put (log, &id, 1));
  TEST_ASSERT_EQUAL_size_t (1, objstore_txn_log_entry_count (log));
  TEST_ASSERT_EQUAL_INT (SQLITE_OK, objstore_txn_log_rollback_frame (log));
  TEST_ASSERT_EQUAL_size_t (0, objstore_txn_log_entry_count (log));

  TEST_ASSERT_EQUAL_INT (SQLITE_OK, objstore_txn_log_begin_frame (log));
  TEST_ASSERT_EQUAL_INT (SQLITE_OK,
                         objstore_txn_log_append_put (log, &id, 2));
  TEST_ASSERT_EQUAL_INT (SQLITE_OK, objstore_txn_log_release_frame (log));
  TEST_ASSERT_EQUAL_size_t (1, objstore_txn_log_entry_count (log));

  objstore_txn_log_destroy (log);
  objstore_close_ephemeral_db (db);
}

static void
test_txn_log_snapshot_preserves_order_across_delete_reinsert (void)
{
  sqlite3 *db = NULL;
  objstore_txn_log *log = txn_log_create (&db);
  objstore_id id1 = txn_make_id (0x60);
  objstore_id id2 = txn_make_id (0x70);

  TEST_ASSERT_EQUAL_INT (SQLITE_OK,
                         objstore_txn_log_append_put (log, &id1, 10));
  TEST_ASSERT_EQUAL_INT (SQLITE_OK,
                         objstore_txn_log_append_put (log, &id2, 20));
  TEST_ASSERT_EQUAL_INT (SQLITE_OK,
                         objstore_txn_log_append_delete (log, &id1));
  TEST_ASSERT_EQUAL_INT (SQLITE_OK,
                         objstore_txn_log_append_put (log, &id1, 30));

  objstore_txn_snapshot *snapshot = objstore_txn_snapshot_build (log, 4);
  TEST_ASSERT_NOT_NULL (snapshot);
  TEST_ASSERT_EQUAL_size_t (2, objstore_txn_snapshot_count (snapshot));

  const objstore_txn_entry *first = objstore_txn_snapshot_entry (snapshot, 0);
  const objstore_txn_entry *second = objstore_txn_snapshot_entry (snapshot, 1);
  TEST_ASSERT_NOT_NULL (first);
  TEST_ASSERT_NOT_NULL (second);
  TEST_ASSERT_EQUAL_MEMORY (id2.bytes, first->id.bytes, OBJSTORE_ID_SIZE);
  TEST_ASSERT_EQUAL_INT (20, first->payload_size);
  TEST_ASSERT_EQUAL_MEMORY (id1.bytes, second->id.bytes, OBJSTORE_ID_SIZE);
  TEST_ASSERT_EQUAL_INT (30, second->payload_size);

  objstore_txn_snapshot_destroy (snapshot);
  objstore_txn_log_destroy (log);
  objstore_close_ephemeral_db (db);
}

static void
test_txn_log_drop_last_entry (void)
{
  sqlite3 *db = NULL;
  objstore_txn_log *log = txn_log_create (&db);
  objstore_id id1 = txn_make_id (0x40);
  objstore_id id2 = txn_make_id (0x50);

  TEST_ASSERT_EQUAL_INT (SQLITE_OK,
                         objstore_txn_log_append_put (log, &id1, 5));
  TEST_ASSERT_EQUAL_INT (SQLITE_OK,
                         objstore_txn_log_append_put (log, &id2, 6));
  TEST_ASSERT_EQUAL_size_t (2, objstore_txn_log_entry_count (log));

  TEST_ASSERT_EQUAL_INT (SQLITE_OK, objstore_txn_log_drop_last (log));
  TEST_ASSERT_EQUAL_size_t (1, objstore_txn_log_entry_count (log));
  TEST_ASSERT_EQUAL_MEMORY (
      id1.bytes,
      objstore_txn_log_entry_at (log, 0)->id.bytes,
      OBJSTORE_ID_SIZE);

  TEST_ASSERT_EQUAL_INT (SQLITE_OK, objstore_txn_log_drop_last (log));
  TEST_ASSERT_EQUAL_size_t (0, objstore_txn_log_entry_count (log));

  objstore_txn_log_destroy (log);
  objstore_close_ephemeral_db (db);
}

void
txn_log_register_tests (void)
{
  RUN_TEST (test_txn_log_tracks_state_by_sequence);
  RUN_TEST (test_txn_log_snapshots_filter_by_sequence);
  RUN_TEST (test_txn_log_savepoint_frames);
  RUN_TEST (test_txn_log_snapshot_preserves_order_across_delete_reinsert);
  RUN_TEST (test_txn_log_drop_last_entry);
}
