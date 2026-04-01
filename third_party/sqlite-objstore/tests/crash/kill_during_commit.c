// Copyright 2024 sqlite-objstore
// SPDX-License-Identifier: Apache-2.0

#include <errno.h>
#include <signal.h>
#include <sqlite3.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include "objstore/objstore.h"

static void
fatalf (const char *msg)
{
  perror (msg);
  exit (EXIT_FAILURE);
}

static char *
create_temp_db_path (void)
{
  char tmpl[] = "/tmp/objstore-crash-dbXXXXXX.sqlite3";
  int fd = mkstemps (tmpl, 8); /* keep .sqlite3 suffix */
  if (fd < 0)
    {
      fatalf ("mkstemps");
    }
  close (fd);
  unlink (tmpl);
  char *path = strdup (tmpl);
  if (path == NULL)
    {
      fatalf ("strdup");
    }
  return path;
}

static void
configure_objstore (sqlite3 *db)
{
  objstore_config cfg = {
    .backend = OBJSTORE_BACKEND_SQLITE,
    .storage_root = NULL,
    .chunk_size_bytes = 0,
    .shard_width = 0,
    .sync_mode = OBJSTORE_SYNC_FULL,
    .reserved_flags = 0,
  };
  if (objstore_register (db, &cfg) != SQLITE_OK)
    {
      fprintf (stderr, "objstore_register failed: %s\n", sqlite3_errmsg (db));
      exit (EXIT_FAILURE);
    }
  char *errmsg = NULL;
  int rc
      = sqlite3_exec (db, "CREATE VIRTUAL TABLE IF NOT EXISTS objstore USING "
                           "objstore();",
                      NULL, NULL, &errmsg);
  if (rc != SQLITE_OK)
    {
      fprintf (stderr, "CREATE VIRTUAL TABLE failed: %s\n",
               errmsg ? errmsg : "(null)");
      sqlite3_free (errmsg);
      exit (EXIT_FAILURE);
    }
  sqlite3_free (errmsg);
}

static void
child_stage_objects (const char *db_path)
{
  sqlite3 *db = NULL;
  if (sqlite3_open (db_path, &db) != SQLITE_OK)
    {
      fprintf (stderr, "child sqlite_open failed: %s\n",
               sqlite3_errmsg (db));
      _exit (EXIT_FAILURE);
    }
  configure_objstore (db);
  if (sqlite3_exec (db, "BEGIN IMMEDIATE;", NULL, NULL, NULL) != SQLITE_OK)
    {
      fprintf (stderr, "child BEGIN failed: %s\n", sqlite3_errmsg (db));
      _exit (EXIT_FAILURE);
    }

  sqlite3_stmt *stmt = NULL;
  if (sqlite3_prepare_v2 (db, "INSERT INTO objstore(data) VALUES (?1);", -1,
                          &stmt, NULL)
      != SQLITE_OK)
    {
      fprintf (stderr, "prepare insert failed: %s\n", sqlite3_errmsg (db));
      _exit (EXIT_FAILURE);
    }

  const sqlite3_uint64 payload_size = 2 * 1024 * 1024; /* 2 MiB */
  const int insert_count = 32;
  for (int i = 0; i < insert_count; ++i)
    {
      sqlite3_reset (stmt);
      sqlite3_clear_bindings (stmt);
      if (sqlite3_bind_zeroblob64 (stmt, 1, payload_size) != SQLITE_OK)
        {
          fprintf (stderr, "bind zeroblob failed: %s\n", sqlite3_errmsg (db));
          _exit (EXIT_FAILURE);
        }
      const int rc = sqlite3_step (stmt);
      if (rc != SQLITE_DONE)
        {
          fprintf (stderr, "insert step failed rc=%d: %s\n", rc,
                   sqlite3_errmsg (db));
          _exit (EXIT_FAILURE);
        }
    }
  sqlite3_finalize (stmt);

  /* Crash without committing to simulate power-loss mid-transaction. */
  fprintf (stderr, "child staged %d inserts; simulating crash\n", insert_count);
  kill (getpid (), SIGKILL);
  /* unreachable */
  _exit (EXIT_FAILURE);
}

static int
count_rows (sqlite3 *db, const char *sql)
{
  sqlite3_stmt *stmt = NULL;
  if (sqlite3_prepare_v2 (db, sql, -1, &stmt, NULL) != SQLITE_OK)
    {
      fprintf (stderr, "prepare failed for %s: %s\n", sql, sqlite3_errmsg (db));
      exit (EXIT_FAILURE);
    }
  int rc = sqlite3_step (stmt);
  if (rc != SQLITE_ROW)
    {
      fprintf (stderr, "unexpected step rc=%d for %s\n", rc, sql);
      exit (EXIT_FAILURE);
    }
  int count = sqlite3_column_int (stmt, 0);
  sqlite3_finalize (stmt);
  return count;
}

static void
validate_after_crash (const char *db_path)
{
  sqlite3 *db = NULL;
  if (sqlite3_open (db_path, &db) != SQLITE_OK)
    {
      fatalf ("sqlite_open validator");
    }
  configure_objstore (db);

  int objstore_rows = count_rows (db, "SELECT COUNT(*) FROM objstore;");
  int data_rows = count_rows (db, "SELECT COUNT(*) FROM objstore_data;");
  sqlite3_close (db);
  if (objstore_rows != 0 || data_rows != 0)
    {
      fprintf (stderr,
               "Validation failed: objstore rows=%d objstore_data rows=%d\n",
               objstore_rows, data_rows);
      exit (EXIT_FAILURE);
    }
  fprintf (stderr, "Validation succeeded: crash left no committed rows\n");
}

int
main (void)
{
  char *db_path = create_temp_db_path ();
  pid_t child = fork ();
  if (child < 0)
    {
      fatalf ("fork");
    }
  if (child == 0)
    {
      child_stage_objects (db_path);
    }

  int status = 0;
  if (waitpid (child, &status, 0) < 0)
    {
      fatalf ("waitpid");
    }
  if (!WIFSIGNALED (status) || WTERMSIG (status) != SIGKILL)
    {
      fprintf (stderr,
               "Child did not terminate via SIGKILL (status=%d). "
               "Test invalid.\n",
               status);
      return EXIT_FAILURE;
    }

  /* Validate that the SQLite backend did not commit any queued writes. */
  validate_after_crash (db_path);

  unlink (db_path);
  free (db_path);
  return EXIT_SUCCESS;
}

