#include <objstore/objstore.h>
#include <sqlite3.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void
fatal_sqlite (sqlite3 *db, const char *msg, int rc)
{
  fprintf (stderr, "%s: %s (rc=%d)\n", msg,
           db != NULL ? sqlite3_errmsg (db) : "(null)", rc);
  sqlite3_close (db);
  exit (EXIT_FAILURE);
}

static void
exec_or_die (sqlite3 *db, const char *sql)
{
  char *errmsg = NULL;
  int rc = sqlite3_exec (db, sql, NULL, NULL, &errmsg);
  if (rc != SQLITE_OK)
    {
      fprintf (stderr, "SQL error while running \"%s\": %s\n", sql,
               errmsg != NULL ? errmsg : "(null)");
      sqlite3_free (errmsg);
      fatal_sqlite (db, "exec_or_die", rc);
    }
}

int
main (void)
{
  sqlite3 *db = NULL;
  int rc = sqlite3_open (":memory:", &db);
  if (rc != SQLITE_OK)
    {
      fatal_sqlite (db, "sqlite3_open", rc);
    }

  rc = objstore_register (db, NULL);
  if (rc != SQLITE_OK)
    {
      fatal_sqlite (db, "objstore_register", rc);
    }

  exec_or_die (db, "CREATE VIRTUAL TABLE objstore USING objstore();");
  exec_or_die (
      db,
      "CREATE TABLE files ("
      "  id BLOB PRIMARY KEY REFERENCES objstore(id),"
      "  filename TEXT NOT NULL,"
      "  size INTEGER NOT NULL,"
      "  created_at INTEGER NOT NULL"
      ");");

  const char *filename = "demo.txt";
  const unsigned char payload[] = "Hello from sqlite-objstore!";
  sqlite3_stmt *insert_stmt = NULL;
  rc = sqlite3_prepare_v2 (
      db,
      "WITH new_file AS ("
      "    SELECT objstore_put(?1) AS id, length(?1) AS size"
      ")"
      "INSERT INTO files (id, filename, size, created_at)"
      "SELECT id, ?2, size, strftime('%s','now') FROM new_file;",
      -1, &insert_stmt, NULL);
  if (rc != SQLITE_OK)
    {
      fatal_sqlite (db, "prepare insert", rc);
    }
  sqlite3_bind_blob (insert_stmt, 1, payload, (int)strlen ((const char *)payload),
                     SQLITE_STATIC);
  sqlite3_bind_text (insert_stmt, 2, filename, -1, SQLITE_STATIC);
  rc = sqlite3_step (insert_stmt);
  if (rc != SQLITE_DONE)
    {
      fatal_sqlite (db, "insert file", rc);
    }
  sqlite3_finalize (insert_stmt);

  sqlite3_stmt *select_stmt = NULL;
  rc = sqlite3_prepare_v2 (
      db,
      "SELECT filename, size, objstore_get(id)"
      "FROM files"
      " WHERE filename = ?1;",
      -1, &select_stmt, NULL);
  if (rc != SQLITE_OK)
    {
      fatal_sqlite (db, "prepare select", rc);
    }
  sqlite3_bind_text (select_stmt, 1, filename, -1, SQLITE_STATIC);

  if ((rc = sqlite3_step (select_stmt)) == SQLITE_ROW)
    {
      const unsigned char *name = sqlite3_column_text (select_stmt, 0);
      sqlite3_int64 size = sqlite3_column_int64 (select_stmt, 1);
      const unsigned char *data = sqlite3_column_blob (select_stmt, 2);
      int bytes = sqlite3_column_bytes (select_stmt, 2);
      printf ("Restored %s (%lld bytes): %.*s\n", name,
              (long long)size, bytes, data != NULL ? (const char *)data : "");
    }
  else
    {
      fatal_sqlite (db, "fetch metadata", rc);
    }
  sqlite3_finalize (select_stmt);

  sqlite3_close (db);
  return 0;
}

