#include <objstore/objstore.h>
#include <sqlite3.h>

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
      fprintf (stderr, "SQL error \"%s\": %s\n", sql,
               errmsg != NULL ? errmsg : "(null)");
      sqlite3_free (errmsg);
      fatal_sqlite (db, "exec_or_die", rc);
    }
}

static void
cache_put (sqlite3 *db, sqlite3_stmt *delete_stmt, sqlite3_stmt *insert_stmt,
           const char *key, const char *value, int ttl_seconds)
{
  sqlite3_reset (delete_stmt);
  sqlite3_clear_bindings (delete_stmt);
  sqlite3_bind_text (delete_stmt, 1, key, -1, SQLITE_STATIC);
  int rc = sqlite3_step (delete_stmt);
  if (rc != SQLITE_DONE)
    {
      fatal_sqlite (db, "cache_put delete", rc);
    }

  sqlite3_reset (insert_stmt);
  sqlite3_clear_bindings (insert_stmt);
  sqlite3_bind_text (insert_stmt, 1, key, -1, SQLITE_STATIC);
  sqlite3_bind_blob (insert_stmt, 2, value, (int)strlen (value), SQLITE_STATIC);
  sqlite3_bind_int (insert_stmt, 3, ttl_seconds);
  rc = sqlite3_step (insert_stmt);
  if (rc != SQLITE_DONE)
    {
      fatal_sqlite (db, "cache_put", rc);
    }
}

static void
cache_get (sqlite3 *db, sqlite3_stmt *stmt, const char *key)
{
  sqlite3_reset (stmt);
  sqlite3_clear_bindings (stmt);
  sqlite3_bind_text (stmt, 1, key, -1, SQLITE_STATIC);
  int rc = sqlite3_step (stmt);
  if (rc == SQLITE_ROW)
    {
      const unsigned char *value = sqlite3_column_blob (stmt, 0);
      int size = sqlite3_column_bytes (stmt, 0);
      printf ("cache hit for %s -> %.*s\n", key, size,
              value != NULL ? (const char *)value : "");
    }
  else if (rc == SQLITE_DONE)
    {
      printf ("cache miss for %s\n", key);
    }
  else
    {
      fatal_sqlite (db, "cache_get", rc);
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
      "CREATE TABLE cache_entries ("
      "  cache_key TEXT PRIMARY KEY,"
      "  obj_id BLOB NOT NULL REFERENCES objstore(id),"
      "  expires_at INTEGER NOT NULL"
      ");");

  sqlite3_stmt *delete_stmt = NULL;
  rc = sqlite3_prepare_v2 (
      db, "DELETE FROM cache_entries WHERE cache_key = ?1;", -1, &delete_stmt,
      NULL);
  if (rc != SQLITE_OK)
    {
      fatal_sqlite (db, "prepare delete", rc);
    }

  sqlite3_stmt *insert_stmt = NULL;
  rc = sqlite3_prepare_v2 (
      db,
      "WITH new_value AS (SELECT objstore_put(?2) AS id)"
      "INSERT INTO cache_entries(cache_key, obj_id, expires_at)"
      "SELECT ?1, id, strftime('%s','now') + ?3 FROM new_value"
      ";",
      -1, &insert_stmt, NULL);
  if (rc != SQLITE_OK)
    {
      fatal_sqlite (db, "prepare put", rc);
    }

  sqlite3_stmt *get_stmt = NULL;
  rc = sqlite3_prepare_v2 (
      db,
      "SELECT objstore_get(obj_id)"
      "  FROM cache_entries"
      " WHERE cache_key = ?1"
      "   AND expires_at >= strftime('%s','now');",
      -1, &get_stmt, NULL);
  if (rc != SQLITE_OK)
    {
      fatal_sqlite (db, "prepare get", rc);
    }

  cache_put (db, delete_stmt, insert_stmt, "weather:nyc", "sunny", 3600);
  cache_get (db, get_stmt, "weather:nyc");

  cache_put (db, delete_stmt, insert_stmt, "weather:nyc", "light rain", 3600);
  cache_get (db, get_stmt, "weather:nyc");

  cache_get (db, get_stmt, "weather:sfo");

  sqlite3_finalize (delete_stmt);
  sqlite3_finalize (insert_stmt);
  sqlite3_finalize (get_stmt);
  sqlite3_close (db);
  return 0;
}

