#include "objstore/objstore.h"

#include <sqlite3.h>

#include <errno.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct wasm_fixture_result
{
  const char *fixture_id;
  uint64_t assertions;
  uint64_t failures;
  const char *error;
} wasm_fixture_result;

typedef struct wasm_runner_context
{
  wasm_fixture_result *result;
} wasm_runner_context;

static int
wasm_trace_stmt (unsigned trace_event, void *ctx, void *p, void *x)
{
  (void)ctx;
  (void)x;
  if (trace_event != SQLITE_TRACE_STMT)
    {
      return SQLITE_OK;
    }
  sqlite3_stmt *stmt = (sqlite3_stmt *)p;
  const char *sql = sqlite3_sql (stmt);
  if (sql != NULL)
    {
      fprintf (stderr, "[wasm-matrix] SQL: %s\n", sql);
    }
  return SQLITE_OK;
}
static void
json_escape_and_print (const char *key, const char *value)
{
  if (value == NULL)
    {
      printf ("\"%s\":null", key);
      return;
    }
  printf ("\"%s\":\"", key);
  for (const char *p = value; *p != '\0'; ++p)
    {
      switch (*p)
        {
        case '\"':
        case '\\':
          putchar ('\\');
          putchar (*p);
          break;
        case '\n':
          printf ("\\n");
          break;
        case '\r':
          printf ("\\r");
          break;
        case '\t':
          printf ("\\t");
          break;
        default:
          putchar (*p);
          break;
        }
    }
  putchar ('\"');
}

static bool
values_equal (sqlite3_value *lhs, sqlite3_value *rhs)
{
  if (lhs == NULL || rhs == NULL)
    {
      return lhs == rhs;
    }
  const int lhs_type = sqlite3_value_type (lhs);
  const int rhs_type = sqlite3_value_type (rhs);
  if (lhs_type == SQLITE_NULL && rhs_type == SQLITE_NULL)
    {
      return true;
    }
  if (lhs_type == SQLITE_BLOB || rhs_type == SQLITE_BLOB)
    {
      const int lhs_size = sqlite3_value_bytes (lhs);
      const int rhs_size = sqlite3_value_bytes (rhs);
      if (lhs_size != rhs_size)
        {
          return false;
        }
      const void *lhs_blob = sqlite3_value_blob (lhs);
      const void *rhs_blob = sqlite3_value_blob (rhs);
      return lhs_blob != NULL && rhs_blob != NULL
             && memcmp (lhs_blob, rhs_blob, (size_t)lhs_size) == 0;
    }
  if (lhs_type == SQLITE_FLOAT || rhs_type == SQLITE_FLOAT)
    {
      return sqlite3_value_double (lhs) == sqlite3_value_double (rhs);
    }
  if (lhs_type == SQLITE_INTEGER || rhs_type == SQLITE_INTEGER)
    {
      return sqlite3_value_int64 (lhs) == sqlite3_value_int64 (rhs);
    }
  const unsigned char *lhs_text = sqlite3_value_text (lhs);
  const unsigned char *rhs_text = sqlite3_value_text (rhs);
  if (lhs_text == NULL || rhs_text == NULL)
    {
      return lhs_text == rhs_text;
    }
  return strcmp ((const char *)lhs_text, (const char *)rhs_text) == 0;
}

static void
wasm_expect_fn (sqlite3_context *ctx, int argc, sqlite3_value **argv)
{
  wasm_fixture_result *result = (wasm_fixture_result *)sqlite3_user_data (ctx);
  if (result == NULL || argc != 3)
    {
      sqlite3_result_error_code (ctx, SQLITE_MISUSE);
      return;
    }
  result->assertions++;
  sqlite3_value *label_val = argv[0];
  sqlite3_value *actual = argv[1];
  sqlite3_value *expected = argv[2];
  if (!values_equal (actual, expected))
    {
      result->failures++;
      if (result->error == NULL)
        {
          const unsigned char *label = sqlite3_value_text (label_val);
          if (label != NULL)
            {
              result->error = (const char *)label;
            }
          else
            {
              result->error = "assertion_failed";
            }
        }
      sqlite3_result_int (ctx, 0);
      return;
    }
  sqlite3_result_int (ctx, 1);
}

static char *
read_file_contents (const char *path, size_t *out_size)
{
  FILE *fp = fopen (path, "rb");
  if (fp == NULL)
    {
      return NULL;
    }
  if (fseek (fp, 0, SEEK_END) != 0)
    {
      fclose (fp);
      return NULL;
    }
  long len = ftell (fp);
  if (len < 0)
    {
      fclose (fp);
      return NULL;
    }
  if (fseek (fp, 0, SEEK_SET) != 0)
    {
      fclose (fp);
      return NULL;
    }
  char *buffer = (char *)malloc ((size_t)len + 1);
  if (buffer == NULL)
    {
      fclose (fp);
      return NULL;
    }
  size_t read_total = fread (buffer, 1, (size_t)len, fp);
  fclose (fp);
  buffer[read_total] = '\0';
  if (out_size != NULL)
    {
      *out_size = read_total;
    }
  return buffer;
}

static void
emit_result_json (const wasm_fixture_result *result)
{
  const bool success = (result->failures == 0) && (result->error == NULL);
  printf ("{");
  printf ("\"fixture\":\"%s\",", result->fixture_id ? result->fixture_id : "");
  printf ("\"status\":\"%s\",", success ? "ok" : "failed");
  printf ("\"assertions\":%llu,", (unsigned long long)result->assertions);
  printf ("\"failures\":%llu", (unsigned long long)result->failures);
  if (result->error != NULL)
    {
      printf (",");
      json_escape_and_print ("error", result->error);
    }
  printf ("}\n");
}

int
main (int argc, char **argv)
{
  const char *bundle_root = "/bundle";
  const char *fixture_rel = NULL;
  const char *fixture_id = NULL;

  for (int i = 1; i < argc; ++i)
    {
      if (strcmp (argv[i], "--bundle-root") == 0 && (i + 1) < argc)
        {
          bundle_root = argv[++i];
        }
      else if (strcmp (argv[i], "--fixture") == 0 && (i + 1) < argc)
        {
          fixture_rel = argv[++i];
        }
      else if (strcmp (argv[i], "--fixture-id") == 0 && (i + 1) < argc)
        {
          fixture_id = argv[++i];
        }
    }

  wasm_fixture_result result = {
    .fixture_id = fixture_id != NULL ? fixture_id : "(unknown)",
    .assertions = 0,
    .failures = 0,
    .error = NULL,
  };

  if (fixture_rel == NULL)
    {
      result.error = "missing --fixture argument";
      result.failures = 1;
      emit_result_json (&result);
      return EXIT_FAILURE;
    }

  size_t full_len = strlen (bundle_root) + 1 + strlen (fixture_rel) + 1;
  char *full_path = (char *)malloc (full_len);
  if (full_path == NULL)
    {
      result.error = "out_of_memory";
      result.failures = 1;
      emit_result_json (&result);
      return EXIT_FAILURE;
    }
  snprintf (full_path, full_len, "%s/%s", bundle_root, fixture_rel);

  size_t sql_size = 0;
  char *sql = read_file_contents (full_path, &sql_size);
  free (full_path);
  if (sql == NULL)
    {
      result.error = "unable_to_read_fixture";
      result.failures = 1;
      emit_result_json (&result);
      return EXIT_FAILURE;
    }

  sqlite3 *db = NULL;
  int rc = sqlite3_open (":memory:", &db);
  if (rc != SQLITE_OK)
    {
      free (sql);
      result.error = "sqlite_open_failed";
      result.failures = 1;
      emit_result_json (&result);
      return EXIT_FAILURE;
    }

  objstore_config cfg = {
    .backend = OBJSTORE_BACKEND_AUTO,
    .storage_root = NULL,
    .chunk_size_bytes = 0,
    .shard_width = 0,
    .sync_mode = OBJSTORE_SYNC_FULL,
    .reserved_flags = 0,
  };

  rc = objstore_register (db, &cfg);
  const char *trace_env = getenv ("OBJSTORE_WASM_TRACE");
  if (trace_env != NULL && trace_env[0] != '\0')
    {
      sqlite3_trace_v2 (db, SQLITE_TRACE_STMT, wasm_trace_stmt, NULL);
    }

  if (rc != SQLITE_OK)
    {
      sqlite3_close (db);
      free (sql);
      result.error = "objstore_register_failed";
      result.failures = 1;
      emit_result_json (&result);
      return EXIT_FAILURE;
    }

  rc = sqlite3_create_function_v2 (db, "wasm_expect", 3,
                                   SQLITE_UTF8 | SQLITE_DETERMINISTIC, &result,
                                   wasm_expect_fn, NULL, NULL, NULL);
  if (rc != SQLITE_OK)
    {
      sqlite3_close (db);
      free (sql);
      result.error = "wasm_expect_registration_failed";
      result.failures = 1;
      emit_result_json (&result);
      return EXIT_FAILURE;
    }

  char *errmsg = NULL;
  rc = sqlite3_exec (db, sql, NULL, NULL, &errmsg);
  free (sql);
  if (rc != SQLITE_OK)
    {
      char *errmsg_copy = errmsg;
      result.error = errmsg_copy != NULL ? errmsg_copy : "sqlite_exec_failed";
      result.failures = 1;
      sqlite3_close (db);
      emit_result_json (&result);
      if (errmsg_copy != NULL)
        {
          sqlite3_free (errmsg_copy);
        }
      return EXIT_FAILURE;
    }

  sqlite3_close (db);
  emit_result_json (&result);
  return (result.failures == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}
