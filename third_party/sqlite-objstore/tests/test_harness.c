#if !defined(_WIN32)
#define _POSIX_C_SOURCE 200809L
#endif

#include "test_harness.h"

#include <ctype.h>
#include <dirent.h>
#include <errno.h>
#include <limits.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "objstore/blake3.h"
#include "objstore/objstore.h"
#include "test_support.h"
#include "unity.h"

#ifndef PATH_MAX
#define PATH_MAX 4096
#endif

static objstore_fixture_kind g_fixture_kind = OBJSTORE_FIXTURE_SMOKE;
static sqlite3 *g_sqlite_backend_db = NULL;
static const objstore_backend *g_sqlite_backend = NULL;
static objstore_backend_env *g_sqlite_backend_env = NULL;

int
objstore_open_shared_memory_db (sqlite3 **out_db, const char *label)
{
  if (out_db == NULL)
    {
      return SQLITE_MISUSE;
    }
  static int counter = 0;
  const char *tag = (label != NULL) ? label : "smoke";
  char path[PATH_MAX];
  sqlite3_snprintf (sizeof (path), path, "/tmp/objstore_%s_%d.sqlite3", tag,
                    counter++);
  const int flags
      = SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE | SQLITE_OPEN_SHAREDCACHE;
  unlink (path);
  return sqlite3_open_v2 (path, out_db, flags, NULL);
}

void
objstore_tests_set_fixture (objstore_fixture_kind kind)
{
  g_fixture_kind = kind;
}

const objstore_backend *
objstore_tests_sqlite_backend (void)
{
  return g_sqlite_backend;
}

objstore_backend_env *
objstore_tests_backend_env (void)
{
  return g_sqlite_backend_env;
}

sqlite3 *
objstore_tests_sqlite_db (void)
{
  return g_sqlite_backend_db;
}

void
setUp (void)
{
  if (g_fixture_kind == OBJSTORE_FIXTURE_SQLITE_BACKEND)
    {
      TEST_ASSERT_EQUAL_INT (
          SQLITE_OK, objstore_open_shared_memory_db (&g_sqlite_backend_db,
                                                     "fixture"));
      g_sqlite_backend = objstore_backend_by_kind (OBJSTORE_BACKEND_SQLITE);
      TEST_ASSERT_NOT_NULL (g_sqlite_backend);
      objstore_config cfg = {
        .backend = OBJSTORE_BACKEND_SQLITE,
        .storage_root = NULL,
        .chunk_size_bytes = 0,
        .reserved_flags = 0,
      };
      TEST_ASSERT_EQUAL_INT (
          SQLITE_OK, g_sqlite_backend->open_env (g_sqlite_backend_db, &cfg,
                                                 &g_sqlite_backend_env));
    }
}

void
tearDown (void)
{
  if (g_fixture_kind == OBJSTORE_FIXTURE_SQLITE_BACKEND)
    {
      if (g_sqlite_backend && g_sqlite_backend_env)
        {
          g_sqlite_backend->close_env (g_sqlite_backend_env);
        }
      if (g_sqlite_backend_db != NULL)
        {
          sqlite3_close (g_sqlite_backend_db);
        }
      g_sqlite_backend_env = NULL;
      g_sqlite_backend = NULL;
      g_sqlite_backend_db = NULL;
    }
}

sqlite3 *
objstore_open_ephemeral_db (void)
{
  sqlite3 *db = NULL;
  TEST_ASSERT_EQUAL_INT (
      SQLITE_OK, objstore_open_shared_memory_db (&db, "ephemeral"));
  objstore_config cfg = {
    .backend = OBJSTORE_BACKEND_SQLITE,
    .storage_root = NULL,
    .chunk_size_bytes = 0,
    .reserved_flags = 0,
  };
  TEST_ASSERT_EQUAL_INT (SQLITE_OK, objstore_register (db, &cfg));
  char *errmsg = NULL;
  const int rc
      = sqlite3_exec (db, "CREATE VIRTUAL TABLE objstore USING objstore();",
                      NULL, NULL, &errmsg);
  if (rc != SQLITE_OK)
    {
      fprintf (stderr, "CREATE VIRTUAL TABLE failed: %s\n",
               errmsg != NULL ? errmsg : "(null)");
      fflush (stderr);
    }
  TEST_ASSERT_EQUAL_INT_MESSAGE (SQLITE_OK, rc,
                                 errmsg ? errmsg : "objstore vtable create");
  sqlite3_free (errmsg);
  return db;
}

void
objstore_close_ephemeral_db (sqlite3 *db)
{
  if (db != NULL)
    {
      const char *path = sqlite3_db_filename (db, "main");
      if (path != NULL && path[0] != '\0')
        {
          unlink (path);
        }
      sqlite3_close (db);
    }
}

int
objstore_remove_tree (const char *path)
{
  DIR *dir = opendir (path);
  if (dir == NULL)
    {
      return (errno == ENOENT) ? 0 : -1;
    }
  struct dirent *entry = NULL;
  while ((entry = readdir (dir)) != NULL)
    {
      if (strcmp (entry->d_name, ".") == 0
          || strcmp (entry->d_name, "..") == 0)
        {
          continue;
        }
      char *child = sqlite3_mprintf ("%s/%s", path, entry->d_name);
      if (child == NULL)
        {
          closedir (dir);
          return -1;
        }
      struct stat st;
      if (stat (child, &st) == 0 && S_ISDIR (st.st_mode))
        {
          objstore_remove_tree (child);
        }
      else
        {
          unlink (child);
        }
      sqlite3_free (child);
    }
  closedir (dir);
  return rmdir (path);
}

char *
objstore_create_temp_root (const char *tag)
{
  if (tag == NULL)
    {
      tag = "objstore-temp-";
    }
  for (int attempt = 0; attempt < 16; ++attempt)
    {
      unsigned char random_bytes[6] = { 0 };
      char random_hex[sizeof (random_bytes) * 2 + 1];
      sqlite3_randomness ((int)sizeof (random_bytes), random_bytes);
      for (size_t i = 0; i < sizeof (random_bytes); ++i)
        {
          sqlite3_snprintf (3, &random_hex[i * 2], "%02x", random_bytes[i]);
        }
      random_hex[sizeof (random_hex) - 1] = '\0';

      char *tmpl = sqlite3_mprintf ("/tmp/%s%s", tag, random_hex);
      if (tmpl == NULL)
        {
          return NULL;
        }
      if (mkdir (tmpl, 0700) == 0)
        {
          return tmpl;
        }
      sqlite3_free (tmpl);
      if (errno != EEXIST)
        {
          return NULL;
        }
    }
  return NULL;
}

void
objstore_ensure_dir (const char *path)
{
  if (path == NULL || path[0] == '\0')
    {
      return;
    }
  char *mutable_path = sqlite3_mprintf ("%s", path);
  if (mutable_path == NULL)
    {
      return;
    }
  for (char *p = mutable_path + 1; *p != '\0'; ++p)
    {
      if (*p == '/')
        {
          *p = '\0';
          mkdir (mutable_path, 0777);
          *p = '/';
        }
    }
  mkdir (mutable_path, 0777);
  sqlite3_free (mutable_path);
}

void
objstore_prepare_file_roots (const char *base_path, char **out_objects_root,
                             char **out_staging_root)
{
  TEST_ASSERT_NOT_NULL (base_path);
  TEST_ASSERT_NOT_NULL (out_objects_root);
  TEST_ASSERT_NOT_NULL (out_staging_root);
  char *objects_root = sqlite3_mprintf ("%s/objects", base_path);
  TEST_ASSERT_NOT_NULL (objects_root);
  objstore_ensure_dir (objects_root);
  char *staging_root = sqlite3_mprintf ("%s/.staging", objects_root);
  TEST_ASSERT_NOT_NULL (staging_root);
  objstore_ensure_dir (staging_root);
  char *active_root = sqlite3_mprintf ("%s/active", staging_root);
  char *commit_root = sqlite3_mprintf ("%s/commit", staging_root);
  TEST_ASSERT_NOT_NULL (active_root);
  TEST_ASSERT_NOT_NULL (commit_root);
  objstore_ensure_dir (active_root);
  objstore_ensure_dir (commit_root);
  sqlite3_free (active_root);
  sqlite3_free (commit_root);
  *out_objects_root = objects_root;
  *out_staging_root = staging_root;
}

char *
objstore_build_sharded_path (const char *objects_root, const char *hex,
                             uint8_t shard_width)
{
  char *path = sqlite3_mprintf ("%s", objects_root);
  if (path == NULL)
    {
      return NULL;
    }
  for (uint8_t i = 0; i < shard_width; i += 2)
    {
      char segment[3] = { hex[i], hex[i + 1], '\0' };
      char *next = sqlite3_mprintf ("%s/%s", path, segment);
      sqlite3_free (path);
      path = next;
      if (path == NULL)
        {
          return NULL;
        }
    }
  char *full = sqlite3_mprintf ("%s/%s.dat", path, hex);
  sqlite3_free (path);
  return full;
}

void
objstore_exec_or_fail (sqlite3 *db, const char *sql)
{
  TEST_ASSERT_NOT_NULL (db);
  TEST_ASSERT_NOT_NULL (sql);
  char *errmsg = NULL;
  int rc = sqlite3_exec (db, sql, NULL, NULL, &errmsg);
  if (rc != SQLITE_OK)
    {
      if (errmsg != NULL)
        {
          TEST_FAIL_MESSAGE (errmsg);
        }
      TEST_FAIL_MESSAGE ("sqlite3_exec failed");
    }
  sqlite3_free (errmsg);
}

void
objstore_exec_expect_error (sqlite3 *db, const char *sql)
{
  TEST_ASSERT_NOT_NULL (db);
  TEST_ASSERT_NOT_NULL (sql);
  char *errmsg = NULL;
  int rc = sqlite3_exec (db, sql, NULL, NULL, &errmsg);
  if (rc == SQLITE_OK)
    {
      sqlite3_free (errmsg);
      TEST_FAIL_MESSAGE ("sqlite3_exec unexpectedly succeeded");
    }
  sqlite3_free (errmsg);
}

void
objstore_id_to_hex (const objstore_id *id, char *hex_out)
{
  static const char digits[] = "0123456789abcdef";
  for (size_t i = 0; i < OBJSTORE_ID_SIZE; ++i)
    {
      const uint8_t byte = id->bytes[i];
      hex_out[i * 2] = digits[(byte >> 4) & 0xF];
      hex_out[i * 2 + 1] = digits[byte & 0xF];
    }
  hex_out[OBJSTORE_ID_SIZE * 2] = '\0';
}

void
objstore_compute_zero_payload_id (size_t size, objstore_id *out_id)
{
  objstore_blake3 hash;
  objstore_blake3_init (&hash);
  uint8_t zero = 0;
  for (size_t i = 0; i < size; ++i)
    {
      objstore_blake3_update (&hash, &zero, 1);
    }
  objstore_blake3_final (&hash, out_id);
}

sqlite3 *
objstore_open_file_backend_db (char **out_base, char **out_objects,
                               char **out_staging, const char *label)
{
  char *base = objstore_create_temp_root (
      label != NULL ? label : "objstore-file-vtab-");
  TEST_ASSERT_NOT_NULL (base);
  char *objects_root = NULL;
  char *staging_root = NULL;
  objstore_prepare_file_roots (base, &objects_root, &staging_root);

  sqlite3 *db = NULL;
  TEST_ASSERT_EQUAL_INT (
      SQLITE_OK, objstore_open_shared_memory_db (
                     &db, label != NULL ? label : "file_backend_vtab"));
  objstore_config cfg = {
    .backend = OBJSTORE_BACKEND_FILE,
    .storage_root = objects_root,
    .chunk_size_bytes = 0,
    .reserved_flags = 0,
  };
  TEST_ASSERT_EQUAL_INT (SQLITE_OK, objstore_register (db, &cfg));
  *out_base = base;
  *out_objects = objects_root;
  if (out_staging != NULL)
    {
      *out_staging = staging_root;
    }
  else
    {
      sqlite3_free (staging_root);
    }
  return db;
}

int
objstore_exists_helper (sqlite3 *db, const uint8_t *id_bytes)
{
  sqlite3_stmt *stmt = NULL;
  TEST_ASSERT_EQUAL_INT (
      SQLITE_OK,
      sqlite3_prepare_v2 (db, "SELECT objstore_exists(?1);", -1, &stmt, NULL));
  TEST_ASSERT_EQUAL_INT (SQLITE_OK, sqlite3_bind_blob (stmt, 1, id_bytes,
                                                       OBJSTORE_ID_SIZE,
                                                       SQLITE_TRANSIENT));
  const int rc = sqlite3_step (stmt);
  if (rc != SQLITE_ROW)
    {
      const char *errmsg = sqlite3_errmsg (db);
      const int errcode = sqlite3_extended_errcode (db);
      char message[128];
      if (errmsg != NULL)
        {
          snprintf (message, sizeof (message), "%s (code %d)", errmsg,
                    errcode);
        }
      else
        {
          snprintf (message, sizeof (message), "sqlite3_step failed (code %d)",
                    errcode);
        }
      sqlite3_finalize (stmt);
      TEST_FAIL_MESSAGE (message);
    }
  const int result = sqlite3_column_int (stmt, 0);
  sqlite3_finalize (stmt);
  return result;
}

int
main (void)
{
  UNITY_BEGIN ();
  objstore_tests_set_fixture (OBJSTORE_FIXTURE_SMOKE);
  core_api_register_tests ();
  vtab_crud_register_tests ();
  scalar_function_register_tests ();
  sqlite_helper_register_tests ();
  file_backend_vtab_register_tests ();
  txn_log_register_tests ();
  backend_sqlite_register_tests ();
  backend_file_register_tests ();
  backend_vfs_register_tests ();
  object_manager_register_tests ();
  return UNITY_END ();
}
