// Copyright 2024 sqlite-objstore
// SPDX-License-Identifier: Apache-2.0

#include "perf_common.h"

#include <errno.h>
#include <dirent.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

#include "objstore/backend.h"

static char *
perf_dup_string (const char *input)
{
  if (input == NULL)
    {
      return NULL;
    }
  char *copy = sqlite3_mprintf ("%s", input);
  return copy;
}

static int
perf_prepare_db_path (char **path_out)
{
  char tmpl[] = "/tmp/objstore-perf-db-XXXXXX.sqlite3";
  char *path = sqlite3_mprintf ("%s", tmpl);
  if (path == NULL)
    {
      return SQLITE_NOMEM;
    }
  int fd = mkstemps (path, 8); /* ".sqlite3" suffix */
  if (fd < 0)
    {
      sqlite3_free (path);
      return SQLITE_IOERR;
    }
  close (fd);
  if (unlink (path) != 0 && errno != ENOENT)
    {
      sqlite3_free (path);
      return SQLITE_IOERR;
    }
  *path_out = path;
  return SQLITE_OK;
}

char *
perf_create_temp_dir (const char *prefix)
{
  const char *use_prefix = (prefix != NULL) ? prefix : "/tmp/objstore-perf-";
  char *tmpl = sqlite3_mprintf ("%sXXXXXX", use_prefix);
  if (tmpl == NULL)
    {
      return NULL;
    }
  if (mkdtemp (tmpl) == NULL)
    {
      sqlite3_free (tmpl);
      return NULL;
    }
  return tmpl;
}

void
perf_ensure_dir (const char *path)
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
  for (char *cursor = mutable_path + 1; *cursor != '\0'; ++cursor)
    {
      if (*cursor == '/')
        {
          *cursor = '\0';
          mkdir (mutable_path, 0777);
          *cursor = '/';
        }
    }
  mkdir (mutable_path, 0777);
  sqlite3_free (mutable_path);
}

int
perf_remove_tree (const char *path)
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
          perf_remove_tree (child);
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

static int
perf_prepare_file_roots (perf_env *env, const char *explicit_root)
{
  if (explicit_root != NULL)
    {
      env->owned_storage_root = perf_dup_string (explicit_root);
      if (env->owned_storage_root == NULL)
        {
          return SQLITE_NOMEM;
        }
      env->owns_storage_tree = false;
      perf_ensure_dir (env->owned_storage_root);
      char *staging = sqlite3_mprintf ("%s/.staging", env->owned_storage_root);
      if (staging == NULL)
        {
          return SQLITE_NOMEM;
        }
      perf_ensure_dir (staging);
      sqlite3_free (staging);
      return SQLITE_OK;
    }

  char *base = perf_create_temp_dir ("/tmp/objstore-perf-file-");
  if (base == NULL)
    {
      return SQLITE_IOERR;
    }
  env->owned_base_dir = base;
  env->owns_storage_tree = true;
  char *objects = sqlite3_mprintf ("%s/objects", base);
  if (objects == NULL)
    {
      return SQLITE_NOMEM;
    }
  perf_ensure_dir (objects);
  char *staging = sqlite3_mprintf ("%s/.staging", objects);
  if (staging == NULL)
    {
      sqlite3_free (objects);
      return SQLITE_NOMEM;
    }
  perf_ensure_dir (staging);
  sqlite3_free (staging);
  env->owned_storage_root = objects;
  return SQLITE_OK;
}

int
perf_env_open (const perf_backend_options *opts, perf_env *env)
{
  if (opts == NULL || env == NULL)
    {
      return SQLITE_MISUSE;
    }
  memset (env, 0, sizeof (*env));
  env->backend = opts->backend;

  if (env->backend == OBJSTORE_BACKEND_FILE)
    {
      const int prep_rc
          = perf_prepare_file_roots (env, opts->storage_root);
      if (prep_rc != SQLITE_OK)
        {
          perf_env_close (env);
          return prep_rc;
        }
    }

  if (opts->db_path != NULL)
    {
      env->owned_db_path = perf_dup_string (opts->db_path);
      if (env->owned_db_path == NULL)
        {
          perf_env_close (env);
          return SQLITE_NOMEM;
        }
      env->delete_db_path = false;
    }
  else
    {
      const int rc = perf_prepare_db_path (&env->owned_db_path);
      if (rc != SQLITE_OK)
        {
          perf_env_close (env);
          return rc;
        }
      env->delete_db_path = true;
    }

  sqlite3 *db = NULL;
  const int flags = SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE
                    | SQLITE_OPEN_FULLMUTEX | SQLITE_OPEN_URI;
  int rc = sqlite3_open_v2 (env->owned_db_path, &db, flags, NULL);
  if (rc != SQLITE_OK)
    {
      perf_env_close (env);
      return rc;
    }

  objstore_config cfg = {
    .backend = env->backend,
    .storage_root = env->owned_storage_root,
    .chunk_size_bytes = opts->chunk_size_bytes,
    .shard_width = opts->shard_width,
    .sync_mode = opts->sync_mode,
    .reserved_flags = 0,
  };
  rc = objstore_register (db, &cfg);
  if (rc != SQLITE_OK)
    {
      sqlite3_close (db);
      perf_env_close (env);
      return rc;
    }

  rc = perf_exec_sql (
      db, "CREATE VIRTUAL TABLE objstore USING objstore();");
  if (rc != SQLITE_OK)
    {
      sqlite3_close (db);
      perf_env_close (env);
      return rc;
    }

  env->db = db;
  return SQLITE_OK;
}

void
perf_env_close (perf_env *env)
{
  if (env == NULL)
    {
      return;
    }
  if (env->db != NULL)
    {
      sqlite3_close (env->db);
      env->db = NULL;
    }
  if (env->owned_db_path != NULL)
    {
      if (env->delete_db_path)
        {
          unlink (env->owned_db_path);
        }
      sqlite3_free (env->owned_db_path);
      env->owned_db_path = NULL;
    }
  if (env->owned_base_dir != NULL)
    {
      perf_remove_tree (env->owned_base_dir);
      sqlite3_free (env->owned_base_dir);
      env->owned_base_dir = NULL;
      env->owns_storage_tree = false;
    }
  else if (env->owns_storage_tree && env->owned_storage_root != NULL)
    {
      perf_remove_tree (env->owned_storage_root);
      env->owns_storage_tree = false;
    }
  if (env->owned_storage_root != NULL)
    {
      sqlite3_free (env->owned_storage_root);
      env->owned_storage_root = NULL;
    }
}

int
perf_parse_backend (const char *name, objstore_backend_kind *out)
{
  if (name == NULL || out == NULL)
    {
      return SQLITE_MISUSE;
    }
  if (strcasecmp (name, "sqlite") == 0)
    {
      *out = OBJSTORE_BACKEND_SQLITE;
      return SQLITE_OK;
    }
  if (strcasecmp (name, "file") == 0)
    {
      *out = OBJSTORE_BACKEND_FILE;
      return SQLITE_OK;
    }
  if (strcasecmp (name, "auto") == 0)
    {
      *out = OBJSTORE_BACKEND_AUTO;
      return SQLITE_OK;
    }
  return SQLITE_ERROR;
}

double
perf_now_seconds (void)
{
  struct timespec ts;
  clock_gettime (CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

void
perf_fill_pattern (unsigned char *buffer, size_t size, unsigned int seed)
{
  if (buffer == NULL)
    {
      return;
    }
  unsigned int state = seed ? seed : 0xA5A5u;
  for (size_t i = 0; i < size; ++i)
    {
      state = 1664525u * state + 1013904223u;
      buffer[i] = (unsigned char)(state >> 24);
    }
}

int
perf_exec_sql (sqlite3 *db, const char *sql)
{
  if (db == NULL || sql == NULL)
    {
      return SQLITE_MISUSE;
    }
  char *errmsg = NULL;
  const int rc = sqlite3_exec (db, sql, NULL, NULL, &errmsg);
  if (rc != SQLITE_OK)
    {
      if (errmsg != NULL)
        {
          fprintf (stderr, "SQL error: %s\n", errmsg);
        }
    }
  sqlite3_free (errmsg);
  return rc;
}

