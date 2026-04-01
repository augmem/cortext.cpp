// Copyright 2024 sqlite-objstore
// SPDX-License-Identifier: Apache-2.0

#ifndef OBJSTORE_TESTS_PERF_COMMON_H
#define OBJSTORE_TESTS_PERF_COMMON_H

#include <stdbool.h>
#include <stddef.h>

#include <sqlite3.h>

#include "objstore/objstore.h"

typedef struct perf_backend_options
{
  objstore_backend_kind backend;
  const char *db_path;
  const char *storage_root;
  size_t chunk_size_bytes;
  uint8_t shard_width;
  objstore_sync_mode sync_mode;
} perf_backend_options;

typedef struct perf_env
{
  sqlite3 *db;
  objstore_backend_kind backend;
  char *owned_db_path;
  char *owned_storage_root;
  char *owned_base_dir;
  bool delete_db_path;
  bool owns_storage_tree;
} perf_env;

int perf_env_open (const perf_backend_options *opts, perf_env *env);
void perf_env_close (perf_env *env);

int perf_parse_backend (const char *name, objstore_backend_kind *out);

double perf_now_seconds (void);
void perf_fill_pattern (unsigned char *buffer, size_t size, unsigned int seed);

int perf_exec_sql (sqlite3 *db, const char *sql);

char *perf_create_temp_dir (const char *prefix);
int perf_remove_tree (const char *path);
void perf_ensure_dir (const char *path);

#endif /* OBJSTORE_TESTS_PERF_COMMON_H */

