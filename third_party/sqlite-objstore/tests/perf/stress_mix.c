// Copyright 2024 sqlite-objstore
// SPDX-License-Identifier: Apache-2.0

#include <stdarg.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "objstore/backend.h"
#include "objstore/blake3.h"
#include "perf_common.h"

typedef struct stress_options
{
  objstore_backend_kind backend;
  size_t operations;
  size_t min_payload;
  size_t max_payload;
  size_t txn_batch;
  const char *db_path;
  const char *storage_root;
  unsigned int seed;
  bool verbose;
} stress_options;

typedef struct stress_entry
{
  unsigned char id[OBJSTORE_ID_SIZE];
  size_t size;
  bool alive;
} stress_entry;

typedef struct stress_pool
{
  stress_entry *entries;
  size_t *live_indices;
  size_t entry_count;
  size_t live_count;
  size_t capacity;
} stress_pool;

typedef struct stress_stats
{
  size_t inserts;
  size_t reads;
  size_t deletes;
  size_t bytes_written;
  size_t bytes_read;
} stress_stats;

static void
stress_usage (const char *prog)
{
  fprintf (stderr,
           "Usage: %s [--backend sqlite|file] [--ops N] [--min BYTES]\n"
           "          [--max BYTES] [--txn-batch N] [--seed N]\n"
           "          [--db PATH] [--storage-root DIR] [--verbose]\n",
           prog);
}

static const char *
stress_backend_name (objstore_backend_kind kind)
{
  switch (kind)
    {
    case OBJSTORE_BACKEND_SQLITE:
      return "sqlite";
    case OBJSTORE_BACKEND_FILE:
      return "file";
    case OBJSTORE_BACKEND_AUTO:
      return "auto";
    default:
      return "unknown";
    }
}

static void
stress_parse_args (int argc, char **argv, stress_options *opts)
{
  opts->backend = OBJSTORE_BACKEND_SQLITE;
  opts->operations = 5000;
  opts->min_payload = 4 * 1024;
  opts->max_payload = 64 * 1024;
  opts->txn_batch = 64;
  opts->db_path = NULL;
  opts->storage_root = NULL;
  opts->seed = 0x1234ABCDu;
  opts->verbose = false;

  for (int i = 1; i < argc; ++i)
    {
      const char *arg = argv[i];
      if (strcmp (arg, "--backend") == 0 && i + 1 < argc)
        {
          if (perf_parse_backend (argv[++i], &opts->backend) != SQLITE_OK)
            {
              fprintf (stderr, "Unknown backend '%s'\n", argv[i]);
              exit (EXIT_FAILURE);
            }
        }
      else if (strncmp (arg, "--backend=", 10) == 0)
        {
          if (perf_parse_backend (arg + 10, &opts->backend) != SQLITE_OK)
            {
              fprintf (stderr, "Unknown backend '%s'\n", arg + 10);
              exit (EXIT_FAILURE);
            }
        }
      else if (strcmp (arg, "--ops") == 0 && i + 1 < argc)
        {
          opts->operations = (size_t)strtoull (argv[++i], NULL, 10);
        }
      else if (strncmp (arg, "--ops=", 6) == 0)
        {
          opts->operations = (size_t)strtoull (arg + 6, NULL, 10);
        }
      else if (strcmp (arg, "--min") == 0 && i + 1 < argc)
        {
          opts->min_payload = (size_t)strtoull (argv[++i], NULL, 10);
        }
      else if (strncmp (arg, "--min=", 6) == 0)
        {
          opts->min_payload = (size_t)strtoull (arg + 6, NULL, 10);
        }
      else if (strcmp (arg, "--max") == 0 && i + 1 < argc)
        {
          opts->max_payload = (size_t)strtoull (argv[++i], NULL, 10);
        }
      else if (strncmp (arg, "--max=", 6) == 0)
        {
          opts->max_payload = (size_t)strtoull (arg + 6, NULL, 10);
        }
      else if (strcmp (arg, "--txn-batch") == 0 && i + 1 < argc)
        {
          opts->txn_batch = (size_t)strtoull (argv[++i], NULL, 10);
        }
      else if (strncmp (arg, "--txn-batch=", 12) == 0)
        {
          opts->txn_batch = (size_t)strtoull (arg + 12, NULL, 10);
        }
      else if (strcmp (arg, "--seed") == 0 && i + 1 < argc)
        {
          opts->seed = (unsigned int)strtoul (argv[++i], NULL, 10);
        }
      else if (strncmp (arg, "--seed=", 7) == 0)
        {
          opts->seed = (unsigned int)strtoul (arg + 7, NULL, 10);
        }
      else if (strcmp (arg, "--db") == 0 && i + 1 < argc)
        {
          opts->db_path = argv[++i];
        }
      else if (strncmp (arg, "--db=", 5) == 0)
        {
          opts->db_path = arg + 5;
        }
      else if (strcmp (arg, "--storage-root") == 0 && i + 1 < argc)
        {
          opts->storage_root = argv[++i];
        }
      else if (strncmp (arg, "--storage-root=", 15) == 0)
        {
          opts->storage_root = arg + 15;
        }
      else if (strcmp (arg, "--verbose") == 0)
        {
          opts->verbose = true;
        }
      else if (strcmp (arg, "--help") == 0)
        {
          stress_usage (argv[0]);
          exit (EXIT_SUCCESS);
        }
      else
        {
          fprintf (stderr, "Unknown argument: %s\n", arg);
          stress_usage (argv[0]);
          exit (EXIT_FAILURE);
        }
    }

  if (opts->operations == 0 || opts->min_payload == 0
      || opts->max_payload < opts->min_payload || opts->txn_batch == 0)
    {
      fprintf (stderr, "Invalid options (check ops/min/max/txn-batch)\n");
      exit (EXIT_FAILURE);
    }
}

static uint32_t
stress_next (uint32_t *state)
{
  *state = 1664525u * (*state) + 1013904223u;
  return *state;
}

static size_t
stress_rand_between (uint32_t *state, size_t min, size_t max)
{
  const uint32_t span = (uint32_t)(max - min + 1);
  return min + (stress_next (state) % span);
}

static void
stress_pool_init (stress_pool *pool, size_t capacity)
{
  pool->entries = calloc (capacity, sizeof (stress_entry));
  pool->live_indices = calloc (capacity, sizeof (size_t));
  pool->entry_count = 0;
  pool->live_count = 0;
  pool->capacity = capacity;
  if (pool->entries == NULL || pool->live_indices == NULL)
    {
      fprintf (stderr, "stress_pool allocation failure\n");
      exit (EXIT_FAILURE);
    }
}

static void
stress_pool_destroy (stress_pool *pool)
{
  free (pool->entries);
  free (pool->live_indices);
  pool->entries = NULL;
  pool->live_indices = NULL;
  pool->entry_count = 0;
  pool->live_count = 0;
  pool->capacity = 0;
}

static stress_entry *
stress_pool_add (stress_pool *pool)
{
  if (pool->entry_count >= pool->capacity)
    {
      fprintf (stderr, "stress_pool exhausted capacity=%zu\n", pool->capacity);
      exit (EXIT_FAILURE);
    }
  stress_entry *entry = &pool->entries[pool->entry_count];
  entry->alive = true;
  pool->live_indices[pool->live_count++] = pool->entry_count;
  ++pool->entry_count;
  return entry;
}

static stress_entry *
stress_pool_pick (stress_pool *pool, uint32_t *rand_state)
{
  if (pool->live_count == 0)
    {
      return NULL;
    }
  const size_t idx
      = pool->live_indices[stress_next (rand_state) % pool->live_count];
  return &pool->entries[idx];
}

static void
stress_pool_remove (stress_pool *pool, stress_entry *entry)
{
  if (!entry->alive)
    {
      return;
    }
  entry->alive = false;
  for (size_t i = 0; i < pool->live_count; ++i)
    {
      const size_t idx = pool->live_indices[i];
      if (&pool->entries[idx] == entry)
        {
          pool->live_indices[i] = pool->live_indices[pool->live_count - 1];
          --pool->live_count;
          return;
        }
    }
}

static sqlite3_stmt *
stress_prepare (sqlite3 *db, const char *sql)
{
  sqlite3_stmt *stmt = NULL;
  if (sqlite3_prepare_v3 (db, sql, -1, SQLITE_PREPARE_PERSISTENT, &stmt, NULL)
      != SQLITE_OK)
    {
      fprintf (stderr, "Failed to prepare statement: %s\n", sql);
      exit (EXIT_FAILURE);
    }
  return stmt;
}

static void
stress_run_insert (sqlite3 *db, const stress_options *opts,
                   sqlite3_stmt *stmt_insert, stress_pool *pool,
                   uint8_t *payload, uint32_t *rand_state, stress_stats *stats)
{
  const size_t payload_size
      = stress_rand_between (rand_state, opts->min_payload, opts->max_payload);
  perf_fill_pattern (payload, payload_size, stress_next (rand_state));
  objstore_id computed_id;
  objstore_blake3_hash_blob (payload, payload_size, &computed_id);
  sqlite3_reset (stmt_insert);
  sqlite3_clear_bindings (stmt_insert);
  if (sqlite3_bind_blob (stmt_insert, 1, computed_id.bytes, OBJSTORE_ID_SIZE,
                         SQLITE_TRANSIENT)
      != SQLITE_OK)
    {
      fprintf (stderr, "sqlite3_bind_blob failed for insert id\n");
      exit (EXIT_FAILURE);
    }
  if (sqlite3_bind_blob64 (stmt_insert, 2, payload,
                           (sqlite3_uint64)payload_size, SQLITE_TRANSIENT)
      != SQLITE_OK)
    {
      fprintf (stderr, "sqlite3_bind_blob64 failed for insert\n");
      exit (EXIT_FAILURE);
    }
  const int rc = sqlite3_step (stmt_insert);
  if (rc != SQLITE_DONE)
    {
      fprintf (stderr, "INSERT failed rc=%d\n", rc);
      exit (EXIT_FAILURE);
    }
  stress_entry *entry = stress_pool_add (pool);
  memcpy (entry->id, computed_id.bytes, OBJSTORE_ID_SIZE);
  entry->size = payload_size;
  ++stats->inserts;
  stats->bytes_written += payload_size;
}

static void
stress_run_read (sqlite3 *db, sqlite3_stmt *stmt_get, stress_entry *entry,
                 stress_stats *stats)
{
  sqlite3_reset (stmt_get);
  sqlite3_clear_bindings (stmt_get);
  if (sqlite3_bind_blob (stmt_get, 1, entry->id, OBJSTORE_ID_SIZE,
                         SQLITE_TRANSIENT)
      != SQLITE_OK)
    {
      fprintf (stderr, "bind id failed for read\n");
      exit (EXIT_FAILURE);
    }
  int rc = sqlite3_step (stmt_get);
  if (rc != SQLITE_ROW)
    {
      fprintf (stderr, "SELECT failed rc=%d\n", rc);
      exit (EXIT_FAILURE);
    }
  const int blob_bytes = sqlite3_column_bytes (stmt_get, 0);
  if (blob_bytes != (int)entry->size)
    {
      fprintf (stderr, "Unexpected blob size %d (expected %zu)\n", blob_bytes,
               entry->size);
      exit (EXIT_FAILURE);
    }
  (void)sqlite3_column_blob (stmt_get, 0);
  rc = sqlite3_step (stmt_get);
  if (rc != SQLITE_DONE)
    {
      fprintf (stderr, "SELECT finalize rc=%d\n", rc);
      exit (EXIT_FAILURE);
    }
  ++stats->reads;
  stats->bytes_read += entry->size;
}

static void
stress_run_delete (sqlite3 *db, sqlite3_stmt *stmt_delete, stress_pool *pool,
                   stress_entry *entry, stress_stats *stats)
{
  sqlite3_reset (stmt_delete);
  sqlite3_clear_bindings (stmt_delete);
  if (sqlite3_bind_blob (stmt_delete, 1, entry->id, OBJSTORE_ID_SIZE,
                         SQLITE_TRANSIENT)
      != SQLITE_OK)
    {
      fprintf (stderr, "bind id failed for delete\n");
      exit (EXIT_FAILURE);
    }
  const int rc = sqlite3_step (stmt_delete);
  if (rc != SQLITE_DONE)
    {
      fprintf (stderr, "DELETE failed rc=%d\n", rc);
      exit (EXIT_FAILURE);
    }
  stress_pool_remove (pool, entry);
  ++stats->deletes;
}

static void
stress_log (const stress_options *opts, const char *fmt, ...)
{
  if (!opts->verbose)
    {
      return;
    }
  va_list args;
  va_start (args, fmt);
  vfprintf (stdout, fmt, args);
  va_end (args);
}

int
main (int argc, char **argv)
{
  stress_options options;
  stress_parse_args (argc, argv, &options);

  perf_backend_options backend_opts = {
    .backend = options.backend,
    .db_path = options.db_path,
    .storage_root = options.storage_root,
    .chunk_size_bytes = 0,
    .shard_width = 0,
    .sync_mode = OBJSTORE_SYNC_FULL,
  };

  perf_env env;
  const int env_rc = perf_env_open (&backend_opts, &env);
  if (env_rc != SQLITE_OK)
    {
      fprintf (stderr, "perf_env_open failed rc=%d\n", env_rc);
      return EXIT_FAILURE;
    }

  stress_pool pool;
  stress_pool_init (&pool, options.operations + 1);
  uint8_t *payload = malloc (options.max_payload);
  if (payload == NULL)
    {
      fprintf (stderr, "payload allocation failed\n");
      perf_env_close (&env);
      return EXIT_FAILURE;
    }

  sqlite3_stmt *stmt_insert
      = stress_prepare (env.db, "INSERT INTO objstore(id, data) VALUES(?1, ?2);");
  sqlite3_stmt *stmt_get
      = stress_prepare (env.db, "SELECT data FROM objstore WHERE id = ?1;");
  sqlite3_stmt *stmt_delete
      = stress_prepare (env.db, "DELETE FROM objstore WHERE id = ?1;");

  stress_stats stats = { 0 };
  uint32_t rand_state = options.seed;
  const double start = perf_now_seconds ();

  size_t executed = 0;
  while (executed < options.operations)
    {
      const size_t batch
          = (options.operations - executed < options.txn_batch)
                ? (options.operations - executed)
                : options.txn_batch;
      if (perf_exec_sql (env.db, "BEGIN IMMEDIATE;") != SQLITE_OK)
        {
          fprintf (stderr, "BEGIN failed inside stress loop\n");
          exit (EXIT_FAILURE);
        }
      for (size_t i = 0; i < batch; ++i)
        {
          const uint32_t roll = stress_next (&rand_state) % 100;
          if (roll < 40 || pool.live_count == 0)
            {
              stress_log (&options, "[%zu] insert\n", executed + i);
              stress_run_insert (env.db, &options, stmt_insert, &pool, payload,
                                 &rand_state, &stats);
            }
          else if (roll < 75)
            {
              stress_entry *entry = stress_pool_pick (&pool, &rand_state);
              if (entry != NULL)
                {
                  stress_log (&options, "[%zu] read\n", executed + i);
                  stress_run_read (env.db, stmt_get, entry, &stats);
                }
            }
          else
            {
              stress_entry *entry = stress_pool_pick (&pool, &rand_state);
              if (entry != NULL)
                {
                  stress_log (&options, "[%zu] delete\n", executed + i);
                  stress_run_delete (env.db, stmt_delete, &pool, entry, &stats);
                }
            }
        }
      if (perf_exec_sql (env.db, "COMMIT;") != SQLITE_OK)
        {
          fprintf (stderr, "COMMIT failed inside stress loop\n");
          exit (EXIT_FAILURE);
        }
      executed += batch;
    }

  const double end = perf_now_seconds ();
  const double elapsed = end - start;
  sqlite3_finalize (stmt_insert);
  sqlite3_finalize (stmt_get);
  sqlite3_finalize (stmt_delete);
  free (payload);

  sqlite3_stmt *count_stmt
      = stress_prepare (env.db, "SELECT COUNT(*) FROM objstore;");
  int rc = sqlite3_step (count_stmt);
  size_t row_count = 0;
  if (rc == SQLITE_ROW)
    {
      row_count = (size_t)sqlite3_column_int64 (count_stmt, 0);
    }
  sqlite3_finalize (count_stmt);

  const double ops_per_sec
      = elapsed > 0.0 ? (double)options.operations / elapsed : 0.0;
  const size_t expected_live = pool.live_count;
  printf ("backend=%s ops=%zu inserts=%zu reads=%zu deletes=%zu "
          "live=%zu rows=%zu seconds=%.2f ops/s=%.2f "
          "written=%.2f MiB read=%.2f MiB\n",
          stress_backend_name (options.backend), options.operations,
          stats.inserts, stats.reads, stats.deletes, expected_live, row_count,
          elapsed, ops_per_sec,
          (double)stats.bytes_written / (1024.0 * 1024.0),
          (double)stats.bytes_read / (1024.0 * 1024.0));

  stress_pool_destroy (&pool);
  perf_env_close (&env);

  if (row_count != expected_live)
    {
      fprintf (stderr,
               "Live set mismatch: internal=%zu sqlite=%zu (possible leak)\n",
               expected_live, row_count);
      return EXIT_FAILURE;
    }
  return EXIT_SUCCESS;
}

