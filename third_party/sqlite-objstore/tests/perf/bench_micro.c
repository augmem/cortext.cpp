// Copyright 2024 sqlite-objstore
// SPDX-License-Identifier: Apache-2.0

#include <inttypes.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "objstore/backend.h"
#include "objstore/blake3.h"
#include "perf_common.h"

typedef struct bench_options
{
  objstore_backend_kind backend;
  size_t operations;
  size_t payload_bytes;
  const char *db_path;
  const char *storage_root;
  unsigned int seed;
  bool quiet;
} bench_options;

typedef struct phase_stats
{
  const char *label;
  double seconds;
  size_t operations;
  size_t bytes;
} phase_stats;

static void
bench_usage (const char *prog)
{
  fprintf (stderr,
           "Usage: %s [--backend sqlite|file] [--ops N] [--payload BYTES]\n"
           "          [--db PATH] [--storage-root DIR] [--seed N] [--quiet]\n",
           prog);
}

static const char *
bench_backend_name (objstore_backend_kind kind)
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
bench_parse_args (int argc, char **argv, bench_options *opts)
{
  opts->backend = OBJSTORE_BACKEND_SQLITE;
  opts->operations = 256;
  opts->payload_bytes = 64 * 1024;
  opts->db_path = NULL;
  opts->storage_root = NULL;
  opts->seed = 0xC0FFEEu;
  opts->quiet = false;

  for (int i = 1; i < argc; ++i)
    {
      const char *arg = argv[i];
      if (strcmp (arg, "--backend") == 0 && i + 1 < argc)
        {
          if (perf_parse_backend (argv[++i], &opts->backend) != SQLITE_OK)
            {
              fprintf (stderr, "Unknown backend: %s\n", argv[i]);
              exit (EXIT_FAILURE);
            }
        }
      else if (strncmp (arg, "--backend=", 10) == 0)
        {
          if (perf_parse_backend (arg + 10, &opts->backend) != SQLITE_OK)
            {
              fprintf (stderr, "Unknown backend: %s\n", arg + 10);
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
      else if (strcmp (arg, "--payload") == 0 && i + 1 < argc)
        {
          opts->payload_bytes = (size_t)strtoull (argv[++i], NULL, 10);
        }
      else if (strncmp (arg, "--payload=", 10) == 0)
        {
          opts->payload_bytes = (size_t)strtoull (arg + 10, NULL, 10);
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
      else if (strcmp (arg, "--seed") == 0 && i + 1 < argc)
        {
          opts->seed = (unsigned int)strtoul (argv[++i], NULL, 10);
        }
      else if (strncmp (arg, "--seed=", 7) == 0)
        {
          opts->seed = (unsigned int)strtoul (arg + 7, NULL, 10);
        }
      else if (strcmp (arg, "--quiet") == 0)
        {
          opts->quiet = true;
        }
      else if (strcmp (arg, "--help") == 0)
        {
          bench_usage (argv[0]);
          exit (EXIT_SUCCESS);
        }
      else
        {
          fprintf (stderr, "Unknown argument: %s\n", arg);
          bench_usage (argv[0]);
          exit (EXIT_FAILURE);
        }
    }

  if (opts->operations == 0 || opts->payload_bytes == 0)
    {
      fprintf (stderr, "--ops and --payload must be > 0\n");
      exit (EXIT_FAILURE);
    }
}

static int
bench_prepare_statement (sqlite3 *db, const char *sql, sqlite3_stmt **out)
{
  return sqlite3_prepare_v3 (db, sql, -1, SQLITE_PREPARE_PERSISTENT, out, NULL);
}

static void
bench_print_stats (const bench_options *opts, const phase_stats *stats)
{
  if (opts->quiet)
    {
      return;
    }
  const double ops_per_sec
      = stats->seconds > 0.0 ? (double)stats->operations / stats->seconds : 0.0;
  const double mb_per_sec
      = stats->seconds > 0.0 ? (double)stats->bytes / (1024.0 * 1024.0)
                                  / stats->seconds
                             : 0.0;
  printf ("phase=%s backend=%s ops=%zu bytes=%zu time=%.3fs ops/s=%.2f "
          "throughput=%.2f MiB/s\n",
          stats->label, bench_backend_name (opts->backend), stats->operations,
          stats->bytes, stats->seconds, ops_per_sec, mb_per_sec);
}

static void
bench_run_inserts (sqlite3 *db, const bench_options *opts, unsigned char *ids,
                   unsigned char *payload, phase_stats *stats)
{
  sqlite3_stmt *stmt = NULL;
  if (bench_prepare_statement (
          db, "INSERT INTO objstore(id, data) VALUES (?1, ?2);", &stmt)
      != SQLITE_OK)
    {
      fprintf (stderr, "Failed to prepare INSERT statement\n");
      exit (EXIT_FAILURE);
    }
  if (perf_exec_sql (db, "BEGIN IMMEDIATE;") != SQLITE_OK)
    {
      fprintf (stderr, "BEGIN failed\n");
      exit (EXIT_FAILURE);
    }

  const double start = perf_now_seconds ();
  for (size_t i = 0; i < opts->operations; ++i)
    {
      perf_fill_pattern (payload, opts->payload_bytes, opts->seed + i);
      objstore_id computed_id;
      objstore_blake3_hash_blob (payload, opts->payload_bytes, &computed_id);
      sqlite3_reset (stmt);
      sqlite3_clear_bindings (stmt);
      if (sqlite3_bind_blob (stmt, 1, computed_id.bytes, OBJSTORE_ID_SIZE,
                             SQLITE_TRANSIENT)
          != SQLITE_OK)
        {
          fprintf (stderr, "sqlite3_bind_blob failed for id at op %zu\n", i);
          exit (EXIT_FAILURE);
        }
      if (sqlite3_bind_blob64 (stmt, 2, payload,
                               (sqlite3_uint64)opts->payload_bytes,
                               SQLITE_TRANSIENT)
          != SQLITE_OK)
        {
          fprintf (stderr, "sqlite3_bind_blob64 failed at op %zu\n", i);
          exit (EXIT_FAILURE);
        }
      const int rc = sqlite3_step (stmt);
      if (rc != SQLITE_DONE)
        {
          fprintf (stderr, "INSERT finalization failed (rc=%d)\n", rc);
          exit (EXIT_FAILURE);
        }
      memcpy (ids + i * OBJSTORE_ID_SIZE, computed_id.bytes, OBJSTORE_ID_SIZE);
    }
  const double end = perf_now_seconds ();
  if (perf_exec_sql (db, "COMMIT;") != SQLITE_OK)
    {
      fprintf (stderr, "COMMIT failed after insert phase\n");
      exit (EXIT_FAILURE);
    }
  sqlite3_finalize (stmt);

  stats->label = "insert";
  stats->seconds = end - start;
  stats->operations = opts->operations;
  stats->bytes = opts->operations * opts->payload_bytes;
}

static void
bench_run_reads (sqlite3 *db, const bench_options *opts, const unsigned char *ids,
                 phase_stats *stats)
{
  sqlite3_stmt *stmt = NULL;
  if (bench_prepare_statement (db, "SELECT data FROM objstore WHERE id = ?1;",
                               &stmt)
      != SQLITE_OK)
    {
      fprintf (stderr, "Failed to prepare SELECT statement\n");
      exit (EXIT_FAILURE);
    }
  const double start = perf_now_seconds ();
  size_t bytes_seen = 0;
  for (size_t i = 0; i < opts->operations; ++i)
    {
      sqlite3_reset (stmt);
      sqlite3_clear_bindings (stmt);
      if (sqlite3_bind_blob (stmt, 1, ids + i * OBJSTORE_ID_SIZE,
                             OBJSTORE_ID_SIZE, SQLITE_TRANSIENT)
          != SQLITE_OK)
        {
          fprintf (stderr, "sqlite3_bind_blob failed during read\n");
          exit (EXIT_FAILURE);
        }
      int rc = sqlite3_step (stmt);
      if (rc != SQLITE_ROW)
        {
          fprintf (stderr, "SELECT failed at op %zu (rc=%d)\n", i, rc);
          exit (EXIT_FAILURE);
        }
      const void *blob = sqlite3_column_blob (stmt, 0);
      const int blob_bytes = sqlite3_column_bytes (stmt, 0);
      if (blob == NULL || blob_bytes != (int)opts->payload_bytes)
        {
          fprintf (stderr, "Unexpected blob size %d\n", blob_bytes);
          exit (EXIT_FAILURE);
        }
      const unsigned char *bytes = (const unsigned char *)blob;
      unsigned int checksum = 0;
      for (int b = 0; b < blob_bytes; ++b)
        {
          checksum += bytes[b];
        }
      (void)checksum;
      bytes_seen += (size_t)blob_bytes;
      rc = sqlite3_step (stmt);
      if (rc != SQLITE_DONE)
        {
          fprintf (stderr, "SELECT finalization failed (rc=%d)\n", rc);
          exit (EXIT_FAILURE);
        }
    }
  const double end = perf_now_seconds ();
  sqlite3_finalize (stmt);

  stats->label = "read";
  stats->seconds = end - start;
  stats->operations = opts->operations;
  stats->bytes = bytes_seen;
}

static void
bench_run_deletes (sqlite3 *db, const bench_options *opts,
                   const unsigned char *ids, phase_stats *stats)
{
  sqlite3_stmt *stmt = NULL;
  if (bench_prepare_statement (db, "DELETE FROM objstore WHERE id = ?1;", &stmt)
      != SQLITE_OK)
    {
      fprintf (stderr, "Failed to prepare DELETE statement\n");
      exit (EXIT_FAILURE);
    }
  if (perf_exec_sql (db, "BEGIN IMMEDIATE;") != SQLITE_OK)
    {
      fprintf (stderr, "BEGIN failed for delete phase\n");
      exit (EXIT_FAILURE);
    }
  const double start = perf_now_seconds ();
  for (size_t i = 0; i < opts->operations; ++i)
    {
      sqlite3_reset (stmt);
      sqlite3_clear_bindings (stmt);
      if (sqlite3_bind_blob (stmt, 1, ids + i * OBJSTORE_ID_SIZE,
                             OBJSTORE_ID_SIZE, SQLITE_TRANSIENT)
          != SQLITE_OK)
        {
          fprintf (stderr, "sqlite3_bind_blob failed during delete\n");
          exit (EXIT_FAILURE);
        }
      const int rc = sqlite3_step (stmt);
      if (rc != SQLITE_DONE)
        {
          fprintf (stderr, "DELETE failed at op %zu (rc=%d)\n", i, rc);
          exit (EXIT_FAILURE);
        }
      if (sqlite3_changes (db) != 1)
        {
          fprintf (stderr, "DELETE removed %d rows\n", sqlite3_changes (db));
          exit (EXIT_FAILURE);
        }
    }
  const double end = perf_now_seconds ();
  if (perf_exec_sql (db, "COMMIT;") != SQLITE_OK)
    {
      fprintf (stderr, "COMMIT failed for delete phase\n");
      exit (EXIT_FAILURE);
    }
  sqlite3_finalize (stmt);

  stats->label = "delete";
  stats->seconds = end - start;
  stats->operations = opts->operations;
  stats->bytes = opts->operations * OBJSTORE_ID_SIZE;
}

int
main (int argc, char **argv)
{
  bench_options options;
  bench_parse_args (argc, argv, &options);

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
      fprintf (stderr, "perf_env_open failed (%d)\n", env_rc);
      return EXIT_FAILURE;
    }

  unsigned char *payload = malloc (options.payload_bytes);
  unsigned char *ids
      = malloc (options.operations * OBJSTORE_ID_SIZE);
  if (payload == NULL || ids == NULL)
    {
      fprintf (stderr, "Allocation failure\n");
      perf_env_close (&env);
      return EXIT_FAILURE;
    }

  phase_stats insert_stats;
  bench_run_inserts (env.db, &options, ids, payload, &insert_stats);
  bench_print_stats (&options, &insert_stats);

  phase_stats read_stats;
  bench_run_reads (env.db, &options, ids, &read_stats);
  bench_print_stats (&options, &read_stats);

  phase_stats delete_stats;
  bench_run_deletes (env.db, &options, ids, &delete_stats);
  bench_print_stats (&options, &delete_stats);

  free (payload);
  free (ids);
  perf_env_close (&env);
  return EXIT_SUCCESS;
}

