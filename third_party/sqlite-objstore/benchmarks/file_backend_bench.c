#include "objstore/backend.h"
#include "objstore/object_manager.h"
#include "objstore/objstore.h"
#include "objstore/vtab.h"

#include <errno.h>
#include <math.h>
#include <sqlite3.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

static int ensure_dir (const char *path);
static int ensure_staging_tree (const char *objects_root);

typedef struct benchmark_config
{
  const char *objects_root;
  size_t object_size;
  size_t count;
  objstore_sync_mode sync_mode;
  uint8_t shard_width;
  size_t chunk_size;
  int runs;
  bool clear_cache;
  const char *workload;
  bool sync_matrix;
} benchmark_config;

typedef struct benchmark_stats
{
  double seconds;
  double mib_per_sec;
  double ops_per_sec;
} benchmark_stats;

typedef struct run_metrics
{
  double throughput_mib;
  double elapsed_sec;
  double p50_ms;
  double p95_ms;
  double p99_ms;
  double ops_per_sec;
} run_metrics;

typedef struct workload_spec
{
  const char *name;
  size_t count;
  size_t size_bytes;
  bool track_ops;
  double throughput_target_mib;
  double ops_target;
} workload_spec;

static const workload_spec kWorkloadPresets[] = {
  { "large", 10, 1073741824ull, false, 500.0, 0.0 }, /* 10 × 1 GiB */
  { "medium", 1000, 10485760ull, false, 0.0, 0.0 },  /* 1000 × 10 MiB */
  { "small", 10000, 1024ull, true, 0.0, 10000.0 },   /* 10k × 1 KiB */
};

typedef struct workload_run
{
  workload_spec spec;
  bool custom;
} workload_run;

typedef struct benchmark_state
{
  sqlite3 *db;
  objstore_connection *conn;
  size_t chunk_bytes;
} benchmark_state;

typedef struct chunk_reader
{
  size_t remaining;
  size_t chunk_bytes;
  uint64_t state;
  uint64_t word_cache;
  size_t word_bytes_used;
} chunk_reader;

typedef struct sink_writer
{
  sqlite3_uint64 total_bytes;
} sink_writer;

static void print_usage (const char *prog);
static int ensure_dir (const char *path);
static void fatal (const char *message);
static void sqlite_die (sqlite3 *db, const char *message);
static void sqlite_exec_or_die (sqlite3 *db, const char *sql);
static double now_seconds (void);
static bool clear_os_cache (void);
static void chunk_reader_init (chunk_reader *reader, uint64_t seed,
                               size_t object_size, size_t chunk_bytes);
static uint64_t chunk_reader_next (uint64_t *state);
static int chunk_reader_pull (void *ctx, void *buffer, size_t capacity,
                              size_t *nread);
static int sink_writer_push (void *ctx, const void *buffer, size_t nread);
static double compute_mean (const double *values, int n);
static double compute_stddev (const double *values, int n, double mean);
static double percentile_from_sorted (const double *values, size_t count,
                                      double fraction);
static int compare_double (const void *lhs, const void *rhs);
static void delete_workload_objects (benchmark_state *state,
                                     const objstore_id *ids, size_t count);
static int run_write_benchmark (benchmark_state *state, size_t object_size,
                                size_t count, uint64_t run_seed,
                                objstore_id *out_ids, run_metrics *metrics);
static benchmark_stats run_read_benchmark (benchmark_state *state,
                                           const objstore_id *ids,
                                           size_t count);
static benchmark_stats run_rowid_lookup_benchmark (benchmark_state *state,
                                                   const objstore_id *ids,
                                                   size_t count);
static void run_workload (benchmark_state *state, const benchmark_config *cfg,
                          const workload_run *workload);
static bool select_workloads (const benchmark_config *cfg,
                              workload_run **out_runs, size_t *out_count);
static const char *sync_mode_label (objstore_sync_mode mode);
static int benchmark_run (const benchmark_config *cfg);

int
main (int argc, char **argv)
{
  benchmark_config cfg = {
    .objects_root = "bench-output/objects",
    .object_size = 1 << 16,
    .count = 200,
    .sync_mode = OBJSTORE_SYNC_FULL,
    .shard_width = 2,
    .chunk_size = 0,
    .runs = 3,
    .clear_cache = true,
    .workload = "all",
    .sync_matrix = false,
  };

  for (int i = 1; i < argc; ++i)
    {
      if (strcmp (argv[i], "--objects") == 0 && i + 1 < argc)
        {
          cfg.objects_root = argv[++i];
        }
      else if (strcmp (argv[i], "--count") == 0 && i + 1 < argc)
        {
          cfg.count = (size_t)strtoull (argv[++i], NULL, 10);
        }
      else if (strcmp (argv[i], "--size") == 0 && i + 1 < argc)
        {
          cfg.object_size = (size_t)strtoull (argv[++i], NULL, 10);
        }
      else if (strcmp (argv[i], "--sync") == 0 && i + 1 < argc)
        {
          const char *mode = argv[++i];
          if (strcmp (mode, "full") == 0)
            {
              cfg.sync_mode = OBJSTORE_SYNC_FULL;
            }
          else if (strcmp (mode, "metadata") == 0)
            {
              cfg.sync_mode = OBJSTORE_SYNC_METADATA;
            }
          else if (strcmp (mode, "off") == 0)
            {
              cfg.sync_mode = OBJSTORE_SYNC_OFF;
            }
          else
            {
              fatal ("sync mode must be full, metadata, or off");
            }
        }
      else if (strcmp (argv[i], "--shard") == 0 && i + 1 < argc)
        {
          cfg.shard_width = (uint8_t)atoi (argv[++i]);
        }
      else if (strcmp (argv[i], "--chunk") == 0 && i + 1 < argc)
        {
          cfg.chunk_size = (size_t)strtoull (argv[++i], NULL, 10);
        }
      else if (strcmp (argv[i], "--runs") == 0 && i + 1 < argc)
        {
          cfg.runs = atoi (argv[++i]);
        }
      else if (strcmp (argv[i], "--no-cache-clear") == 0)
        {
          cfg.clear_cache = false;
        }
      else if (strcmp (argv[i], "--workload") == 0 && i + 1 < argc)
        {
          cfg.workload = argv[++i];
        }
      else if (strcmp (argv[i], "--sync-matrix") == 0)
        {
          cfg.sync_matrix = true;
        }
      else if (strcmp (argv[i], "--help") == 0)
        {
          print_usage (argv[0]);
          return EXIT_SUCCESS;
        }
      else
        {
          print_usage (argv[0]);
          return EXIT_FAILURE;
        }
    }

  if (cfg.count == 0 || cfg.object_size == 0)
    {
      fatal ("count and size must be positive");
    }
  if (cfg.chunk_size != 0
      && (cfg.chunk_size < OBJSTORE_MIN_CHUNK_SIZE
          || cfg.chunk_size > OBJSTORE_MAX_CHUNK_SIZE))
    {
      fatal ("chunk must be between 1024 and 65536 bytes (or zero for auto)");
    }
  if (cfg.runs <= 0)
    {
      fatal ("runs must be >= 1");
    }

  if (!cfg.sync_matrix)
    {
      return benchmark_run (&cfg);
    }

  const objstore_sync_mode modes[] = {
    OBJSTORE_SYNC_FULL,
    OBJSTORE_SYNC_METADATA,
    OBJSTORE_SYNC_OFF,
  };
  for (size_t i = 0; i < sizeof (modes) / sizeof (modes[0]); ++i)
    {
      benchmark_config mode_cfg = cfg;
      mode_cfg.sync_mode = modes[i];
      printf ("\n=== sync_mode=%s ===\n", sync_mode_label (modes[i]));
      int rc = benchmark_run (&mode_cfg);
      if (rc != EXIT_SUCCESS)
        {
          return rc;
        }
    }
  return EXIT_SUCCESS;
}

static void
print_usage (const char *prog)
{
  fprintf (
      stderr,
      "Usage: %s [--objects PATH] [--count N] [--size BYTES]\n"
      "           [--sync full|metadata|off] [--shard WIDTH] [--chunk BYTES]\n"
      "           [--runs N] [--workload all|large|medium|small|custom]\n"
      "           [--no-cache-clear] [--sync-matrix]\n",
      prog);
}

static int
ensure_dir (const char *path)
{
  if (path == NULL || *path == '\0')
    {
      return -1;
    }
  char *mutable_path = sqlite3_mprintf ("%s", path);
  if (mutable_path == NULL)
    {
      return -1;
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
  int rc = mkdir (mutable_path, 0777);
  if (rc != 0 && errno != EEXIST)
    {
      sqlite3_free (mutable_path);
      return -1;
    }
  sqlite3_free (mutable_path);
  return 0;
}

static int
ensure_staging_tree (const char *objects_root)
{
  if (objects_root == NULL)
    {
      return -1;
    }
  char *staging_root = sqlite3_mprintf ("%s/.staging", objects_root);
  if (staging_root == NULL)
    {
      return -1;
    }
  int rc = ensure_dir (staging_root);
  if (rc == 0)
    {
      char *active_root = sqlite3_mprintf ("%s/active", staging_root);
      char *commit_root = sqlite3_mprintf ("%s/commit", staging_root);
      if (active_root == NULL || commit_root == NULL)
        {
          rc = -1;
        }
      else if (ensure_dir (active_root) != 0 || ensure_dir (commit_root) != 0)
        {
          rc = -1;
        }
      sqlite3_free (active_root);
      sqlite3_free (commit_root);
    }
  sqlite3_free (staging_root);
  return rc;
}

static void
fatal (const char *message)
{
  fprintf (stderr, "bench error: %s\n", message);
  exit (EXIT_FAILURE);
}

static void
sqlite_die (sqlite3 *db, const char *message)
{
  fprintf (stderr, "%s: %s\n", message,
           db != NULL ? sqlite3_errmsg (db) : "unknown");
  exit (EXIT_FAILURE);
}

static void
sqlite_exec_or_die (sqlite3 *db, const char *sql)
{
  char *errmsg = NULL;
  if (sqlite3_exec (db, sql, NULL, NULL, &errmsg) != SQLITE_OK)
    {
      fprintf (stderr, "sqlite exec failed: %s (%s)\n",
               errmsg ? errmsg : "(null)", sql);
      sqlite3_free (errmsg);
      exit (EXIT_FAILURE);
    }
  sqlite3_free (errmsg);
}

static double
now_seconds (void)
{
  struct timespec ts;
  clock_gettime (CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

static bool
clear_os_cache (void)
{
#if defined(__APPLE__)
  int rc = system ("sync && sudo purge >/dev/null 2>&1");
#elif defined(__linux__)
  int rc = system ("sync && echo 3 | sudo tee /proc/sys/vm/drop_caches "
                   ">/dev/null 2>&1");
#else
  int rc = -1;
#endif
  if (rc != 0)
    {
      fprintf (stderr,
               "warning: cache clearing command failed (rc=%d)—results may be "
               "cached\n",
               rc);
      return false;
    }
  sleep (1);
  return true;
}

static void
chunk_reader_init (chunk_reader *reader, uint64_t seed, size_t object_size,
                   size_t chunk_bytes)
{
  if (reader == NULL)
    {
      return;
    }
  reader->remaining = object_size;
  if (chunk_bytes == 0 || chunk_bytes > OBJSTORE_MAX_CHUNK_SIZE)
    {
      reader->chunk_bytes = OBJSTORE_MAX_CHUNK_SIZE;
    }
  else if (chunk_bytes < OBJSTORE_MIN_CHUNK_SIZE)
    {
      reader->chunk_bytes = OBJSTORE_MIN_CHUNK_SIZE;
    }
  else
    {
      reader->chunk_bytes = chunk_bytes;
    }
  if (reader->chunk_bytes == 0)
    {
      reader->chunk_bytes = OBJSTORE_DEFAULT_CHUNK_SIZE;
    }
  reader->state = seed ^ 0xA5A5A5A5A5A5A5A5ull;
  reader->word_cache = 0;
  reader->word_bytes_used = sizeof (reader->word_cache);
}

static uint64_t
chunk_reader_next (uint64_t *state)
{
  uint64_t x = *state + 0x9E3779B97F4A7C15ull;
  *state = x;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
  return x ^ (x >> 31);
}

static int
chunk_reader_pull (void *ctx, void *buffer, size_t capacity, size_t *nread)
{
  chunk_reader *reader = (chunk_reader *)ctx;
  if (reader == NULL || buffer == NULL || nread == NULL)
    {
      return SQLITE_MISUSE;
    }
  if (reader->remaining == 0)
    {
      *nread = 0;
      return SQLITE_DONE;
    }
  size_t limit = reader->chunk_bytes;
  if (limit == 0 || limit > capacity)
    {
      limit = capacity;
    }
  if (limit == 0)
    {
      *nread = 0;
      return SQLITE_DONE;
    }
  size_t to_emit = reader->remaining < limit ? reader->remaining : limit;
  uint8_t *out = (uint8_t *)buffer;
  size_t produced = 0;
  while (produced < to_emit)
    {
      if (reader->word_bytes_used >= sizeof (reader->word_cache))
        {
          reader->word_cache = chunk_reader_next (&reader->state);
          reader->word_bytes_used = 0;
        }
      out[produced++] = (uint8_t)(reader->word_cache & 0xFFu);
      reader->word_cache >>= 8;
      ++reader->word_bytes_used;
    }
  reader->remaining -= to_emit;
  *nread = to_emit;
  return SQLITE_OK;
}

static int
sink_writer_push (void *ctx, const void *buffer, size_t nread)
{
  (void)buffer;
  sink_writer *writer = (sink_writer *)ctx;
  if (writer == NULL)
    {
      return SQLITE_MISUSE;
    }
  writer->total_bytes += (sqlite3_uint64)nread;
  return SQLITE_OK;
}

static double
compute_mean (const double *values, int n)
{
  if (n <= 0)
    {
      return 0.0;
    }
  double sum = 0.0;
  for (int i = 0; i < n; ++i)
    {
      sum += values[i];
    }
  return sum / (double)n;
}

static double
compute_stddev (const double *values, int n, double mean)
{
  if (n <= 1)
    {
      return 0.0;
    }
  double sum = 0.0;
  for (int i = 0; i < n; ++i)
    {
      double diff = values[i] - mean;
      sum += diff * diff;
    }
  return sqrt (sum / (double)(n - 1));
}

static int
compare_double (const void *lhs, const void *rhs)
{
  const double a = *(const double *)lhs;
  const double b = *(const double *)rhs;
  return (a > b) - (a < b);
}

static void
delete_workload_objects (benchmark_state *state, const objstore_id *ids,
                         size_t count)
{
  if (state == NULL || state->db == NULL || state->conn == NULL || ids == NULL
      || count == 0)
    {
      return;
    }
  sqlite_exec_or_die (state->db, "BEGIN;");
  for (size_t i = 0; i < count; ++i)
    {
      int rc = objstore_object_delete (state->conn, &ids[i]);
      if (rc != SQLITE_OK && rc != SQLITE_NOTFOUND)
        {
          sqlite3_exec (state->db, "ROLLBACK;", NULL, NULL, NULL);
          fprintf (stderr, "delete failed for object %zu (rc=%d)\n", i, rc);
          fatal ("delete failed");
        }
    }
  sqlite_exec_or_die (state->db, "COMMIT;");
}

static double
percentile_from_sorted (const double *values, size_t count, double fraction)
{
  if (count == 0)
    {
      return 0.0;
    }
  size_t idx = (size_t)((count - 1) * fraction);
  if (idx >= count)
    {
      idx = count - 1;
    }
  return values[idx];
}

static int
run_write_benchmark (benchmark_state *state, size_t object_size, size_t count,
                     uint64_t run_seed, objstore_id *out_ids,
                     run_metrics *metrics)
{
  if (state == NULL || state->db == NULL || state->conn == NULL
      || out_ids == NULL || metrics == NULL)
    {
      return SQLITE_MISUSE;
    }

  double *latencies = sqlite3_malloc64 (sizeof (double) * count);
  if (latencies == NULL)
    {
      return SQLITE_NOMEM;
    }

  sqlite_exec_or_die (state->db, "BEGIN;");
  const double bench_start = now_seconds ();
  size_t progress_step = count / 10u;
  if (progress_step == 0)
    {
      progress_step = 1;
    }
  for (size_t i = 0; i < count; ++i)
    {
      chunk_reader reader_state;
      uint64_t seed = (run_seed << 32) ^ (uint64_t)i;
      chunk_reader_init (&reader_state, seed, object_size, state->chunk_bytes);
      objstore_stream_reader stream_reader = {
        .ctx = &reader_state,
        .pull = chunk_reader_pull,
        .size_hint = (sqlite3_int64)object_size,
      };
      const double op_start = now_seconds ();
      int rc = objstore_object_put_reader (state->conn, &stream_reader,
                                           &out_ids[i]);
      const double op_end = now_seconds ();
      if (rc != SQLITE_OK)
        {
          sqlite3_exec (state->db, "ROLLBACK;", NULL, NULL, NULL);
          sqlite3_free (latencies);
          fprintf (stderr, "write failed for object %zu (rc=%d)\n", i, rc);
          return rc;
        }
      latencies[i] = (op_end - op_start) * 1000.0;

      if (((i + 1) % progress_step) == 0 || (i + 1) == count)
        {
          double pct = ((double)(i + 1) * 100.0) / (double)count;
          printf ("\r  write progress: %zu/%zu (%.1f%%)", i + 1, count, pct);
          fflush (stdout);
        }
    }
  printf ("\r  write progress: %zu/%zu (100.0%%)        \n", count, count);
  const double bench_end = now_seconds ();
  sqlite_exec_or_die (state->db, "COMMIT;");

  qsort (latencies, count, sizeof (double), compare_double);

  const double elapsed = bench_end - bench_start;
  const double total_mib
      = ((double)object_size * (double)count) / (1024.0 * 1024.0);
  metrics->elapsed_sec = elapsed;
  metrics->throughput_mib = (elapsed > 0.0) ? (total_mib / elapsed) : 0.0;
  metrics->ops_per_sec = (elapsed > 0.0) ? ((double)count / elapsed) : 0.0;
  metrics->p50_ms = percentile_from_sorted (latencies, count, 0.50);
  metrics->p95_ms = percentile_from_sorted (latencies, count, 0.95);
  metrics->p99_ms = percentile_from_sorted (latencies, count, 0.99);

  sqlite3_free (latencies);
  return SQLITE_OK;
}

static benchmark_stats
run_read_benchmark (benchmark_state *state, const objstore_id *ids,
                    size_t count)
{
  benchmark_stats stats = { 0 };
  if (state == NULL || state->conn == NULL || ids == NULL || count == 0)
    {
      return stats;
    }

  objstore_backend_txn *txn = NULL;
  int rc = objstore_begin_read_txn (state->conn, &txn);
  if (rc != SQLITE_OK)
    {
      sqlite_die (state->db, "objstore_begin_read_txn");
    }

  sink_writer writer_ctx = { .total_bytes = 0 };
  objstore_stream_writer writer = {
    .ctx = &writer_ctx,
    .push = sink_writer_push,
  };

  const double start = now_seconds ();
  size_t progress_step = count / 10u;
  if (progress_step == 0)
    {
      progress_step = 1;
    }
  for (size_t i = 0; i < count; ++i)
    {
      rc = objstore_object_read_stream (state->conn, txn, &ids[i],
                                        (sqlite3_uint64)-1, &writer);
      if (rc != SQLITE_OK)
        {
          objstore_end_read_txn (state->conn, txn, rc);
          fprintf (stderr, "read failed for object %zu (rc=%d)\n", i, rc);
          fatal ("read failed");
        }
      if (((i + 1) % progress_step) == 0 || (i + 1) == count)
        {
          double pct = ((double)(i + 1) * 100.0) / (double)count;
          printf ("\r  read progress : %zu/%zu (%.1f%%)", i + 1, count, pct);
          fflush (stdout);
        }
    }
  printf ("\r  read progress : %zu/%zu (100.0%%)        \n", count, count);
  rc = objstore_end_read_txn (state->conn, txn, SQLITE_OK);
  if (rc != SQLITE_OK)
    {
      sqlite_die (state->db, "objstore_end_read_txn");
    }
  const double end = now_seconds ();

  const double elapsed = end - start;
  const double total_mib = (double)writer_ctx.total_bytes / (1024.0 * 1024.0);
  stats.seconds = elapsed;
  stats.mib_per_sec = (elapsed > 0.0) ? (total_mib / elapsed) : 0.0;
  return stats;
}

static benchmark_stats
run_rowid_lookup_benchmark (benchmark_state *state, const objstore_id *ids,
                            size_t count)
{
  benchmark_stats stats = { 0 };
  if (state == NULL || state->db == NULL || ids == NULL || count == 0)
    {
      return stats;
    }
  sqlite3_stmt *stmt = NULL;
  if (sqlite3_prepare_v3 (state->db,
                          "SELECT id FROM objstore WHERE rowid = ?1;", -1,
                          SQLITE_PREPARE_PERSISTENT, &stmt, NULL)
      != SQLITE_OK)
    {
      sqlite_die (state->db, "prepare rowid lookup benchmark");
    }
  const double start = now_seconds ();
  for (size_t i = 0; i < count; ++i)
    {
      const sqlite3_int64 rowid = objstore_rowid_from_id (&ids[i]);
      if (sqlite3_bind_int64 (stmt, 1, rowid) != SQLITE_OK)
        {
          sqlite_die (state->db, "bind rowid lookup benchmark");
        }
      int rc = sqlite3_step (stmt);
      if (rc != SQLITE_ROW)
        {
          sqlite_die (state->db, "rowid lookup benchmark step");
        }
      const void *blob = sqlite3_column_blob (stmt, 0);
      const int size = sqlite3_column_bytes (stmt, 0);
      if (blob == NULL || size != OBJSTORE_ID_SIZE
          || memcmp (blob, ids[i].bytes, OBJSTORE_ID_SIZE) != 0)
        {
          sqlite_die (state->db, "rowid lookup benchmark mismatch");
        }
      sqlite3_clear_bindings (stmt);
      sqlite3_reset (stmt);
    }
  const double end = now_seconds ();
  sqlite3_finalize (stmt);
  stats.seconds = end - start;
  stats.ops_per_sec
      = (stats.seconds > 0.0) ? ((double)count / stats.seconds) : 0.0;
  return stats;
}

static void
run_workload (benchmark_state *state, const benchmark_config *cfg,
              const workload_run *workload)
{
  if (state == NULL || cfg == NULL || workload == NULL)
    {
      return;
    }

  const workload_spec *spec = &workload->spec;
  const size_t count = spec->count;
  const size_t object_size = spec->size_bytes;
  const double dataset_gib
      = ((double)count * (double)object_size) / (1024.0 * 1024.0 * 1024.0);

  double *write_tp = calloc ((size_t)cfg->runs, sizeof (double));
  double *read_tp = calloc ((size_t)cfg->runs, sizeof (double));
  double *rowid_ops = calloc ((size_t)cfg->runs, sizeof (double));
  double *ops_values
      = spec->track_ops ? calloc ((size_t)cfg->runs, sizeof (double)) : NULL;
  objstore_id *ids
      = (count > 0) ? sqlite3_malloc64 (sizeof (objstore_id) * count) : NULL;
  if (write_tp == NULL || read_tp == NULL || rowid_ops == NULL || ids == NULL
      || (spec->track_ops && ops_values == NULL))
    {
      fprintf (stderr, "allocation failure for workload %s\n", spec->name);
      free (write_tp);
      free (read_tp);
      free (rowid_ops);
      free (ops_values);
      sqlite3_free (ids);
      return;
    }

  double p50_sum = 0.0;
  double p95_sum = 0.0;
  double p99_sum = 0.0;

  for (int run = 0; run < cfg->runs; ++run)
    {
      if (cfg->clear_cache)
        {
          clear_os_cache ();
        }
      printf ("Run %d/%d: write phase (%zu objects, %.2f GiB)\n", run + 1,
              cfg->runs, count, dataset_gib);
      fflush (stdout);
      run_metrics metrics = { 0 };
      int rc = run_write_benchmark (state, object_size, count, (uint64_t)run,
                                    ids, &metrics);
      if (rc != SQLITE_OK)
        {
          fprintf (stderr, "write benchmark failed for workload %s (rc=%d)\n",
                   spec->name, rc);
          break;
        }
      write_tp[run] = metrics.throughput_mib;
      p50_sum += metrics.p50_ms;
      p95_sum += metrics.p95_ms;
      p99_sum += metrics.p99_ms;
      if (spec->track_ops && ops_values != NULL)
        {
          ops_values[run] = metrics.ops_per_sec;
        }

      if (cfg->clear_cache)
        {
          clear_os_cache ();
        }
      printf ("Run %d/%d: read phase\n", run + 1, cfg->runs);
      fflush (stdout);
      benchmark_stats read_stats = run_read_benchmark (state, ids, count);
      read_tp[run] = read_stats.mib_per_sec;

      benchmark_stats rowid_stats
          = run_rowid_lookup_benchmark (state, ids, count);
      rowid_ops[run] = rowid_stats.ops_per_sec;
      printf ("  rowid lookups : %.0f ops/sec (%.2f s)\n",
              rowid_stats.ops_per_sec, rowid_stats.seconds);

      delete_workload_objects (state, ids, count);
    }

  const double write_mean = compute_mean (write_tp, cfg->runs);
  const double write_stddev = compute_stddev (write_tp, cfg->runs, write_mean);
  const double read_mean = compute_mean (read_tp, cfg->runs);
  const double read_stddev = compute_stddev (read_tp, cfg->runs, read_mean);
  const double rowid_mean = compute_mean (rowid_ops, cfg->runs);
  const double lat_p50 = p50_sum / (double)cfg->runs;
  const double lat_p95 = p95_sum / (double)cfg->runs;
  const double lat_p99 = p99_sum / (double)cfg->runs;

  double ops_mean = 0.0;
  double ops_stddev = 0.0;
  if (spec->track_ops && ops_values != NULL)
    {
      ops_mean = compute_mean (ops_values, cfg->runs);
      ops_stddev = compute_stddev (ops_values, cfg->runs, ops_mean);
    }

  printf ("### %s workload (%zu × %zu bytes, %.2f GiB total)\n", spec->name,
          count, object_size, dataset_gib);
  printf ("Write throughput : %.2f ± %.2f MiB/s\n", write_mean, write_stddev);
  printf ("Read throughput  : %.2f ± %.2f MiB/s\n", read_mean, read_stddev);
  printf ("Rowid lookups    : %.0f ops/sec (mean)\n", rowid_mean);
  printf ("Latency (ms)     : p50=%.2f  p95=%.2f  p99=%.2f\n", lat_p50,
          lat_p95, lat_p99);
  if (spec->track_ops && ops_values != NULL)
    {
      printf ("Operations/sec  : %.0f ± %.0f\n", ops_mean, ops_stddev);
    }

  if (!workload->custom)
    {
      if (spec->throughput_target_mib > 0.0)
        {
          printf ("Throughput target (%.0f MiB/s): %s\n",
                  spec->throughput_target_mib,
                  (write_mean >= spec->throughput_target_mib) ? "PASS"
                                                              : "FAIL");
        }
      if (spec->ops_target > 0.0 && spec->track_ops)
        {
          printf ("Ops/sec target (%.0f): %s\n", spec->ops_target,
                  (ops_mean >= spec->ops_target) ? "PASS" : "FAIL");
        }
    }
  printf ("\n");

  free (write_tp);
  free (read_tp);
  free (rowid_ops);
  free (ops_values);
  sqlite3_free (ids);
}

static const char *
sync_mode_label (objstore_sync_mode mode)
{
  switch (mode)
    {
    case OBJSTORE_SYNC_FULL:
      return "full";
    case OBJSTORE_SYNC_METADATA:
      return "metadata";
    case OBJSTORE_SYNC_OFF:
      return "off";
    default:
      return "unknown";
    }
}

static bool
select_workloads (const benchmark_config *cfg, workload_run **out_runs,
                  size_t *out_count)
{
  workload_run *runs = NULL;
  size_t count = 0;

  bool custom_only
      = cfg->workload != NULL && strcmp (cfg->workload, "custom") == 0;
  bool run_all = (cfg->workload == NULL) || strcmp (cfg->workload, "all") == 0;

  if (custom_only)
    {
      runs = calloc (1, sizeof (*runs));
      if (runs == NULL)
        {
          return false;
        }
      runs[0].spec.name = "custom";
      runs[0].spec.count = cfg->count;
      runs[0].spec.size_bytes = cfg->object_size;
      runs[0].spec.track_ops = (cfg->object_size <= 2048);
      runs[0].spec.throughput_target_mib = 0.0;
      runs[0].spec.ops_target = 0.0;
      runs[0].custom = true;
      count = 1;
    }
  else if (run_all)
    {
      const size_t preset_count
          = sizeof (kWorkloadPresets) / sizeof (kWorkloadPresets[0]);
      runs = calloc (preset_count, sizeof (*runs));
      if (runs == NULL)
        {
          return false;
        }
      for (size_t i = 0; i < preset_count; ++i)
        {
          runs[i].spec = kWorkloadPresets[i];
          runs[i].custom = false;
        }
      count = preset_count;
    }
  else
    {
      for (size_t i = 0;
           i < sizeof (kWorkloadPresets) / sizeof (kWorkloadPresets[0]); ++i)
        {
          if (strcmp (cfg->workload, kWorkloadPresets[i].name) == 0)
            {
              runs = calloc (1, sizeof (*runs));
              if (runs == NULL)
                {
                  return false;
                }
              runs[0].spec = kWorkloadPresets[i];
              runs[0].custom = false;
              count = 1;
              break;
            }
        }
      if (count == 0)
        {
          fprintf (stderr, "unknown workload preset '%s'\n", cfg->workload);
          return false;
        }
    }

  *out_runs = runs;
  *out_count = count;
  return true;
}

static int
benchmark_run (const benchmark_config *cfg)
{
  if (ensure_dir (cfg->objects_root) != 0
      || ensure_staging_tree (cfg->objects_root) != 0)
    {
      fatal ("failed to create benchmark directories");
    }

  sqlite3 *db = NULL;
  if (sqlite3_open (":memory:", &db) != SQLITE_OK)
    {
      sqlite_die (db, "open database");
    }

  objstore_config store_cfg = {
    .backend = OBJSTORE_BACKEND_FILE,
    .storage_root = cfg->objects_root,
    .chunk_size_bytes = cfg->chunk_size,
    .shard_width = cfg->shard_width,
    .sync_mode = cfg->sync_mode,
    .reserved_flags = 0,
  };

  const objstore_backend *backend = NULL;
  objstore_backend_kind resolved_kind = OBJSTORE_BACKEND_AUTO;
  if (objstore_backend_resolve (&store_cfg, &backend, &resolved_kind)
      != SQLITE_OK)
    {
      sqlite_die (db, "objstore_backend_resolve");
    }
  size_t chunk_bytes = objstore_effective_chunk_size (&store_cfg);

  objstore_connection *conn = NULL;
  if (objstore_connection_create (db, backend, &store_cfg, chunk_bytes, &conn)
      != SQLITE_OK)
    {
      sqlite_die (db, "objstore_connection_create");
    }
  if (objstore_module_register (db, conn) != SQLITE_OK)
    {
      sqlite_die (db, "objstore_module_register");
    }
  sqlite_exec_or_die (db, "CREATE VIRTUAL TABLE objstore USING objstore();");

  benchmark_state state = {
    .db = db,
    .conn = conn,
    .chunk_bytes = chunk_bytes,
  };
  (void)resolved_kind;

  workload_run *runs = NULL;
  size_t workload_count = 0;
  if (!select_workloads (cfg, &runs, &workload_count))
    {
      objstore_connection_destroy (conn);
      sqlite3_close (db);
      return EXIT_FAILURE;
    }

  printf ("Benchmark configuration:\n");
  printf ("  shard_width : %u hex chars\n", (unsigned)cfg->shard_width);
  printf ("  sync_mode   : %d\n", cfg->sync_mode);
  if (cfg->chunk_size == 0)
    {
      printf ("  chunk_size  : auto (%zu bytes)\n", state.chunk_bytes);
    }
  else
    {
      printf ("  chunk_size  : %zu bytes\n", cfg->chunk_size);
    }
  printf ("  runs        : %d\n", cfg->runs);
  printf ("  cache_clear : %s\n", cfg->clear_cache ? "enabled" : "disabled");
  printf ("\n");

  for (size_t i = 0; i < workload_count; ++i)
    {
      run_workload (&state, cfg, &runs[i]);
    }

  free (runs);
  objstore_connection_destroy (conn);
  sqlite3_close (db);
  return EXIT_SUCCESS;
}
