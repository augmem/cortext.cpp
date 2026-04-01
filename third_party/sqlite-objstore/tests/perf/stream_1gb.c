// Copyright 2024 sqlite-objstore
// SPDX-License-Identifier: Apache-2.0

#include <errno.h>
#include <sqlite3.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <unistd.h>

#include "objstore/backend.h"
#include "objstore/object_manager.h"
#include "objstore/vtab.h"
#include "perf_common.h"

typedef struct stream_options
{
  objstore_backend_kind backend;
  sqlite3_uint64 payload_bytes;
  const char *db_path;
  const char *storage_root;
  size_t chunk_size_bytes;
  uint8_t shard_width;
  objstore_sync_mode sync_mode;
} stream_options;

typedef struct temp_paths
{
  char *db_path;
  bool owns_db;
  char *objects_root;
  char *objects_base;
} temp_paths;

static void
usage (const char *prog)
{
  fprintf (stderr,
           "Usage: %s [--backend sqlite|file] [--bytes N] [--db PATH]\n"
           "          [--storage-root DIR] [--chunk-size BYTES]\n"
           "          [--shard-width HEX_DIGITS] [--sync full|metadata|off]\n",
           prog);
}

static objstore_sync_mode
parse_sync_mode (const char *value)
{
  if (value == NULL)
    {
      return OBJSTORE_SYNC_FULL;
    }
  if (strcasecmp (value, "full") == 0)
    {
      return OBJSTORE_SYNC_FULL;
    }
  if (strcasecmp (value, "metadata") == 0)
    {
      return OBJSTORE_SYNC_METADATA;
    }
  if (strcasecmp (value, "off") == 0)
    {
      return OBJSTORE_SYNC_OFF;
    }
  fprintf (stderr, "Unknown sync mode: %s\n", value);
  exit (EXIT_FAILURE);
}

static void
parse_args (int argc, char **argv, stream_options *opts)
{
  opts->backend = OBJSTORE_BACKEND_FILE;
  opts->payload_bytes = (sqlite3_uint64)1 << 30; /* 1 GiB default */
  opts->db_path = NULL;
  opts->storage_root = NULL;
  opts->chunk_size_bytes = 0;
  opts->shard_width = 0;
  opts->sync_mode = OBJSTORE_SYNC_FULL;

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
          if (opts->backend == OBJSTORE_BACKEND_AUTO)
            {
              fprintf (stderr, "Auto backend not supported for this harness\n");
              exit (EXIT_FAILURE);
            }
        }
      else if (strncmp (arg, "--backend=", 10) == 0)
        {
          const char *value = arg + 10;
          if (perf_parse_backend (value, &opts->backend) != SQLITE_OK
              || opts->backend == OBJSTORE_BACKEND_AUTO)
            {
              fprintf (stderr, "Unknown backend: %s\n", value);
              exit (EXIT_FAILURE);
            }
        }
      else if (strcmp (arg, "--bytes") == 0 && i + 1 < argc)
        {
          opts->payload_bytes = (sqlite3_uint64)strtoull (argv[++i], NULL, 10);
        }
      else if (strncmp (arg, "--bytes=", 8) == 0)
        {
          opts->payload_bytes = (sqlite3_uint64)strtoull (arg + 8, NULL, 10);
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
      else if (strcmp (arg, "--chunk-size") == 0 && i + 1 < argc)
        {
          opts->chunk_size_bytes = (size_t)strtoull (argv[++i], NULL, 10);
        }
      else if (strncmp (arg, "--chunk-size=", 13) == 0)
        {
          opts->chunk_size_bytes = (size_t)strtoull (arg + 13, NULL, 10);
        }
      else if (strcmp (arg, "--shard-width") == 0 && i + 1 < argc)
        {
          opts->shard_width = (uint8_t)strtoul (argv[++i], NULL, 10);
        }
      else if (strncmp (arg, "--shard-width=", 14) == 0)
        {
          opts->shard_width = (uint8_t)strtoul (arg + 14, NULL, 10);
        }
      else if (strcmp (arg, "--sync") == 0 && i + 1 < argc)
        {
          opts->sync_mode = parse_sync_mode (argv[++i]);
        }
      else if (strncmp (arg, "--sync=", 7) == 0)
        {
          opts->sync_mode = parse_sync_mode (arg + 7);
        }
      else if (strcmp (arg, "--help") == 0)
        {
          usage (argv[0]);
          exit (EXIT_SUCCESS);
        }
      else
        {
          fprintf (stderr, "Unknown argument: %s\n", arg);
          usage (argv[0]);
          exit (EXIT_FAILURE);
        }
    }

  if (opts->payload_bytes == 0)
    {
      fprintf (stderr, "--bytes must be greater than zero\n");
      exit (EXIT_FAILURE);
    }
  if (opts->backend == OBJSTORE_BACKEND_FILE
      && opts->shard_width != 0 && (opts->shard_width % 2u) != 0)
    {
      fprintf (stderr, "shard width must be even (hex digits)\n");
      exit (EXIT_FAILURE);
    }
}

static char *
create_temp_db_path (temp_paths *paths)
{
  char tmpl[] = "/tmp/objstore-stream-dbXXXXXX.sqlite3";
  int fd = mkstemps (tmpl, 8);
  if (fd < 0)
    {
      perror ("mkstemps");
      exit (EXIT_FAILURE);
    }
  close (fd);
  unlink (tmpl);
  char *copy = strdup (tmpl);
  if (copy == NULL)
    {
      perror ("strdup");
      exit (EXIT_FAILURE);
    }
  paths->db_path = copy;
  paths->owns_db = true;
  return copy;
}

static void
prepare_file_backend_root (const stream_options *opts, temp_paths *paths,
                           objstore_config *cfg)
{
  if (opts->backend != OBJSTORE_BACKEND_FILE)
    {
      cfg->storage_root = NULL;
      return;
    }

  if (opts->storage_root != NULL)
    {
      perf_ensure_dir (opts->storage_root);
      cfg->storage_root = strdup (opts->storage_root);
      if (cfg->storage_root == NULL)
        {
          perror ("strdup storage_root");
          exit (EXIT_FAILURE);
        }
      paths->objects_root = (char *)cfg->storage_root;
      return;
    }

  char *base = perf_create_temp_dir ("/tmp/objstore-stream-file-");
  if (base == NULL)
    {
      perror ("perf_create_temp_dir");
      exit (EXIT_FAILURE);
    }
  char *objects = sqlite3_mprintf ("%s/objects", base);
  char *staging = sqlite3_mprintf ("%s/.staging", objects);
  char *active = sqlite3_mprintf ("%s/active", staging);
  char *commit = sqlite3_mprintf ("%s/commit", staging);
  if (objects == NULL || staging == NULL || active == NULL || commit == NULL)
    {
      fprintf (stderr, "Allocation failure preparing file backend roots\n");
      exit (EXIT_FAILURE);
    }
  perf_ensure_dir (objects);
  perf_ensure_dir (staging);
  perf_ensure_dir (active);
  perf_ensure_dir (commit);
  sqlite3_free (active);
  sqlite3_free (commit);
  sqlite3_free (staging);

  cfg->storage_root = objects;
  paths->objects_root = objects;
  paths->objects_base = base;
}

typedef struct zero_reader_state
{
  sqlite3_uint64 remaining;
} zero_reader_state;

static int
zero_reader_pull (void *ctx, void *buffer, size_t capacity, size_t *nread)
{
  zero_reader_state *state = (zero_reader_state *)ctx;
  if (state->remaining == 0)
    {
      *nread = 0;
      return SQLITE_DONE;
    }
  size_t chunk
      = (state->remaining < capacity) ? (size_t)state->remaining : capacity;
  memset (buffer, 0, chunk);
  state->remaining -= chunk;
  *nread = chunk;
  return SQLITE_OK;
}

typedef struct sink_writer_state
{
  sqlite3_uint64 expected;
  sqlite3_uint64 seen;
} sink_writer_state;

static int
sink_writer_push (void *ctx, const void *buffer, size_t nread)
{
  (void)buffer;
  sink_writer_state *state = (sink_writer_state *)ctx;
  state->seen += (sqlite3_uint64)nread;
  if (state->seen > state->expected)
    {
      fprintf (stderr, "Sink writer saw more bytes than expected\n");
      return SQLITE_CORRUPT;
    }
  return SQLITE_OK;
}

static void
cleanup_paths (temp_paths *paths)
{
  if (paths->owns_db && paths->db_path != NULL)
    {
      unlink (paths->db_path);
      free (paths->db_path);
      paths->db_path = NULL;
    }
  if (paths->objects_root != NULL && paths->objects_base == NULL)
    {
      free (paths->objects_root);
      paths->objects_root = NULL;
    }
  if (paths->objects_base != NULL)
    {
      perf_remove_tree (paths->objects_base);
      sqlite3_free (paths->objects_root);
      sqlite3_free (paths->objects_base);
      paths->objects_root = NULL;
      paths->objects_base = NULL;
    }
}

int
main (int argc, char **argv)
{
  stream_options options;
  parse_args (argc, argv, &options);

  temp_paths paths = { 0 };
  const char *db_path = options.db_path;
  if (db_path == NULL)
    {
      db_path = create_temp_db_path (&paths);
    }

  objstore_config cfg = {
    .backend = options.backend,
    .storage_root = NULL,
    .chunk_size_bytes = options.chunk_size_bytes,
    .shard_width = options.shard_width,
    .sync_mode = options.sync_mode,
    .reserved_flags = 0,
  };
  prepare_file_backend_root (&options, &paths, &cfg);

  sqlite3 *db = NULL;
  if (sqlite3_open (db_path, &db) != SQLITE_OK)
    {
      fprintf (stderr, "sqlite_open failed: %s\n", sqlite3_errmsg (db));
      cleanup_paths (&paths);
      return EXIT_FAILURE;
    }

  const objstore_backend *backend = objstore_backend_by_kind (cfg.backend);
  if (backend == NULL)
    {
      fprintf (stderr, "Requested backend not compiled in\n");
      sqlite3_close (db);
      cleanup_paths (&paths);
      return EXIT_FAILURE;
    }

  size_t chunk_size = objstore_effective_chunk_size (&cfg);
  objstore_connection *conn = NULL;
  if (objstore_connection_create (db, backend, &cfg, chunk_size, &conn)
      != SQLITE_OK)
    {
      fprintf (stderr, "objstore_connection_create failed\n");
      sqlite3_close (db);
      cleanup_paths (&paths);
      return EXIT_FAILURE;
    }

  if (sqlite3_exec (db, "BEGIN IMMEDIATE;", NULL, NULL, NULL) != SQLITE_OK)
    {
      fprintf (stderr, "BEGIN IMMEDIATE failed: %s\n", sqlite3_errmsg (db));
      objstore_connection_destroy (conn);
      sqlite3_close (db);
      cleanup_paths (&paths);
      return EXIT_FAILURE;
    }

  zero_reader_state reader_state = { .remaining = options.payload_bytes };
  objstore_stream_reader reader = {
    .ctx = &reader_state,
    .pull = zero_reader_pull,
    .size_hint = (sqlite3_int64)options.payload_bytes,
  };
  objstore_id id;
  double start_put = perf_now_seconds ();
  int rc = objstore_object_put_reader (conn, &reader, &id);
  double end_put = perf_now_seconds ();
  if (rc != SQLITE_OK)
    {
      fprintf (stderr, "objstore_object_put_reader failed rc=%d\n", rc);
      sqlite3_exec (db, "ROLLBACK;", NULL, NULL, NULL);
      objstore_connection_destroy (conn);
      sqlite3_close (db);
      cleanup_paths (&paths);
      return EXIT_FAILURE;
    }

  if (sqlite3_exec (db, "COMMIT;", NULL, NULL, NULL) != SQLITE_OK)
    {
      fprintf (stderr, "COMMIT failed: %s\n", sqlite3_errmsg (db));
      objstore_connection_destroy (conn);
      sqlite3_close (db);
      cleanup_paths (&paths);
      return EXIT_FAILURE;
    }

  double write_seconds = end_put - start_put;
  double write_throughput
      = (write_seconds > 0.0)
            ? ((double)options.payload_bytes / (1024.0 * 1024.0 * write_seconds))
            : 0.0;

  objstore_backend_txn *read_txn = NULL;
  rc = objstore_begin_read_txn (conn, &read_txn);
  if (rc != SQLITE_OK)
    {
      fprintf (stderr, "objstore_begin_read_txn failed rc=%d\n", rc);
      objstore_connection_destroy (conn);
      sqlite3_close (db);
      cleanup_paths (&paths);
      return EXIT_FAILURE;
    }

  sink_writer_state writer_state = {
    .expected = options.payload_bytes,
    .seen = 0,
  };
  objstore_stream_writer writer
      = { .ctx = &writer_state, .push = sink_writer_push };
  double start_read = perf_now_seconds ();
  rc = objstore_object_read_stream (conn, read_txn, &id, (sqlite3_uint64)-1,
                                    &writer);
  double end_read = perf_now_seconds ();
  objstore_end_read_txn (conn, read_txn, rc);
  if (rc != SQLITE_OK || writer_state.seen != writer_state.expected)
    {
      fprintf (stderr,
               "objstore_object_read_stream failed rc=%d seen=%llu expected=%llu\n",
               rc, (unsigned long long)writer_state.seen,
               (unsigned long long)writer_state.expected);
      objstore_connection_destroy (conn);
      sqlite3_close (db);
      cleanup_paths (&paths);
      return EXIT_FAILURE;
    }

  double read_seconds = end_read - start_read;
  double read_throughput
      = (read_seconds > 0.0)
            ? ((double)options.payload_bytes / (1024.0 * 1024.0 * read_seconds))
            : 0.0;

  printf ("stream_1gb backend=%s bytes=%llu write_seconds=%.3f "
          "write_mib_per_s=%.2f read_seconds=%.3f read_mib_per_s=%.2f "
          "chunk_size=%zu\n",
          (options.backend == OBJSTORE_BACKEND_FILE) ? "file" : "sqlite",
          (unsigned long long)options.payload_bytes, write_seconds,
          write_throughput, read_seconds, read_throughput, chunk_size);

  objstore_connection_destroy (conn);
  sqlite3_close (db);
  cleanup_paths (&paths);
  return EXIT_SUCCESS;
}

