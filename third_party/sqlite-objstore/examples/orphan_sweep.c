#if !defined(_WIN32)
#define _POSIX_C_SOURCE 200809L
#endif

#include "objstore/backend.h"

#include "backend_fs_common.h"

#include <dirent.h>
#include <errno.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

typedef struct string_list
{
  char **items;
  size_t count;
  size_t capacity;
} string_list;

typedef struct orphan_record
{
  char *hex_id;
  char *payload_path;
  sqlite3_int64 size_bytes;
} orphan_record;

typedef struct orphan_list
{
  orphan_record *items;
  size_t count;
  size_t capacity;
} orphan_list;

typedef struct sweep_options
{
  const char *db_path;
  const char *storage_root;
  const char *live_query;
  bool delete_mode;
} sweep_options;

static void string_list_destroy (string_list *list);
static void orphan_list_destroy (orphan_list *list);

static int
compare_string_ptrs (const void *lhs, const void *rhs)
{
  const char *const *left = (const char *const *)lhs;
  const char *const *right = (const char *const *)rhs;
  return strcmp (*left, *right);
}

static int
string_list_append (string_list *list, const char *value, size_t value_len)
{
  if (list == NULL || value == NULL)
    {
      return SQLITE_MISUSE;
    }
  if (list->count == list->capacity)
    {
      size_t new_capacity = (list->capacity == 0) ? 16u : list->capacity * 2u;
      char **resized
          = sqlite3_realloc64 (list->items, new_capacity * sizeof (*resized));
      if (resized == NULL)
        {
          return SQLITE_NOMEM;
        }
      list->items = resized;
      list->capacity = new_capacity;
    }
  char *copy = sqlite3_malloc64 (value_len + 1u);
  if (copy == NULL)
    {
      return SQLITE_NOMEM;
    }
  memcpy (copy, value, value_len);
  copy[value_len] = '\0';
  list->items[list->count++] = copy;
  return SQLITE_OK;
}

static void
string_list_sort (string_list *list)
{
  if (list != NULL && list->count > 1u)
    {
      qsort (list->items, list->count, sizeof (*list->items),
             compare_string_ptrs);
    }
}

static bool
string_list_contains (const string_list *list, const char *value)
{
  if (list == NULL || value == NULL || list->count == 0u)
    {
      return false;
    }
  return bsearch (&value, list->items, list->count, sizeof (*list->items),
                  compare_string_ptrs)
         != NULL;
}

static void
string_list_destroy (string_list *list)
{
  if (list == NULL)
    {
      return;
    }
  for (size_t i = 0; i < list->count; ++i)
    {
      sqlite3_free (list->items[i]);
    }
  sqlite3_free (list->items);
  list->items = NULL;
  list->count = 0u;
  list->capacity = 0u;
}

static int
orphan_list_append (orphan_list *list, const char *hex_id,
                    const char *payload_path, sqlite3_int64 size_bytes)
{
  if (list == NULL || hex_id == NULL || payload_path == NULL)
    {
      return SQLITE_MISUSE;
    }
  if (list->count == list->capacity)
    {
      size_t new_capacity = (list->capacity == 0u) ? 16u : list->capacity * 2u;
      orphan_record *resized = sqlite3_realloc64 (
          list->items, new_capacity * sizeof (*resized));
      if (resized == NULL)
        {
          return SQLITE_NOMEM;
        }
      list->items = resized;
      list->capacity = new_capacity;
    }
  orphan_record *record = &list->items[list->count];
  record->hex_id = sqlite3_mprintf ("%s", hex_id);
  record->payload_path = sqlite3_mprintf ("%s", payload_path);
  if (record->hex_id == NULL || record->payload_path == NULL)
    {
      sqlite3_free (record->hex_id);
      sqlite3_free (record->payload_path);
      record->hex_id = NULL;
      record->payload_path = NULL;
      return SQLITE_NOMEM;
    }
  record->size_bytes = size_bytes;
  ++list->count;
  return SQLITE_OK;
}

static void
orphan_list_destroy (orphan_list *list)
{
  if (list == NULL)
    {
      return;
    }
  for (size_t i = 0; i < list->count; ++i)
    {
      sqlite3_free (list->items[i].hex_id);
      sqlite3_free (list->items[i].payload_path);
    }
  sqlite3_free (list->items);
  list->items = NULL;
  list->count = 0u;
  list->capacity = 0u;
}

static void
usage (const char *argv0)
{
  fprintf (stderr,
           "usage: %s --db <db.sqlite3> --storage-root <objects-root> "
           "--live-query <sql> [--delete]\n"
           "\n"
           "The live query must return objstore ids as either 32-byte BLOBs or\n"
           "64-char lowercase/uppercase hex strings. Use UNION/UNION ALL when\n"
           "your schema references objstore ids from multiple tables.\n",
           argv0);
}

static int
parse_args (int argc, char **argv, sweep_options *out)
{
  if (out == NULL)
    {
      return SQLITE_MISUSE;
    }
  memset (out, 0, sizeof (*out));
  for (int i = 1; i < argc; ++i)
    {
      const char *arg = argv[i];
      if (strcmp (arg, "--db") == 0 && i + 1 < argc)
        {
          out->db_path = argv[++i];
        }
      else if (strcmp (arg, "--storage-root") == 0 && i + 1 < argc)
        {
          out->storage_root = argv[++i];
        }
      else if (strcmp (arg, "--live-query") == 0 && i + 1 < argc)
        {
          out->live_query = argv[++i];
        }
      else if (strcmp (arg, "--delete") == 0)
        {
          out->delete_mode = true;
        }
      else if (strcmp (arg, "--help") == 0)
        {
          usage (argv[0]);
          exit (0);
        }
      else
        {
          fprintf (stderr, "unknown argument: %s\n", arg);
          return SQLITE_MISUSE;
        }
    }
  if (out->db_path == NULL || out->storage_root == NULL
      || out->live_query == NULL)
    {
      return SQLITE_MISUSE;
    }
  return SQLITE_OK;
}

static void
bytes_to_hex (const unsigned char *bytes, size_t count, char *out)
{
  static const char digits[] = "0123456789abcdef";
  for (size_t i = 0; i < count; ++i)
    {
      out[i * 2u] = digits[(bytes[i] >> 4u) & 0x0Fu];
      out[i * 2u + 1u] = digits[bytes[i] & 0x0Fu];
    }
  out[count * 2u] = '\0';
}

static int
normalize_live_id (sqlite3_stmt *stmt, string_list *live_ids)
{
  const int type = sqlite3_column_type (stmt, 0);
  if (type == SQLITE_BLOB)
    {
      const void *blob = sqlite3_column_blob (stmt, 0);
      const int bytes = sqlite3_column_bytes (stmt, 0);
      if (blob == NULL || bytes != OBJSTORE_ID_SIZE)
        {
          return SQLITE_MISMATCH;
        }
      char hex[OBJSTORE_ID_SIZE * 2u + 1u];
      bytes_to_hex ((const unsigned char *)blob, OBJSTORE_ID_SIZE, hex);
      return string_list_append (live_ids, hex, sizeof (hex) - 1u);
    }
  if (type == SQLITE_TEXT)
    {
      const unsigned char *text = sqlite3_column_text (stmt, 0);
      const int bytes = sqlite3_column_bytes (stmt, 0);
      if (text == NULL || bytes != (int)(OBJSTORE_ID_SIZE * 2u))
        {
          return SQLITE_MISMATCH;
        }
      objstore_id id = { 0 };
      if (objstore_fs_hex_to_id ((const char *)text, &id) != SQLITE_OK)
        {
          return SQLITE_MISMATCH;
        }
      char hex[OBJSTORE_ID_SIZE * 2u + 1u];
      objstore_fs_id_to_hex (&id, hex);
      return string_list_append (live_ids, hex, sizeof (hex) - 1u);
    }
  return SQLITE_MISMATCH;
}

static int
collect_live_ids (sqlite3 *db, const char *sql, string_list *live_ids)
{
  sqlite3_stmt *stmt = NULL;
  int rc = sqlite3_prepare_v2 (db, sql, -1, &stmt, NULL);
  if (rc != SQLITE_OK)
    {
      return rc;
    }
  while ((rc = sqlite3_step (stmt)) == SQLITE_ROW)
    {
      rc = normalize_live_id (stmt, live_ids);
      if (rc != SQLITE_OK)
        {
          break;
        }
    }
  if (rc == SQLITE_DONE)
    {
      rc = SQLITE_OK;
    }
  sqlite3_finalize (stmt);
  if (rc == SQLITE_OK)
    {
      string_list_sort (live_ids);
    }
  return rc;
}

static int
collect_orphan_payloads (const char *path, const string_list *live_ids,
                         orphan_list *orphans)
{
  DIR *dir = opendir (path);
  if (dir == NULL)
    {
      return (errno == ENOENT) ? SQLITE_OK : SQLITE_IOERR;
    }
  const size_t suffix_len = strlen (OBJSTORE_FILE_SUFFIX);
  int rc = SQLITE_OK;
  struct dirent *entry = NULL;
  while ((entry = readdir (dir)) != NULL && rc == SQLITE_OK)
    {
      if (strcmp (entry->d_name, ".") == 0
          || strcmp (entry->d_name, "..") == 0)
        {
          continue;
        }
      char *child = objstore_fs_path_join (path, entry->d_name);
      if (child == NULL)
        {
          rc = SQLITE_NOMEM;
          break;
        }
      struct stat st;
      if (stat (child, &st) != 0)
        {
          sqlite3_free (child);
          continue;
        }
      if (S_ISDIR (st.st_mode))
        {
          if (strcmp (entry->d_name, ".staging") == 0
              || strcmp (entry->d_name, "rowidx") == 0)
            {
              sqlite3_free (child);
              continue;
            }
          rc = collect_orphan_payloads (child, live_ids, orphans);
          sqlite3_free (child);
          continue;
        }
      if (!S_ISREG (st.st_mode))
        {
          sqlite3_free (child);
          continue;
        }
      const size_t name_len = strlen (entry->d_name);
      if (name_len <= suffix_len
          || strcmp (entry->d_name + name_len - suffix_len, OBJSTORE_FILE_SUFFIX)
                 != 0)
        {
          sqlite3_free (child);
          continue;
        }
      const size_t hex_len = name_len - suffix_len;
      if (hex_len != OBJSTORE_ID_SIZE * 2u)
        {
          sqlite3_free (child);
          continue;
        }
      char hex[OBJSTORE_ID_SIZE * 2u + 1u];
      memcpy (hex, entry->d_name, hex_len);
      hex[hex_len] = '\0';
      objstore_id id = { 0 };
      if (objstore_fs_hex_to_id (hex, &id) != SQLITE_OK)
        {
          sqlite3_free (child);
          continue;
        }
      objstore_fs_id_to_hex (&id, hex);
      if (!string_list_contains (live_ids, hex))
        {
          rc = orphan_list_append (orphans, hex, child, (sqlite3_int64)st.st_size);
        }
      sqlite3_free (child);
    }
  closedir (dir);
  return rc;
}

static char *
rowidx_entry_path (const char *storage_root, const char *hex_id)
{
  objstore_id id = { 0 };
  if (objstore_fs_hex_to_id (hex_id, &id) != SQLITE_OK)
    {
      return NULL;
    }
  unsigned char prefix[OBJSTORE_ROWID_PREFIX_SIZE];
  objstore_rowid_prefix_from_id (&id, prefix);
  char prefix_hex[OBJSTORE_ROWID_HEX_CHARS + 1u];
  bytes_to_hex (prefix, OBJSTORE_ROWID_PREFIX_SIZE, prefix_hex);
  char shard[3] = { prefix_hex[0], prefix_hex[1], '\0' };
  char *row_dir
      = sqlite3_mprintf ("%s/rowidx/%s/%s", storage_root, shard, prefix_hex);
  if (row_dir == NULL)
    {
      return NULL;
    }
  char *row_path = sqlite3_mprintf ("%s/%s", row_dir, hex_id);
  sqlite3_free (row_dir);
  return row_path;
}

static int
delete_orphan_record (const char *storage_root, const orphan_record *record)
{
  if (storage_root == NULL || record == NULL || record->payload_path == NULL)
    {
      return SQLITE_MISUSE;
    }
  if (unlink (record->payload_path) != 0 && errno != ENOENT)
    {
      return SQLITE_IOERR_DELETE;
    }
  char *rowidx_path = rowidx_entry_path (storage_root, record->hex_id);
  if (rowidx_path != NULL)
    {
      if (unlink (rowidx_path) != 0 && errno != ENOENT)
        {
          sqlite3_free (rowidx_path);
          return SQLITE_IOERR_DELETE;
        }
      sqlite3_free (rowidx_path);
    }
  return SQLITE_OK;
}

int
main (int argc, char **argv)
{
  sweep_options opts;
  int rc = parse_args (argc, argv, &opts);
  if (rc != SQLITE_OK)
    {
      usage (argv[0]);
      return 1;
    }

  sqlite3 *db = NULL;
  rc = sqlite3_open_v2 (opts.db_path, &db, SQLITE_OPEN_READWRITE | SQLITE_OPEN_URI,
                        NULL);
  if (rc != SQLITE_OK)
    {
      fprintf (stderr, "failed to open database: %s\n",
               db != NULL ? sqlite3_errmsg (db) : "sqlite_open_failed");
      sqlite3_close (db);
      return 1;
    }

  string_list live_ids = { 0 };
  orphan_list orphans = { 0 };

  rc = collect_live_ids (db, opts.live_query, &live_ids);
  if (rc != SQLITE_OK)
    {
      fprintf (stderr,
               "live-query failed or returned non-id values (expected 32-byte "
               "BLOBs or 64-char hex text): %s\n",
               sqlite3_errmsg (db));
      string_list_destroy (&live_ids);
      orphan_list_destroy (&orphans);
      sqlite3_close (db);
      return 1;
    }

  rc = collect_orphan_payloads (opts.storage_root, &live_ids, &orphans);
  if (rc != SQLITE_OK)
    {
      fprintf (stderr, "failed to scan storage root: %s\n", opts.storage_root);
      string_list_destroy (&live_ids);
      orphan_list_destroy (&orphans);
      sqlite3_close (db);
      return 1;
    }

  sqlite3_int64 orphan_bytes = 0;
  for (size_t i = 0; i < orphans.count; ++i)
    {
      orphan_bytes += orphans.items[i].size_bytes;
      printf ("%s %s (%lld bytes)\n",
              opts.delete_mode ? "deleted_orphan" : "orphan",
              orphans.items[i].payload_path,
              (long long)orphans.items[i].size_bytes);
      if (opts.delete_mode)
        {
          rc = delete_orphan_record (opts.storage_root, &orphans.items[i]);
          if (rc != SQLITE_OK)
            {
              fprintf (stderr, "failed to delete orphan: %s\n",
                       orphans.items[i].payload_path);
              string_list_destroy (&live_ids);
              orphan_list_destroy (&orphans);
              sqlite3_close (db);
              return 1;
            }
        }
    }

  printf ("summary live_ids=%zu orphan_objects=%zu orphan_bytes=%lld mode=%s\n",
          live_ids.count, orphans.count, (long long)orphan_bytes,
          opts.delete_mode ? "delete" : "report");

  string_list_destroy (&live_ids);
  orphan_list_destroy (&orphans);
  sqlite3_close (db);
  return (opts.delete_mode || orphans.count == 0u) ? 0 : 2;
}
