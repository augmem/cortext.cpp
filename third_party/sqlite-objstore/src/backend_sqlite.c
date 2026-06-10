#include "objstore/backend.h"

#include <assert.h>
#include <errno.h>
#include <limits.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define OBJSTORE_SQLITE_QUEUE_LIMIT_BYTES (16 * 1024 * 1024)

static int sqlite_backend_compare_existing_buffer (
    sqlite3 *db, const objstore_id *id, const unsigned char *payload,
    sqlite3_int64 payload_size, size_t chunk_size, int *out_equal);

typedef struct sqlite_backend_env
{
  sqlite3 *primary_db;
  size_t chunk_size;
} sqlite_backend_env;

typedef enum
{
  SQLITE_BACKEND_WRITE_DELETE = 0,
  SQLITE_BACKEND_WRITE_PUT = 1,
} sqlite_backend_write_kind;

typedef struct sqlite_backend_write_entry
{
  sqlite_backend_write_kind kind;
  objstore_id id;
  unsigned char *payload;
  sqlite3_int64 size;
} sqlite_backend_write_entry;

typedef struct sqlite_backend_txn
{
  sqlite_backend_env *env;
  sqlite_backend_write_entry *write_queue;
  size_t write_count;
  size_t write_capacity;
  size_t *frame_write_counts;
  sqlite3_int64 *frame_buffered_bytes;
  size_t frame_count;
  size_t frame_capacity;
  sqlite3_int64 buffered_bytes;
  bool staged_flushed;
} sqlite_backend_txn;

typedef struct sqlite_backend_cursor
{
  sqlite3_stmt *stmt;
  sqlite_backend_env *env;
} sqlite_backend_cursor;

typedef struct sqlite_backend_staged_writer
{
  sqlite_backend_txn *txn;
  unsigned char *payload;
  sqlite3_int64 size;
  sqlite3_int64 capacity;
} sqlite_backend_staged_writer;

static inline sqlite_backend_staged_writer *
sqlite_writer_from (objstore_backend_staged_writer *writer)
{
  return (sqlite_backend_staged_writer *)writer;
}

static int
sqlite_backend_ensure_schema (sqlite3 *db)
{
  static const char *kSchema = "CREATE TABLE IF NOT EXISTS objstore_data ("
                               "id BLOB PRIMARY KEY,"
                               "data BLOB NOT NULL"
                               ");"
                               "CREATE TABLE IF NOT EXISTS objstore_rowidx ("
                               "rowid_prefix BLOB NOT NULL,"
                               "id BLOB NOT NULL,"
                               "PRIMARY KEY(rowid_prefix, id)"
                               ");"
                               "CREATE INDEX IF NOT EXISTS objstore_rowidx_prefix "
                               "ON objstore_rowidx(rowid_prefix);";
  return sqlite3_exec (db, kSchema, NULL, NULL, NULL);
}

static int sqlite_backend_rowidx_upsert (sqlite3 *db,
                                         const objstore_id *id);
static int sqlite_backend_rowidx_delete (sqlite3 *db,
                                         const objstore_id *id);
static int sqlite_backend_rowidx_backfill (sqlite3 *db);

static int
sqlite_backend_bind_id (sqlite3_stmt *stmt, int index, const objstore_id *id)
{
  return sqlite3_bind_blob (stmt, index, id->bytes, OBJSTORE_ID_SIZE,
                            SQLITE_STATIC);
}

static bool
sqlite_backend_ids_equal (const objstore_id *a, const objstore_id *b)
{
  return memcmp (a->bytes, b->bytes, OBJSTORE_ID_SIZE) == 0;
}

static int
sqlite_backend_rowidx_upsert (sqlite3 *db, const objstore_id *id)
{
  if (db == NULL || id == NULL)
    {
      return SQLITE_MISUSE;
    }
  static const char *kSql
      = "INSERT OR IGNORE INTO objstore_rowidx(rowid_prefix, id) "
        "VALUES (?1, ?2);";
  sqlite3_stmt *stmt = NULL;
  int rc = sqlite3_prepare_v3 (db, kSql, -1, SQLITE_PREPARE_PERSISTENT, &stmt,
                               NULL);
  if (rc != SQLITE_OK)
    {
      return rc;
    }
  unsigned char prefix[OBJSTORE_ROWID_PREFIX_SIZE];
  objstore_rowid_prefix_from_id (id, prefix);
  rc = sqlite3_bind_blob (stmt, 1, prefix, OBJSTORE_ROWID_PREFIX_SIZE,
                          SQLITE_TRANSIENT);
  if (rc == SQLITE_OK)
    {
      rc = sqlite3_bind_blob (stmt, 2, id->bytes, OBJSTORE_ID_SIZE,
                              SQLITE_TRANSIENT);
    }
  if (rc == SQLITE_OK)
    {
      rc = sqlite3_step (stmt);
      if (rc == SQLITE_DONE)
        {
          rc = SQLITE_OK;
        }
    }
  sqlite3_finalize (stmt);
  return rc;
}

static int
sqlite_backend_rowidx_delete (sqlite3 *db, const objstore_id *id)
{
  if (db == NULL || id == NULL)
    {
      return SQLITE_MISUSE;
    }
  static const char *kSql
      = "DELETE FROM objstore_rowidx WHERE rowid_prefix = ?1 AND id = ?2;";
  sqlite3_stmt *stmt = NULL;
  int rc = sqlite3_prepare_v3 (db, kSql, -1, SQLITE_PREPARE_PERSISTENT, &stmt,
                               NULL);
  if (rc != SQLITE_OK)
    {
      return rc;
    }
  unsigned char prefix[OBJSTORE_ROWID_PREFIX_SIZE];
  objstore_rowid_prefix_from_id (id, prefix);
  rc = sqlite3_bind_blob (stmt, 1, prefix, OBJSTORE_ROWID_PREFIX_SIZE,
                          SQLITE_TRANSIENT);
  if (rc == SQLITE_OK)
    {
      rc = sqlite3_bind_blob (stmt, 2, id->bytes, OBJSTORE_ID_SIZE,
                              SQLITE_TRANSIENT);
    }
  if (rc == SQLITE_OK)
    {
      rc = sqlite3_step (stmt);
      if (rc == SQLITE_DONE)
        {
          rc = SQLITE_OK;
        }
    }
  sqlite3_finalize (stmt);
  return rc;
}

static int
sqlite_backend_rowidx_backfill (sqlite3 *db)
{
  if (db == NULL)
    {
      return SQLITE_MISUSE;
    }
  sqlite3_stmt *stmt = NULL;
  sqlite3_int64 rowidx_count = 0;
  sqlite3_int64 data_count = 0;
  int rc = sqlite3_prepare_v3 (db, "SELECT COUNT(*) FROM objstore_rowidx;", -1,
                               SQLITE_PREPARE_PERSISTENT, &stmt, NULL);
  if (rc != SQLITE_OK)
    {
      return rc;
    }
  if (sqlite3_step (stmt) == SQLITE_ROW)
    {
      rowidx_count = sqlite3_column_int64 (stmt, 0);
    }
  rc = sqlite3_finalize (stmt);
  if (rc != SQLITE_OK)
    {
      return rc;
    }
  rc = sqlite3_prepare_v3 (db, "SELECT COUNT(*) FROM objstore_data;", -1,
                           SQLITE_PREPARE_PERSISTENT, &stmt, NULL);
  if (rc != SQLITE_OK)
    {
      return rc;
    }
  if (sqlite3_step (stmt) == SQLITE_ROW)
    {
      data_count = sqlite3_column_int64 (stmt, 0);
    }
  rc = sqlite3_finalize (stmt);
  if (rc != SQLITE_OK)
    {
      return rc;
    }
  if (rowidx_count == data_count)
    {
      return SQLITE_OK;
    }
  rc = sqlite3_exec (db, "BEGIN IMMEDIATE;", NULL, NULL, NULL);
  if (rc != SQLITE_OK)
    {
      return rc;
    }
  rc = sqlite3_exec (db, "DELETE FROM objstore_rowidx;", NULL, NULL, NULL);
  if (rc != SQLITE_OK)
    {
      sqlite3_exec (db, "ROLLBACK;", NULL, NULL, NULL);
      return rc;
    }
  rc = sqlite3_prepare_v3 (db, "SELECT id FROM objstore_data;", -1,
                           SQLITE_PREPARE_PERSISTENT, &stmt, NULL);
  if (rc != SQLITE_OK)
    {
      sqlite3_exec (db, "ROLLBACK;", NULL, NULL, NULL);
      return rc;
    }
  while ((rc = sqlite3_step (stmt)) == SQLITE_ROW)
    {
      const void *blob = sqlite3_column_blob (stmt, 0);
      const int size = sqlite3_column_bytes (stmt, 0);
      if (blob == NULL || size != OBJSTORE_ID_SIZE)
        {
          rc = SQLITE_CORRUPT;
          break;
        }
      objstore_id id = { 0 };
      memcpy (id.bytes, blob, OBJSTORE_ID_SIZE);
      rc = sqlite_backend_rowidx_upsert (db, &id);
      if (rc != SQLITE_OK)
        {
          break;
        }
    }
  sqlite3_finalize (stmt);
  if (rc == SQLITE_DONE)
    {
      rc = SQLITE_OK;
    }
  if (rc == SQLITE_OK)
    {
      rc = sqlite3_exec (db, "COMMIT;", NULL, NULL, NULL);
    }
  else
    {
      sqlite3_exec (db, "ROLLBACK;", NULL, NULL, NULL);
    }
  return rc;
}

static size_t
sqlite_backend_effective_chunk (const sqlite_backend_env *env)
{
  return (env != NULL && env->chunk_size > 0) ? env->chunk_size
                                              : OBJSTORE_DEFAULT_CHUNK_SIZE;
}

static int
sqlite_backend_buffer_reader (sqlite_backend_txn *tx,
                              const objstore_stream_reader *reader,
                              sqlite3_int64 max_bytes,
                              unsigned char **out_data,
                              sqlite3_int64 *out_size)
{
  if (tx == NULL || reader == NULL || reader->pull == NULL || out_data == NULL
      || out_size == NULL)
    {
      return SQLITE_MISUSE;
    }
  if (max_bytes <= 0)
    {
      return SQLITE_FULL;
    }

  const size_t chunk = sqlite_backend_effective_chunk (tx->env);
  size_t capacity
      = (size_t)((max_bytes < (sqlite3_int64)chunk) ? max_bytes
                                                    : (sqlite3_int64)chunk);
  if (capacity == 0)
    {
      capacity = (chunk > 0) ? chunk : OBJSTORE_DEFAULT_CHUNK_SIZE;
    }

  unsigned char *buffer = (unsigned char *)sqlite3_malloc64 (capacity);
  if (buffer == NULL)
    {
      return SQLITE_NOMEM;
    }

  sqlite3_int64 total = 0;
  int rc = SQLITE_OK;

  for (;;)
    {
      const sqlite3_int64 remaining_limit = max_bytes - total;
      if (remaining_limit <= 0)
        {
          rc = SQLITE_FULL;
          break;
        }
      size_t request = chunk;
      if ((sqlite3_int64)request > remaining_limit)
        {
          request = (size_t)remaining_limit;
        }
      while ((sqlite3_int64)(capacity - (size_t)total)
             < (sqlite3_int64)request)
        {
          size_t new_capacity = capacity == 0 ? request : capacity * 2;
          if ((sqlite3_int64)new_capacity > max_bytes)
            {
              new_capacity = (size_t)max_bytes;
            }
          unsigned char *resized
              = (unsigned char *)sqlite3_realloc64 (buffer, new_capacity);
          if (resized == NULL)
            {
              sqlite3_free (buffer);
              return SQLITE_NOMEM;
            }
          buffer = resized;
          capacity = new_capacity;
        }

      size_t nread = 0;
      rc = reader->pull (reader->ctx, buffer + (size_t)total, request, &nread);
      if (rc == SQLITE_DONE)
        {
          rc = SQLITE_OK;
          break;
        }
      if (rc != SQLITE_OK)
        {
          break;
        }
      total += (sqlite3_int64)nread;
    }

  if (rc != SQLITE_OK)
    {
      sqlite3_free (buffer);
      return rc;
    }

  *out_data = buffer;
  *out_size = total;
  return SQLITE_OK;
}

static int
sqlite_backend_write_blob_from_buffer (sqlite3_blob *blob,
                                       const unsigned char *payload,
                                       sqlite3_int64 length, size_t chunk_size)
{
  if (blob == NULL || (payload == NULL && length > 0))
    {
      return SQLITE_MISUSE;
    }

  const size_t chunk
      = chunk_size > 0 ? chunk_size : OBJSTORE_DEFAULT_CHUNK_SIZE;
  sqlite3_int64 offset = 0;
  while (offset < length)
    {
      const sqlite3_int64 remaining = length - offset;
      size_t to_write = (size_t)((remaining < (sqlite3_int64)chunk)
                                     ? remaining
                                     : (sqlite3_int64)chunk);
      int rc = sqlite3_blob_write (blob, payload + (size_t)offset,
                                   (int)to_write, (int)offset);
      if (rc != SQLITE_OK)
        {
          return rc;
        }
      offset += (sqlite3_int64)to_write;
    }
  return SQLITE_OK;
}

static int
sqlite_backend_stream_buffer_to_writer (const sqlite_backend_env *env,
                                        const unsigned char *payload,
                                        sqlite3_int64 length,
                                        const objstore_stream_writer *writer)
{
  if (writer == NULL || writer->push == NULL)
    {
      return SQLITE_MISUSE;
    }
  if (length == 0)
    {
      return SQLITE_OK;
    }

  const size_t chunk = sqlite_backend_effective_chunk (env);
  sqlite3_int64 offset = 0;
  while (offset < length)
    {
      const sqlite3_int64 remaining = length - offset;
      size_t to_write = (size_t)((remaining < (sqlite3_int64)chunk)
                                     ? remaining
                                     : (sqlite3_int64)chunk);
      int rc = writer->push (writer->ctx, payload + (size_t)offset, to_write);
      if (rc != SQLITE_OK)
        {
          return rc;
        }
      offset += (sqlite3_int64)to_write;
    }
  return SQLITE_OK;
}

static int
sqlite_backend_stream_buffer_range (const sqlite_backend_env *env,
                                    const unsigned char *payload,
                                    sqlite3_int64 total_length,
                                    sqlite3_uint64 offset,
                                    sqlite3_uint64 length,
                                    const objstore_stream_writer *writer)
{
  if (payload == NULL || writer == NULL || writer->push == NULL)
    {
      return SQLITE_MISUSE;
    }
  if (total_length < 0)
    {
      return SQLITE_IOERR;
    }
  const sqlite3_uint64 total = (sqlite3_uint64)total_length;
  if (offset > total || length > total
      || offset > UINT64_MAX - length || offset + length > total)
    {
      return SQLITE_RANGE;
    }
  if (length == 0)
    {
      return SQLITE_OK;
    }
  return sqlite_backend_stream_buffer_to_writer (
      env, payload + (size_t)offset, (sqlite3_int64)length, writer);
}

static sqlite3_int64
sqlite_backend_queue_remaining_bytes (const sqlite_backend_txn *tx,
                                      sqlite3_int64 reclaim_bytes)
{
  sqlite3_int64 used = tx != NULL ? tx->buffered_bytes : 0;
  used -= reclaim_bytes;
  if (used < 0)
    {
      used = 0;
    }
  return OBJSTORE_SQLITE_QUEUE_LIMIT_BYTES - used;
}

static int
sqlite_backend_queue_ensure_capacity (sqlite_backend_txn *tx)
{
  if (tx == NULL)
    {
      return SQLITE_MISUSE;
    }
  if (tx->write_count < tx->write_capacity)
    {
      return SQLITE_OK;
    }
  size_t new_capacity = tx->write_capacity == 0 ? 4 : tx->write_capacity * 2;
  sqlite_backend_write_entry *resized
      = (sqlite_backend_write_entry *)sqlite3_realloc64 (
          tx->write_queue, new_capacity * sizeof (*resized));
  if (resized == NULL)
    {
      return SQLITE_NOMEM;
    }
  tx->write_queue = resized;
  tx->write_capacity = new_capacity;
  return SQLITE_OK;
}

static int
sqlite_backend_frames_ensure_capacity (sqlite_backend_txn *tx)
{
  if (tx == NULL)
    {
      return SQLITE_MISUSE;
    }
  if (tx->frame_count < tx->frame_capacity)
    {
      return SQLITE_OK;
    }
  size_t new_capacity = tx->frame_capacity == 0 ? 4 : tx->frame_capacity * 2;
  size_t *counts = sqlite3_realloc64 (tx->frame_write_counts,
                                      new_capacity * sizeof (*counts));
  if (counts == NULL)
    {
      return SQLITE_NOMEM;
    }
  sqlite3_int64 *buffered = sqlite3_realloc64 (tx->frame_buffered_bytes,
                                               new_capacity * sizeof (*buffered));
  if (buffered == NULL)
    {
      return SQLITE_NOMEM;
    }
  tx->frame_write_counts = counts;
  tx->frame_buffered_bytes = buffered;
  tx->frame_capacity = new_capacity;
  return SQLITE_OK;
}

static sqlite_backend_write_entry *
sqlite_backend_queue_find_latest (sqlite_backend_txn *tx,
                                  const objstore_id *id, size_t *out_index)
{
  if (tx == NULL || id == NULL || tx->write_count == 0)
    {
      return NULL;
    }
  for (size_t i = tx->write_count; i > 0; --i)
    {
      size_t idx = i - 1;
      if (sqlite_backend_ids_equal (&tx->write_queue[idx].id, id))
        {
          if (out_index != NULL)
            {
              *out_index = idx;
            }
          return &tx->write_queue[idx];
        }
    }
  return NULL;
}

static void
sqlite_backend_write_entry_release (sqlite_backend_txn *tx,
                                    sqlite_backend_write_entry *entry)
{
  if (tx == NULL || entry == NULL)
    {
      return;
    }
  if (entry->kind == SQLITE_BACKEND_WRITE_PUT)
    {
      tx->buffered_bytes -= entry->size;
      if (tx->buffered_bytes < 0)
        {
          tx->buffered_bytes = 0;
        }
      sqlite3_free (entry->payload);
      entry->payload = NULL;
      entry->size = 0;
    }
}

static void
sqlite_backend_queue_remove_at (sqlite_backend_txn *tx, size_t index)
{
  if (tx == NULL || index >= tx->write_count)
    {
      return;
    }
  sqlite_backend_write_entry_release (tx, &tx->write_queue[index]);
  if (index + 1 < tx->write_count)
    {
      memmove (&tx->write_queue[index], &tx->write_queue[index + 1],
               (tx->write_count - index - 1)
                   * sizeof (sqlite_backend_write_entry));
    }
  tx->write_count--;
}

static void
sqlite_backend_txn_clear_queue (sqlite_backend_txn *tx)
{
  if (tx == NULL)
    {
      return;
    }
  for (size_t i = 0; i < tx->write_count; ++i)
    {
      sqlite_backend_write_entry_release (tx, &tx->write_queue[i]);
    }
  sqlite3_free (tx->write_queue);
  tx->write_queue = NULL;
  tx->write_count = 0;
  tx->write_capacity = 0;
  tx->buffered_bytes = 0;
}

static void
sqlite_backend_txn_clear_frames (sqlite_backend_txn *tx)
{
  if (tx == NULL)
    {
      return;
    }
  sqlite3_free (tx->frame_write_counts);
  sqlite3_free (tx->frame_buffered_bytes);
  tx->frame_write_counts = NULL;
  tx->frame_buffered_bytes = NULL;
  tx->frame_count = 0;
  tx->frame_capacity = 0;
}

static int
sqlite_backend_savepoint_begin (objstore_backend_txn *txn)
{
  if (txn == NULL)
    {
      return SQLITE_MISUSE;
    }
  sqlite_backend_txn *tx = (sqlite_backend_txn *)txn;
  int rc = sqlite_backend_frames_ensure_capacity (tx);
  if (rc != SQLITE_OK)
    {
      return rc;
    }
  tx->frame_write_counts[tx->frame_count] = tx->write_count;
  tx->frame_buffered_bytes[tx->frame_count] = tx->buffered_bytes;
  ++tx->frame_count;
  return SQLITE_OK;
}

static int
sqlite_backend_savepoint_release (objstore_backend_txn *txn)
{
  if (txn == NULL)
    {
      return SQLITE_MISUSE;
    }
  sqlite_backend_txn *tx = (sqlite_backend_txn *)txn;
  if (tx->frame_count > 0)
    {
      --tx->frame_count;
    }
  return SQLITE_OK;
}

static int
sqlite_backend_savepoint_rollback (objstore_backend_txn *txn)
{
  if (txn == NULL)
    {
      return SQLITE_MISUSE;
    }
  sqlite_backend_txn *tx = (sqlite_backend_txn *)txn;
  if (tx->frame_count == 0)
    {
      return SQLITE_OK;
    }
  const size_t target_count = tx->frame_write_counts[tx->frame_count - 1];
  while (tx->write_count > target_count)
    {
      sqlite_backend_queue_remove_at (tx, tx->write_count - 1);
    }
  tx->buffered_bytes = tx->frame_buffered_bytes[tx->frame_count - 1];
  if (tx->buffered_bytes < 0)
    {
      tx->buffered_bytes = 0;
    }
  return SQLITE_OK;
}

static int
sqlite_backend_queue_append_delete (sqlite_backend_txn *tx,
                                    const objstore_id *id)
{
  int rc = sqlite_backend_queue_ensure_capacity (tx);
  if (rc != SQLITE_OK)
    {
      return rc;
    }
  sqlite_backend_write_entry *entry = &tx->write_queue[tx->write_count++];
  entry->kind = SQLITE_BACKEND_WRITE_DELETE;
  entry->id = *id;
  entry->payload = NULL;
  entry->size = 0;
  return SQLITE_OK;
}

static int
sqlite_backend_queue_append_put (sqlite_backend_txn *tx, const objstore_id *id,
                                 unsigned char *payload,
                                 sqlite3_int64 payload_size)
{
  int rc = sqlite_backend_queue_ensure_capacity (tx);
  if (rc != SQLITE_OK)
    {
      return rc;
    }
  sqlite_backend_write_entry *entry = &tx->write_queue[tx->write_count++];
  entry->kind = SQLITE_BACKEND_WRITE_PUT;
  entry->id = *id;
  entry->payload = payload;
  entry->size = payload_size;
  tx->buffered_bytes += payload_size;
  return SQLITE_OK;
}

static int
sqlite_backend_validate_put (sqlite_backend_txn *tx, const objstore_id *id,
                             const unsigned char *payload,
                             sqlite3_int64 payload_size, bool *skip_enqueue)
{
  if (skip_enqueue != NULL)
    {
      *skip_enqueue = false;
    }
  size_t latest_index = 0;
  sqlite_backend_write_entry *latest
      = sqlite_backend_queue_find_latest (tx, id, &latest_index);
  bool delete_pending = false;
  if (latest != NULL)
    {
      if (latest->kind == SQLITE_BACKEND_WRITE_PUT)
        {
          if (latest->size == payload_size
              && memcmp (latest->payload, payload, (size_t)payload_size) == 0)
            {
              if (skip_enqueue != NULL)
                {
                  *skip_enqueue = true;
                }
              return SQLITE_OK;
            }
          return SQLITE_CONSTRAINT;
        }
      if (latest->kind == SQLITE_BACKEND_WRITE_DELETE)
        {
          delete_pending = true;
        }
    }

  if (!delete_pending)
    {
      int equal = 0;
      int rc = sqlite_backend_compare_existing_buffer (
          tx->env->primary_db, id, payload, payload_size,
          sqlite_backend_effective_chunk (tx->env), &equal);
      if (rc == SQLITE_OK && equal)
        {
          if (skip_enqueue != NULL)
            {
              *skip_enqueue = true;
            }
          return SQLITE_OK;
        }
      if (rc != SQLITE_NOTFOUND)
        {
          return rc;
        }
    }
  return SQLITE_OK;
}

static int
sqlite_backend_staged_write_begin (objstore_backend_txn *txn,
                                   objstore_backend_staged_writer **out_writer)
{
  if (txn == NULL || out_writer == NULL)
    {
      return SQLITE_MISUSE;
    }
  sqlite_backend_staged_writer *writer
      = (sqlite_backend_staged_writer *)sqlite3_malloc (
          sizeof (sqlite_backend_staged_writer));
  if (writer == NULL)
    {
      return SQLITE_NOMEM;
    }
  memset (writer, 0, sizeof (*writer));
  writer->txn = (sqlite_backend_txn *)txn;
  *out_writer = (objstore_backend_staged_writer *)writer;
  return SQLITE_OK;
}

static int
sqlite_backend_staged_write_push (
    objstore_backend_staged_writer *writer_handle, const void *buffer,
    size_t nread)
{
  if (writer_handle == NULL)
    {
      return SQLITE_MISUSE;
    }
  sqlite_backend_staged_writer *writer = sqlite_writer_from (writer_handle);
  if (writer->txn == NULL || (nread > 0 && buffer == NULL))
    {
      return SQLITE_MISUSE;
    }
  if (nread == 0)
    {
      return SQLITE_OK;
    }

  sqlite3_int64 remaining
      = sqlite_backend_queue_remaining_bytes (writer->txn, writer->size);
  if ((sqlite3_int64)nread > remaining)
    {
      return SQLITE_FULL;
    }

  sqlite3_int64 needed = writer->size + (sqlite3_int64)nread;
  if (needed > writer->capacity)
    {
      sqlite3_int64 new_capacity
          = (writer->capacity == 0) ? (sqlite3_int64)nread : writer->capacity;
      while (new_capacity < needed)
        {
          if (new_capacity > (OBJSTORE_SQLITE_QUEUE_LIMIT_BYTES / 2))
            {
              new_capacity = needed;
              break;
            }
          new_capacity *= 2;
        }
      unsigned char *resized = (unsigned char *)sqlite3_realloc64 (
          writer->payload, (size_t)new_capacity);
      if (resized == NULL)
        {
          return SQLITE_NOMEM;
        }
      writer->payload = resized;
      writer->capacity = new_capacity;
    }

  memcpy (writer->payload + writer->size, buffer, nread);
  writer->size += (sqlite3_int64)nread;
  return SQLITE_OK;
}

static int
sqlite_backend_staged_write_finalize (
    objstore_backend_staged_writer *writer_handle, const objstore_id *id)
{
  sqlite_backend_staged_writer *writer = sqlite_writer_from (writer_handle);
  if (writer == NULL)
    {
      return SQLITE_MISUSE;
    }
  if (id == NULL)
    {
      sqlite3_free (writer->payload);
      sqlite3_free (writer);
      return SQLITE_OK;
    }

  bool skip_enqueue = false;
  int rc = sqlite_backend_validate_put (writer->txn, id, writer->payload,
                                        writer->size, &skip_enqueue);
  if (rc != SQLITE_OK)
    {
      sqlite3_free (writer->payload);
      sqlite3_free (writer);
      return rc;
    }
  if (skip_enqueue)
    {
      sqlite3_free (writer->payload);
      sqlite3_free (writer);
      return SQLITE_OK;
    }

  rc = sqlite_backend_queue_append_put (writer->txn, id, writer->payload,
                                        writer->size);
  if (rc != SQLITE_OK)
    {
      sqlite3_free (writer->payload);
    }
  writer->payload = NULL;
  writer->capacity = 0;
  sqlite3_free (writer);
  return rc;
}

static int
sqlite_backend_insert_placeholder (sqlite3 *db, const objstore_id *id,
                                   sqlite3_int64 length,
                                   sqlite3_int64 *out_rowid)
{
  const char *sql
      = "INSERT INTO objstore_data(id, data) VALUES (?1, zeroblob(?2));";
  sqlite3_stmt *stmt = NULL;
  int rc = sqlite3_prepare_v3 (db, sql, -1, SQLITE_PREPARE_PERSISTENT, &stmt,
                               NULL);
  if (rc != SQLITE_OK)
    {
      return rc;
    }

  rc = sqlite_backend_bind_id (stmt, 1, id);
  if (rc == SQLITE_OK)
    {
      rc = sqlite3_bind_int64 (stmt, 2, length);
    }

  if (rc == SQLITE_OK)
    {
      rc = sqlite3_step (stmt);
    }

  sqlite3_finalize (stmt);

  if (rc == SQLITE_CONSTRAINT)
    {
      return SQLITE_CONSTRAINT;
    }
  if (rc != SQLITE_DONE)
    {
      return (rc == SQLITE_OK) ? SQLITE_ERROR : rc;
    }

  if (out_rowid != NULL)
    {
      *out_rowid = sqlite3_last_insert_rowid (db);
    }
  return SQLITE_OK;
}

static int
sqlite_backend_open_blob (sqlite3 *db, sqlite3_int64 rowid, int writeable,
                          sqlite3_blob **out_blob)
{
  return sqlite3_blob_open (db, "main", "objstore_data", "data", rowid,
                            writeable, out_blob);
}

static int
sqlite_backend_lookup_rowid (sqlite3 *db, const objstore_id *id,
                             sqlite3_int64 *out_rowid)
{
  const char *sql = "SELECT rowid FROM objstore_data WHERE id = ?1;";
  sqlite3_stmt *stmt = NULL;
  int rc = sqlite3_prepare_v3 (db, sql, -1, SQLITE_PREPARE_PERSISTENT, &stmt,
                               NULL);
  if (rc != SQLITE_OK)
    {
      return rc;
    }

  rc = sqlite_backend_bind_id (stmt, 1, id);
  if (rc == SQLITE_OK)
    {
      rc = sqlite3_step (stmt);
    }

  if (rc == SQLITE_ROW)
    {
      if (out_rowid != NULL)
        {
          *out_rowid = sqlite3_column_int64 (stmt, 0);
        }
      rc = SQLITE_OK;
    }
  else if (rc == SQLITE_DONE)
    {
      rc = SQLITE_NOTFOUND;
    }
  else if (rc != SQLITE_OK)
    {
      /* fall through */
    }

  sqlite3_finalize (stmt);
  return rc;
}

static int
sqlite_backend_compare_existing_buffer (sqlite3 *db, const objstore_id *id,
                                        const unsigned char *payload,
                                        sqlite3_int64 length,
                                        size_t chunk_size, int *out_equal)
{
  if (out_equal != NULL)
    {
      *out_equal = 0;
    }

  sqlite3_int64 rowid = 0;
  int rc = sqlite_backend_lookup_rowid (db, id, &rowid);
  if (rc != SQLITE_OK)
    {
      return rc;
    }

  sqlite3_blob *blob = NULL;
  rc = sqlite_backend_open_blob (db, rowid, 0, &blob);
  if (rc != SQLITE_OK)
    {
      return rc;
    }

  const int blob_size = sqlite3_blob_bytes (blob);
  if ((sqlite3_int64)blob_size != length)
    {
      sqlite3_blob_close (blob);
      return SQLITE_CONSTRAINT;
    }

  const size_t chunk
      = chunk_size > 0 ? chunk_size : OBJSTORE_DEFAULT_CHUNK_SIZE;
  unsigned char *buffer = (unsigned char *)sqlite3_malloc64 (chunk);
  if (buffer == NULL)
    {
      sqlite3_blob_close (blob);
      return SQLITE_NOMEM;
    }

  sqlite3_int64 offset = 0;
  int cmp_rc = SQLITE_OK;
  while (offset < length)
    {
      const sqlite3_int64 remaining = length - offset;
      size_t to_read = (size_t)((remaining < (sqlite3_int64)chunk)
                                    ? remaining
                                    : (sqlite3_int64)chunk);
      cmp_rc = sqlite3_blob_read (blob, buffer, (int)to_read, (int)offset);
      if (cmp_rc != SQLITE_OK)
        {
          break;
        }
      if (memcmp (payload + (size_t)offset, buffer, to_read) != 0)
        {
          cmp_rc = SQLITE_CONSTRAINT;
          break;
        }
      offset += (sqlite3_int64)to_read;
    }

  sqlite3_free (buffer);
  sqlite3_blob_close (blob);
  if (cmp_rc == SQLITE_OK && out_equal != NULL)
    {
      *out_equal = 1;
    }
  return cmp_rc;
}

static int
sqlite_backend_delete_row (sqlite3 *db, const objstore_id *id)
{
  const char *sql = "DELETE FROM objstore_data WHERE id = ?1;";
  sqlite3_stmt *stmt = NULL;
  int rc = sqlite3_prepare_v3 (db, sql, -1, SQLITE_PREPARE_PERSISTENT, &stmt,
                               NULL);
  if (rc != SQLITE_OK)
    {
      return rc;
    }
  rc = sqlite_backend_bind_id (stmt, 1, id);
  if (rc == SQLITE_OK)
    {
      rc = sqlite3_step (stmt);
      if (rc == SQLITE_DONE)
        {
          const int deleted = sqlite3_changes (db);
          if (deleted > 0)
            {
              int rowidx_rc = sqlite_backend_rowidx_delete (db, id);
              rc = (rowidx_rc == SQLITE_OK) ? SQLITE_OK : rowidx_rc;
            }
          else
            {
              rc = SQLITE_NOTFOUND;
            }
        }
    }
  sqlite3_finalize (stmt);
  return rc;
}

static int
sqlite_backend_apply_put_entry (sqlite_backend_txn *tx,
                                const sqlite_backend_write_entry *entry)
{
  sqlite3 *db = tx->env->primary_db;
  sqlite3_int64 rowid = 0;
  int rc = sqlite_backend_insert_placeholder (db, &entry->id, entry->size,
                                              &rowid);
  if (rc == SQLITE_CONSTRAINT)
    {
      int equal = 0;
      rc = sqlite_backend_compare_existing_buffer (
          db, &entry->id, entry->payload, entry->size,
          sqlite_backend_effective_chunk (tx->env), &equal);
      if (rc == SQLITE_OK && equal)
        {
          return SQLITE_OK;
        }
      return rc;
    }
  if (rc != SQLITE_OK)
    {
      return rc;
    }

  sqlite3_blob *blob = NULL;
  rc = sqlite_backend_open_blob (db, rowid, 1, &blob);
  if (rc != SQLITE_OK)
    {
      return rc;
    }

  rc = sqlite_backend_write_blob_from_buffer (
      blob, entry->payload, entry->size,
      sqlite_backend_effective_chunk (tx->env));
  sqlite3_blob_close (blob);
  if (rc != SQLITE_OK)
    {
      return rc;
    }
  return sqlite_backend_rowidx_upsert (db, &entry->id);
}

static int
sqlite_backend_flush_queue (sqlite_backend_txn *tx)
{
  if (tx == NULL)
    {
      return SQLITE_MISUSE;
    }

  sqlite3 *db = tx->env->primary_db;
  for (size_t i = 0; i < tx->write_count; ++i)
    {
      sqlite_backend_write_entry *entry = &tx->write_queue[i];
      int rc = SQLITE_OK;
      if (entry->kind == SQLITE_BACKEND_WRITE_DELETE)
        {
          rc = sqlite_backend_delete_row (db, &entry->id);
          if (rc == SQLITE_NOTFOUND)
            {
              rc = SQLITE_OK;
            }
        }
      else
        {
          rc = sqlite_backend_apply_put_entry (tx, entry);
        }

      if (rc != SQLITE_OK)
        {
          return rc;
        }
    }
  return SQLITE_OK;
}

static int
sqlite_backend_put (objstore_backend_txn *txn, const objstore_id *id,
                    const objstore_stream_reader *reader)
{
  if (txn == NULL || id == NULL || reader == NULL || reader->pull == NULL)
    {
      return SQLITE_MISUSE;
    }

  sqlite_backend_txn *tx = (sqlite_backend_txn *)txn;
  assert (tx->env != NULL);
  assert (tx->env->primary_db != NULL);
  int rc = SQLITE_OK;

  size_t latest_index = 0;
  sqlite_backend_write_entry *latest
      = sqlite_backend_queue_find_latest (tx, id, &latest_index);
  sqlite3_int64 reclaim_bytes = 0;
  if (latest != NULL)
    {
      if (latest->kind == SQLITE_BACKEND_WRITE_PUT)
        {
          reclaim_bytes = latest->size;
        }
    }

  const sqlite3_int64 max_payload
      = sqlite_backend_queue_remaining_bytes (tx, reclaim_bytes);
  if (max_payload <= 0)
    {
      return SQLITE_FULL;
    }

  unsigned char *payload = NULL;
  sqlite3_int64 payload_size = 0;
  rc = sqlite_backend_buffer_reader (tx, reader, max_payload, &payload,
                                     &payload_size);
  if (rc != SQLITE_OK)
    {
      return rc;
    }

  bool skip_enqueue = false;
  rc = sqlite_backend_validate_put (tx, id, payload, payload_size,
                                    &skip_enqueue);
  if (rc != SQLITE_OK)
    {
      sqlite3_free (payload);
      return rc;
    }
  if (skip_enqueue)
    {
      sqlite3_free (payload);
      return SQLITE_OK;
    }

  rc = sqlite_backend_queue_append_put (tx, id, payload, payload_size);
  if (rc != SQLITE_OK)
    {
      sqlite3_free (payload);
      return rc;
    }
  return SQLITE_OK;
}

static int
sqlite_backend_get (objstore_backend_txn *txn, const objstore_id *id,
                    const objstore_stream_writer *writer)
{
  if (txn == NULL || id == NULL || writer == NULL || writer->push == NULL)
    {
      return SQLITE_MISUSE;
    }

  sqlite_backend_txn *tx = (sqlite_backend_txn *)txn;
  size_t latest_index = 0;
  sqlite_backend_write_entry *latest
      = sqlite_backend_queue_find_latest (tx, id, &latest_index);
  if (latest != NULL)
    {
      if (latest->kind == SQLITE_BACKEND_WRITE_DELETE)
        {
          return SQLITE_NOTFOUND;
        }
      return sqlite_backend_stream_buffer_to_writer (tx->env, latest->payload,
                                                     latest->size, writer);
    }

  sqlite3 *db = tx->env->primary_db;
  sqlite3_int64 rowid = 0;
  int rc = sqlite_backend_lookup_rowid (db, id, &rowid);
  if (rc != SQLITE_OK)
    {
      return rc;
    }

  sqlite3_blob *blob = NULL;
  rc = sqlite_backend_open_blob (db, rowid, 0, &blob);
  if (rc != SQLITE_OK)
    {
      return rc;
    }

  const int blob_size = sqlite3_blob_bytes (blob);
  unsigned char *buffer
      = (unsigned char *)sqlite3_malloc64 (tx->env->chunk_size);
  if (buffer == NULL)
    {
      sqlite3_blob_close (blob);
      return SQLITE_NOMEM;
    }

  int offset = 0;
  while (offset < blob_size)
    {
      const int remaining = blob_size - offset;
      const int to_read = remaining < (int)tx->env->chunk_size
                              ? remaining
                              : (int)tx->env->chunk_size;
      rc = sqlite3_blob_read (blob, buffer, to_read, offset);
      if (rc != SQLITE_OK)
        {
          break;
        }
      rc = writer->push (writer->ctx, buffer, (size_t)to_read);
      if (rc != SQLITE_OK)
        {
          break;
        }
      offset += to_read;
    }

  sqlite3_free (buffer);
  sqlite3_blob_close (blob);
  return rc;
}

static int
sqlite_backend_get_range (objstore_backend_txn *txn, const objstore_id *id,
                          sqlite3_uint64 offset, sqlite3_uint64 length,
                          const objstore_stream_writer *writer)
{
  if (txn == NULL || id == NULL || writer == NULL || writer->push == NULL)
    {
      return SQLITE_MISUSE;
    }
  if (length == 0)
    {
      return SQLITE_OK;
    }

  sqlite_backend_txn *tx = (sqlite_backend_txn *)txn;
  size_t latest_index = 0;
  sqlite_backend_write_entry *latest
      = sqlite_backend_queue_find_latest (tx, id, &latest_index);
  if (latest != NULL)
    {
      if (latest->kind == SQLITE_BACKEND_WRITE_DELETE)
        {
          return SQLITE_NOTFOUND;
        }
      return sqlite_backend_stream_buffer_range (tx->env, latest->payload,
                                                 latest->size, offset, length,
                                                 writer);
    }

  sqlite3 *db = tx->env->primary_db;
  sqlite3_int64 rowid = 0;
  int rc = sqlite_backend_lookup_rowid (db, id, &rowid);
  if (rc != SQLITE_OK)
    {
      return rc;
    }

  sqlite3_blob *blob = NULL;
  rc = sqlite_backend_open_blob (db, rowid, 0, &blob);
  if (rc != SQLITE_OK)
    {
      return rc;
    }

  const int blob_size = sqlite3_blob_bytes (blob);
  if (blob_size < 0)
    {
      sqlite3_blob_close (blob);
      return SQLITE_IOERR;
    }
  if (offset > (sqlite3_uint64)INT_MAX
      || length > (sqlite3_uint64)INT_MAX)
    {
      sqlite3_blob_close (blob);
      return SQLITE_RANGE;
    }
  if (offset > UINT64_MAX - length)
    {
      sqlite3_blob_close (blob);
      return SQLITE_RANGE;
    }
  if (offset + length > (sqlite3_uint64)blob_size)
    {
      sqlite3_blob_close (blob);
      return SQLITE_RANGE;
    }

  unsigned char *buffer
      = (unsigned char *)sqlite3_malloc64 (tx->env->chunk_size);
  if (buffer == NULL)
    {
      sqlite3_blob_close (blob);
      return SQLITE_NOMEM;
    }

  sqlite3_uint64 remaining = length;
  sqlite3_int64 cursor = (sqlite3_int64)offset;
  while (remaining > 0)
    {
      const sqlite3_uint64 max_chunk = (sqlite3_uint64)tx->env->chunk_size;
      sqlite3_uint64 to_read_u64
          = remaining < max_chunk ? remaining : max_chunk;
      const int to_read = (int)to_read_u64;
      rc = sqlite3_blob_read (blob, buffer, to_read, (int)cursor);
      if (rc != SQLITE_OK)
        {
          break;
        }
      rc = writer->push (writer->ctx, buffer, (size_t)to_read);
      if (rc != SQLITE_OK)
        {
          break;
        }
      remaining -= (sqlite3_uint64)to_read;
      cursor += (sqlite3_int64)to_read;
    }

  sqlite3_free (buffer);
  sqlite3_blob_close (blob);
  return rc;
}

static int
sqlite_backend_get_size (objstore_backend_txn *txn, const objstore_id *id,
                         sqlite3_int64 *out_size)
{
  if (txn == NULL || id == NULL || out_size == NULL)
    {
      return SQLITE_MISUSE;
    }
  sqlite_backend_txn *tx = (sqlite_backend_txn *)txn;
  size_t latest_index = 0;
  sqlite_backend_write_entry *latest
      = sqlite_backend_queue_find_latest (tx, id, &latest_index);
  if (latest != NULL)
    {
      if (latest->kind == SQLITE_BACKEND_WRITE_DELETE)
        {
          return SQLITE_NOTFOUND;
        }
      *out_size = latest->size;
      return SQLITE_OK;
    }

  sqlite3 *db = tx->env->primary_db;
  sqlite3_int64 rowid = 0;
  int rc = sqlite_backend_lookup_rowid (db, id, &rowid);
  if (rc != SQLITE_OK)
    {
      return rc;
    }
  sqlite3_blob *blob = NULL;
  rc = sqlite_backend_open_blob (db, rowid, 0, &blob);
  if (rc != SQLITE_OK)
    {
      return rc;
    }
  *out_size = sqlite3_blob_bytes (blob);
  if (*out_size < 0)
    {
      sqlite3_blob_close (blob);
      return SQLITE_IOERR;
    }
  sqlite3_blob_close (blob);
  return SQLITE_OK;
}

static int
sqlite_backend_delete (objstore_backend_txn *txn, const objstore_id *id)
{
  if (txn == NULL || id == NULL)
    {
      return SQLITE_MISUSE;
    }

  sqlite_backend_txn *tx = (sqlite_backend_txn *)txn;

  size_t latest_index = 0;
  sqlite_backend_write_entry *latest
      = sqlite_backend_queue_find_latest (tx, id, &latest_index);
  if (latest != NULL)
    {
      if (latest->kind == SQLITE_BACKEND_WRITE_PUT)
        {
          if (tx->frame_count > 0)
            {
              return sqlite_backend_queue_append_delete (tx, id);
            }
          sqlite_backend_queue_remove_at (tx, latest_index);
          return SQLITE_OK;
        }
      return SQLITE_NOTFOUND;
    }

  sqlite3 *db = tx->env->primary_db;
  int rc = sqlite_backend_lookup_rowid (db, id, NULL);
  if (rc == SQLITE_NOTFOUND)
    {
      return SQLITE_NOTFOUND;
    }
  if (rc != SQLITE_OK)
    {
      return rc;
    }
  return sqlite_backend_queue_append_delete (tx, id);
}

static int
sqlite_backend_exists (objstore_backend_txn *txn, const objstore_id *id)
{
  if (txn == NULL || id == NULL)
    {
      return SQLITE_MISUSE;
    }

  sqlite_backend_txn *tx = (sqlite_backend_txn *)txn;
  size_t latest_index = 0;
  sqlite_backend_write_entry *latest
      = sqlite_backend_queue_find_latest (tx, id, &latest_index);
  if (latest != NULL)
    {
      return (latest->kind == SQLITE_BACKEND_WRITE_PUT) ? SQLITE_OK
                                                        : SQLITE_NOTFOUND;
    }

  sqlite3 *db = tx->env->primary_db;
  return sqlite_backend_lookup_rowid (db, id, NULL);
}

static int
sqlite_backend_lookup_id_by_rowid (objstore_backend_txn *txn,
                                   sqlite3_int64 rowid, objstore_id *out)
{
  if (txn == NULL || out == NULL)
    {
      return SQLITE_MISUSE;
    }
  sqlite_backend_txn *tx = (sqlite_backend_txn *)txn;
  sqlite3 *db = tx->env->primary_db;
  static const char *kSql
      = "SELECT d.id FROM objstore_rowidx AS r "
        "JOIN objstore_data AS d ON r.id = d.id "
        "WHERE r.rowid_prefix = ?1 LIMIT 1;";
  sqlite3_stmt *stmt = NULL;
  int rc = sqlite3_prepare_v3 (db, kSql, -1, SQLITE_PREPARE_PERSISTENT, &stmt,
                               NULL);
  if (rc != SQLITE_OK)
    {
      return rc;
    }
  unsigned char prefix[OBJSTORE_ROWID_PREFIX_SIZE];
  objstore_rowid_to_bytes (rowid, prefix);
  rc = sqlite3_bind_blob (stmt, 1, prefix, OBJSTORE_ROWID_PREFIX_SIZE,
                          SQLITE_TRANSIENT);
  if (rc != SQLITE_OK)
    {
      sqlite3_finalize (stmt);
      return rc;
    }
  rc = sqlite3_step (stmt);
  if (rc == SQLITE_ROW)
    {
      const void *blob = sqlite3_column_blob (stmt, 0);
      const int size = sqlite3_column_bytes (stmt, 0);
      if (blob == NULL || size != OBJSTORE_ID_SIZE)
        {
          rc = SQLITE_CORRUPT;
        }
      else
        {
          memcpy (out->bytes, blob, OBJSTORE_ID_SIZE);
          rc = SQLITE_OK;
        }
    }
  else if (rc == SQLITE_DONE)
    {
      rc = SQLITE_NOTFOUND;
    }
  sqlite3_finalize (stmt);
  return rc;
}

static int
sqlite_backend_scan_open (objstore_backend_txn *txn,
                          objstore_backend_cursor **out_cursor)
{
  if (txn == NULL || out_cursor == NULL)
    {
      return SQLITE_MISUSE;
    }

  sqlite_backend_txn *tx = (sqlite_backend_txn *)txn;
  const char *sql = "SELECT id FROM objstore_data ORDER BY id;";
  sqlite3_stmt *stmt = NULL;
  sqlite3 *db = tx->env->primary_db;
  int rc = sqlite3_prepare_v3 (db, sql, -1, SQLITE_PREPARE_PERSISTENT, &stmt,
                               NULL);
  if (rc != SQLITE_OK)
    {
      return rc;
    }

  sqlite_backend_cursor *cursor = sqlite3_malloc (sizeof (*cursor));
  if (cursor == NULL)
    {
      sqlite3_finalize (stmt);
      return SQLITE_NOMEM;
    }
  cursor->stmt = stmt;
  cursor->env = tx->env;

  *out_cursor = (objstore_backend_cursor *)cursor;
  return SQLITE_OK;
}

static int
sqlite_backend_scan_next (objstore_backend_cursor *cursor_base,
                          objstore_id *out_id)
{
  if (cursor_base == NULL || out_id == NULL)
    {
      return SQLITE_MISUSE;
    }

  sqlite_backend_cursor *cursor = (sqlite_backend_cursor *)cursor_base;
  int rc = sqlite3_step (cursor->stmt);
  if (rc == SQLITE_DONE)
    {
      return SQLITE_DONE;
    }
  if (rc != SQLITE_ROW)
    {
      return rc;
    }

  const void *blob = sqlite3_column_blob (cursor->stmt, 0);
  const int size = sqlite3_column_bytes (cursor->stmt, 0);
  if (blob == NULL || size != OBJSTORE_ID_SIZE)
    {
      return SQLITE_CORRUPT;
    }

  memcpy (out_id->bytes, blob, OBJSTORE_ID_SIZE);
  return SQLITE_OK;
}

static void
sqlite_backend_scan_close (objstore_backend_cursor *cursor_base)
{
  if (cursor_base == NULL)
    {
      return;
    }
  sqlite_backend_cursor *cursor = (sqlite_backend_cursor *)cursor_base;
  sqlite3_finalize (cursor->stmt);
  sqlite3_free (cursor);
}

static int
sqlite_backend_open_env (sqlite3 *db, const objstore_config *config,
                         objstore_backend_env **out_env)
{
  if (db == NULL || out_env == NULL)
    {
      return SQLITE_MISUSE;
    }

  int rc = sqlite_backend_ensure_schema (db);
  if (rc != SQLITE_OK)
    {
      return rc;
    }

  rc = sqlite_backend_rowidx_backfill (db);
  if (rc != SQLITE_OK)
    {
      return rc;
    }

  sqlite_backend_env *env = sqlite3_malloc (sizeof (*env));
  if (env == NULL)
    {
      return SQLITE_NOMEM;
    }
  env->primary_db = db;
  env->chunk_size = objstore_effective_chunk_size (config);

  *out_env = (objstore_backend_env *)env;
  return SQLITE_OK;
}

static void
sqlite_backend_close_env (objstore_backend_env *env)
{
  if (env == NULL)
    {
      return;
    }
  sqlite_backend_env *sqlite_env = (sqlite_backend_env *)env;
  sqlite3_free (sqlite_env);
}

static int
sqlite_backend_begin_txn (objstore_backend_env *env,
                          objstore_backend_txn **out_txn)
{
  if (env == NULL || out_txn == NULL)
    {
      return SQLITE_MISUSE;
    }

  sqlite_backend_txn *txn = sqlite3_malloc (sizeof (*txn));
  if (txn == NULL)
    {
      return SQLITE_NOMEM;
    }
  txn->env = (sqlite_backend_env *)env;
  txn->write_queue = NULL;
  txn->write_count = 0;
  txn->write_capacity = 0;
  txn->frame_write_counts = NULL;
  txn->frame_buffered_bytes = NULL;
  txn->frame_count = 0;
  txn->frame_capacity = 0;
  txn->buffered_bytes = 0;
  txn->staged_flushed = false;
  *out_txn = (objstore_backend_txn *)txn;
  return SQLITE_OK;
}

static int
sqlite_backend_commit_txn (objstore_backend_txn *txn)
{
  if (txn == NULL)
    {
      return SQLITE_MISUSE;
    }
  sqlite_backend_txn *tx = (sqlite_backend_txn *)txn;
  int rc = SQLITE_OK;
  if (!tx->staged_flushed)
    {
      if (tx->write_count > 0)
        {
          /* Apply buffered mutations before COMMIT so the backend still
             commits first and SQLite can cancel its own commit if the flush
             fails. */
          rc = sqlite_backend_flush_queue (tx);
        }
      sqlite_backend_txn_clear_queue (tx);
    }
  sqlite_backend_txn_clear_frames (tx);
  sqlite3_free (tx);
  return rc;
}

static void
sqlite_backend_rollback_txn (objstore_backend_txn *txn)
{
  if (txn == NULL)
    {
      return;
    }
  sqlite_backend_txn *tx = (sqlite_backend_txn *)txn;
  sqlite_backend_txn_clear_queue (tx);
  sqlite_backend_txn_clear_frames (tx);
  sqlite3_free (tx);
}

static int
sqlite_backend_commit_staged (objstore_backend_txn *txn)
{
  if (txn == NULL)
    {
      return SQLITE_MISUSE;
    }
  sqlite_backend_txn *tx = (sqlite_backend_txn *)txn;
  if (tx->write_count == 0)
    {
      tx->staged_flushed = true;
      return SQLITE_OK;
    }
  int rc = sqlite_backend_flush_queue (tx);
  if (rc == SQLITE_OK)
    {
      sqlite_backend_txn_clear_queue (tx);
      tx->staged_flushed = true;
    }
  return rc;
}

static void
sqlite_backend_rollback_staged (objstore_backend_txn *txn)
{
  if (txn == NULL)
    {
      return;
    }
  sqlite_backend_txn *tx = (sqlite_backend_txn *)txn;
  sqlite_backend_txn_clear_queue (tx);
  sqlite_backend_txn_clear_frames (tx);
  tx->staged_flushed = true;
}

const objstore_backend objstore_backend_sqlite = {
  .kind = OBJSTORE_BACKEND_SQLITE,
  .name = "sqlite",
  .open_env = sqlite_backend_open_env,
  .close_env = sqlite_backend_close_env,
  .begin_txn = sqlite_backend_begin_txn,
  .commit_txn = sqlite_backend_commit_txn,
  .rollback_txn = sqlite_backend_rollback_txn,
  .savepoint_begin = sqlite_backend_savepoint_begin,
  .savepoint_release = sqlite_backend_savepoint_release,
  .savepoint_rollback = sqlite_backend_savepoint_rollback,
  .staged_write_begin = sqlite_backend_staged_write_begin,
  .staged_write_push = sqlite_backend_staged_write_push,
  .staged_write_finalize = sqlite_backend_staged_write_finalize,
  .staged_write_set_size_hint = NULL,
  .commit_staged = sqlite_backend_commit_staged,
  .rollback_staged = sqlite_backend_rollback_staged,
  .put = sqlite_backend_put,
  .get = sqlite_backend_get,
  .get_range = sqlite_backend_get_range,
  .get_size = sqlite_backend_get_size,
  .delete_fn = sqlite_backend_delete,
  .exists = sqlite_backend_exists,
  .scan_open = sqlite_backend_scan_open,
  .scan_next = sqlite_backend_scan_next,
  .scan_close = sqlite_backend_scan_close,
  .lookup_id_by_rowid = sqlite_backend_lookup_id_by_rowid,
};
