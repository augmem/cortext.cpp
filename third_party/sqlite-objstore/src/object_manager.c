#include "objstore/object_manager.h"

#include <stdbool.h>
#include <stdint.h>
#include <string.h>

#include "objstore/blake3.h"
#include "objstore/txn.h"
#include "objstore/vtab.h"

typedef struct objstore_value_reader
{
  const uint8_t *data;
  size_t size;
  size_t offset;
} objstore_value_reader;

typedef struct objstore_blob_builder
{
  sqlite3_context *ctx;
  uint8_t *data;
  size_t size;
  size_t capacity;
} objstore_blob_builder;

typedef struct objstore_hash_writer
{
  objstore_blake3 *hash;
} objstore_hash_writer;

typedef struct objstore_range_writer
{
  const objstore_stream_writer *inner;
  sqlite3_uint64 offset;
  sqlite3_uint64 remaining;
  sqlite3_uint64 skipped;
} objstore_range_writer;

static int objstore_value_reader_pull (void *ctx, void *buffer,
                                       size_t capacity, size_t *nread);
static int objstore_blob_builder_reserve (objstore_blob_builder *builder,
                                          size_t needed);
static int objstore_blob_builder_push (void *ctx, const void *buffer,
                                       size_t nread);
static void objstore_blob_builder_reset (objstore_blob_builder *builder);
static int objstore_range_writer_push (void *ctx, const void *buffer,
                                       size_t nread);
static int objstore_range_resolve (const objstore_range_spec *range,
                                   sqlite3_int64 size,
                                   sqlite3_uint64 *out_offset,
                                   sqlite3_uint64 *out_length);
static int objstore_connection_backend_exists (objstore_connection *conn,
                                               const objstore_id *id,
                                               int *out_present);
static int objstore_object_stage_stream (objstore_connection *conn,
                                         const objstore_stream_reader *reader,
                                         const objstore_id *explicit_id,
                                         objstore_id *out_id);
static int objstore_hash_writer_push (void *ctx, const void *buffer,
                                      size_t nread);

int
objstore_object_validate_data_value (sqlite3_value *value)
{
  if (value == NULL)
    {
      return SQLITE_MISUSE;
    }
  if (sqlite3_value_type (value) == SQLITE_NULL)
    {
      return SQLITE_MISMATCH;
    }
  if (sqlite3_value_type (value) != SQLITE_BLOB)
    {
      return SQLITE_MISMATCH;
    }
  return SQLITE_OK;
}

int
objstore_object_put_value (objstore_connection *conn, sqlite3_value *value,
                           objstore_id *out_id)
{
  int rc = objstore_object_validate_data_value (value);
  if (rc != SQLITE_OK)
    {
      return rc;
    }
  const uint8_t *blob = (const uint8_t *)sqlite3_value_blob (value);
  size_t blob_size = (size_t)sqlite3_value_bytes (value);
  if (blob_size > 0 && blob == NULL)
    {
      return SQLITE_NOMEM;
    }

  objstore_value_reader reader_state
      = { .data = blob, .size = blob_size, .offset = 0 };
  objstore_stream_reader reader = {
    .ctx = &reader_state,
    .pull = objstore_value_reader_pull,
    .size_hint = (sqlite3_int64)blob_size,
  };

  objstore_id staged_id;
  rc = objstore_object_stage_stream (conn, &reader, NULL, &staged_id);
  if (rc == SQLITE_OK && out_id != NULL)
    {
      *out_id = staged_id;
    }
  return rc;
}

int
objstore_object_put_value_with_id (objstore_connection *conn,
                                   const objstore_id *id, sqlite3_value *value)
{
  if (id == NULL)
    {
      return SQLITE_MISUSE;
    }
  int rc = objstore_object_validate_data_value (value);
  if (rc != SQLITE_OK)
    {
      return rc;
    }
  const uint8_t *blob = (const uint8_t *)sqlite3_value_blob (value);
  size_t blob_size = (size_t)sqlite3_value_bytes (value);
  if (blob_size > 0 && blob == NULL)
    {
      return SQLITE_NOMEM;
    }
  objstore_value_reader reader_state
      = { .data = blob, .size = blob_size, .offset = 0 };
  objstore_stream_reader reader = {
    .ctx = &reader_state,
    .pull = objstore_value_reader_pull,
    .size_hint = (sqlite3_int64)blob_size,
  };
  return objstore_object_stage_stream (conn, &reader, id, NULL);
}

static int
objstore_object_stage_stream (objstore_connection *conn,
                              const objstore_stream_reader *reader,
                              const objstore_id *explicit_id,
                              objstore_id *out_id)
{
  if (conn == NULL || reader == NULL || reader->pull == NULL)
    {
      return SQLITE_MISUSE;
    }

  objstore_backend_txn *write_txn = NULL;
  int rc = objstore_connection_begin_write (conn, &write_txn);
  if (rc != SQLITE_OK)
    {
      return rc;
    }

  objstore_backend_staged_writer *writer = NULL;
  rc = conn->backend->staged_write_begin (write_txn, &writer);
  if (rc != SQLITE_OK)
    {
      return rc;
    }
  if (reader->size_hint >= 0
      && conn->backend->staged_write_set_size_hint != NULL)
    {
      rc = conn->backend->staged_write_set_size_hint (writer,
                                                      reader->size_hint);
      if (rc != SQLITE_OK)
        {
          conn->backend->staged_write_finalize (writer, NULL);
          return rc;
        }
    }

  bool need_hash = (explicit_id == NULL);
  objstore_blake3 hash;
  if (need_hash)
    {
      objstore_blake3_init (&hash);
    }

  sqlite3_int64 total_bytes = 0;
  const size_t chunk_size = conn->chunk_size;
  unsigned char *buffer = sqlite3_malloc64 (chunk_size);
  if (buffer == NULL)
    {
      conn->backend->staged_write_finalize (writer, NULL);
      return SQLITE_NOMEM;
    }

  while (true)
    {
      size_t nread = 0;
      rc = reader->pull (reader->ctx, buffer, chunk_size, &nread);
      if (rc == SQLITE_DONE)
        {
          rc = SQLITE_OK;
          break;
        }
      if (rc != SQLITE_OK)
        {
          conn->backend->staged_write_finalize (writer, NULL);
          return rc;
        }
      if (nread == 0)
        {
          continue;
        }
      if (need_hash)
        {
          objstore_blake3_update (&hash, buffer, nread);
        }
      total_bytes += (sqlite3_int64)nread;
      rc = conn->backend->staged_write_push (writer, buffer, nread);
      if (rc != SQLITE_OK)
        {
          conn->backend->staged_write_finalize (writer, NULL);
          return rc;
        }
    }

  sqlite3_free (buffer);

  objstore_id final_id;
  if (explicit_id != NULL)
    {
      final_id = *explicit_id;
    }
  else
    {
      objstore_blake3_final (&hash, &final_id);
    }

  rc = objstore_txn_log_append_put (conn->txn_log, &final_id, total_bytes);
  if (rc != SQLITE_OK)
    {
      conn->backend->staged_write_finalize (writer, NULL);
      return rc;
    }

  int finalize_rc = conn->backend->staged_write_finalize (writer, &final_id);
  if (finalize_rc != SQLITE_OK)
    {
      objstore_txn_log_drop_last (conn->txn_log);
      return finalize_rc;
    }

  if (out_id != NULL)
    {
      *out_id = final_id;
    }
  return SQLITE_OK;
}

int
objstore_object_put_reader (objstore_connection *conn,
                            const objstore_stream_reader *reader,
                            objstore_id *out_id)
{
  if (conn == NULL || reader == NULL || reader->pull == NULL)
    {
      return SQLITE_MISUSE;
    }
  return objstore_object_stage_stream (conn, reader, NULL, out_id);
}

int
objstore_object_put_reader_with_id (objstore_connection *conn,
                                    const objstore_id *id,
                                    const objstore_stream_reader *reader)
{
  if (id == NULL || reader == NULL)
    {
      return SQLITE_MISUSE;
    }
  return objstore_object_stage_stream (conn, reader, id, NULL);
}

int
objstore_object_delete (objstore_connection *conn, const objstore_id *id)
{
  if (conn == NULL || id == NULL)
    {
      return SQLITE_MISUSE;
    }
  objstore_backend_txn *write_txn = NULL;
  int rc = objstore_connection_begin_write (conn, &write_txn);
  if (rc != SQLITE_OK)
    {
      return rc;
    }
  rc = objstore_txn_log_append_delete (conn->txn_log, id);
  if (rc != SQLITE_OK)
    {
      return rc;
    }
  rc = conn->backend->delete_fn (write_txn, id);
  if (rc == SQLITE_NOTFOUND)
    {
      rc = SQLITE_OK;
    }
  if (rc != SQLITE_OK)
    {
      objstore_txn_log_drop_last (conn->txn_log);
      return rc;
    }
  return SQLITE_OK;
}

int
objstore_object_exists (objstore_connection *conn, const objstore_id *id,
                        int *out_present)
{
  if (conn == NULL || id == NULL || out_present == NULL)
    {
      return SQLITE_MISUSE;
    }
  if (conn->txn_log != NULL)
    {
      objstore_txn_lookup_state state = objstore_txn_log_state_for_id (
          conn->txn_log, id, (sqlite3_uint64)-1);
      if (state == OBJSTORE_TXN_LOOKUP_PUT)
        {
          *out_present = 1;
          return SQLITE_OK;
        }
      if (state == OBJSTORE_TXN_LOOKUP_DELETE)
        {
          *out_present = 0;
          return SQLITE_OK;
        }
    }
  return objstore_connection_backend_exists (conn, id, out_present);
}

int
objstore_object_read_stream (objstore_connection *conn,
                             objstore_backend_txn *txn, const objstore_id *id,
                             sqlite3_uint64 sequence_limit,
                             const objstore_stream_writer *writer)
{
  if (conn == NULL || txn == NULL || id == NULL || writer == NULL
      || writer->push == NULL)
    {
      return SQLITE_MISUSE;
    }
  objstore_txn_lookup_state state
      = objstore_txn_log_state_for_id (conn->txn_log, id, sequence_limit);
  if (state == OBJSTORE_TXN_LOOKUP_PUT && conn->txn_log != NULL)
    {
      if (conn->write_txn == NULL)
        {
          return SQLITE_MISUSE;
        }
      return conn->backend->get (conn->write_txn, id, writer);
    }
  if (state == OBJSTORE_TXN_LOOKUP_DELETE)
    {
      return SQLITE_NOTFOUND;
    }
  return conn->backend->get (txn, id, writer);
}

int
objstore_object_read_stream_range (objstore_connection *conn,
                                   objstore_backend_txn *txn,
                                   const objstore_id *id,
                                   sqlite3_uint64 sequence_limit,
                                   const objstore_range_spec *range,
                                   const objstore_stream_writer *writer)
{
  if (conn == NULL || txn == NULL || id == NULL || range == NULL
      || writer == NULL || writer->push == NULL)
    {
      return SQLITE_MISUSE;
    }
  objstore_txn_lookup_state state
      = objstore_txn_log_state_for_id (conn->txn_log, id, sequence_limit);
  objstore_backend_txn *target_txn = txn;
  if (state == OBJSTORE_TXN_LOOKUP_PUT && conn->txn_log != NULL)
    {
      if (conn->write_txn == NULL)
        {
          return SQLITE_MISUSE;
        }
      target_txn = conn->write_txn;
    }
  else if (state == OBJSTORE_TXN_LOOKUP_DELETE)
    {
      return SQLITE_NOTFOUND;
    }

  sqlite3_int64 size = 0;
  int rc = conn->backend->get_size (target_txn, id, &size);
  if (rc != SQLITE_OK)
    {
      return rc;
    }
  sqlite3_uint64 offset = 0;
  sqlite3_uint64 length = 0;
  rc = objstore_range_resolve (range, size, &offset, &length);
  if (rc != SQLITE_OK)
    {
      return rc;
    }
  if (conn->backend->get_range != NULL)
    {
      return conn->backend->get_range (target_txn, id, offset, length, writer);
    }
  objstore_range_writer range_writer = {
    .inner = writer,
    .offset = offset,
    .remaining = length,
    .skipped = 0,
  };
  objstore_stream_writer wrapper = {
    .ctx = &range_writer,
    .push = objstore_range_writer_push,
  };
  return conn->backend->get (target_txn, id, &wrapper);
}

int
objstore_object_read_blob (sqlite3_context *ctx, objstore_connection *conn,
                           objstore_backend_txn *txn, const objstore_id *id,
                           sqlite3_uint64 sequence_limit)
{
  if (ctx == NULL)
    {
      return SQLITE_MISUSE;
    }
  objstore_blob_builder builder
      = { .ctx = ctx, .data = NULL, .size = 0, .capacity = 0 };
  objstore_stream_writer writer = {
    .ctx = &builder,
    .push = objstore_blob_builder_push,
  };
  int rc
      = objstore_object_read_stream (conn, txn, id, sequence_limit, &writer);
  if (rc == SQLITE_NOTFOUND)
    {
      sqlite3_result_null (ctx);
      objstore_blob_builder_reset (&builder);
      return SQLITE_OK;
    }
  if (rc != SQLITE_OK)
    {
      objstore_blob_builder_reset (&builder);
      return rc;
    }
  sqlite3_result_blob (ctx, builder.data, (int)builder.size, SQLITE_TRANSIENT);
  objstore_blob_builder_reset (&builder);
  return SQLITE_OK;
}

int
objstore_object_read_blob_range (sqlite3_context *ctx, objstore_connection *conn,
                                 objstore_backend_txn *txn,
                                 const objstore_id *id,
                                 sqlite3_uint64 sequence_limit,
                                 const objstore_range_spec *range)
{
  if (ctx == NULL)
    {
      return SQLITE_MISUSE;
    }
  objstore_blob_builder builder
      = { .ctx = ctx, .data = NULL, .size = 0, .capacity = 0 };
  objstore_stream_writer writer = {
    .ctx = &builder,
    .push = objstore_blob_builder_push,
  };
  int rc = objstore_object_read_stream_range (conn, txn, id, sequence_limit,
                                              range, &writer);
  if (rc == SQLITE_NOTFOUND)
    {
      sqlite3_result_null (ctx);
      objstore_blob_builder_reset (&builder);
      return SQLITE_OK;
    }
  if (rc != SQLITE_OK)
    {
      objstore_blob_builder_reset (&builder);
      return rc;
    }
  sqlite3_result_blob (ctx, builder.data, (int)builder.size, SQLITE_TRANSIENT);
  objstore_blob_builder_reset (&builder);
  return SQLITE_OK;
}

int
objstore_object_verify (objstore_connection *conn, const objstore_id *id,
                        int *out_valid)
{
  if (conn == NULL || id == NULL || out_valid == NULL)
    {
      return SQLITE_MISUSE;
    }
  *out_valid = 0;
  objstore_backend_txn *txn = NULL;
  int rc = objstore_begin_read_txn (conn, &txn);
  if (rc != SQLITE_OK)
    {
      return rc;
    }

  objstore_blake3 hash_ctx;
  objstore_blake3_init (&hash_ctx);
  objstore_hash_writer writer_ctx = { .hash = &hash_ctx };
  objstore_stream_writer writer = {
    .ctx = &writer_ctx,
    .push = objstore_hash_writer_push,
  };

  rc = objstore_object_read_stream (conn, txn, id, (sqlite3_uint64)-1,
                                    &writer);
  int end_rc = objstore_end_read_txn (conn, txn, rc);
  if (rc == SQLITE_NOTFOUND)
    {
      return SQLITE_NOTFOUND;
    }
  if (rc != SQLITE_OK)
    {
      return rc;
    }

  objstore_id computed;
  objstore_blake3_final (&hash_ctx, &computed);
  *out_valid = (memcmp (computed.bytes, id->bytes, OBJSTORE_ID_SIZE) == 0);
  return end_rc;
}

static int
objstore_connection_backend_exists (objstore_connection *conn,
                                    const objstore_id *id, int *out_present)
{
  if (conn == NULL || id == NULL || out_present == NULL)
    {
      return SQLITE_MISUSE;
    }
  objstore_backend_txn *read_txn = NULL;
  int rc = objstore_begin_read_txn (conn, &read_txn);
  if (rc != SQLITE_OK)
    {
      return rc;
    }
  rc = conn->backend->exists (read_txn, id);
  int end_rc = objstore_end_read_txn (
      conn, read_txn, (rc == SQLITE_NOTFOUND) ? SQLITE_DONE : rc);
  if (rc == SQLITE_NOTFOUND)
    {
      *out_present = 0;
      return (end_rc == SQLITE_OK) ? SQLITE_OK : end_rc;
    }
  if (rc != SQLITE_OK)
    {
      return rc;
    }
  *out_present = 1;
  return end_rc;
}

static int
objstore_value_reader_pull (void *ctx, void *buffer, size_t capacity,
                            size_t *nread)
{
  objstore_value_reader *reader = (objstore_value_reader *)ctx;
  if (reader == NULL || buffer == NULL || nread == NULL)
    {
      return SQLITE_MISUSE;
    }
  if (reader->offset >= reader->size)
    {
      *nread = 0;
      return SQLITE_DONE;
    }
  size_t remaining = reader->size - reader->offset;
  size_t to_copy = remaining < capacity ? remaining : capacity;
  if (to_copy > 0)
    {
      memcpy (buffer, reader->data + reader->offset, to_copy);
      reader->offset += to_copy;
    }
  *nread = to_copy;
  return SQLITE_OK;
}

static int
objstore_blob_builder_reserve (objstore_blob_builder *builder, size_t needed)
{
  if (builder->capacity >= needed)
    {
      return SQLITE_OK;
    }
  size_t new_capacity = builder->capacity == 0 ? OBJSTORE_DEFAULT_CHUNK_SIZE
                                               : builder->capacity;
  while (new_capacity < needed)
    {
      new_capacity *= 2u;
    }
  uint8_t *resized
      = (uint8_t *)sqlite3_realloc64 (builder->data, new_capacity);
  if (resized == NULL)
    {
      return SQLITE_NOMEM;
    }
  builder->data = resized;
  builder->capacity = new_capacity;
  return SQLITE_OK;
}

static int
objstore_blob_builder_push (void *ctx, const void *buffer, size_t nread)
{
  objstore_blob_builder *builder = (objstore_blob_builder *)ctx;
  if (builder == NULL || (nread > 0 && buffer == NULL))
    {
      return SQLITE_MISUSE;
    }
  if (nread == 0)
    {
      return SQLITE_OK;
    }
  int rc = objstore_blob_builder_reserve (builder, builder->size + nread);
  if (rc != SQLITE_OK)
    {
      return rc;
    }
  memcpy (builder->data + builder->size, buffer, nread);
  builder->size += nread;
  return SQLITE_OK;
}

static void
objstore_blob_builder_reset (objstore_blob_builder *builder)
{
  if (builder == NULL)
    {
      return;
    }
  sqlite3_free (builder->data);
  builder->data = NULL;
  builder->size = 0;
  builder->capacity = 0;
}

static int
objstore_range_writer_push (void *ctx, const void *buffer, size_t nread)
{
  objstore_range_writer *range = (objstore_range_writer *)ctx;
  if (range == NULL || range->inner == NULL || range->inner->push == NULL
      || (nread > 0 && buffer == NULL))
    {
      return SQLITE_MISUSE;
    }
  if (nread == 0 || range->remaining == 0)
    {
      return SQLITE_OK;
    }
  const uint8_t *cursor = (const uint8_t *)buffer;
  size_t available = nread;
  if (range->skipped < range->offset)
    {
      sqlite3_uint64 to_skip = range->offset - range->skipped;
      size_t skip_now = (size_t)((to_skip < (sqlite3_uint64)available)
                                     ? to_skip
                                     : available);
      range->skipped += skip_now;
      cursor += skip_now;
      available -= skip_now;
      if (available == 0)
        {
          return SQLITE_OK;
        }
    }
  size_t to_send = available;
  if ((sqlite3_uint64)to_send > range->remaining)
    {
      to_send = (size_t)range->remaining;
    }
  int rc = SQLITE_OK;
  if (to_send > 0)
    {
      rc = range->inner->push (range->inner->ctx, cursor, to_send);
      if (rc != SQLITE_OK)
        {
          return rc;
        }
      range->remaining -= (sqlite3_uint64)to_send;
    }
  return SQLITE_OK;
}

static int
objstore_range_resolve (const objstore_range_spec *range, sqlite3_int64 size,
                        sqlite3_uint64 *out_offset, sqlite3_uint64 *out_length)
{
  if (range == NULL || out_offset == NULL || out_length == NULL)
    {
      return SQLITE_MISUSE;
    }
  if (!range->has_start && !range->has_end)
    {
      return SQLITE_MISUSE;
    }
  if (size < 0)
    {
      return SQLITE_IOERR;
    }
  const sqlite3_uint64 total = (sqlite3_uint64)size;
  if (total == 0)
    {
      return SQLITE_RANGE;
    }
  if (range->has_start && range->has_end)
    {
      if (range->end < range->start)
        {
          return SQLITE_RANGE;
        }
      if (range->start >= total)
        {
          return SQLITE_RANGE;
        }
      sqlite3_uint64 end = range->end;
      if (end >= total)
        {
          end = total - 1u;
        }
      if (end < range->start)
        {
          return SQLITE_RANGE;
        }
      *out_offset = range->start;
      *out_length = end - range->start + 1u;
      return SQLITE_OK;
    }
  if (range->has_start)
    {
      if (range->start >= total)
        {
          return SQLITE_RANGE;
        }
      *out_offset = range->start;
      *out_length = total - range->start;
      return SQLITE_OK;
    }
  if (range->end == 0)
    {
      return SQLITE_RANGE;
    }
  if (range->end >= total)
    {
      *out_offset = 0;
      *out_length = total;
      return SQLITE_OK;
    }
  *out_offset = total - range->end;
  *out_length = range->end;
  return SQLITE_OK;
}

static int
objstore_hash_writer_push (void *ctx, const void *buffer, size_t nread)
{
  objstore_hash_writer *writer = (objstore_hash_writer *)ctx;
  if (writer == NULL || writer->hash == NULL)
    {
      return SQLITE_MISUSE;
    }
  if (nread > 0 && buffer != NULL)
    {
      objstore_blake3_update (writer->hash, buffer, nread);
    }
  return SQLITE_OK;
}
