#include "objstore/txn.h"

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define OBJSTORE_TXN_SEQUENCE_MAX ((sqlite3_uint64) - 1)

typedef struct objstore_txn_entry_internal
{
  objstore_txn_entry pub;
} objstore_txn_entry_internal;

struct objstore_txn_log
{
  sqlite3 *db;
  objstore_txn_entry_internal *entries;
  size_t entry_count;
  size_t entry_capacity;
  size_t *frames;
  size_t frame_count;
  sqlite3_uint64 operation_count;
};

struct objstore_txn_snapshot
{
  const objstore_txn_log *log;
  size_t *op_indexes;
  size_t count;
};

typedef enum objstore_txn_snapshot_slot_state
{
  OBJSTORE_TXN_SNAPSHOT_SLOT_EMPTY = 0,
  OBJSTORE_TXN_SNAPSHOT_SLOT_OCCUPIED = 1,
  OBJSTORE_TXN_SNAPSHOT_SLOT_TOMBSTONE = 2,
} objstore_txn_snapshot_slot_state;

typedef struct objstore_txn_snapshot_slot
{
  objstore_id id;
  size_t snapshot_index;
  objstore_txn_snapshot_slot_state state;
} objstore_txn_snapshot_slot;

static int
objstore_txn_log_reserve_entries (objstore_txn_log *log)
{
  if (log->entry_count < log->entry_capacity)
    {
      return SQLITE_OK;
    }
  size_t new_capacity
      = (log->entry_capacity == 0) ? 8 : log->entry_capacity * 2;
  objstore_txn_entry_internal *resized = sqlite3_realloc64 (
      log->entries, new_capacity * sizeof (*log->entries));
  if (resized == NULL)
    {
      return SQLITE_NOMEM;
    }
  log->entries = resized;
  log->entry_capacity = new_capacity;
  return SQLITE_OK;
}

static void
objstore_txn_entry_internal_dispose (objstore_txn_entry_internal *entry)
{
  if (entry == NULL)
    {
      return;
    }
  memset (&entry->pub, 0, sizeof (entry->pub));
}

static void
objstore_txn_log_wipe_entries (objstore_txn_log *log)
{
  if (log == NULL || log->entries == NULL)
    {
      return;
    }
  for (size_t i = 0; i < log->entry_count; ++i)
    {
      objstore_txn_entry_internal_dispose (&log->entries[i]);
    }
  sqlite3_free (log->entries);
  log->entries = NULL;
  log->entry_count = 0;
  log->entry_capacity = 0;
}

static void
objstore_txn_log_wipe_frames (objstore_txn_log *log)
{
  sqlite3_free (log->frames);
  log->frames = NULL;
  log->frame_count = 0;
}

static ptrdiff_t
objstore_txn_log_find_latest_index (const objstore_txn_log *log,
                                    const objstore_id *id,
                                    sqlite3_uint64 sequence_limit);

static sqlite3_uint64
objstore_txn_snapshot_hash_id (const objstore_id *id)
{
  sqlite3_uint64 hash = 1469598103934665603ull;
  if (id == NULL)
    {
      return hash;
    }
  for (size_t i = 0; i < OBJSTORE_ID_SIZE; ++i)
    {
      hash ^= (sqlite3_uint64)id->bytes[i];
      hash *= 1099511628211ull;
    }
  return hash;
}

static size_t
objstore_txn_snapshot_table_capacity (size_t entry_count)
{
  size_t capacity = 16;
  while (capacity < entry_count * 2)
    {
      capacity *= 2;
    }
  return capacity;
}

static size_t
objstore_txn_snapshot_find_slot (objstore_txn_snapshot_slot *slots,
                                 size_t slot_count, const objstore_id *id,
                                 bool *out_found)
{
  size_t first_tombstone = SIZE_MAX;
  size_t index
      = (size_t)(objstore_txn_snapshot_hash_id (id) & (slot_count - 1));
  for (;;)
    {
      objstore_txn_snapshot_slot *slot = &slots[index];
      if (slot->state == OBJSTORE_TXN_SNAPSHOT_SLOT_EMPTY)
        {
          *out_found = false;
          return (first_tombstone != SIZE_MAX) ? first_tombstone : index;
        }
      if (slot->state == OBJSTORE_TXN_SNAPSHOT_SLOT_TOMBSTONE)
        {
          if (first_tombstone == SIZE_MAX)
            {
              first_tombstone = index;
            }
        }
      else if (memcmp (slot->id.bytes, id->bytes, OBJSTORE_ID_SIZE) == 0)
        {
          *out_found = true;
          return index;
        }
      index = (index + 1) & (slot_count - 1);
    }
}

int
objstore_txn_log_create (sqlite3 *db, const objstore_config *config,
                         objstore_txn_log **out_log)
{
  if (db == NULL || out_log == NULL)
    {
      return SQLITE_MISUSE;
    }
  objstore_txn_log *log = sqlite3_malloc (sizeof (objstore_txn_log));
  if (log == NULL)
    {
      return SQLITE_NOMEM;
    }
  memset (log, 0, sizeof (*log));
  log->db = db;
  log->operation_count = 0;
  (void)config;
  *out_log = log;
  return SQLITE_OK;
}

void
objstore_txn_log_destroy (objstore_txn_log *log)
{
  if (log == NULL)
    {
      return;
    }
  objstore_txn_log_clear (log);
  sqlite3_free (log);
}

void
objstore_txn_log_clear (objstore_txn_log *log)
{
  if (log == NULL)
    {
      return;
    }
  objstore_txn_log_wipe_entries (log);
  objstore_txn_log_wipe_frames (log);
}

bool
objstore_txn_log_is_empty (const objstore_txn_log *log)
{
  return (log == NULL) || (log->entry_count == 0);
}

sqlite3_uint64
objstore_txn_log_operation_count (const objstore_txn_log *log)
{
  return (log != NULL) ? log->operation_count : 0;
}

int
objstore_txn_log_append_put (objstore_txn_log *log, const objstore_id *id,
                             sqlite3_int64 payload_size)
{
  if (log == NULL || id == NULL)
    {
      return SQLITE_MISUSE;
    }
  int rc = objstore_txn_log_reserve_entries (log);
  if (rc != SQLITE_OK)
    {
      return rc;
    }
  objstore_txn_entry_internal *entry = &log->entries[log->entry_count];
  memset (entry, 0, sizeof (*entry));
  entry->pub.kind = OBJSTORE_TXN_ENTRY_PUT;
  entry->pub.id = *id;
  entry->pub.payload_size = payload_size;
  entry->pub.sequence = log->operation_count++;
  ++log->entry_count;
  return SQLITE_OK;
}

int
objstore_txn_log_append_delete (objstore_txn_log *log, const objstore_id *id)
{
  if (log == NULL || id == NULL)
    {
      return SQLITE_MISUSE;
    }
  int rc = objstore_txn_log_reserve_entries (log);
  if (rc != SQLITE_OK)
    {
      return rc;
    }
  objstore_txn_entry_internal *entry = &log->entries[log->entry_count];
  memset (entry, 0, sizeof (*entry));
  entry->pub.kind = OBJSTORE_TXN_ENTRY_DELETE;
  entry->pub.id = *id;
  entry->pub.payload_size = 0;
  entry->pub.sequence = log->operation_count++;
  ++log->entry_count;
  return SQLITE_OK;
}

static ptrdiff_t
objstore_txn_log_find_latest_index (const objstore_txn_log *log,
                                    const objstore_id *id,
                                    sqlite3_uint64 sequence_limit)
{
  if (log == NULL || id == NULL)
    {
      return -1;
    }
  for (ptrdiff_t i = (ptrdiff_t)log->entry_count - 1; i >= 0; --i)
    {
      const objstore_txn_entry_internal *entry = &log->entries[(size_t)i];
      if (entry->pub.sequence >= sequence_limit)
        {
          continue;
        }
      if (memcmp (entry->pub.id.bytes, id->bytes, OBJSTORE_ID_SIZE) == 0)
        {
          return i;
        }
    }
  return -1;
}

objstore_txn_lookup_state
objstore_txn_log_state_for_id (const objstore_txn_log *log,
                               const objstore_id *id,
                               sqlite3_uint64 sequence_limit)
{
  ptrdiff_t index = objstore_txn_log_find_latest_index (log, id,
                                                        sequence_limit);
  if (index < 0)
    {
      return OBJSTORE_TXN_LOOKUP_NONE;
    }
  const objstore_txn_entry_internal *entry = &log->entries[(size_t)index];
  return (entry->pub.kind == OBJSTORE_TXN_ENTRY_PUT)
             ? OBJSTORE_TXN_LOOKUP_PUT
             : OBJSTORE_TXN_LOOKUP_DELETE;
}

int
objstore_txn_log_lookup_rowid (const objstore_txn_log *log,
                               sqlite3_int64 rowid, objstore_id *out_id)
{
  if (log == NULL || out_id == NULL)
    {
      return SQLITE_MISUSE;
    }
  for (ptrdiff_t i = (ptrdiff_t)log->entry_count - 1; i >= 0; --i)
    {
      const objstore_txn_entry_internal *entry = &log->entries[(size_t)i];
      sqlite3_int64 entry_rowid = objstore_rowid_from_id (&entry->pub.id);
      if (entry_rowid != rowid)
        {
          continue;
        }
      if (entry->pub.kind == OBJSTORE_TXN_ENTRY_DELETE)
        {
          return SQLITE_NOTFOUND;
        }
      *out_id = entry->pub.id;
      return SQLITE_OK;
    }
  return SQLITE_NOTFOUND;
}

int
objstore_txn_log_begin_frame (objstore_txn_log *log)
{
  if (log == NULL)
    {
      return SQLITE_MISUSE;
    }
  size_t *resized = sqlite3_realloc64 (
      log->frames, (log->frame_count + 1) * sizeof (*log->frames));
  if (resized == NULL)
    {
      return SQLITE_NOMEM;
    }
  log->frames = resized;
  log->frames[log->frame_count++] = log->entry_count;
  return SQLITE_OK;
}

int
objstore_txn_log_release_frame (objstore_txn_log *log)
{
  if (log == NULL)
    {
      return SQLITE_MISUSE;
    }
  if (log->frame_count == 0)
    {
      return SQLITE_OK;
    }
  --log->frame_count;
  return SQLITE_OK;
}

int
objstore_txn_log_rollback_frame (objstore_txn_log *log)
{
  if (log == NULL)
    {
      return SQLITE_MISUSE;
    }
  if (log->frame_count == 0)
    {
      objstore_txn_log_clear (log);
      return SQLITE_OK;
    }
  size_t target = log->frames[log->frame_count - 1];
  while (log->entry_count > target)
    {
      objstore_txn_entry_internal_dispose (
          &log->entries[log->entry_count - 1]);
      --log->entry_count;
    }
  log->frames[log->frame_count - 1] = log->entry_count;
  return SQLITE_OK;
}

size_t
objstore_txn_log_entry_count (const objstore_txn_log *log)
{
  return (log != NULL) ? log->entry_count : 0;
}

const objstore_txn_entry *
objstore_txn_log_entry_at (const objstore_txn_log *log, size_t index)
{
  if (log == NULL || index >= log->entry_count)
    {
      return NULL;
    }
  return &log->entries[index].pub;
}

int
objstore_txn_log_drop_last (objstore_txn_log *log)
{
  if (log == NULL || log->entry_count == 0)
    {
      return SQLITE_OK;
    }
  objstore_txn_entry_internal_dispose (&log->entries[log->entry_count - 1]);
  --log->entry_count;
  return SQLITE_OK;
}

objstore_txn_snapshot *
objstore_txn_snapshot_build (const objstore_txn_log *log,
                             sqlite3_uint64 sequence_limit)
{
  objstore_txn_snapshot *snapshot
      = sqlite3_malloc (sizeof (objstore_txn_snapshot));
  if (snapshot == NULL)
    {
      return NULL;
    }
  memset (snapshot, 0, sizeof (*snapshot));
  snapshot->log = log;
  if (log == NULL || log->entry_count == 0 || sequence_limit == 0)
    {
      return snapshot;
    }
  snapshot->op_indexes
      = sqlite3_malloc64 (log->entry_count * sizeof (*snapshot->op_indexes));
  if (snapshot->op_indexes == NULL)
    {
      objstore_txn_snapshot_destroy (snapshot);
      return NULL;
    }
  size_t slot_count = objstore_txn_snapshot_table_capacity (log->entry_count);
  objstore_txn_snapshot_slot *slots
      = sqlite3_malloc64 (slot_count * sizeof (*slots));
  if (slots == NULL)
    {
      objstore_txn_snapshot_destroy (snapshot);
      return NULL;
    }
  memset (slots, 0, slot_count * sizeof (*slots));
  size_t append_count = 0;
  size_t live_count = 0;
  for (size_t i = 0; i < log->entry_count; ++i)
    {
      const objstore_txn_entry_internal *entry = &log->entries[i];
      if (entry->pub.sequence >= sequence_limit)
        {
          break;
        }
      bool found = false;
      size_t slot_index = objstore_txn_snapshot_find_slot (
          slots, slot_count, &entry->pub.id, &found);
      if (entry->pub.kind == OBJSTORE_TXN_ENTRY_DELETE)
        {
          if (found)
            {
              size_t snapshot_index = slots[slot_index].snapshot_index;
              snapshot->op_indexes[snapshot_index] = SIZE_MAX;
              slots[slot_index].state = OBJSTORE_TXN_SNAPSHOT_SLOT_TOMBSTONE;
              slots[slot_index].snapshot_index = SIZE_MAX;
              --live_count;
            }
          continue;
        }
      if (found)
        {
          snapshot->op_indexes[slots[slot_index].snapshot_index] = i;
        }
      else
        {
          slots[slot_index].id = entry->pub.id;
          slots[slot_index].snapshot_index = append_count;
          slots[slot_index].state = OBJSTORE_TXN_SNAPSHOT_SLOT_OCCUPIED;
          snapshot->op_indexes[append_count++] = i;
          ++live_count;
        }
    }
  if (live_count != append_count)
    {
      size_t write_index = 0;
      for (size_t read_index = 0; read_index < append_count; ++read_index)
        {
          if (snapshot->op_indexes[read_index] == SIZE_MAX)
            {
              continue;
            }
          snapshot->op_indexes[write_index++] = snapshot->op_indexes[read_index];
        }
      snapshot->count = write_index;
    }
  else
    {
      snapshot->count = append_count;
    }
  sqlite3_free (slots);
  return snapshot;
}

void
objstore_txn_snapshot_destroy (objstore_txn_snapshot *snapshot)
{
  if (snapshot == NULL)
    {
      return;
    }
  sqlite3_free (snapshot->op_indexes);
  sqlite3_free (snapshot);
}

size_t
objstore_txn_snapshot_count (const objstore_txn_snapshot *snapshot)
{
  return (snapshot != NULL) ? snapshot->count : 0;
}

const objstore_txn_entry *
objstore_txn_snapshot_entry (const objstore_txn_snapshot *snapshot,
                             size_t index)
{
  if (snapshot == NULL || snapshot->log == NULL || index >= snapshot->count)
    {
      return NULL;
    }
  size_t op_index = snapshot->op_indexes[index];
  return &snapshot->log->entries[op_index].pub;
}

bool
objstore_txn_snapshot_contains (const objstore_txn_snapshot *snapshot,
                                const objstore_id *id)
{
  if (snapshot == NULL || snapshot->log == NULL || id == NULL)
    {
      return false;
    }
  for (size_t i = 0; i < snapshot->count; ++i)
    {
      const objstore_txn_entry_internal *entry
          = &snapshot->log->entries[snapshot->op_indexes[i]];
      if (memcmp (entry->pub.id.bytes, id->bytes, OBJSTORE_ID_SIZE) == 0)
        {
          return true;
        }
    }
  return false;
}
