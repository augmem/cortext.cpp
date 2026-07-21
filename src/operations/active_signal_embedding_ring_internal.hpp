#pragma once

#include "cortext/store/store.hpp"
#include "cortext/store/utils.hpp"
#include "sparse_retrieval_knobs_internal.hpp"

#include <algorithm>
#include <any>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace cortext::operations::active_signal_embedding_ring_internal
{

inline int
Capacity (double focus, double sensitivity, double stability)
{
  return sparse_retrieval_knobs_internal::ActiveSignalEmbeddingCapacity (
      focus, sensitivity, stability);
}

inline void
InsertIntoSlot (Transaction &tx, long long slot, long long signal_id,
                const std::any &embedding, long long created_at, int capacity)
{
  tx.Execute (
      "INSERT INTO cortext_active_signal_embeddings("
      "slot, signal_id, embedding, created_at, capacity) "
      "VALUES (?, ?, ?, ?, ?) "
      "ON CONFLICT(slot) DO UPDATE SET "
      "signal_id = excluded.signal_id, "
      "embedding = excluded.embedding, "
      "created_at = excluded.created_at, "
      "capacity = excluded.capacity",
      { slot, signal_id, embedding, created_at,
        static_cast<long long> (capacity) });
}

inline void
InsertAtCapacity (Transaction &tx, long long signal_id,
                  const std::any &embedding, long long created_at,
                  int capacity)
{
  if (signal_id <= 0 || capacity <= 0)
    throw std::invalid_argument ("invalid active signal embedding ring key");

  const auto existing = tx.Execute (
      "SELECT slot FROM cortext_active_signal_embeddings "
      "WHERE signal_id = ? LIMIT 1",
      { signal_id });
  if (!existing.empty ())
    {
      const auto slot
          = store::AnyToLongLong (existing.front ().at ("slot")).value_or (-1);
      if (slot < 0)
        throw std::runtime_error ("invalid active signal embedding slot");
      InsertIntoSlot (tx, slot, signal_id, embedding, created_at, capacity);
      return;
    }

  const auto count_rows = tx.Execute (
      "SELECT COUNT(*) AS row_count "
      "FROM cortext_active_signal_embeddings");
  const auto row_count
      = count_rows.empty ()
            ? 0
            : store::AnyToLongLong (
                  count_rows.front ().at ("row_count")).value_or (-1);
  if (row_count < 0 || row_count > capacity)
    throw std::runtime_error ("invalid active signal embedding row count");

  if (row_count < capacity)
    {
      // A signal deletion can leave a physical slot gap. Fill it with the
      // next exact vector presented to this bounded ring; once full, the
      // timestamp frontier below—not row identity—owns replacement order.
      const auto free_rows = tx.Execute (
          "WITH RECURSIVE slots(slot) AS ("
          "  SELECT 0 UNION ALL "
          "  SELECT slot + 1 FROM slots WHERE slot + 1 < ?"
          ") "
          "SELECT slots.slot FROM slots "
          "LEFT JOIN cortext_active_signal_embeddings a "
          "ON a.slot = slots.slot "
          "WHERE a.slot IS NULL ORDER BY slots.slot LIMIT 1",
          { static_cast<long long> (capacity) });
      if (free_rows.empty ())
        throw std::runtime_error ("active signal embedding slot unavailable");
      const auto slot = store::AnyToLongLong (
          free_rows.front ().at ("slot")).value_or (-1);
      if (slot < 0)
        throw std::runtime_error ("invalid active signal embedding slot");
      InsertIntoSlot (tx, slot, signal_id, embedding, created_at, capacity);
      return;
    }

  const auto oldest_rows = tx.Execute (
      "SELECT slot, signal_id, created_at "
      "FROM cortext_active_signal_embeddings "
      "ORDER BY created_at ASC, signal_id ASC LIMIT 1");
  if (oldest_rows.empty ())
    throw std::runtime_error ("active signal embedding frontier unavailable");
  const auto oldest_slot
      = store::AnyToLongLong (oldest_rows.front ().at ("slot")).value_or (-1);
  const auto oldest_signal_id = store::AnyToLongLong (
      oldest_rows.front ().at ("signal_id")).value_or (0);
  const auto oldest_created_at = store::AnyToLongLong (
      oldest_rows.front ().at ("created_at")).value_or (0);
  if (oldest_slot < 0 || oldest_signal_id <= 0)
    throw std::runtime_error ("invalid active signal embedding frontier");
  if (created_at < oldest_created_at
      || (created_at == oldest_created_at && signal_id <= oldest_signal_id))
    return;
  InsertIntoSlot (tx, oldest_slot, signal_id, embedding, created_at, capacity);
}

inline void
EnsureCapacity (Transaction &tx, int capacity)
{
  if (capacity <= 0)
    throw std::invalid_argument ("active signal embedding capacity is zero");
  const auto capacity_rows = tx.Execute (
      "SELECT DISTINCT capacity FROM cortext_active_signal_embeddings "
      "ORDER BY capacity LIMIT 2");
  if (capacity_rows.empty ())
    {
      // Migration 30 starts with an empty ring while pre-migration signal
      // rows still reference their exact per-signal embeddings. Capture the
      // newest legacy window before the first post-migration signal switches
      // signals.embedding_id to the aggregate memory embedding.
      auto legacy_rows = tx.Execute (
          "SELECT s.signal_id, e.embedding, s.timestamp AS created_at "
          "FROM signals s "
          "JOIN embeddings e ON e.embedding_id = s.embedding_id "
          "ORDER BY s.timestamp DESC, s.signal_id DESC LIMIT ?",
          { static_cast<long long> (capacity) });
      std::reverse (legacy_rows.begin (), legacy_rows.end ());
      for (const auto &row : legacy_rows)
        {
          const auto signal_id
              = store::AnyToLongLong (row.at ("signal_id")).value_or (0);
          const auto created_at
              = store::AnyToLongLong (row.at ("created_at")).value_or (0);
          if (signal_id <= 0 || !row.at ("embedding").has_value ())
            throw std::runtime_error (
                "invalid legacy active signal embedding row");
          InsertAtCapacity (tx, signal_id, row.at ("embedding"), created_at,
                            capacity);
        }
      return;
    }
  if (capacity_rows.size () != 1)
    throw std::runtime_error ("mixed active signal embedding capacities");
  const auto stored_capacity = store::AnyToLongLong (
      capacity_rows.front ().at ("capacity"));
  if (!stored_capacity || *stored_capacity <= 0)
    throw std::runtime_error ("invalid active signal embedding capacity");
  if (*stored_capacity == capacity)
    return;

  // This table is already bounded by the previous knob-derived capacity.
  // Rehash only the newest rows that fit the new knob-derived capacity.
  auto rows = tx.Execute (
      "SELECT signal_id, embedding, created_at "
      "FROM cortext_active_signal_embeddings "
      "ORDER BY created_at DESC, signal_id DESC LIMIT ?",
      { static_cast<long long> (capacity) });
  tx.Execute ("DELETE FROM cortext_active_signal_embeddings");
  std::reverse (rows.begin (), rows.end ());
  for (const auto &row : rows)
    {
      const auto signal_id
          = store::AnyToLongLong (row.at ("signal_id")).value_or (0);
      const auto created_at
          = store::AnyToLongLong (row.at ("created_at")).value_or (0);
      if (signal_id <= 0 || !row.at ("embedding").has_value ())
        throw std::runtime_error ("invalid active signal embedding row");
      InsertAtCapacity (tx, signal_id, row.at ("embedding"), created_at,
                        capacity);
    }
}

inline void
Upsert (Transaction &tx, long long signal_id,
        const std::vector<float> &embedding, long long created_at,
        double focus, double sensitivity, double stability)
{
  const int capacity = Capacity (focus, sensitivity, stability);
  EnsureCapacity (tx, capacity);
  InsertAtCapacity (tx, signal_id, embedding, created_at, capacity);
}

} // namespace cortext::operations::active_signal_embedding_ring_internal
