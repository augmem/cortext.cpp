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

inline long long
SlotFor (long long signal_id, int capacity)
{
  if (signal_id <= 0 || capacity <= 0)
    throw std::invalid_argument ("invalid active signal embedding ring key");
  return (signal_id - 1) % static_cast<long long> (capacity);
}

inline void
InsertAtCapacity (Transaction &tx, long long signal_id,
                  const std::any &embedding, long long created_at,
                  int capacity)
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
      { SlotFor (signal_id, capacity), signal_id, embedding, created_at,
        static_cast<long long> (capacity) });
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
          "SELECT s.signal_id, e.embedding, "
          "       COALESCE(s.created_at, s.timestamp) AS created_at "
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
      "ORDER BY signal_id DESC LIMIT ?",
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
