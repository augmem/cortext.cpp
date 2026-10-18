#pragma once

// Member vectors: the stored shared-space vectors (memory_signal_vectors) of
// the signals of multi-signal LONG_TERM memories, resident for queries.
//
// A memory grouped from several signals (a Natural unit: a stretch of
// conversation) carries one pooled retrieval vector, which sits between its
// members and is close to none of them; a query about one member ranks the
// memory far below where that member alone would rank. Queries therefore
// also rank memories by their best member (graph_retrieval.cpp, member
// channel), from this index.
//
// The index reads new rows on demand (signal ids above the highest it has
// read, of memories that are LONG_TERM with more than one signal when read)
// and keeps each vector as int8 codes with one scale (dim + 4 bytes per
// member). Single-signal memories are never read: their member vector is
// their memory vector. Callers resolve each hit's current memory through the
// store before use, so a signal that moved or left since it was read never
// yields a stale memory.

#include "cortext/store/store.hpp"

#include <Eigen/Dense>

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace cortext::internal
{

struct MemberVectorIndex
{
  int dim = 0;
  long long read_through = 0; // highest signal_id read
  std::vector<long long> signal_ids;
  std::vector<std::int8_t> codes; // dim per member
  std::vector<float> scales;      // 0 = dropped member
};

} // namespace cortext::internal

namespace cortext::operations::member_vector_index_internal
{

/// Reads member rows stored since the last refresh. Returns rows added.
std::size_t Refresh (internal::MemberVectorIndex &index, Transaction &tx,
                     int dim);

/// The `limit` best members by cosine to `query` (normalised inside), best
/// first, as (signal_id, cosine).
std::vector<std::pair<long long, float> >
Search (const internal::MemberVectorIndex &index,
        const Eigen::VectorXf &query, std::size_t limit);

/// Marks a member dropped (its signal left the store or its memory).
void Drop (internal::MemberVectorIndex &index, long long signal_id);

} // namespace cortext::operations::member_vector_index_internal
