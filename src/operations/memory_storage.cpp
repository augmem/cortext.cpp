#include "cortext/operations/memory_storage.hpp"
#include "constructive_recall_internal.hpp"
#include "historical_surface_search_cache_internal.hpp"
#include "association_fanout_cache_internal.hpp"
#include "emotional_metadata_cache_internal.hpp"
#include "signal_record_rollback_internal.hpp"
#include "cortext/core/algorithms.hpp"
#include "cortext/core/knobs.hpp"
#include "cortext/core/utils.hpp"
#include "cortext/processor/accumulator_state.hpp"
#include "cortext/processor/operation_context.hpp"
#include "cortext/signal.hpp"
#include "cortext/core/sparse.hpp"
#include "cortext/store/object_store.hpp"
#include "cortext/store/schema_helpers.hpp"
#include "cortext/store/store.hpp"
#include "cortext/store/utils.hpp"
#include "cortext/telemetry/telemetry.hpp"
#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace cortext::operations
{

namespace
{
using SteadyClock = std::chrono::steady_clock;

double
ElapsedMillis (SteadyClock::time_point start)
{
  return std::chrono::duration<double, std::milli> (SteadyClock::now () - start)
      .count ();
}

/// @brief Determine primary modality from signal records (majority vote)
std::string
GetPrimaryModality (const std::vector<SignalRecord> &signals,
                    const std::string &fallback_modality)
{
  if (signals.empty ())
    {
      return fallback_modality;
    }

  std::unordered_map<std::string, int> counts;
  for (const auto &sig : signals)
    {
      counts[sig.modality]++;
    }

  std::string primary = signals[0].modality;
  int max_count = 0;
  for (const auto &kv : counts)
    {
      if (kv.second > max_count)
        {
          max_count = kv.second;
          primary = kv.first;
        }
    }
  return primary;
}

/// @brief Serialize 6d emotion/mood vector to blob
std::vector<char>
SerializeEmotionVector (const std::array<double, 6> &vec)
{
  std::vector<char> blob (sizeof (double) * 6);
  std::memcpy (blob.data (), vec.data (), blob.size ());
  return blob;
}

std::string
SourceOriginFor ()
{
  return "source";
}

double
SourcePriorReliability (double F, double S, double T)
{
  return core::SourceReliabilityPrior (F, S, T);
}

bool
DecodeEmbeddingAny (const std::any &value, int expected_dim,
                    Eigen::VectorXf &out)
{
  if (expected_dim <= 0)
    {
      return false;
    }
  if (value.type () == typeid (std::vector<float>))
    {
      const auto &vec = std::any_cast<const std::vector<float> &> (value);
      if (vec.size () != static_cast<std::size_t> (expected_dim))
        {
          return false;
        }
      out.resize (static_cast<Eigen::Index> (expected_dim));
      for (int i = 0; i < expected_dim; ++i)
        {
          out[static_cast<Eigen::Index> (i)]
              = vec[static_cast<std::size_t> (i)];
        }
      return true;
    }
  return core::DecodeFloatBlob (value, expected_dim, out);
}

std::vector<float>
ToFloatVector (const Eigen::VectorXf &v)
{
  std::vector<float> out;
  out.resize (static_cast<std::size_t> (v.size ()));
  for (int i = 0; i < v.size (); ++i)
    {
      out[static_cast<std::size_t> (i)] = v[i];
    }
  return out;
}

struct BorrowedSupersessionCandidate
{
  long long memory_id = 0;
  const Eigen::VectorXf *embedding = nullptr;
};

struct CurrentSupersessionSearchProof
{
  bool valid = false;
  std::vector<long long> ranked_memory_ids;
  float cutoff_distance = 0.0f;
  long long cutoff_memory_id = 0;
};

using historical_surface_search_cache_internal::SupersessionCandidate;
using historical_surface_search_cache_internal::SupersessionCandidateRows;

void
AppendUniqueBorrowedSupersessionRows (
    std::vector<SupersessionCandidate> &rows,
    std::vector<BorrowedSupersessionCandidate> extra_rows,
    std::unordered_set<long long> &seen_memory_ids,
    std::size_t *accepted_count = nullptr)
{
  for (const auto &row : extra_rows)
    {
      if (row.memory_id <= 0
          || !seen_memory_ids.insert (row.memory_id).second)
        {
          continue;
        }
      rows.push_back ({ row.memory_id, row.embedding, std::nullopt });
      if (accepted_count)
        ++*accepted_count;
    }
}

void
AppendUniqueOwnedSupersessionRows (
    std::vector<SupersessionCandidate> &rows,
    std::vector<std::map<std::string, std::any>> extra_rows,
    std::unordered_set<long long> &seen_memory_ids, int embedding_dim,
    std::size_t *accepted_count = nullptr)
{
  for (auto &row : extra_rows)
    {
      auto memory_it = row.find ("memory_id");
      if (memory_it == row.end ())
        {
          continue;
        }
      const auto memory_id = store::AnyToLongLong (memory_it->second);
      if (!memory_id || *memory_id <= 0
          || !seen_memory_ids.insert (*memory_id).second)
        {
          continue;
        }
      SupersessionCandidate candidate;
      candidate.memory_id = *memory_id;
      const auto embedding_it = row.find ("embedding");
      if (embedding_it != row.end ())
        {
          Eigen::VectorXf decoded;
          if (DecodeEmbeddingAny (embedding_it->second, embedding_dim,
                                  decoded))
            {
              candidate.owned_embedding = std::move (decoded);
            }
        }
      rows.push_back (std::move (candidate));
      if (accepted_count)
        ++*accepted_count;
    }
}

std::optional<bool>
HistoricalCandidatePopulationCovered (
    const historical_surface_search_cache_internal::State &state,
    const Eigen::VectorXf &query_embedding, long long memory_id,
    long long end_ts, int candidate_limit,
    const std::unordered_set<long long> &seen_memory_ids,
    std::size_t *rows_visited)
{
  if (rows_visited)
    *rows_visited += state.supersession_entry_indices.size ();
  auto &ranked = state.ranked_scratch;
  ranked.clear ();
  ranked.reserve (state.supersession_entry_indices.size ());
  for (const std::size_t index : state.supersession_entry_indices)
    {
      if (index >= state.entries.size ())
        return std::nullopt;
      const auto &entry = state.entries[index];
      if (!historical_surface_search_cache_internal::
              IsSupersessionCandidateEntry (entry)
          || entry.embedding.size () != query_embedding.size ())
        return std::nullopt;
      ranked.push_back (
          { index, (entry.embedding - query_embedding).squaredNorm () });
    }
  auto by_distance = [&state] (
                         const historical_surface_search_cache_internal::RankedEntry &a,
                         const historical_surface_search_cache_internal::RankedEntry &b) {
    if (a.distance != b.distance)
      return a.distance < b.distance;
    return state.entries[a.index].embedding_id
           < state.entries[b.index].embedding_id;
  };
  if (static_cast<int> (ranked.size ()) > candidate_limit)
    {
      std::nth_element (ranked.begin (), ranked.begin () + candidate_limit,
                        ranked.end (), by_distance);
      ranked.resize (static_cast<std::size_t> (candidate_limit));
    }
  std::sort (ranked.begin (), ranked.end (), by_distance);
  return std::all_of (
      ranked.begin (), ranked.end (), [&] (const auto &candidate) {
        const auto &entry = state.entries[candidate.index];
        return std::all_of (
            entry.memory_references.begin (),
            entry.memory_references.end (), [&] (const auto &reference) {
              if (reference.kind != "LONG_TERM"
                  && reference.kind != "ASSOCIATION")
                return true;
              return reference.memory_id == memory_id
                     || reference.start_ts >= end_ts
                     || seen_memory_ids.count (reference.memory_id) != 0;
            });
      });
}

bool
HistoricalCandidatePopulationCoveredByCurrentProof (
    const historical_surface_search_cache_internal::State &state,
    const Eigen::VectorXf &query_embedding, long long excluded_memory_id,
    long long end_ts, int candidate_limit,
    const std::unordered_set<long long> &seen_memory_ids,
    const CurrentSupersessionSearchProof &proof)
{
  if (!proof.valid || state.supersession_population_ambiguous
      || state.supersession_embedding_fanout
      || !state.supersession_tie_order_equivalent)
    return false;
  std::unordered_set<long long> ranked_current (
      proof.ranked_memory_ids.begin (), proof.ranked_memory_ids.end ());
  auto base_order_less = [&] (float left_distance, long long left_memory_id,
                              float right_distance,
                              long long right_memory_id) {
    return left_distance < right_distance
           || (left_distance == right_distance
               && left_memory_id < right_memory_id);
  };
  const bool current_population_exhausted
      = static_cast<int> (proof.ranked_memory_ids.size ()) < candidate_limit;

  for (const long long memory_id :
       state.supersession_population_mismatches)
    {
      const auto base_it
          = state.supersession_entry_by_memory.find (memory_id);
      const auto current_it = state.current_memory_index.find (memory_id);
      const bool in_current_top_k
          = ranked_current.count (memory_id) != 0;
      if (base_it == state.supersession_entry_by_memory.end ())
        {
          if (!current_population_exhausted && in_current_top_k)
            return false;
          continue;
        }
      if (base_it->second >= state.entries.size ())
        return false;
      const auto &base = state.entries[base_it->second];
      const auto *reference
          = historical_surface_search_cache_internal::
              FindSupersessionMemoryReference (base, memory_id);
      if (!reference)
        return false;
      if (reference->memory_id == excluded_memory_id
          || reference->start_ts >= end_ts)
        continue;
      if (current_population_exhausted)
        {
          if (seen_memory_ids.count (reference->memory_id) == 0)
            return false;
          continue;
        }
      if (base.embedding.size () != query_embedding.size ())
        return false;
      const float base_distance
          = (base.embedding - query_embedding).squaredNorm ();
      const bool base_before_cutoff = base_order_less (
          base_distance, reference->memory_id, proof.cutoff_distance,
          proof.cutoff_memory_id);
      const bool cutoff_before_base = base_order_less (
          proof.cutoff_distance, proof.cutoff_memory_id, base_distance,
          reference->memory_id);
      if (current_it == state.current_memory_index.end ())
        {
          if (base_before_cutoff)
            return false;
          continue;
        }
      if (in_current_top_k)
        {
          if (cutoff_before_base
              || seen_memory_ids.count (reference->memory_id) == 0)
            return false;
        }
      else if (base_before_cutoff)
        {
          return false;
        }
    }

  for (const long long memory_id : proof.ranked_memory_ids)
    {
      const auto base_it
          = state.supersession_entry_by_memory.find (memory_id);
      if (base_it == state.supersession_entry_by_memory.end ())
        continue;
      if (base_it->second >= state.entries.size ())
        return false;
      const auto &base = state.entries[base_it->second];
      const auto *reference
          = historical_surface_search_cache_internal::
              FindSupersessionMemoryReference (base, memory_id);
      if (!reference)
        return false;
      if (reference->memory_id != excluded_memory_id
          && reference->start_ts < end_ts
          && seen_memory_ids.count (reference->memory_id) == 0)
        return false;
    }
  return true;
}

std::optional<std::vector<BorrowedSupersessionCandidate>>
LoadHistoricalSupersessionRowsFromCache (
    const std::shared_ptr<
        const historical_surface_search_cache_internal::State> &state,
    const Eigen::VectorXf &query_embedding,
    long long memory_id, long long end_ts, int candidate_limit,
    const std::unordered_set<long long> &seen_memory_ids,
    const CurrentSupersessionSearchProof &current_search_proof,
    bool *coverage_proven, std::size_t *rows_visited)
{
  if (coverage_proven)
    *coverage_proven = false;
  if (rows_visited)
    *rows_visited = 0;
  const std::size_t embedding_count = state ? state->entries.size () : 0;
  const int embedding_dim = static_cast<int> (query_embedding.size ());
  if (!state || embedding_dim <= 0 || candidate_limit <= 0
      || state->embedding_dim != embedding_dim
      || state->search.size ()
             != embedding_count * static_cast<std::size_t> (embedding_dim))
    {
      return std::nullopt;
    }

  if (historical_surface_search_cache_internal::
          CurrentPopulationCoversHistorical (*state, memory_id))
    {
      if (coverage_proven)
        *coverage_proven = true;
      return std::vector<BorrowedSupersessionCandidate>{};
    }

  if (HistoricalCandidatePopulationCoveredByCurrentProof (
          *state, query_embedding, memory_id, end_ts, candidate_limit,
          seen_memory_ids, current_search_proof))
    {
      if (coverage_proven)
        *coverage_proven = true;
      return std::vector<BorrowedSupersessionCandidate>{};
    }

  const auto candidate_population_covered
      = HistoricalCandidatePopulationCovered (
          *state, query_embedding, memory_id, end_ts, candidate_limit,
          seen_memory_ids, rows_visited);
  if (!candidate_population_covered)
    return std::nullopt;
  if (*candidate_population_covered)
    {
      if (coverage_proven)
        *coverage_proven = true;
      return std::vector<BorrowedSupersessionCandidate>{};
    }

  using RowMajorMatrix
      = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  const Eigen::Map<const RowMajorMatrix> matrix (
      state->search.data (), static_cast<Eigen::Index> (embedding_count),
      embedding_dim);
  if (rows_visited)
    *rows_visited += embedding_count;
  auto &distance_scratch = state->distance_scratch;
  distance_scratch.resize (embedding_count);
  Eigen::Map<Eigen::VectorXf> distances (
      distance_scratch.data (), static_cast<Eigen::Index> (embedding_count));
  distances = (matrix.rowwise () - query_embedding.transpose ())
                  .rowwise ()
                  .squaredNorm ();
  auto &ranked = state->ranked_scratch;
  ranked.clear ();
  ranked.reserve (embedding_count);
  for (std::size_t index = 0; index < embedding_count; ++index)
    {
      const auto &entry = state->entries[index];
      if (entry.embedding_id <= 0 || entry.embedding.size () != embedding_dim)
        {
          return std::nullopt;
        }
      ranked.push_back (
          { index, distances[static_cast<Eigen::Index> (index)] });
    }
  auto by_distance = [&state] (
                         const historical_surface_search_cache_internal::RankedEntry &a,
                         const historical_surface_search_cache_internal::RankedEntry &b) {
    if (a.distance != b.distance)
      {
        return a.distance < b.distance;
      }
    return state->entries[a.index].embedding_id
           < state->entries[b.index].embedding_id;
  };
  if (static_cast<int> (ranked.size ()) > candidate_limit)
    {
      std::nth_element (ranked.begin (), ranked.begin () + candidate_limit,
                        ranked.end (), by_distance);
      ranked.resize (static_cast<std::size_t> (candidate_limit));
    }
  std::sort (ranked.begin (), ranked.end (), by_distance);

  std::vector<BorrowedSupersessionCandidate> rows;
  rows.reserve (ranked.size ());
  for (const auto &candidate : ranked)
    {
      const auto &entry = state->entries[candidate.index];
      for (const auto &reference : entry.memory_references)
        {
          if (reference.memory_id <= 0 || reference.memory_id == memory_id
              || (reference.kind != "LONG_TERM"
                  && reference.kind != "ASSOCIATION")
              || reference.start_ts >= end_ts
              || seen_memory_ids.count (reference.memory_id) != 0)
            continue;
          rows.push_back ({ reference.memory_id, &entry.embedding });
        }
    }
  return rows;
}

std::optional<std::vector<BorrowedSupersessionCandidate>>
LoadCurrentSupersessionRowsFromCache (
    const std::shared_ptr<
        const historical_surface_search_cache_internal::State> &state,
    const ProcessorContext &p_ctx, const Eigen::VectorXf &query_embedding,
    long long memory_id, long long end_ts, int candidate_limit,
    CurrentSupersessionSearchProof *proof, std::size_t *rows_visited)
{
  if (proof)
    *proof = {};
  if (rows_visited)
    *rows_visited = 0;
  const std::size_t entry_count = state ? state->current_entries.size () : 0;
  const int embedding_dim = static_cast<int> (query_embedding.size ());
  if (!state || embedding_dim <= 0 || candidate_limit <= 0
      || state->embedding_dim != embedding_dim
      || state->current_search.size ()
             != entry_count * static_cast<std::size_t> (embedding_dim))
    {
      return std::nullopt;
    }
  if (entry_count == 0)
    {
      if (proof)
        proof->valid = true;
      return std::vector<BorrowedSupersessionCandidate>{};
    }
  using RowMajorMatrix
      = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  const Eigen::Map<const RowMajorMatrix> matrix (
      state->current_search.data (), static_cast<Eigen::Index> (entry_count),
      embedding_dim);
  if (rows_visited)
    *rows_visited = entry_count;
  auto &distance_scratch = state->distance_scratch;
  distance_scratch.resize (entry_count);
  Eigen::Map<Eigen::VectorXf> distances (
      distance_scratch.data (), static_cast<Eigen::Index> (entry_count));
  distances = (matrix.rowwise () - query_embedding.transpose ())
                  .rowwise ()
                  .squaredNorm ();
  auto &ranked = state->ranked_scratch;
  ranked.clear ();
  ranked.reserve (entry_count);
  for (std::size_t index = 0; index < entry_count; ++index)
    {
      const auto &entry = state->current_entries[index];
      if (entry.memory_id <= 0 || entry.embedding_id <= 0
          || entry.embedding.size () != embedding_dim)
        return std::nullopt;
      ranked.push_back (
          { index, distances[static_cast<Eigen::Index> (index)] });
    }
  auto by_distance = [&state] (
                         const historical_surface_search_cache_internal::RankedEntry &a,
                         const historical_surface_search_cache_internal::RankedEntry &b) {
    if (a.distance != b.distance)
      return a.distance < b.distance;
    return state->current_entries[a.index].memory_id
           < state->current_entries[b.index].memory_id;
  };
  if (static_cast<int> (ranked.size ()) > candidate_limit)
    {
      std::nth_element (ranked.begin (), ranked.begin () + candidate_limit,
                        ranked.end (), by_distance);
      ranked.resize (static_cast<std::size_t> (candidate_limit));
    }
  std::sort (ranked.begin (), ranked.end (), by_distance);

  if (proof)
    {
      proof->valid = true;
      proof->ranked_memory_ids.reserve (ranked.size ());
      for (const auto &candidate : ranked)
        proof->ranked_memory_ids.push_back (
            state->current_entries[candidate.index].memory_id);
      if (!ranked.empty ())
        {
          proof->cutoff_distance = ranked.back ().distance;
          proof->cutoff_memory_id
              = state->current_entries[ranked.back ().index].memory_id;
        }
    }

  std::vector<BorrowedSupersessionCandidate> rows;
  rows.reserve (ranked.size ());
  for (const auto &candidate : ranked)
    {
      const auto &entry = state->current_entries[candidate.index];
      const auto surface_it = p_ctx.retrieval_surface_index.find (
          entry.memory_id);
      if (surface_it == p_ctx.retrieval_surface_index.end ()
          || surface_it->second >= p_ctx.retrieval_surface_cache.size ())
        {
          return std::nullopt;
        }
      const auto &surface = p_ctx.retrieval_surface_cache[surface_it->second];
      if (entry.memory_id == memory_id
          || (surface.kind != "LONG_TERM" && surface.kind != "ASSOCIATION")
          || surface.start_ts >= end_ts)
        {
          continue;
        }
      rows.push_back ({ entry.memory_id, &entry.embedding });
    }
  return rows;
}

SupersessionCandidateRows
LoadSupersessionCandidateRows (OperationContext &context, Transaction &tx,
                               const ProcessorContext &p_ctx,
                               const Eigen::VectorXf &embedding_to_store,
                               long long memory_id, long long end_ts,
                               int candidate_limit)
{
  SupersessionCandidateRows result;
  result.cache_owner
      = historical_surface_search_cache_internal::Find (p_ctx);
  std::unordered_set<long long> seen_memory_ids;
  std::size_t current_execution_count = 0;
  std::size_t historical_execution_count = 0;
  std::size_t current_rows_visited = 0;
  std::size_t historical_rows_visited = 0;
  std::size_t sql_fallback_count = 0;
  CurrentSupersessionSearchProof current_search_proof;
  const std::vector<float> query_embedding = ToFloatVector (
      embedding_to_store);
  const auto current_start = SteadyClock::now ();
  auto current_cache_rows = LoadCurrentSupersessionRowsFromCache (
      result.cache_owner, p_ctx, embedding_to_store, memory_id, end_ts,
      candidate_limit, &current_search_proof, &current_rows_visited);
  if (current_cache_rows)
    {
      AppendUniqueBorrowedSupersessionRows (
          result.rows, std::move (*current_cache_rows), seen_memory_ids,
          &current_execution_count);
    }
  else
    {
      ++sql_fallback_count;
      try
        {
          auto current_rows = tx.Execute (
              "SELECT m.memory_id, cme.embedding "
              "FROM ("
              "  SELECT memory_id, embedding "
              "  FROM current_memory_embeddings "
              "  WHERE embedding MATCH ? "
              "    AND k = ?"
              ") cme "
              "JOIN memories m ON m.memory_id = cme.memory_id "
              "WHERE m.memory_id != ? "
              "  AND m.kind IN ('LONG_TERM', 'ASSOCIATION') "
              "  AND COALESCE(m.start_ts, 0) < ?",
              { query_embedding, static_cast<long long> (candidate_limit),
                memory_id, end_ts });
          AppendUniqueOwnedSupersessionRows (
              result.rows, std::move (current_rows), seen_memory_ids,
              static_cast<int> (embedding_to_store.size ()),
              &current_execution_count);
        }
      catch (const std::exception &e)
        {
          telemetry::LogDebug (
              "cortext.memory_storage.supersession_current_knn_unavailable",
              { telemetry::Attribute::String ("error", e.what ()) });
        }
    }
  context.AddOperationTiming ("MemoryStorage.supersession_current_load",
                              ElapsedMillis (current_start));
  const auto current_count = result.rows.size ();

  const auto historical_start = SteadyClock::now ();
  bool historical_coverage_proven = false;
  auto historical_cache_rows = LoadHistoricalSupersessionRowsFromCache (
      result.cache_owner, embedding_to_store, memory_id, end_ts,
      candidate_limit, seen_memory_ids, current_search_proof,
      &historical_coverage_proven, &historical_rows_visited);
  if (historical_cache_rows)
    {
      AppendUniqueBorrowedSupersessionRows (
          result.rows, std::move (*historical_cache_rows), seen_memory_ids,
          &historical_execution_count);
    }
  else
    {
      ++sql_fallback_count;
      try
        {
          auto historical_rows = tx.Execute (
              "SELECT m.memory_id, e.embedding "
              "FROM embeddings e "
              "JOIN memories m ON m.embedding_id = e.embedding_id "
              "WHERE e.embedding MATCH ? "
              "  AND k = ? "
              "  AND m.memory_id != ? "
              "  AND m.embedding_id IS NOT NULL "
              "  AND m.kind IN ('LONG_TERM', 'ASSOCIATION') "
              "  AND COALESCE(m.start_ts, 0) < ?",
              { query_embedding, static_cast<long long> (candidate_limit),
                memory_id, end_ts });
          AppendUniqueOwnedSupersessionRows (
              result.rows, std::move (historical_rows), seen_memory_ids,
              static_cast<int> (embedding_to_store.size ()),
              &historical_execution_count);
        }
      catch (const std::exception &e)
        {
          telemetry::LogDebug (
              "cortext.memory_storage.supersession_historical_knn_unavailable",
              { telemetry::Attribute::String ("error", e.what ()) });
        }
    }
  context.AddOperationTiming ("MemoryStorage.supersession_historical_load",
                              ElapsedMillis (historical_start));
  context.AddOperationTiming (
      "MemoryStorage.supersession_historical_coverage_proven",
      historical_coverage_proven ? 1.0 : 0.0);
  if (result.cache_owner)
    {
      context.AddOperationTiming (
          "MemoryStorage.supersession_population_mismatch_count",
          static_cast<double> (
              result.cache_owner->supersession_population_mismatches.size ()));
      context.AddOperationTiming (
          "MemoryStorage.supersession_population_ambiguous",
          result.cache_owner->supersession_population_ambiguous ? 1.0 : 0.0);
      context.AddOperationTiming (
          "MemoryStorage.supersession_tie_order_equivalent",
          result.cache_owner->supersession_tie_order_equivalent ? 1.0 : 0.0);
    }
  context.AddOperationTiming (
      "MemoryStorage.supersession_current_candidate_count",
      static_cast<double> (current_count));
  context.AddOperationTiming (
      "MemoryStorage.supersession_current_candidate_activity",
      current_count > 0 ? 1.0 : 0.0);
  context.AddOperationTiming (
      "MemoryStorage.supersession_historical_candidate_count",
      static_cast<double> (result.rows.size () - current_count));
  context.AddOperationTiming (
      "MemoryStorage.supersession_historical_candidate_activity",
      result.rows.size () > current_count ? 1.0 : 0.0);
  context.AddOperationTiming (
      "MemoryStorage.supersession_current_candidate_execution_count",
      static_cast<double> (current_execution_count));
  context.AddOperationTiming (
      "MemoryStorage.supersession_historical_candidate_execution_count",
      static_cast<double> (historical_execution_count));
  context.AddOperationTiming (
      "MemoryStorage.supersession_current_rows_visited",
      static_cast<double> (current_rows_visited));
  context.AddOperationTiming (
      "MemoryStorage.supersession_historical_rows_visited",
      static_cast<double> (historical_rows_visited));

  // When both cache searches completed and historical coverage was proven,
  // an empty result is exact even when the just-inserted memory is the sole
  // cached entry and is excluded from its own candidate set.
  const bool certified_empty_population
      = current_cache_rows.has_value ()
        && historical_cache_rows.has_value () && historical_coverage_proven;
  if (result.rows.empty () && !certified_empty_population)
    {
      ++sql_fallback_count;
      auto recent_rows = tx.Execute (
          "SELECT m.memory_id, "
          "       CASE WHEN cme.memory_id IS NOT NULL "
          "            THEN cme.embedding ELSE e.embedding END AS embedding "
          "FROM memories m "
          "LEFT JOIN current_memory_embeddings cme "
          "  ON cme.memory_id = m.memory_id "
          "JOIN embeddings e "
          "  ON e.embedding_id = COALESCE(cme.embedding_id, m.embedding_id) "
          "WHERE m.memory_id != ? "
          "  AND m.embedding_id IS NOT NULL "
          "  AND m.kind IN ('LONG_TERM', 'ASSOCIATION') "
          "  AND COALESCE(m.start_ts, 0) < ? "
          "ORDER BY m.memory_id DESC "
          "LIMIT ?",
          { memory_id, end_ts, static_cast<long long> (candidate_limit) });
      AppendUniqueOwnedSupersessionRows (
          result.rows, std::move (recent_rows), seen_memory_ids,
          static_cast<int> (embedding_to_store.size ()));
    }

  context.AddOperationTiming (
      "MemoryStorage.supersession_sql_fallback_count",
      static_cast<double> (sql_fallback_count));

  return result;
}

struct SupersessionEdge
{
  long long target_memory_id = 0;
  double similarity = 0.0;
  double weight = 0.0;
};

std::vector<SupersessionEdge>
WriteSupersessionEdges (OperationContext &context, Transaction &tx,
                        long long memory_id,
                        const Eigen::VectorXf &embedding_to_store,
                        long long end_ts)
{
  if (memory_id <= 0 || embedding_to_store.size () <= 0)
    {
      return {};
    }

  const auto &cfg = context.GetConfig ();
  const int embedding_dim = static_cast<int> (embedding_to_store.size ());
  const int candidate_limit = std::max (
      1, core::SupersessionCandidateLimit (cfg.focus, cfg.sensitivity,
                                           cfg.stability));
  const int max_edges = core::SupersessionMaxEdges (
      cfg.focus, cfg.sensitivity, cfg.stability);
  const auto candidate_load_start = SteadyClock::now ();
  const auto candidate_rows = LoadSupersessionCandidateRows (
      context, tx, context.GetProcessorContext (), embedding_to_store,
      memory_id, end_ts, candidate_limit);
  context.AddOperationTiming ("MemoryStorage.supersession_candidate_load",
                              ElapsedMillis (candidate_load_start));

  const double similarity_threshold = core::SupersessionSimilarityThreshold (
      cfg.focus, cfg.sensitivity, cfg.stability);
  const double duplicate_threshold = core::SupersessionDuplicateThreshold (
      cfg.focus, cfg.sensitivity, cfg.stability);
  int scanned_count = 0;
  int decoded_count = 0;
  int below_topic_count = 0;
  int duplicate_count = 0;
  double best_similarity = -1.0;
  std::vector<SupersessionEdge> edges;
  edges.reserve (static_cast<std::size_t> (max_edges));
  int edge_count = 0;
  const double query_norm = embedding_to_store.norm ();
  const auto score_start = SteadyClock::now ();
  for (const auto &row : candidate_rows.rows)
    {
      ++scanned_count;
      if (row.memory_id <= 0)
        {
          continue;
        }
      const Eigen::VectorXf *target_embedding = row.Embedding ();
      if (target_embedding == nullptr
          || target_embedding->size () != embedding_dim)
        {
          continue;
        }
      ++decoded_count;
      const double similarity
          = historical_surface_search_cache_internal::
              SupersessionCosineSimilarity (embedding_to_store, query_norm,
                                             *target_embedding);
      best_similarity = std::max (best_similarity, similarity);
      if (similarity < similarity_threshold)
        {
          ++below_topic_count;
          continue;
        }
      if (similarity >= duplicate_threshold)
        {
          ++duplicate_count;
          continue;
        }
      const double weight = core::SupersessionEdgeWeight (
          similarity, cfg.focus, cfg.sensitivity, cfg.stability);
      edges.push_back ({ row.memory_id, similarity, weight });
    }

  std::sort (edges.begin (), edges.end (),
             [] (const SupersessionEdge &a, const SupersessionEdge &b) {
               if (a.similarity != b.similarity)
                 {
                   return a.similarity > b.similarity;
                 }
               return a.target_memory_id > b.target_memory_id;
             });
  if (static_cast<int> (edges.size ()) > max_edges)
    {
      edges.resize (static_cast<std::size_t> (max_edges));
    }
  context.AddOperationTiming ("MemoryStorage.supersession_score",
                              ElapsedMillis (score_start));

  const auto write_start = SteadyClock::now ();
  for (const auto &edge : edges)
    {
      tx.Execute (
          "INSERT OR REPLACE INTO associations "
          "(source_memory_id, target_memory_id, edge_type, weight, "
          "last_reinforced) "
          "VALUES (?, ?, 'supersedes', ?, ?)",
          { memory_id, edge.target_memory_id, edge.weight, end_ts });
      tx.Execute (
          "UPDATE memories "
          "SET source_contradiction_count = source_contradiction_count + 1 "
          "WHERE memory_id = ?",
          { edge.target_memory_id });
      ++edge_count;
    }
  context.AddOperationTiming ("MemoryStorage.supersession_write",
                              ElapsedMillis (write_start));

  telemetry::LogDebug (
      "cortext.memory_storage.supersession_scan",
      { telemetry::Attribute::Int64 ("source_memory_id", memory_id),
        telemetry::Attribute::Int64 ("candidate_count", scanned_count),
        telemetry::Attribute::Int64 ("decoded_count", decoded_count),
        telemetry::Attribute::Int64 ("below_topic_count", below_topic_count),
        telemetry::Attribute::Int64 ("duplicate_count", duplicate_count),
        telemetry::Attribute::Int64 ("edge_count", edge_count),
        telemetry::Attribute::Int64 ("max_edges", max_edges),
        telemetry::Attribute::Double ("best_similarity", best_similarity),
        telemetry::Attribute::Double ("similarity_threshold",
                                      similarity_threshold),
        telemetry::Attribute::Double ("duplicate_threshold",
                                      duplicate_threshold) });
  return edges;
}

} // namespace

void
MemoryStorage::Execute (OperationContext &context, Transaction &tx) const
{
  // Check write gate decision - if rejected, discard entirely
  if (!context.GetAccumulatorWriteDecision ())
    {
      telemetry::AddCounter ("cortext.memory_storage.rejected_total", 1);
      return;
    }

  const auto &signal = context.GetSignal ();
  const auto &cfg = context.GetConfig ();
  if (signal.retention == Retention::Ephemeral)
    {
      telemetry::AddCounter ("cortext.memory_storage.ephemeral_skip_total", 1);
      return;
    }

  Store *store = context.GetStore ();
  if (!store)
    {
      telemetry::AddCounter ("cortext.memory_storage.no_store_total", 1);
      return;
    }

  auto &p_ctx = context.GetProcessorContext ();
  signal_record_rollback_internal::EnsureBackedUp (p_ctx);

  // Get accumulator state for this source
  auto acc_it = p_ctx.accumulator_states.find (signal.source_id);
  if (acc_it == p_ctx.accumulator_states.end ())
    {
      telemetry::AddCounter ("cortext.memory_storage.no_accumulator_total", 1);
      return;
    }
  auto &acc = acc_it->second;

  // Use nested transaction for atomicity - all writes succeed or none
  const auto savepoint_begin_start = SteadyClock::now ();
  auto savepoint = tx.Begin ();
  auto object_savepoint = context.GetObjectTransaction ()
                              ? context.GetObjectTransaction ()->Begin ()
                              : nullptr;
  context.AddOperationTiming ("MemoryStorage.begin_savepoints",
                              ElapsedMillis (savepoint_begin_start));
  bool savepoint_finished = false;
  bool object_savepoint_finished = false;
  std::vector<SignalRecord *> deferred_payload_records;
  auto rollback_savepoints = [&] {
    std::exception_ptr rollback_error;
    if (object_savepoint && !object_savepoint_finished)
      {
        try
          {
            object_savepoint->Rollback ();
            object_savepoint_finished = true;
          }
        catch (...)
          {
            if (!rollback_error)
              {
                rollback_error = std::current_exception ();
              }
          }
      }
    if (!savepoint_finished)
      {
        try
          {
            savepoint->Rollback ();
            savepoint_finished = true;
          }
        catch (...)
          {
            rollback_error = std::current_exception ();
          }
      }
    // Preserve deferred bytes for a retry. A rolled-back object transaction
    // invalidates ids assigned during this attempted write.
    for (auto *rec : deferred_payload_records)
      {
        rec->blob_id.clear ();
      }
    if (rollback_error)
      {
        std::rethrow_exception (rollback_error);
      }
  };

  try
    {
      using store::AnyToLongLong;
      using store::BlobFromAny;
      using store::EigenToFloatVec;

      // Section 4.4: Use representative embedding if available (memory write)
      const auto &rep_emb = context.GetRepresentativeEmbedding ();
      const Eigen::VectorXf &embedding_to_store
          = rep_emb.has_value () ? *rep_emb : acc.mu_acc;

      // Convert embedding from Eigen to vector<float> for SQL storage
      const std::vector<float> emb_vec = EigenToFloatVec (embedding_to_store);

      // 1. Compute memory-level aggregates from accumulator (Section 4.4)
      const int n_signals = std::max (acc.n_signals, 1);
      const double s_avg = acc.s_sum / static_cast<double> (n_signals);
      const double s_max = acc.s_max;
      const double s_emotion_max = acc.s_emotion_max;
      const double s_arousal_avg
          = acc.s_arousal_sum / static_cast<double> (n_signals);
      const uint64_t start_ts = acc.t_start;
      const uint64_t end_ts = signal.timestamp;
      const double drift_mag = acc.drift_acc;
      const double boundary_score
          = context.GetBoundaryScore ().value_or (0.0);

      // 2. Get emotion and mood from context (Section 6.1.1)
      const auto &emotion_probs = context.GetEmotionProbabilities ();
      const auto &mood_vec = p_ctx.mood_vector;
      const std::vector<char> emotion_blob = SerializeEmotionVector (emotion_probs);
      const std::vector<char> mood_blob = SerializeEmotionVector (mood_vec);

      // 3. Determine primary modality from tracked signals
      const std::string primary_modality
          = GetPrimaryModality (acc.signals, signal.modality);

      const std::string origin = SourceOriginFor ();
      const double source_reliability
          = SourcePriorReliability (cfg.focus, cfg.sensitivity, cfg.stability);

      // 4. Require tracked per-signal records for persistence.
      if (acc.signals.empty ())
        {
          rollback_savepoints ();
          telemetry::AddCounter (
              "cortext.memory_storage.missing_signals_total", 1);
          return;
        }

      // 5. Persist deferred per-signal payloads only after the write gate has
      // accepted this accumulator. This keeps rejected Natural signals out of
      // objstore while preserving one object per persisted signal.
      double signal_payload_put_ms = 0.0;
      for (auto &rec : acc.signals)
        {
          if (rec.blob_id.empty () && !rec.payload.empty ())
            {
              const auto object_put_start = SteadyClock::now ();
              rec.blob_id = PutObject (object_savepoint.get (), *savepoint,
                                       rec.payload);
              deferred_payload_records.push_back (&rec);
              signal_payload_put_ms += ElapsedMillis (object_put_start);
            }
        }
      context.AddOperationTiming ("MemoryStorage.signal_payload_put",
                                  signal_payload_put_ms);

      // 6. Store memory-level content blob by concatenating signal blobs.
      const auto content_start = SteadyClock::now ();
      double content_get_ms = 0.0;
      double content_put_ms = 0.0;
      std::vector<unsigned char> content_payload;
      std::vector<const SignalRecord *> ordered_signals;
      ordered_signals.reserve (acc.signals.size ());
      for (const auto &rec : acc.signals)
        {
          ordered_signals.push_back (&rec);
        }
      std::sort (ordered_signals.begin (), ordered_signals.end (),
                 [] (const SignalRecord *a, const SignalRecord *b) {
                   return a->serial_position < b->serial_position;
                 });
      const bool text_mode = std::all_of (
          ordered_signals.begin (), ordered_signals.end (),
          [] (const SignalRecord *rec) {
            return rec && rec->modality == "text";
          });
      const bool audio_mode = std::all_of (
          ordered_signals.begin (), ordered_signals.end (),
          [] (const SignalRecord *rec) {
            return rec && rec->modality == "audio";
          });
      const bool single_payload_mode = ordered_signals.size () == 1;
      const bool memory_blob_supported
          = text_mode || audio_mode || single_payload_mode;
      std::vector<unsigned char> content_blob_id;
      if (single_payload_mode && !ordered_signals.empty ()
          && ordered_signals[0] && !ordered_signals[0]->blob_id.empty ())
        {
          content_blob_id = ordered_signals[0]->blob_id;
        }
      else if (memory_blob_supported)
        {
          for (const auto *rec : ordered_signals)
            {
              if (!rec || rec->blob_id.empty ())
                {
                  continue;
                }
              const auto object_get_start = SteadyClock::now ();
              auto bytes_opt = GetObject (object_savepoint.get (), *savepoint,
                                          rec->blob_id);
              content_get_ms += ElapsedMillis (object_get_start);
              if (bytes_opt)
                {
                  const auto &bytes = *bytes_opt;
                  if (!bytes.empty ())
                    {
                      if (text_mode && !content_payload.empty ())
                        {
                          content_payload.push_back ('\n');
                        }
                      content_payload.insert (content_payload.end (),
                                              bytes.begin (), bytes.end ());
                    }
                }
            }
        }
      if (memory_blob_supported && content_payload.empty () && signal.payload
          && !signal.payload->empty ())
        {
          content_payload = *signal.payload;
        }
      if (content_blob_id.empty () && !content_payload.empty ())
        {
          const auto object_put_start = SteadyClock::now ();
          content_blob_id
              = PutObject (object_savepoint.get (), *savepoint,
                           content_payload);
          content_put_ms += ElapsedMillis (object_put_start);
        }
      context.AddOperationTiming ("MemoryStorage.content_get_objects",
                                  content_get_ms);
      context.AddOperationTiming ("MemoryStorage.content_put_object",
                                  content_put_ms);
      context.AddOperationTiming ("MemoryStorage.content_blob",
                                  ElapsedMillis (content_start));
      // 7. Insert memory embedding (v2: minimal sqlite-vec table)
      const std::string insert_sql = std::string ("INSERT INTO embeddings (")
                                     + store::kEmbeddingsInsertColumns
                                     + ") VALUES ("
                                     + store::kEmbeddingsInsertDefaults + ")";
      const auto insert_embedding_start = SteadyClock::now ();
      savepoint->Execute (insert_sql,
                          { emb_vec, static_cast<long long> (end_ts) });
      context.AddOperationTiming ("MemoryStorage.insert_memory_embedding",
                                  ElapsedMillis (insert_embedding_start));

      const auto embedding_id_start = SteadyClock::now ();
      auto id_rows = savepoint->Execute ("SELECT last_insert_rowid() AS id", {});
      context.AddOperationTiming ("MemoryStorage.select_embedding_id",
                                  ElapsedMillis (embedding_id_start));
      if (id_rows.empty () || id_rows[0].count ("id") == 0)
        {
          rollback_savepoints ();
          telemetry::AddCounter (
              "cortext.memory_storage.embedding_error_total", 1);
          return;
        }
      const auto id_opt = AnyToLongLong (id_rows[0].at ("id"));
      if (!id_opt)
        {
          rollback_savepoints ();
          telemetry::AddCounter (
              "cortext.memory_storage.embedding_error_total", 1);
          return;
        }
      const long long embedding_id = *id_opt;

      // 8. Insert MEMORIES row with aggregated metadata (v2 schema)
      std::any episode_id_any;
      if (p_ctx.episode_start_ts > 0)
        {
          episode_id_any = static_cast<long long> (p_ctx.episode_start_ts);
        }

      std::vector<float> ctx_vec;
      if (acc.c_t.size () > 0)
        {
          ctx_vec = EigenToFloatVec (acc.c_t);
        }

      const auto initial_trace = core::MemoryInitialTracePolicy (
          cfg.focus, cfg.sensitivity, cfg.stability);
      const double initial_strength = core::MemoryInitialStrengthPolicy (
          cfg.focus, cfg.sensitivity, cfg.stability);
      const double initial_stability = core::MemoryInitialStabilityPolicy (
          cfg.focus, cfg.sensitivity, cfg.stability);
      const auto insert_memory_start = SteadyClock::now ();
      savepoint->Execute (
          "INSERT INTO memories ("
          "  embedding_id, source_id, kind, start_ts, end_ts, n_signals, "
          "  modality, s_max, s_avg, s_emotion_max, s_arousal_avg, boundary_score, "
          "  drift_mag, emotion, ambient_mood, episode_id, "
          "  blob_id, created_at, context, source_origin, source_reliability, "
          "  strength, stability, "
          "  trace_fast, trace_med, trace_slow, trace_ultra"
          ") VALUES (?, ?, 'LONG_TERM', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
          { embedding_id, signal.source_id, static_cast<long long> (start_ts),
            static_cast<long long> (end_ts),
            static_cast<long long> (n_signals), primary_modality,
            s_max, s_avg, s_emotion_max, s_arousal_avg, boundary_score,
            drift_mag, emotion_blob, mood_blob, episode_id_any,
            content_blob_id.empty () ? std::any () : std::any (content_blob_id),
            static_cast<long long> (end_ts),
            ctx_vec.empty () ? std::any () : std::any (ctx_vec),
            origin, source_reliability,
            initial_strength, initial_stability,
            initial_trace.fast, initial_trace.medium, initial_trace.slow,
            initial_trace.ultra });
      context.AddOperationTiming ("MemoryStorage.insert_memory",
                                  ElapsedMillis (insert_memory_start));

      // 8. Get memory_id from inserted memories row
      const auto memory_id_start = SteadyClock::now ();
      auto mem_id_rows
          = savepoint->Execute ("SELECT last_insert_rowid() AS id", {});
      context.AddOperationTiming ("MemoryStorage.select_memory_id",
                                  ElapsedMillis (memory_id_start));
      const long long memory_id
          = mem_id_rows.empty ()
                ? 0
                : AnyToLongLong (mem_id_rows[0].at ("id")).value_or (0);
      emotional_metadata_cache_internal::Upsert (
          p_ctx,
          { memory_id, embedding_id, static_cast<long long> (end_ts), false,
            0.0, s_arousal_avg, 0.0, 0, 0.0 });
      std::vector<historical_surface_search_cache_internal::Entry>
          inserted_search_entries;
      inserted_search_entries.push_back (
          { embedding_id, memory_id, static_cast<long long> (start_ts),
            "LONG_TERM", signal.source_id, embedding_to_store });

      // 10. Insert SIGNALS rows (one per tracked signal)
      std::optional<long long> stored_signal_id;
      std::optional<long long> current_signal_id;
      double signal_embedding_insert_ms = 0.0;
      double signal_row_insert_ms = 0.0;
      double signal_id_select_ms = 0.0;
      for (const auto &sig_rec : acc.signals)
        {
          const std::vector<float> signal_embedding
              = sig_rec.embedding.size () > 0
                    ? EigenToFloatVec (sig_rec.embedding)
                    : emb_vec;
          const auto signal_embedding_start = SteadyClock::now ();
          savepoint->Execute (insert_sql,
                              { signal_embedding,
                                static_cast<long long> (sig_rec.timestamp) });
          signal_embedding_insert_ms += ElapsedMillis (signal_embedding_start);

          const auto signal_embedding_id_start = SteadyClock::now ();
          auto signal_embedding_id_rows
              = savepoint->Execute ("SELECT last_insert_rowid() AS id", {});
          signal_id_select_ms += ElapsedMillis (signal_embedding_id_start);
          if (signal_embedding_id_rows.empty ()
              || signal_embedding_id_rows[0].count ("id") == 0)
            {
              rollback_savepoints ();
              telemetry::AddCounter (
                  "cortext.memory_storage.signal_embedding_error_total", 1);
              return;
            }
          const long long signal_embedding_id
              = AnyToLongLong (signal_embedding_id_rows[0].at ("id"))
                    .value_or (0);
          if (signal_embedding_id <= 0)
            {
              rollback_savepoints ();
              telemetry::AddCounter (
                  "cortext.memory_storage.signal_embedding_error_total", 1);
              return;
            }
          inserted_search_entries.push_back (
              { signal_embedding_id, 0, 0, std::string (), std::string (),
                sig_rec.embedding.size () > 0 ? sig_rec.embedding
                                              : embedding_to_store });

          const auto signal_row_start = SteadyClock::now ();
          savepoint->Execute (
              "INSERT INTO signals ("
              "  memory_id, source_id, embedding_id, timestamp, modality, "
              "  mime, blob_id, serial_position, score, created_at"
              ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
              { memory_id, signal.source_id, signal_embedding_id,
                static_cast<long long> (sig_rec.timestamp), sig_rec.modality,
                sig_rec.mime,
                sig_rec.blob_id.empty () ? std::any ()
                                         : std::any (sig_rec.blob_id),
                static_cast<long long> (sig_rec.serial_position), sig_rec.score,
                static_cast<long long> (sig_rec.timestamp) });
          signal_row_insert_ms += ElapsedMillis (signal_row_start);
          const auto signal_id_start = SteadyClock::now ();
          auto signal_id_rows
              = savepoint->Execute ("SELECT last_insert_rowid() AS id", {});
          signal_id_select_ms += ElapsedMillis (signal_id_start);
          if (!signal_id_rows.empty ()
              && signal_id_rows[0].count ("id") != 0)
            {
              const auto signal_id
                  = AnyToLongLong (signal_id_rows[0].at ("id"));
              if (signal_id && *signal_id > 0)
                {
                  stored_signal_id = *signal_id;
                  if (sig_rec.timestamp == signal.timestamp
                      && sig_rec.modality == signal.modality)
                    {
                      current_signal_id = *signal_id;
                    }
                }
            }
        }
      context.AddOperationTiming ("MemoryStorage.insert_signal_embeddings",
                                  signal_embedding_insert_ms);
      context.AddOperationTiming ("MemoryStorage.insert_signal_rows",
                                  signal_row_insert_ms);
      context.AddOperationTiming ("MemoryStorage.select_signal_ids",
                                  signal_id_select_ms);

      for (auto &entry : inserted_search_entries)
        {
          // Signal-only embeddings participate in sqlite-vec's top-k
          // population even though the subsequent memory join filters them.
          historical_surface_search_cache_internal::Append (
              p_ctx, std::move (entry));
        }

      // 10. Leave signal tracking until accumulator resets (used by WM gating)
      const auto supersession_start = SteadyClock::now ();
      const auto supersession_edges = WriteSupersessionEdges (
          context, *savepoint, memory_id, embedding_to_store,
          static_cast<long long> (end_ts));
      context.AddOperationTiming ("MemoryStorage.supersession_edges",
                                  ElapsedMillis (supersession_start));

      if (!constructive_recall::Disabled () && memory_id > 0)
        {
          const auto reconstruction_start = SteadyClock::now ();
          constructive_recall::AppendReconstructionWithEmbeddingId (
              *savepoint, memory_id, embedding_id, content_blob_id,
              static_cast<long long> (end_ts), 0.0, "initial", 1.0, 1.0);
          context.AddOperationTiming ("MemoryStorage.initial_reconstruction",
                                      ElapsedMillis (reconstruction_start));
        }

      // Commit external object content before releasing the DB savepoint. If
      // DB commit then fails, content-addressed payloads may be orphaned; the
      // reverse order can leave DB rows pointing at uncommitted object content.
      if (object_savepoint)
        {
          const auto object_commit_start = SteadyClock::now ();
          object_savepoint->Commit ();
          object_savepoint_finished = true;
          object_savepoint.reset ();
          context.AddOperationTiming ("MemoryStorage.commit_object_savepoint",
                                      ElapsedMillis (object_commit_start));
        }
      const auto savepoint_commit_start = SteadyClock::now ();
      savepoint->Commit ();
      savepoint_finished = true;
      context.AddOperationTiming ("MemoryStorage.commit_savepoints",
                                  ElapsedMillis (savepoint_commit_start));
      // The object and SQL savepoints both committed, so records can retain
      // only their durable ids before working-memory copies them onward.
      for (auto *rec : deferred_payload_records)
        {
          rec->payload.clear ();
          rec->payload.shrink_to_fit ();
        }

      // Set stored_embedding_id in context for output
      context.SetStoredEmbeddingId (embedding_id);
      if (memory_id > 0)
        {
          context.SetStoredMemoryId (memory_id);
          context.SetStoredSignalId (
              current_signal_id.has_value () ? current_signal_id
                                             : stored_signal_id);
        }
      else
        {
          context.SetStoredMemoryId (std::nullopt);
          context.SetStoredSignalId (std::nullopt);
        }
      p_ctx.memories_since_consolidation += 1;
      if (memory_id > 0 && embedding_to_store.size () > 0)
        {
          const auto surface_start = SteadyClock::now ();
          ProcessorContext::RetrievalSurfaceEntry surface_entry;
          surface_entry.memory_id = memory_id;
          surface_entry.embedding_id = embedding_id;
          surface_entry.created_at = static_cast<long long> (end_ts);
          surface_entry.start_ts = static_cast<long long> (start_ts);
          surface_entry.event_ts = static_cast<long long> (start_ts);
          surface_entry.kind = "LONG_TERM";
          surface_entry.source_id = signal.source_id;
          surface_entry.modality = primary_modality;
          surface_entry.source_reliability = source_reliability;
          if (acc.c_t.size () > 0)
            {
              surface_entry.context_embedding = acc.c_t;
            }
          surface_entry.embedding = embedding_to_store;
          p_ctx.UpsertRetrievalSurface (std::move (surface_entry));
          auto &fanout_cache = p_ctx.association_fanout_cache;
          if (fanout_cache.valid && !supersession_edges.empty ())
            {
              bool incrementally_maintained = true;
              for (const auto &edge : supersession_edges)
                {
                  const auto target_it = p_ctx.retrieval_surface_index.find (
                      edge.target_memory_id);
                  if (target_it == p_ctx.retrieval_surface_index.end ()
                      || target_it->second
                             >= p_ctx.retrieval_surface_cache.size ())
                    {
                      incrementally_maintained = false;
                      break;
                    }
                  const auto &target
                      = p_ctx.retrieval_surface_cache[target_it->second];
                  const long long target_embedding_id
                      = historical_surface_search_cache_internal::
                          BaseEmbeddingIdForMemory (
                              p_ctx, edge.target_memory_id,
                              target.embedding_id);
                  if (target_embedding_id <= 0)
                    {
                      incrementally_maintained = false;
                      break;
                    }
                  association_fanout_cache::UpsertAssociation (
                      p_ctx, fanout_cache, memory_id, edge.target_memory_id,
                      embedding_id, target_embedding_id, "supersedes",
                      edge.weight, static_cast<long long> (end_ts));
                }
              if (incrementally_maintained)
                association_fanout_cache::BuildSupersessionEligibility (
                    fanout_cache, p_ctx);
              else
                fanout_cache.valid = false;
            }
          if (!constructive_recall::Disabled ()
              && !constructive_recall::CurrentSurfaceWritesDisabled ())
            {
              historical_surface_search_cache_internal::UpsertCurrent (
                  p_ctx,
                  { embedding_id, memory_id, 0, std::string (),
                    std::string (), embedding_to_store });
            }
          if (!constructive_recall::Disabled ()
              && constructive_recall::CurrentSurfaceWritesDisabled ())
            {
              historical_surface_search_cache_internal::
                  SetCurrentSurfaceDatabaseCurrent (p_ctx, false);
            }
          const int k_key = core::SparseKeySize (
              context.GetConfig ().focus, context.GetConfig ().sensitivity,
              context.GetConfig ().stability);
          const std::string key = core::SparseKey (embedding_to_store, k_key);
          if (!key.empty ())
            {
              signal_record_rollback_internal::
                  PreserveSparseIndexBeforeInsert (
                      p_ctx, key, memory_id);
              p_ctx.index_store[key].push_back (memory_id);
              p_ctx.index_reverse[memory_id] = key;
            }
          context.AddOperationTiming ("MemoryStorage.retrieval_surface_update",
                                      ElapsedMillis (surface_start));
        }
      telemetry::AddCounter ("cortext.memory_storage.stored_total", 1);
      telemetry::AddCounter ("cortext.memory_storage.signals_written_total",
                             static_cast<int64_t> (n_signals));

      // Debug logging
      telemetry::LogDebug (
          "cortext.memory_storage",
          { telemetry::Attribute::Bool ("stored", true),
            telemetry::Attribute::Int64 ("embedding_id", embedding_id),
            telemetry::Attribute::Int64 ("n_signals", n_signals),
            telemetry::Attribute::String ("primary_modality", primary_modality),
            telemetry::Attribute::Double ("s_max", s_max),
            telemetry::Attribute::Double ("s_avg", s_avg) });
    }
  catch (const std::exception &e)
    {
      rollback_savepoints ();
      telemetry::AddCounter ("cortext.memory_storage.error_total", 1);
      telemetry::LogError (
          "MemoryStorage failed",
          { telemetry::Attribute::String ("error", e.what ()) });
      throw;
    }
}

} // namespace cortext::operations
