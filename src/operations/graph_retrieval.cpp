#include "cortext/operations/graph_retrieval.hpp"

#include "constructive_recall_internal.hpp"
#include "association_fanout_cache_internal.hpp"
#include "historical_surface_search_cache_internal.hpp"
#include "retrieval_trace_state.hpp"
#include "cortext/core/knobs.hpp"
#include "cortext/core/utils.hpp"
#include "cortext/processor/operation_context.hpp"
#include "cortext/store/store.hpp"
#include "cortext/store/utils.hpp"
#include "cortext/telemetry/telemetry.hpp"
#include "../experimental_env.hpp"

#include <algorithm>
#include <array>
#include <any>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <exception>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace cortext::operations
{

namespace
{

struct Candidate
{
  long long memory_id = 0;
  long long embedding_id = 0;
  long long start_ts = 0;
  Eigen::VectorXf embedding;
  double seed_score = 0.0;
  double graph_score = 0.0;
  double temporal_score = 0.0;
  double superseded_penalty = 0.0;
  double score = 0.0;
  bool direct_seed = false;
};

constexpr double kTemporalRankDaySeconds = 24.0 * 60.0 * 60.0;

double
Cosine (const Eigen::VectorXf &a, const Eigen::VectorXf &b)
{
  if (a.size () == 0 || b.size () == 0 || a.size () != b.size ())
    {
      return 0.0;
    }
  const double denom = static_cast<double> (a.norm () * b.norm ());
  if (denom <= 1e-12)
    {
      return 0.0;
    }
  return static_cast<double> (a.dot (b)) / denom;
}

bool
AnyToEmbedding (const std::any &value, int expected_dim, Eigen::VectorXf &out)
{
  if (value.type () == typeid (std::vector<float>))
    {
      const auto &vec = std::any_cast<const std::vector<float> &> (value);
      if (expected_dim > 0 && static_cast<int> (vec.size ()) != expected_dim)
        {
          return false;
        }
      out.resize (static_cast<Eigen::Index> (vec.size ()));
      for (std::size_t i = 0; i < vec.size (); ++i)
        {
          out[static_cast<Eigen::Index> (i)] = vec[i];
        }
      return out.size () > 0;
    }
  if (value.type () == typeid (Eigen::VectorXf))
    {
      out = std::any_cast<Eigen::VectorXf> (value);
      return out.size () > 0
             && (expected_dim <= 0 || out.size () == expected_dim);
    }
  if (expected_dim > 0 && core::DecodeFloatBlob (value, expected_dim, out))
    {
      return true;
    }
  return false;
}

long long
AnyLongLong (const std::map<std::string, std::any> &row,
             const std::string &key)
{
  auto it = row.find (key);
  if (it == row.end ())
    {
      return 0;
    }
  return cortext::store::AnyToLongLong (it->second).value_or (0);
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

long long
LoadCurrentSurfaceEmbeddingId (Transaction &tx, long long memory_id,
                               long long fallback_embedding_id)
{
  const auto latest = constructive_recall::LoadLatestReconstruction (
      tx, memory_id);
  auto rows = tx.Execute (
      "SELECT embedding_id, created_at "
      "FROM current_memory_embeddings WHERE memory_id = ?",
      { memory_id });
  if (!rows.empty () && rows[0].count ("embedding_id") == 1)
    {
      const long long embedding_id = cortext::store::AnyToLongLong (
          rows[0].at ("embedding_id")).value_or (0);
      const bool current_surface_fresh
          = !latest.has_value () || embedding_id == latest->embedding_id;
      if (embedding_id > 0 && current_surface_fresh)
        {
          return embedding_id;
        }
    }
  if (latest.has_value () && latest->embedding_id > 0)
    {
      return latest->embedding_id;
    }
  return fallback_embedding_id;
}

bool
RefreshCandidateToCurrentSurface (Transaction &tx, Candidate &candidate,
                                  int embedding_dim)
{
  if (auto current = constructive_recall::LoadCurrentEmbedding (
          tx, candidate.memory_id, candidate.embedding_id, embedding_dim))
    {
      candidate.embedding_id = LoadCurrentSurfaceEmbeddingId (
          tx, candidate.memory_id, candidate.embedding_id);
      candidate.embedding = std::move (*current);
      return true;
    }
  return false;
}

bool
RefreshCandidateFromProcessorSurface (const ProcessorContext &p_ctx,
                                      Candidate &candidate,
                                      int embedding_dim)
{
  const auto it = p_ctx.retrieval_surface_index.find (candidate.memory_id);
  if (it == p_ctx.retrieval_surface_index.end ()
      || it->second >= p_ctx.retrieval_surface_cache.size ())
    {
      return false;
    }
  const auto &entry = p_ctx.retrieval_surface_cache[it->second];
  if (entry.embedding_id <= 0 || entry.embedding.size () != embedding_dim)
    {
      return false;
    }
  candidate.embedding_id = entry.embedding_id;
  candidate.embedding = entry.embedding;
  return true;
}

void
RefreshProcessorSurfaces (Transaction &tx, ProcessorContext &p_ctx,
                          long long memory_id, long long old_embedding_id,
                          long long new_embedding_id,
                          const Eigen::VectorXf &embedding)
{
  auto cache_it = p_ctx.retrieval_surface_index.find (memory_id);
  if (cache_it != p_ctx.retrieval_surface_index.end ())
    {
      auto &entry = p_ctx.retrieval_surface_cache[cache_it->second];
      if (entry.embedding_id > 0)
        {
          p_ctx.retrieval_surface_embedding_index.erase (entry.embedding_id);
        }
      entry.embedding_id = new_embedding_id;
      entry.embedding = embedding;
      entry.embedding_norm = embedding.norm ();
      p_ctx.retrieval_surface_embedding_index[new_embedding_id]
          = cache_it->second;
    }

  if (old_embedding_id > 0 && old_embedding_id != new_embedding_id
      && p_ctx.retrieval_surface_embedding_index.find (old_embedding_id)
             == p_ctx.retrieval_surface_embedding_index.end ())
    {
      auto rows = tx.Execute (
          "SELECT memory_id FROM memories "
          "WHERE embedding_id = ? AND memory_id != ? "
          "ORDER BY memory_id ASC LIMIT 1",
          { old_embedding_id, memory_id });
      if (!rows.empty () && rows[0].count ("memory_id") == 1)
        {
          const long long sibling_memory_id = cortext::store::AnyToLongLong (
              rows[0].at ("memory_id")).value_or (0);
          auto sibling_it = p_ctx.retrieval_surface_index.find (
              sibling_memory_id);
          if (sibling_memory_id > 0
              && sibling_it != p_ctx.retrieval_surface_index.end ())
            {
              p_ctx.retrieval_surface_embedding_index[old_embedding_id]
                  = sibling_it->second;
            }
        }
    }

  auto assoc_it = p_ctx.association_cache_index.find (memory_id);
  if (assoc_it != p_ctx.association_cache_index.end ())
    {
      auto &entry = p_ctx.association_cache[assoc_it->second];
      entry.embedding_id = new_embedding_id;
      entry.embedding = embedding;
      entry.embedding_norm = embedding.norm ();
    }
  if (!constructive_recall::CurrentSurfaceWritesDisabled ())
    {
      historical_surface_search_cache_internal::UpsertCurrent (
          p_ctx,
          { new_embedding_id, memory_id, 0, std::string (), std::string (),
            embedding });
    }
}

void
AppendUniqueRows (
    std::vector<std::map<std::string, std::any>> &rows,
    std::vector<std::map<std::string, std::any>> extra_rows,
    std::unordered_set<long long> &seen_memory_ids)
{
  for (auto &row : extra_rows)
    {
      const long long memory_id = AnyLongLong (row, "memory_id");
      if (memory_id <= 0 || !seen_memory_ids.insert (memory_id).second)
        {
          continue;
        }
      rows.push_back (std::move (row));
    }
}

constexpr std::size_t kFamilyNormBlockCount = 32;

struct FamilyEmbeddingFeatures
{
  Eigen::VectorXf normalized;
  std::array<double, kFamilyNormBlockCount> block_norms {};
  std::array<double, kFamilyNormBlockCount> hadamard_block_norms {};
  bool has_hadamard_features = false;
};

template <typename ValueAccessor>
void
BuildFamilyBlockNorms (
    Eigen::Index dimension_count, ValueAccessor value_at,
    std::array<double, kFamilyNormBlockCount> &block_norms)
{
  for (Eigen::Index dimension = 0; dimension < dimension_count; ++dimension)
    {
      const double value = value_at (dimension);
      const auto block = std::min<std::size_t> (
          kFamilyNormBlockCount - 1,
          static_cast<std::size_t> (dimension) * kFamilyNormBlockCount
              / static_cast<std::size_t> (dimension_count));
      block_norms[block] += value * value;
    }
  for (double &block_norm : block_norms)
    {
      block_norm = std::sqrt (block_norm);
    }
}

FamilyEmbeddingFeatures
BuildFamilyEmbeddingFeatures (const Eigen::VectorXf &embedding)
{
  FamilyEmbeddingFeatures features;
  if (embedding.size () <= 0)
    {
      return features;
    }

  double squared_norm = 0.0;
  for (Eigen::Index dimension = 0; dimension < embedding.size (); ++dimension)
    {
      const double value = static_cast<double> (embedding[dimension]);
      squared_norm += value * value;
    }
  if (squared_norm <= 1e-24)
    {
      return features;
    }

  const double inverse_norm = 1.0 / std::sqrt (squared_norm);
  features.normalized.resize (embedding.size ());
  for (Eigen::Index dimension = 0; dimension < embedding.size (); ++dimension)
    {
      features.normalized[dimension] = static_cast<float> (
          static_cast<double> (embedding[dimension]) * inverse_norm);
    }
  BuildFamilyBlockNorms (
      features.normalized.size (),
      [&features] (Eigen::Index dimension) {
        return static_cast<double> (features.normalized[dimension]);
      },
      features.block_norms);

  const std::size_t dimension_count
      = static_cast<std::size_t> (features.normalized.size ());
  if (dimension_count > 0
      && (dimension_count & (dimension_count - 1)) == 0)
    {
      std::vector<double> transformed (dimension_count);
      for (std::size_t dimension = 0; dimension < dimension_count;
           ++dimension)
        {
          transformed[dimension]
              = static_cast<double> (features.normalized[dimension]);
        }
      for (std::size_t span = 1; span < dimension_count; span *= 2)
        {
          for (std::size_t begin = 0; begin < dimension_count;
               begin += span * 2)
            {
              for (std::size_t offset = 0; offset < span; ++offset)
                {
                  const double left = transformed[begin + offset];
                  const double right = transformed[begin + span + offset];
                  transformed[begin + offset] = left + right;
                  transformed[begin + span + offset] = left - right;
                }
            }
        }
      const double inverse_transform_norm
          = 1.0 / std::sqrt (static_cast<double> (dimension_count));
      BuildFamilyBlockNorms (
          features.normalized.size (),
          [&transformed, inverse_transform_norm] (Eigen::Index dimension) {
            return transformed[static_cast<std::size_t> (dimension)]
                   * inverse_transform_norm;
          },
          features.hadamard_block_norms);
      features.has_hadamard_features = true;
    }
  return features;
}

class FamilyRepresentativeIndex
{
public:
  explicit FamilyRepresentativeIndex (double duplicate_threshold)
      : duplicate_threshold_ (
            core::Clamp (duplicate_threshold, -1.0, 1.0))
  {
  }

  bool
  IsDuplicate (const FamilyEmbeddingFeatures &candidate) const
  {
    if (candidate.normalized.size () <= 0)
      {
        return false;
      }
    for (const auto &representative : representatives_)
      {
        if (representative.normalized.size () != candidate.normalized.size ())
          {
            continue;
          }
        double cosine_upper_bound = 0.0;
        for (std::size_t block = 0; block < kFamilyNormBlockCount; ++block)
          {
            cosine_upper_bound += candidate.block_norms[block]
                                  * representative.block_norms[block];
          }
        if (candidate.has_hadamard_features
            && representative.has_hadamard_features)
          {
            double hadamard_upper_bound = 0.0;
            for (std::size_t block = 0; block < kFamilyNormBlockCount;
                 ++block)
              {
                hadamard_upper_bound
                    += candidate.hadamard_block_norms[block]
                       * representative.hadamard_block_norms[block];
              }
            cosine_upper_bound
                = std::min (cosine_upper_bound, hadamard_upper_bound);
          }
        constexpr double kBoundRoundoff = 1e-7;
        if (cosine_upper_bound + kBoundRoundoff < duplicate_threshold_)
          {
            continue;
          }

#ifdef CORTEXT_TESTING
        retrieval_trace::IncrementLastFamilyExactComparisonCount ();
#endif
        double cosine = 0.0;
        for (Eigen::Index dimension = 0;
             dimension < candidate.normalized.size (); ++dimension)
          {
            cosine += static_cast<double> (candidate.normalized[dimension])
                      * static_cast<double> (
                          representative.normalized[dimension]);
          }
        if (cosine >= duplicate_threshold_)
          {
            return true;
          }
      }
    return false;
  }

  void
  Add (FamilyEmbeddingFeatures features)
  {
    representatives_.push_back (std::move (features));
  }

private:
  double duplicate_threshold_ = 1.0;
  std::vector<FamilyEmbeddingFeatures> representatives_;
};

template <typename EntryAccessor>
void
CollapseRankedFamilies (
    std::vector<historical_surface_search_cache_internal::RankedEntry> &ranked,
    double duplicate_threshold, int candidate_limit, EntryAccessor entry_at)
{
  std::vector<historical_surface_search_cache_internal::RankedEntry> selected;
  selected.reserve (std::min<std::size_t> (
      ranked.size (),
      static_cast<std::size_t> (std::max (0, candidate_limit))));
  FamilyRepresentativeIndex families (duplicate_threshold);
  for (const auto &candidate : ranked)
    {
      if (static_cast<int> (selected.size ()) >= candidate_limit)
        {
          break;
      }
      const auto &entry = entry_at (candidate.index);
      auto features = BuildFamilyEmbeddingFeatures (entry.embedding);
      if (families.IsDuplicate (features))
        {
          continue;
        }
      families.Add (std::move (features));
      selected.push_back (candidate);
    }
  ranked = std::move (selected);
}

std::optional<std::vector<std::map<std::string, std::any>>>
LoadHistoricalSeedRowsFromCache (
    const ProcessorContext &p_ctx, const Eigen::VectorXf &query_embedding,
    long long exclusion_ts, int candidate_limit,
    const std::unordered_set<long long> &superseded_memory_ids,
    double duplicate_threshold)
{
  const auto state
      = historical_surface_search_cache_internal::Find (p_ctx);
  const std::size_t embedding_count = state ? state->entries.size () : 0;
  const int embedding_dim = static_cast<int> (query_embedding.size ());
  if (!state || state->recovery_failed || embedding_dim <= 0
      || candidate_limit <= 0
      || state->embedding_dim != embedding_dim
      || state->search.size ()
             != embedding_count * static_cast<std::size_t> (embedding_dim))
    {
      return std::nullopt;
    }

  using RowMajorMatrix
      = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  const Eigen::Map<const RowMajorMatrix> matrix (
      state->search.data (), static_cast<Eigen::Index> (embedding_count),
      embedding_dim);
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
      const auto eligible_reference = std::find_if (
          entry.memory_references.begin (), entry.memory_references.end (),
          [&] (const auto &reference) {
            return reference.memory_id > 0
                   && reference.kind == "LONG_TERM"
                   && reference.start_ts < exclusion_ts
                   && state->current_memory_index.count (
                          reference.memory_id)
                          == 0
                   && superseded_memory_ids.count (reference.memory_id) == 0;
          });
      if (eligible_reference == entry.memory_references.end ())
        {
          continue;
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
  std::sort (ranked.begin (), ranked.end (), by_distance);
  CollapseRankedFamilies (
      ranked, duplicate_threshold, candidate_limit,
      [&state] (std::size_t candidate_index) -> const auto & {
        return state->entries[candidate_index];
      });

  std::vector<std::map<std::string, std::any>> rows;
  rows.reserve (ranked.size ());
  for (const auto &candidate : ranked)
    {
      const auto &entry = state->entries[candidate.index];
      const auto eligible_reference = std::find_if (
          entry.memory_references.begin (), entry.memory_references.end (),
          [&] (const auto &reference) {
            return reference.memory_id > 0
                   && reference.kind == "LONG_TERM"
                   && reference.start_ts < exclusion_ts
                   && state->current_memory_index.count (
                          reference.memory_id)
                          == 0
                   && superseded_memory_ids.count (reference.memory_id) == 0;
          });
      if (eligible_reference == entry.memory_references.end ())
        {
          continue;
        }
      rows.push_back (
          { { "memory_id", eligible_reference->memory_id },
            { "embedding_id", entry.embedding_id },
            { "start_ts", eligible_reference->start_ts },
            { "embedding", ToFloatVector (entry.embedding) } });
    }
  return rows;
}

std::optional<std::vector<std::map<std::string, std::any>>>
LoadCurrentSeedRowsFromCache (
    const ProcessorContext &p_ctx, const Eigen::VectorXf &query_embedding,
    long long exclusion_ts, int candidate_limit,
    const std::unordered_set<long long> &superseded_memory_ids,
    double duplicate_threshold)
{
  const auto state
      = historical_surface_search_cache_internal::Find (p_ctx);
  const std::size_t entry_count = state ? state->current_entries.size () : 0;
  const int embedding_dim = static_cast<int> (query_embedding.size ());
  if (!state || state->recovery_failed || embedding_dim <= 0
      || candidate_limit <= 0
      || state->embedding_dim != embedding_dim
      || state->current_search.size ()
             != entry_count * static_cast<std::size_t> (embedding_dim))
    {
      return std::nullopt;
    }
  if (entry_count == 0)
    {
      return std::vector<std::map<std::string, std::any>>{};
    }

  using RowMajorMatrix
      = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  const Eigen::Map<const RowMajorMatrix> matrix (
      state->current_search.data (), static_cast<Eigen::Index> (entry_count),
      embedding_dim);
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
        {
          return std::nullopt;
        }
      const auto surface_it = p_ctx.retrieval_surface_index.find (
          entry.memory_id);
      if (surface_it == p_ctx.retrieval_surface_index.end ()
          || surface_it->second >= p_ctx.retrieval_surface_cache.size ())
        {
          return std::nullopt;
        }
      const auto &surface = p_ctx.retrieval_surface_cache[surface_it->second];
      if (surface.kind != "LONG_TERM" || !surface.vector_seed_eligible
          || surface.start_ts >= exclusion_ts
          || superseded_memory_ids.count (entry.memory_id) != 0)
        {
          continue;
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
    return state->current_entries[a.index].memory_id
           < state->current_entries[b.index].memory_id;
  };
  std::sort (ranked.begin (), ranked.end (), by_distance);
  CollapseRankedFamilies (
      ranked, duplicate_threshold, candidate_limit,
      [&state] (std::size_t candidate_index) -> const auto & {
        return state->current_entries[candidate_index];
      });

  std::vector<std::map<std::string, std::any>> rows;
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
      rows.push_back (
          { { "memory_id", entry.memory_id },
            { "embedding_id", entry.embedding_id },
            { "start_ts", surface.start_ts },
            { "embedding", ToFloatVector (entry.embedding) } });
    }
  return rows;
}

bool
RebuildSurfaceSearchCache (Transaction &tx, ProcessorContext &p_ctx,
                           int embedding_dim)
{
  try
    {
      auto embedding_rows = tx.Execute (
          "SELECT e.embedding_id, e.embedding, "
          "       COALESCE(m.memory_id, 0) AS memory_id, "
          "       COALESCE(m.start_ts, 0) AS start_ts, "
          "       COALESCE(m.kind, '') AS kind, "
          "       COALESCE(m.source_id, '') AS source_id "
          "FROM embeddings e "
          "LEFT JOIN memories m ON m.embedding_id = e.embedding_id "
          "  AND m.kind = 'LONG_TERM' "
          "ORDER BY e.embedding_id, COALESCE(m.start_ts, 0), m.memory_id");
      std::vector<historical_surface_search_cache_internal::Entry>
          historical_entries;
      historical_entries.reserve (embedding_rows.size ());
      for (const auto &row : embedding_rows)
        {
          const auto embedding_it = row.find ("embedding");
          Eigen::VectorXf embedding;
          if (embedding_it == row.end ()
              || !AnyToEmbedding (embedding_it->second, embedding_dim,
                                  embedding))
            {
              return false;
            }
          const auto string_value = [&row] (const std::string &key) {
            const auto it = row.find (key);
            return it != row.end () && it->second.type () == typeid (std::string)
                       ? std::any_cast<std::string> (it->second)
                       : std::string ();
          };
          historical_entries.push_back (
              { AnyLongLong (row, "embedding_id"),
                AnyLongLong (row, "memory_id"),
                AnyLongLong (row, "start_ts"), string_value ("kind"),
                string_value ("source_id"), std::move (embedding) });
        }

      auto current_rows = tx.Execute (
          "SELECT memory_id, embedding_id, embedding "
          "FROM current_memory_embeddings ORDER BY memory_id");
      std::vector<historical_surface_search_cache_internal::Entry>
          current_entries;
      current_entries.reserve (current_rows.size ());
      for (const auto &row : current_rows)
        {
          const auto embedding_it = row.find ("embedding");
          Eigen::VectorXf embedding;
          if (embedding_it == row.end ()
              || !AnyToEmbedding (embedding_it->second, embedding_dim,
                                  embedding))
            {
              return false;
            }
          current_entries.push_back (
              { AnyLongLong (row, "embedding_id"),
                AnyLongLong (row, "memory_id"), 0, std::string (),
                std::string (), std::move (embedding) });
        }
      return historical_surface_search_cache_internal::Reset (
          p_ctx, std::move (historical_entries), std::move (current_entries));
    }
  catch (const std::exception &e)
    {
      telemetry::LogDebug (
          "cortext.graph_retrieval.cache_rebuild_unavailable",
          { telemetry::Attribute::String ("error", e.what ()) });
      return false;
    }
}

std::vector<std::map<std::string, std::any>>
LoadEligibleSeedRowsFromSqlFallback (
    Transaction &tx, const Eigen::VectorXf &query_embedding,
    long long exclusion_ts, int candidate_limit, int materialization_limit,
    const std::unordered_set<long long> &superseded_memory_ids,
    double duplicate_threshold)
{
  std::vector<std::map<std::string, std::any>> selected;
  selected.reserve (static_cast<std::size_t> (std::max (0, candidate_limit)));
  if (candidate_limit <= 0 || materialization_limit <= 0
      || query_embedding.size () <= 0)
    {
      return selected;
    }

  std::vector<long long> ordered_superseded_ids (
      superseded_memory_ids.begin (), superseded_memory_ids.end ());
  std::sort (ordered_superseded_ids.begin (), ordered_superseded_ids.end ());
  std::string superseded_membership = ",";
  for (const long long memory_id : ordered_superseded_ids)
    {
      superseded_membership += std::to_string (memory_id);
      superseded_membership += ',';
    }

  std::unordered_set<long long> seen_memory_ids;
  FamilyRepresentativeIndex families (duplicate_threshold);
  std::size_t row_offset = 0;
#ifdef CORTEXT_TESTING
  std::size_t materialized_row_count = 0;
#endif
  while (static_cast<int> (selected.size ()) < candidate_limit)
    {
      std::string query
          = "WITH current_eligible AS ("
      "  SELECT m.memory_id, cme.embedding_id, m.start_ts, "
      "         cme.embedding, "
      "         vec_distance_l2(cme.embedding, ?) AS distance "
      "  FROM current_memory_embeddings cme "
      "  JOIN memories m ON m.memory_id = cme.memory_id "
      "  WHERE m.kind = 'LONG_TERM' "
      "    AND COALESCE(m.start_ts, 0) < ? "
      "    AND instr(?, ',' || CAST(m.memory_id AS TEXT) || ',') = 0"
      "), historical_eligible AS ("
      "  SELECT m.memory_id, m.embedding_id, m.start_ts, "
      "         e.embedding, "
      "         vec_distance_l2(e.embedding, ?) AS distance "
      "  FROM memories m "
      "  JOIN embeddings e ON e.embedding_id = m.embedding_id "
      "  WHERE m.kind = 'LONG_TERM' "
      "    AND COALESCE(m.start_ts, 0) < ? "
      "    AND instr(?, ',' || CAST(m.memory_id AS TEXT) || ',') = 0 "
      "    AND NOT EXISTS ("
      "      SELECT 1 FROM current_memory_embeddings cme2 "
      "      WHERE cme2.memory_id = m.memory_id"
      "    )"
      "), historical_ranked AS ("
      "  SELECT memory_id, embedding_id, start_ts, embedding, distance, "
      "         ROW_NUMBER() OVER ("
      "           PARTITION BY embedding "
      "           ORDER BY start_ts ASC, memory_id ASC"
      "         ) AS reference_rank "
      "  FROM historical_eligible"
      "), eligible AS ("
      "  SELECT memory_id, embedding_id, start_ts, embedding, distance "
      "  FROM current_eligible "
      "  UNION ALL "
      "  SELECT memory_id, embedding_id, start_ts, embedding, distance "
      "  FROM historical_ranked WHERE reference_rank = 1"
      ") "
      "SELECT memory_id, embedding_id, start_ts, embedding "
      "FROM eligible ";
      std::vector<std::any> params = {
        ToFloatVector (query_embedding), exclusion_ts, superseded_membership,
        ToFloatVector (query_embedding), exclusion_ts, superseded_membership
      };
      query += "ORDER BY distance ASC, memory_id ASC, embedding_id ASC "
               "LIMIT ?";
      params.push_back (static_cast<long long> (materialization_limit));
      if (row_offset > 0)
        {
          query += " OFFSET ?";
          params.push_back (static_cast<long long> (row_offset));
        }
      auto rows = tx.Execute (query, params);
#ifdef CORTEXT_TESTING
      retrieval_trace::IncrementLastSqlFallbackQueryCount ();
      materialized_row_count += rows.size ();
#endif
      if (rows.empty ())
        {
          break;
        }
      row_offset += rows.size ();

      for (auto &row : rows)
        {
          const long long memory_id = AnyLongLong (row, "memory_id");
          const auto embedding_it = row.find ("embedding");
          Eigen::VectorXf embedding;
          if (memory_id <= 0 || superseded_memory_ids.count (memory_id) != 0
              || !seen_memory_ids.insert (memory_id).second
              || embedding_it == row.end ()
              || !AnyToEmbedding (embedding_it->second,
                                  static_cast<int> (query_embedding.size ()),
                                  embedding))
            {
              continue;
            }
          auto features = BuildFamilyEmbeddingFeatures (embedding);
          if (families.IsDuplicate (features))
            {
              continue;
            }
          families.Add (std::move (features));
          selected.push_back (std::move (row));
          if (static_cast<int> (selected.size ()) >= candidate_limit)
            {
              break;
            }
        }
      if (rows.size () < static_cast<std::size_t> (materialization_limit))
        {
          break;
        }
    }
#ifdef CORTEXT_TESTING
  retrieval_trace::SetLastSqlFallbackMaterializedRowCount (
      materialized_row_count);
#endif
  return selected;
}

double
TemporalScore (std::uint64_t now_ts, long long start_ts, double F, double S,
               double T)
{
  if (start_ts <= 0 || now_ts <= static_cast<std::uint64_t> (start_ts))
    {
      return 0.0;
    }
  const auto age_ms = static_cast<std::uint64_t> (now_ts)
                      - static_cast<std::uint64_t> (start_ts);
  const double f = core::RetrievalFocusBias (F);
  const double s = core::RetrievalSensitivityBias (S);
  const double t = core::RetrievalStabilityBias (T);
  const double tau_seconds
      = core::Clamp (core::Lerp (2.0, 14.0, t)
                         * core::Lerp (1.25, 0.75, f)
                         * core::Lerp (1.15, 0.80, s)
                         * kTemporalRankDaySeconds,
                     kTemporalRankDaySeconds, 21.0 * kTemporalRankDaySeconds);
  const double age_seconds = static_cast<double> (age_ms) / 1000.0;
  return core::Clamp (std::exp (-age_seconds / tau_seconds), 0.0, 1.0);
}

void
InsertOrBoost (std::unordered_map<long long, Candidate> &candidates,
               Candidate candidate)
{
  auto it = candidates.find (candidate.memory_id);
  if (it == candidates.end ())
    {
      candidates.emplace (candidate.memory_id, std::move (candidate));
      return;
    }
  else
    {
      it->second.seed_score = std::max (it->second.seed_score,
                                        candidate.seed_score);
      it->second.graph_score = std::max (it->second.graph_score,
                                         candidate.graph_score);
      it->second.temporal_score = std::max (it->second.temporal_score,
                                            candidate.temporal_score);
      it->second.superseded_penalty = std::max (
          it->second.superseded_penalty, candidate.superseded_penalty);
      it->second.score = std::max (it->second.score, candidate.score);
      it->second.direct_seed = it->second.direct_seed || candidate.direct_seed;
      if (it->second.start_ts <= 0)
        {
          it->second.start_ts = candidate.start_ts;
        }
      if (it->second.embedding.size () == 0)
        {
          it->second.embedding = std::move (candidate.embedding);
          it->second.embedding_id = candidate.embedding_id;
        }
    }
}

double
SemanticFirstScore (double semantic_score, double temporal_score)
{
  constexpr double kTieBreakScale = 1e-6;
  const double semantic = core::Clamp (semantic_score, 0.0, 1.0);
  return semantic
         + (1.0 - semantic) * kTieBreakScale
               * core::Clamp (temporal_score, 0.0, 1.0);
}

bool
RetrievalEdgeType (const std::string &edge_type)
{
  return edge_type == "co_occurs" || edge_type == "similar_to"
         || edge_type == "reinforces" || edge_type == "causes"
         || edge_type == "derived_from" || edge_type == "next_in_episode"
         || edge_type == "prev_in_episode"
         || edge_type == "within_same_event";
}

std::unordered_set<long long>
SupersededMemoryIds (
    const ProcessorContext::AssociationFanoutCache *fanout_cache,
    const ProcessorContext &p_ctx, std::uint64_t exclusion_ts)
{
  std::unordered_set<long long> superseded;
  if (!fanout_cache)
    {
      return superseded;
    }
  superseded.reserve (fanout_cache->in_by_target.size ());
  for (const auto &[target_memory_id, edges] : fanout_cache->in_by_target)
    {
      if (std::any_of (edges.begin (), edges.end (), [] (const auto &edge) {
            return edge.edge_type == "supersedes";
          }))
        {
          const bool has_eligible_replacement = std::any_of (
              edges.begin (), edges.end (), [&] (const auto &edge) {
                if (edge.edge_type != "supersedes")
                  {
                    return false;
                  }
                const auto surface_it = p_ctx.retrieval_surface_index.find (
                    edge.memory_id);
                if (surface_it == p_ctx.retrieval_surface_index.end ()
                    || surface_it->second
                           >= p_ctx.retrieval_surface_cache.size ())
                  {
                    return false;
                  }
                const auto &replacement
                    = p_ctx.retrieval_surface_cache[surface_it->second];
                return replacement.memory_id > 0
                       && replacement.embedding_id > 0
                       && replacement.start_ts >= 0
                       && static_cast<std::uint64_t> (replacement.start_ts)
                              < exclusion_ts
                       && (replacement.kind == "LONG_TERM"
                           || replacement.kind == "ASSOCIATION");
              });
          if (has_eligible_replacement)
            {
              superseded.insert (target_memory_id);
            }
        }
    }
  return superseded;
}

std::vector<Candidate>
SelectFamilyRepresentatives (std::vector<Candidate> ranked,
                             double duplicate_threshold, int limit)
{
  std::sort (ranked.begin (), ranked.end (), [] (const Candidate &a,
                                                  const Candidate &b) {
    if (a.score != b.score)
      {
        return a.score > b.score;
      }
    return a.memory_id > b.memory_id;
  });
  std::vector<Candidate> selected;
  selected.reserve (std::min<std::size_t> (
      ranked.size (), static_cast<std::size_t> (std::max (0, limit))));
  FamilyRepresentativeIndex families (duplicate_threshold);
  for (auto &candidate : ranked)
    {
      if (static_cast<int> (selected.size ()) >= limit)
        {
          break;
        }
      auto features = BuildFamilyEmbeddingFeatures (candidate.embedding);
      if (families.IsDuplicate (features))
        {
          continue;
        }
      families.Add (std::move (features));
      selected.push_back (std::move (candidate));
    }
  return selected;
}

void
SortCandidates (std::vector<Candidate> &ranked)
{
  std::sort (ranked.begin (), ranked.end (), [] (const Candidate &a,
                                                 const Candidate &b) {
    if (a.score != b.score)
      {
        return a.score > b.score;
      }
    return a.memory_id > b.memory_id;
  });
}

} // namespace

void
GraphAugmentedRetrieveCandidates::Execute (OperationContext &context,
                                           Transaction &tx) const
{
  auto started = std::chrono::steady_clock::now ();
  retrieval_trace::ClearLastSelectedEmbeddingOrder ();
  retrieval_trace::ClearLastRankedCandidates ();
  retrieval_trace::ClearLastRejectedCandidates ();
  retrieval_trace::ClearLastEvidencePackets ();
  retrieval_trace::ClearLastRetrievalSummary ();
#ifdef CORTEXT_TESTING
  retrieval_trace::ClearLastFamilyExactComparisonCount ();
  retrieval_trace::ClearLastSqlFallbackQueryCount ();
  retrieval_trace::ClearLastSqlFallbackMaterializedRowCount ();
#endif

  if (!context.GetShouldCheckRetrieval ())
    {
      context.SetRetrievedMemoryEmbeddings ({});
      return;
    }

  const auto &signal = context.GetSignal ();
  const auto &cfg = context.GetConfig ();
  const double F = cfg.focus;
  const double S = cfg.sensitivity;
  const double T = cfg.stability;
  const int seed_limit = std::max (1, core::RetrievalMaxResults (F));
  const int output_limit = std::max (
      1, core::RetrievalGraphExpandedRagMaxItems (F, T));
  const int seed_search_limit
      = core::RetrievalSeedSearchK (F, S, T, seed_limit);
  const int fallback_materialization_limit
      = core::RetrievalSeedSearchK (F, S, T, seed_search_limit);
  const int fanout = std::max (
      0, core::RetrievalGraphExpansionFanout (F, S, T));
  const std::uint64_t exclusion_ts
      = context.GetWriteExclusionTs ().value_or (signal.timestamp);
  const double graph_weight
      = core::RetrievalGraphExpandedRagGraphWeight (F, S, T);
  const double duplicate_threshold
      = core::SupersessionDuplicateThreshold (F, S, T);
  const int embedding_dim = static_cast<int> (signal.embedding.size ());
  const bool constructive_recall_enabled
      = !constructive_recall::Disabled ();
  const bool profile_graph_retrieval = internal::experimental_env::Flag (
      "CORTEXT_PROFILE_GRAPH_RETRIEVAL");
  auto profile_timing =
      [&context, profile_graph_retrieval] (
          std::string_view name,
          std::chrono::steady_clock::time_point section_start) {
    if (!profile_graph_retrieval)
      {
        return;
      }
    context.AddOperationTiming (
        name,
        std::chrono::duration<double, std::milli> (
            std::chrono::steady_clock::now () - section_start)
            .count ());
  };
  std::chrono::steady_clock::time_point section_start;
  auto start_profile_timing = [&section_start, profile_graph_retrieval] {
    if (profile_graph_retrieval)
      {
        section_start = std::chrono::steady_clock::now ();
      }
  };
  auto &p_ctx = context.GetProcessorContext ();

  start_profile_timing ();
  auto *fanout_cache
      = association_fanout_cache::Ensure (context.GetStore (), p_ctx);
  const auto superseded_memory_ids = SupersededMemoryIds (
      fanout_cache, p_ctx, exclusion_ts);
  profile_timing ("GraphRetrieve.family_index", section_start);

  std::vector<std::map<std::string, std::any>> rows;
  std::unordered_set<long long> seen_seed_memory_ids;
  bool cache_search_available = false;
  start_profile_timing ();
  if (auto current_knn_rows = LoadCurrentSeedRowsFromCache (
          p_ctx, signal.embedding, static_cast<long long> (exclusion_ts),
          seed_search_limit, superseded_memory_ids, duplicate_threshold))
    {
      cache_search_available = true;
      AppendUniqueRows (rows, std::move (*current_knn_rows),
                        seen_seed_memory_ids);
    }
  if (auto historical_knn_rows = LoadHistoricalSeedRowsFromCache (
          p_ctx, signal.embedding, static_cast<long long> (exclusion_ts),
          seed_search_limit, superseded_memory_ids, duplicate_threshold))
    {
      cache_search_available = true;
      AppendUniqueRows (rows, std::move (*historical_knn_rows),
                        seen_seed_memory_ids);
    }
  profile_timing ("GraphRetrieve.seed_knn_cache", section_start);

  if (!cache_search_available
      && signal.retention != Retention::Ephemeral
      && !historical_surface_search_cache_internal::RecoveryFailed (p_ctx))
    {
      start_profile_timing ();
      if (RebuildSurfaceSearchCache (tx, p_ctx, embedding_dim))
        {
          if (auto current_knn_rows = LoadCurrentSeedRowsFromCache (
                  p_ctx, signal.embedding,
                  static_cast<long long> (exclusion_ts), seed_search_limit,
                  superseded_memory_ids, duplicate_threshold))
            {
              cache_search_available = true;
              AppendUniqueRows (rows, std::move (*current_knn_rows),
                                seen_seed_memory_ids);
            }
          if (auto historical_knn_rows = LoadHistoricalSeedRowsFromCache (
                  p_ctx, signal.embedding,
                  static_cast<long long> (exclusion_ts), seed_search_limit,
                  superseded_memory_ids, duplicate_threshold))
            {
              cache_search_available = true;
              AppendUniqueRows (rows, std::move (*historical_knn_rows),
                                seen_seed_memory_ids);
            }
        }
      else
        {
          historical_surface_search_cache_internal::MarkRecoveryFailed (
              p_ctx);
        }
      profile_timing ("GraphRetrieve.seed_knn_cache_rebuild", section_start);
    }

  if (!cache_search_available)
    {
      start_profile_timing ();
      try
        {
          auto fallback_rows = LoadEligibleSeedRowsFromSqlFallback (
              tx, signal.embedding, static_cast<long long> (exclusion_ts),
              seed_limit, fallback_materialization_limit,
              superseded_memory_ids, duplicate_threshold);
          AppendUniqueRows (rows, std::move (fallback_rows),
                            seen_seed_memory_ids);
        }
      catch (const std::exception &e)
        {
          telemetry::LogDebug (
              "cortext.graph_retrieval.eligible_knn_unavailable",
              { telemetry::Attribute::String ("error", e.what ()) });
        }
      profile_timing ("GraphRetrieve.seed_knn_sql", section_start);
    }

  start_profile_timing ();
  std::vector<Candidate> seeded;
  seeded.reserve (rows.size ());
  for (const auto &row : rows)
    {
      Candidate candidate;
      candidate.memory_id = AnyLongLong (row, "memory_id");
      candidate.embedding_id = AnyLongLong (row, "embedding_id");
      candidate.start_ts = AnyLongLong (row, "start_ts");
      auto it_embedding = row.find ("embedding");
      if (candidate.memory_id <= 0 || candidate.embedding_id <= 0
          || superseded_memory_ids.count (candidate.memory_id) != 0
          || it_embedding == row.end ()
          || !AnyToEmbedding (it_embedding->second, embedding_dim,
                              candidate.embedding))
        {
          continue;
        }
      if (constructive_recall_enabled)
        {
          if (!RefreshCandidateFromProcessorSurface (
                  p_ctx, candidate, embedding_dim))
            {
              (void)RefreshCandidateToCurrentSurface (
                  tx, candidate, embedding_dim);
            }
        }
      candidate.seed_score = core::Map01 (
          Cosine (signal.embedding, candidate.embedding));
      candidate.temporal_score = TemporalScore (
          signal.timestamp, candidate.start_ts, F, S, T);
      candidate.score = SemanticFirstScore (
          candidate.seed_score, candidate.temporal_score);
      candidate.direct_seed = true;
      seeded.push_back (std::move (candidate));
    }
  seeded = SelectFamilyRepresentatives (
      std::move (seeded), duplicate_threshold, seed_limit);
  profile_timing ("GraphRetrieve.seed_rank", section_start);

  std::unordered_map<long long, Candidate> candidates;
  candidates.reserve (seeded.size ()
                      + static_cast<std::size_t> (std::max (0, fanout)));
  for (const auto &candidate : seeded)
    {
      InsertOrBoost (candidates, candidate);
    }

  if (fanout_cache && fanout > 0 && !seeded.empty ())
    {
      start_profile_timing ();
      auto reinforces_degree = [fanout_cache] (long long memory_id) {
        std::size_t result = 0;
        if (auto it = fanout_cache->out_by_source.find (memory_id);
            it != fanout_cache->out_by_source.end ())
          {
            result += static_cast<std::size_t> (std::count_if (
                it->second.begin (), it->second.end (), [] (const auto &edge) {
                  return edge.edge_type == "reinforces";
                }));
          }
        if (auto it = fanout_cache->in_by_target.find (memory_id);
            it != fanout_cache->in_by_target.end ())
          {
            result += static_cast<std::size_t> (std::count_if (
                it->second.begin (), it->second.end (), [] (const auto &edge) {
                  return edge.edge_type == "reinforces";
                }));
          }
        return std::max<std::size_t> (1, result);
      };
      auto expand_edge = [&] (
                             const Candidate &seed,
                             const ProcessorContext::AssociationFanoutEdge &edge,
                             int &expanded) {
        if (expanded >= fanout || !RetrievalEdgeType (edge.edge_type)
            || edge.memory_id <= 0 || edge.memory_id == seed.memory_id
            || superseded_memory_ids.count (edge.memory_id) != 0)
          {
            return;
          }
        const auto surface_it = p_ctx.retrieval_surface_index.find (
            edge.memory_id);
        if (surface_it == p_ctx.retrieval_surface_index.end ()
            || surface_it->second >= p_ctx.retrieval_surface_cache.size ())
          {
            return;
          }
        const auto &surface
            = p_ctx.retrieval_surface_cache[surface_it->second];
        if (surface.memory_id <= 0 || surface.embedding_id <= 0
            || surface.start_ts >= static_cast<long long> (exclusion_ts)
            || (surface.kind != "LONG_TERM" && surface.kind != "ASSOCIATION")
            || surface.embedding.size () != embedding_dim)
          {
            return;
          }
        Candidate candidate;
        candidate.memory_id = surface.memory_id;
        candidate.embedding_id = surface.embedding_id;
        candidate.start_ts = surface.start_ts;
        candidate.embedding = surface.embedding;
        candidate.seed_score = core::Map01 (
            Cosine (signal.embedding, candidate.embedding));
        candidate.temporal_score = TemporalScore (
            signal.timestamp, candidate.start_ts, F, S, T);
        candidate.graph_score = core::Clamp (edge.weight, 0.0, 1.0);
        if (edge.edge_type == "reinforces")
          {
            const double normalization = std::sqrt (
                static_cast<double> (reinforces_degree (seed.memory_id))
                * static_cast<double> (
                    reinforces_degree (candidate.memory_id)));
            candidate.graph_score /= std::max (1.0, normalization);
          }
        const double semantic_score = SemanticFirstScore (
            candidate.seed_score, candidate.temporal_score);
        const double graph_bonus = std::min (
            0.08, graph_weight * candidate.graph_score);
        candidate.score = semantic_score
                          + (1.0 - semantic_score) * graph_bonus;
        InsertOrBoost (candidates, std::move (candidate));
        ++expanded;
      };

      for (const auto &seed : seeded)
        {
          int expanded = 0;
          if (auto it = fanout_cache->out_by_source.find (seed.memory_id);
              it != fanout_cache->out_by_source.end ())
            {
              for (const auto &edge : it->second)
                {
                  expand_edge (seed, edge, expanded);
                }
            }
          if (auto it = fanout_cache->in_by_target.find (seed.memory_id);
              it != fanout_cache->in_by_target.end ())
            {
              for (const auto &edge : it->second)
                {
                  expand_edge (seed, edge, expanded);
                }
            }
        }
      profile_timing ("GraphRetrieve.association_expansion", section_start);
    }

  start_profile_timing ();
  std::vector<Candidate> direct_candidates;
  std::vector<Candidate> expanded_candidates;
  direct_candidates.reserve (seeded.size ());
  expanded_candidates.reserve (candidates.size ());
  for (auto &[memory_id, candidate] : candidates)
    {
      (void)memory_id;
      if (candidate.direct_seed)
        {
          direct_candidates.push_back (std::move (candidate));
        }
      else
        {
          expanded_candidates.push_back (std::move (candidate));
        }
    }
  // Direct candidates came from the already family-collapsed seed set. Graph
  // boosts can update their scores but cannot introduce another direct family.
  SortCandidates (direct_candidates);
  expanded_candidates = SelectFamilyRepresentatives (
      std::move (expanded_candidates), duplicate_threshold,
      static_cast<int> (candidates.size ()));

  std::vector<Candidate> ranked_all;
  ranked_all.reserve (direct_candidates.size ()
                      + expanded_candidates.size ());
  ranked_all.insert (ranked_all.end (), direct_candidates.begin (),
                     direct_candidates.end ());
  ranked_all.insert (ranked_all.end (), expanded_candidates.begin (),
                     expanded_candidates.end ());
  if (!expanded_candidates.empty ())
    {
      SortCandidates (ranked_all);
    }

  std::vector<Candidate> ranked;
  ranked.reserve (std::min<std::size_t> (
      ranked_all.size (), static_cast<std::size_t> (output_limit)));
  std::unordered_set<long long> selected_memory_ids;
  selected_memory_ids.reserve (static_cast<std::size_t> (output_limit));
  if (expanded_candidates.empty ())
    {
      const auto direct_count = std::min<std::size_t> (
          direct_candidates.size (), static_cast<std::size_t> (output_limit));
      ranked.insert (ranked.end (), direct_candidates.begin (),
                     direct_candidates.begin ()
                         + static_cast<std::ptrdiff_t> (direct_count));
      for (const auto &candidate : ranked)
        {
          selected_memory_ids.insert (candidate.memory_id);
        }
    }
  else
    {
      const int direct_anchor_limit = std::max (
          1, output_limit - std::max (1, output_limit / 4));
      FamilyRepresentativeIndex selected_families (duplicate_threshold);
      std::size_t direct_index = 0;
      for (; direct_index < direct_candidates.size ()
             && static_cast<int> (ranked.size ()) < direct_anchor_limit;
           ++direct_index)
        {
          const auto &candidate = direct_candidates[direct_index];
          selected_memory_ids.insert (candidate.memory_id);
          selected_families.Add (BuildFamilyEmbeddingFeatures (
              candidate.embedding));
          ranked.push_back (candidate);
        }

      std::vector<Candidate> remaining;
      remaining.reserve (direct_candidates.size () - direct_index
                         + expanded_candidates.size ());
      remaining.insert (remaining.end (),
                        direct_candidates.begin ()
                            + static_cast<std::ptrdiff_t> (direct_index),
                        direct_candidates.end ());
      remaining.insert (remaining.end (), expanded_candidates.begin (),
                        expanded_candidates.end ());
      SortCandidates (remaining);
      for (const auto &candidate : remaining)
        {
          if (static_cast<int> (ranked.size ()) >= output_limit)
            {
              break;
            }
          auto features = BuildFamilyEmbeddingFeatures (candidate.embedding);
          if (selected_families.IsDuplicate (features))
            {
              continue;
            }
          if (selected_memory_ids.insert (candidate.memory_id).second)
            {
              selected_families.Add (std::move (features));
              ranked.push_back (candidate);
            }
        }
    }

  std::vector<retrieval_trace::RejectedCandidate> rejected_trace;
  rejected_trace.reserve (std::min<std::size_t> (64, ranked_all.size ()));
  const auto cutoff_it = std::min_element (
      ranked.begin (), ranked.end (), [] (const Candidate &a,
                                          const Candidate &b) {
        return a.score < b.score;
      });
  const double cutoff_score
      = cutoff_it == ranked.end () ? 0.0 : cutoff_it->score;
  for (const auto &candidate : ranked_all)
    {
      if (rejected_trace.size () >= 64)
        {
          break;
        }
      if (selected_memory_ids.count (candidate.memory_id) != 0)
        {
          continue;
        }
      retrieval_trace::RankedCandidate trace;
      trace.embedding_id = candidate.embedding_id;
      trace.memory_id = candidate.memory_id;
      trace.score = candidate.score;
      trace.relevance = candidate.seed_score;
      trace.temporal_score = candidate.temporal_score;
      trace.activation.base_level = candidate.seed_score;
      trace.activation.spreading_activation = candidate.graph_score;
      trace.activation.partial_match_penalty = candidate.superseded_penalty;
      trace.activation.activation_total = candidate.score;
      retrieval_trace::RejectedCandidate rejected;
      rejected.candidate = trace;
      rejected.reason = "below_output_limit";
      rejected.stage = "selection";
      rejected.observed = candidate.score;
      rejected.threshold = cutoff_score;
      rejected_trace.push_back (std::move (rejected));
    }
  profile_timing ("GraphRetrieve.rank_candidates", section_start);

  std::unordered_map<long long, Eigen::VectorXf> out;
  out.reserve (ranked.size ());
  std::vector<long long> selected_order;
  selected_order.reserve (ranked.size ());
  std::vector<OperationContext::RetrievedMemoryCandidate> retrieved_candidates;
  retrieved_candidates.reserve (ranked.size ());
  std::vector<retrieval_trace::RankedCandidate> trace_ranked;
  trace_ranked.reserve (ranked.size ());
  {
    const auto reconstruction_start = std::chrono::steady_clock::now ();
    if (constructive_recall_enabled
        && signal.retention != Retention::Ephemeral
        && signal.embedding.size () > 0)
      {
        const constructive_recall::ReconstructionUpdatePolicy policy {
          core::ReconstructionHistoryLimit (F, S, T),
          core::ReconstructionPruneBatchLimit (F, S, T),
          core::ReconstructionMinUpdateIntervalMs (F, S, T)
        };
        const auto reconstruction_policy
            = core::RetrievalConstructiveReconstructionPolicy (F, S, T);
        const int update_count
            = core::RetrievalReconstructionUpdateCount (F, S, T);
        int updates = 0;
        for (auto &candidate : ranked)
          {
            if (updates >= update_count)
              {
                break;
              }
            if (candidate.memory_id <= 0 || candidate.embedding.size () == 0
                || candidate.embedding.size () != signal.embedding.size ())
              {
                continue;
              }
            if (constructive_recall::ShouldSkipReconstructionUpdate (
                    tx, candidate.memory_id,
                    static_cast<long long> (signal.timestamp), "retrieval",
                    policy))
              {
                ++updates;
                continue;
              }

            const double source_confidence
                = core::Clamp (candidate.seed_score, 0.0, 1.0);
            const double candidate_score
                = core::Clamp (candidate.score, 0.0, 1.0);
            const double context_score
                = core::Clamp (candidate.seed_score, 0.0, 1.0);
            const double uncertainty_raw =
                reconstruction_policy.source_confidence_weight
                    * (1.0 - source_confidence)
                    + reconstruction_policy.candidate_score_weight
                          * (1.0 - candidate_score)
                    + reconstruction_policy.context_score_weight
                          * (1.0 - context_score);
            const double uncertainty = core::Clamp (uncertainty_raw, 0.0, 1.0);
            const double blend_raw =
                reconstruction_policy.min_blend
                + (reconstruction_policy.max_blend
                   - reconstruction_policy.min_blend) * uncertainty;
            const double blend = core::Clamp (
                blend_raw, reconstruction_policy.min_blend,
                reconstruction_policy.max_blend);

            Eigen::VectorXf reconstructed
                = static_cast<float> (1.0 - blend) * candidate.embedding
                  + static_cast<float> (
                        blend * reconstruction_policy.query_weight)
                        * signal.embedding;
            const float norm = reconstructed.norm ();
            if (norm <= 1e-9f)
              {
                continue;
              }
            reconstructed /= norm;

            const auto blob_id
                = constructive_recall::LoadCurrentBlobId (tx,
                                                          candidate.memory_id);
            constructive_recall::ReconstructionAppendTimings timings;
            constructive_recall::AppendReconstructionWithEmbeddingUnchecked (
                tx, candidate.memory_id, reconstructed, blob_id,
                static_cast<long long> (signal.timestamp), uncertainty,
                "retrieval", source_confidence, context_score, policy,
                &timings, &context.GetProcessorContext ());
            const long long current_embedding_id
                = LoadCurrentSurfaceEmbeddingId (tx, candidate.memory_id,
                                                 candidate.embedding_id);
            if (auto current = constructive_recall::LoadCurrentEmbedding (
                    tx, candidate.memory_id, candidate.embedding_id,
                    static_cast<int> (reconstructed.size ())))
              {
                auto &p_ctx = context.GetProcessorContext ();
                RefreshProcessorSurfaces (tx, p_ctx, candidate.memory_id,
                                          candidate.embedding_id,
                                          current_embedding_id, *current);
                candidate.embedding_id = current_embedding_id;
                candidate.embedding = *current;
                candidate.seed_score = core::Map01 (
                    Cosine (signal.embedding, candidate.embedding));
              }
            context.AddOperationTiming (
                "GraphRetrieve.reconstruction_append_insert_embedding",
                timings.insert_embedding_ms);
            context.AddOperationTiming (
                "GraphRetrieve.reconstruction_append_insert_record",
                timings.insert_reconstruction_ms);
            context.AddOperationTiming (
                "GraphRetrieve.reconstruction_append_current_surface",
                timings.current_surface_ms);
            context.AddOperationTiming (
                "GraphRetrieve.reconstruction_append_prune",
                timings.prune_ms);
            ++updates;
          }
      }
    context.AddOperationTiming (
        "GraphRetrieve.reconstruction_versions",
        std::chrono::duration<double, std::milli> (
            std::chrono::steady_clock::now () - reconstruction_start)
            .count ());
  }
  for (const auto &candidate : ranked)
    {
      out.emplace (candidate.embedding_id, candidate.embedding);
      selected_order.push_back (candidate.embedding_id);
      retrieved_candidates.push_back (
          { candidate.memory_id, candidate.embedding_id, candidate.embedding,
            candidate.score });

      retrieval_trace::RankedCandidate trace;
      trace.embedding_id = candidate.embedding_id;
      trace.memory_id = candidate.memory_id;
      trace.score = candidate.score;
      trace.relevance = candidate.seed_score;
      trace.temporal_score = candidate.temporal_score;
      trace.activation.base_level = candidate.seed_score;
      trace.activation.spreading_activation = candidate.graph_score;
      trace.activation.partial_match_penalty = candidate.superseded_penalty;
      trace.activation.activation_total = candidate.score;
      trace_ranked.push_back (trace);
    }

  retrieval_trace::SetLastSelectedEmbeddingOrder (selected_order);
  retrieval_trace::SetLastRankedCandidates (trace_ranked);
  retrieval_trace::SetLastRejectedCandidates (rejected_trace);
  retrieval_trace::SetLastRetrievalSummary ({});
  context.SetRetrievedMemoryEmbeddings (std::move (out));
  context.SetRetrievedMemoryCandidates (std::move (retrieved_candidates));
  context.AddOperationTiming (
      "GraphRetrieve.total",
      std::chrono::duration<double, std::milli> (
          std::chrono::steady_clock::now () - started)
          .count ());

  telemetry::LogDebug ("cortext.graph_retrieval", {
    telemetry::Attribute::Int64 ("selected_count",
                                 static_cast<int64_t> (ranked.size ())),
    telemetry::Attribute::Int64 ("seed_count",
                                 static_cast<int64_t> (seeded.size ()))
  });
}

} // namespace cortext::operations
