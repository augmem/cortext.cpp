#include "cortext/operations/graph_retrieval.hpp"

#include "constructive_recall_internal.hpp"
#include "retrieval_trace_state.hpp"
#include "cortext/core/knobs.hpp"
#include "cortext/core/utils.hpp"
#include "cortext/processor/operation_context.hpp"
#include "cortext/store/store.hpp"
#include "cortext/store/utils.hpp"
#include "cortext/telemetry/telemetry.hpp"
#include "../experimental_env.hpp"

#include <algorithm>
#include <any>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <exception>
#include <limits>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>
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
  std::string source_id;
  long long start_ts = 0;
  Eigen::VectorXf embedding;
  double seed_score = 0.0;
  double graph_score = 0.0;
  double temporal_score = 0.0;
  double score = 0.0;
};

constexpr long long kStaleSourceNeighborGapMs = 72LL * 60LL * 60LL * 1000LL;
constexpr long long kFreshSourceNeighborToSignalGapMs = 10LL * 60LL * 1000LL;
constexpr double kTemporalRankDaySeconds = 24.0 * 60.0 * 60.0;

struct TurnSourceId
{
  std::string turn_prefix;
  long long turn = 0;
  int width = 0;
  bool has_suffix_separator = false;
};

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

double
AnyDouble (const std::map<std::string, std::any> &row,
           const std::string &key)
{
  auto it = row.find (key);
  if (it == row.end ())
    {
      return 0.0;
    }
  const std::any &value = it->second;
  if (value.type () == typeid (double))
    {
      return std::any_cast<double> (value);
    }
  if (value.type () == typeid (float))
    {
      return static_cast<double> (std::any_cast<float> (value));
    }
  return cortext::store::AnyToLongLong (value).value_or (0);
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

std::optional<TurnSourceId>
ParseTurnSourceId (const std::string &source_id)
{
  constexpr std::string_view marker = "/turn-";
  const std::size_t marker_pos = source_id.find (marker);
  if (marker_pos == std::string::npos)
    {
      return std::nullopt;
    }
  const std::size_t digits_begin = marker_pos + marker.size ();
  std::size_t digits_end = digits_begin;
  while (digits_end < source_id.size ()
         && std::isdigit (
             static_cast<unsigned char> (source_id[digits_end])))
    {
      ++digits_end;
    }
  if (digits_end == digits_begin)
    {
      return std::nullopt;
    }

  if (digits_end < source_id.size () && source_id[digits_end] != '/')
    {
      return std::nullopt;
    }
  long long turn = 0;
  const auto parse_result = std::from_chars (
      source_id.data () + digits_begin,
      source_id.data () + digits_end,
      turn);
  if (parse_result.ec != std::errc ()
      || parse_result.ptr != source_id.data () + digits_end
      || turn > std::numeric_limits<int>::max ())
    {
      return std::nullopt;
    }

  TurnSourceId parsed;
  parsed.turn_prefix = source_id.substr (0, digits_begin);
  parsed.turn = turn;
  parsed.width = static_cast<int> (digits_end - digits_begin);
  parsed.has_suffix_separator = digits_end < source_id.size ();
  return parsed;
}

std::string
EscapeLikeLiteral (const std::string &value)
{
  std::string escaped;
  escaped.reserve (value.size ());
  for (char c : value)
    {
      if (c == '\\' || c == '%' || c == '_')
        {
          escaped.push_back ('\\');
        }
      escaped.push_back (c);
    }
  return escaped;
}

std::string
TurnSourceLikePattern (const TurnSourceId &source, long long turn)
{
  if (turn < 0)
    {
      return {};
    }
  std::string digits = std::to_string (turn);
  if (static_cast<int> (digits.size ()) < source.width)
    {
      digits.insert (digits.begin (),
                     static_cast<std::size_t> (source.width)
                         - digits.size (),
                     '0');
    }
  const std::string prefix = EscapeLikeLiteral (source.turn_prefix + digits);
  return source.has_suffix_separator ? prefix + "/%" : prefix;
}

bool
SourceSeedGraphExpansionDisabled ()
{
  return internal::experimental_env::Flag (
      "CORTEXT_DISABLE_SOURCE_SEED_GRAPH_EXPANSION");
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
      it->second.score = std::max (it->second.score, candidate.score);
      if (it->second.source_id.empty ())
        {
          it->second.source_id = std::move (candidate.source_id);
        }
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

bool
SourceExpansionSurfaceEligible (
    const ProcessorContext::RetrievalSurfaceEntry &entry,
    const Candidate &seed, std::uint64_t exclusion_ts, int embedding_dim)
{
  if (entry.memory_id <= 0 || entry.memory_id == seed.memory_id
      || entry.embedding_id <= 0 || entry.start_ts <= 0
      || entry.source_id != seed.source_id
      || (entry.kind != "LONG_TERM" && entry.kind != "ASSOCIATION")
      || entry.embedding.size () != embedding_dim)
    {
      return false;
    }
  return static_cast<std::uint64_t> (entry.start_ts) < exclusion_ts;
}

std::vector<const ProcessorContext::RetrievalSurfaceEntry *>
SourceExpansionRowsFromSurface (const ProcessorContext &p_ctx,
                                const Candidate &seed,
                                std::uint64_t exclusion_ts,
                                int embedding_dim, int limit)
{
  std::vector<const ProcessorContext::RetrievalSurfaceEntry *> rows;
  if (seed.memory_id <= 0 || seed.source_id.empty () || seed.start_ts <= 0
      || limit <= 0)
    {
      return rows;
    }

  for (const auto &entry : p_ctx.retrieval_surface_cache)
    {
      if (SourceExpansionSurfaceEligible (entry, seed, exclusion_ts,
                                          embedding_dim))
        {
          rows.push_back (&entry);
        }
    }

  auto by_source_neighbor_order = [&seed] (
                                      const auto *a,
                                      const auto *b) {
    const long long a_gap = std::llabs (a->start_ts - seed.start_ts);
    const long long b_gap = std::llabs (b->start_ts - seed.start_ts);
    if (a_gap != b_gap)
      {
        return a_gap < b_gap;
      }
    if (a->start_ts != b->start_ts)
      {
        return a->start_ts < b->start_ts;
      }
    return a->memory_id < b->memory_id;
  };

  if (static_cast<int> (rows.size ()) > limit)
    {
      std::partial_sort (
          rows.begin (), rows.begin () + limit, rows.end (),
          by_source_neighbor_order);
      rows.resize (static_cast<std::size_t> (limit));
    }
  else
    {
      std::sort (rows.begin (), rows.end (), by_source_neighbor_order);
    }
  return rows;
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
  const int fanout = std::max (0, core::RetrievalGraphExpansionFanout (F, S, T));
  const std::uint64_t exclusion_ts
      = context.GetWriteExclusionTs ().value_or (signal.timestamp);
  const double seed_weight
      = core::RetrievalGraphExpandedRagSeedWeight (F, S, T);
  const double graph_weight
      = core::RetrievalGraphExpandedRagGraphWeight (F, S, T);
  const double temporal_weight
      = core::RetrievalGraphExpandedRagTemporalWeight (F, S, T);
  const int embedding_dim = static_cast<int> (signal.embedding.size ());
  const bool constructive_recall_enabled
      = !constructive_recall::Disabled ();
  const bool source_seed_expansion_enabled
      = !SourceSeedGraphExpansionDisabled ();
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
  auto &p_ctx = context.GetProcessorContext ();

  std::vector<std::map<std::string, std::any>> rows;
  std::unordered_set<long long> seen_seed_memory_ids;
  std::vector<Candidate> cached_recent_seeded;
  std::chrono::steady_clock::time_point section_start;
  auto start_profile_timing = [&section_start, profile_graph_retrieval] {
    if (profile_graph_retrieval)
      {
        section_start = std::chrono::steady_clock::now ();
      }
  };
  start_profile_timing ();
  const int recent_seed_limit = std::max (seed_search_limit,
                                          output_limit * 4);
  if (constructive_recall_enabled
      && !p_ctx.retrieval_surface_cache.empty ())
    {
      std::vector<const ProcessorContext::RetrievalSurfaceEntry *> recent;
      recent.reserve (static_cast<std::size_t> (recent_seed_limit));
      for (const auto &entry : p_ctx.retrieval_surface_cache)
        {
          const bool outside_time_window
              = entry.start_ts >= 0
                && static_cast<std::uint64_t> (entry.start_ts)
                       >= exclusion_ts;
          if (entry.memory_id <= 0 || entry.embedding_id <= 0
              || entry.embedding.size () != signal.embedding.size ()
              || (entry.kind != "LONG_TERM" && entry.kind != "ASSOCIATION")
              || outside_time_window)
            {
              continue;
            }
          recent.push_back (&entry);
        }
      std::sort (recent.begin (), recent.end (),
                 [] (const auto *a, const auto *b) {
        return a->memory_id > b->memory_id;
      });
      if (static_cast<int> (recent.size ()) > recent_seed_limit)
        {
          recent.resize (static_cast<std::size_t> (recent_seed_limit));
        }
      cached_recent_seeded.reserve (recent.size ());
      for (const auto *entry : recent)
        {
          if (!seen_seed_memory_ids.insert (entry->memory_id).second)
            {
              continue;
            }
          Candidate candidate;
          candidate.memory_id = entry->memory_id;
          candidate.embedding_id = entry->embedding_id;
          candidate.source_id = entry->source_id;
          candidate.start_ts = entry->start_ts;
          candidate.embedding = entry->embedding;
          candidate.seed_score = core::Map01 (
              Cosine (signal.embedding, candidate.embedding));
          candidate.temporal_score = TemporalScore (
              signal.timestamp, candidate.start_ts, F, S, T);
          candidate.score = seed_weight * candidate.seed_score
                            + temporal_weight * candidate.temporal_score;
          cached_recent_seeded.push_back (std::move (candidate));
        }
    }
  else
    {
      auto recent_rows = tx.Execute (
          "SELECT m.memory_id, m.embedding_id, m.start_ts, e.embedding, "
          "       m.source_id "
          "FROM memories m "
          "JOIN embeddings e ON e.embedding_id = m.embedding_id "
          "WHERE m.embedding_id IS NOT NULL "
          "  AND m.kind IN ('LONG_TERM', 'ASSOCIATION') "
          "  AND COALESCE(m.start_ts, 0) < ? "
          "ORDER BY m.memory_id DESC LIMIT ?",
          { static_cast<long long> (exclusion_ts),
            static_cast<long long> (recent_seed_limit) });
      AppendUniqueRows (rows, std::move (recent_rows),
                        seen_seed_memory_ids);
    }
  profile_timing ("GraphRetrieve.seed_recent", section_start);

    {
      bool historical_knn_loaded = false;
      try
        {
          start_profile_timing ();
          auto current_knn_rows = tx.Execute (
              "SELECT m.memory_id, cme.embedding_id, m.start_ts, "
              "       cme.embedding, m.source_id "
              "FROM ("
              "  SELECT memory_id, embedding_id, embedding, distance "
              "  FROM current_memory_embeddings "
              "  WHERE embedding MATCH ? "
              "    AND k = ?"
              ") cme "
              "JOIN memories m ON m.memory_id = cme.memory_id "
              "WHERE m.kind IN ('LONG_TERM', 'ASSOCIATION') "
              "  AND COALESCE(m.start_ts, 0) < ? "
              "ORDER BY distance",
              { ToFloatVector (signal.embedding),
                static_cast<long long> (seed_search_limit),
                static_cast<long long> (exclusion_ts) });
          historical_knn_loaded = historical_knn_loaded
                                  || !current_knn_rows.empty ();
          AppendUniqueRows (rows, std::move (current_knn_rows),
                            seen_seed_memory_ids);
          profile_timing ("GraphRetrieve.seed_current_knn_sql",
                          section_start);
        }
      catch (const std::exception &e)
        {
          telemetry::LogDebug (
              "cortext.graph_retrieval.current_knn_unavailable",
              { telemetry::Attribute::String ("error", e.what ()) });
        }
      try
        {
          start_profile_timing ();
          auto historical_knn_rows = tx.Execute (
              "SELECT m.memory_id, m.embedding_id, m.start_ts, e.embedding, "
              "       m.source_id "
              "FROM embeddings e "
              "JOIN memories m ON m.embedding_id = e.embedding_id "
              "WHERE e.embedding MATCH ? "
              "  AND k = ? "
              "  AND m.embedding_id IS NOT NULL "
              "  AND m.kind IN ('LONG_TERM', 'ASSOCIATION') "
              "  AND COALESCE(m.start_ts, 0) < ? "
              "ORDER BY distance",
              { ToFloatVector (signal.embedding),
                static_cast<long long> (seed_search_limit),
                static_cast<long long> (exclusion_ts) });
          historical_knn_loaded = historical_knn_loaded
                                  || !historical_knn_rows.empty ();
          AppendUniqueRows (rows, std::move (historical_knn_rows),
                            seen_seed_memory_ids);
          profile_timing ("GraphRetrieve.seed_historical_knn_sql",
                          section_start);
        }
      catch (const std::exception &e)
        {
          telemetry::LogDebug (
              "cortext.graph_retrieval.knn_unavailable",
              { telemetry::Attribute::String ("error", e.what ()) });
        }

      start_profile_timing ();
      const auto seed_bounds = tx.Execute (
          "SELECT COALESCE(MIN(m.memory_id), 0) AS min_memory_id, "
          "       COALESCE(MAX(m.memory_id), 0) AS max_memory_id "
          "FROM memories m "
          "WHERE m.embedding_id IS NOT NULL "
          "  AND m.kind IN ('LONG_TERM', 'ASSOCIATION') "
          "  AND COALESCE(m.start_ts, 0) < ?",
          { static_cast<long long> (exclusion_ts) });
      if (!historical_knn_loaded && !seed_bounds.empty ())
        {
          const long long min_memory_id
              = AnyLongLong (seed_bounds[0], "min_memory_id");
          const long long max_memory_id
              = AnyLongLong (seed_bounds[0], "max_memory_id");
          if (min_memory_id > 0 && max_memory_id > min_memory_id)
            {
              const long long span = max_memory_id - min_memory_id + 1;
              const long long sample_count = static_cast<long long> (
                  std::max (1, seed_search_limit));
              const long long stride = std::max (1LL, span / sample_count);
              if (stride > 1)
                {
                  auto historical_rows = tx.Execute (
                      "WITH RECURSIVE sample(target_id, n) AS ("
                      "  SELECT ?1, 0 "
                      "  UNION ALL "
                      "  SELECT target_id + ?2, n + 1 FROM sample "
                      "  WHERE n + 1 < ?3 AND target_id + ?2 <= ?4"
                      "), targets(target_id) AS ("
                      "  SELECT target_id FROM sample "
                      "  UNION ALL "
                      "  SELECT target_id + (?2 / 2) FROM sample "
                      "  WHERE target_id + (?2 / 2) <= ?4"
                      "), sampled_ids AS ("
                      "  SELECT ("
                      "    SELECT m.memory_id FROM memories m "
                      "    WHERE m.memory_id >= targets.target_id "
                      "      AND m.embedding_id IS NOT NULL "
                      "      AND m.kind IN ('LONG_TERM', 'ASSOCIATION') "
                      "      AND COALESCE(m.start_ts, 0) < ?5 "
                      "    ORDER BY m.memory_id ASC LIMIT 1"
                      "  ) AS memory_id "
                      "  FROM targets"
                      ") "
                      "SELECT m.memory_id, m.embedding_id, m.start_ts, "
                      "       e.embedding, m.source_id "
                      "FROM sampled_ids sid "
                      "JOIN memories m ON m.memory_id = sid.memory_id "
                      "JOIN embeddings e ON e.embedding_id = m.embedding_id "
                      "WHERE sid.memory_id IS NOT NULL "
                      "GROUP BY m.memory_id "
                      "ORDER BY m.memory_id ASC",
                      { min_memory_id, stride, sample_count, max_memory_id,
                        static_cast<long long> (exclusion_ts) });
                  AppendUniqueRows (rows, std::move (historical_rows),
                                    seen_seed_memory_ids);
                }
            }
        }
      profile_timing ("GraphRetrieve.seed_fallback_sql", section_start);
    }

  start_profile_timing ();
  std::vector<Candidate> seeded;
  seeded.reserve (rows.size () + cached_recent_seeded.size ());
  std::unordered_map<long long, Candidate> candidates;
  candidates.reserve (rows.size () + cached_recent_seeded.size ());
  for (auto &candidate : cached_recent_seeded)
    {
      seeded.push_back (std::move (candidate));
    }
  for (const auto &row : rows)
    {
      Candidate candidate;
      candidate.memory_id = AnyLongLong (row, "memory_id");
      candidate.embedding_id = AnyLongLong (row, "embedding_id");
      const long long start_ts = AnyLongLong (row, "start_ts");
      candidate.start_ts = start_ts;
      auto it_source = row.find ("source_id");
      if (it_source != row.end ()
          && it_source->second.type () == typeid (std::string))
        {
          candidate.source_id = std::any_cast<std::string> (it_source->second);
        }
      auto it_embedding = row.find ("embedding");
      if (candidate.memory_id <= 0 || candidate.embedding_id <= 0
          || it_embedding == row.end ()
          || !AnyToEmbedding (it_embedding->second, embedding_dim,
                              candidate.embedding))
        {
          continue;
        }
      if (constructive_recall_enabled)
        {
          (void)RefreshCandidateToCurrentSurface (tx, candidate,
                                                  embedding_dim);
        }
      candidate.seed_score = core::Map01 (
          Cosine (signal.embedding, candidate.embedding));
      candidate.temporal_score = TemporalScore (signal.timestamp, start_ts, F,
                                                S, T);
      candidate.score = seed_weight * candidate.seed_score
                        + temporal_weight * candidate.temporal_score;
      seeded.push_back (candidate);
    }
  profile_timing ("GraphRetrieve.seed_materialize", section_start);

  start_profile_timing ();
  std::sort (seeded.begin (), seeded.end (), [] (const Candidate &a,
                                                 const Candidate &b) {
    if (a.score != b.score)
      {
        return a.score > b.score;
      }
    return a.memory_id > b.memory_id;
  });
  if (static_cast<int> (seeded.size ()) > seed_limit)
    {
      seeded.resize (static_cast<std::size_t> (seed_limit));
    }
  for (auto candidate : seeded)
    {
      InsertOrBoost (candidates, std::move (candidate));
    }
  std::vector<Candidate> source_expansion_seeds = seeded;
  profile_timing ("GraphRetrieve.seed_rank", section_start);

  if (fanout > 0 && !seeded.empty ())
    {
      start_profile_timing ();
      std::vector<std::any> params;
      params.reserve (seeded.size () + 2);
      std::string values;
      for (const auto &candidate : seeded)
        {
          if (!values.empty ())
            {
              values += ",";
            }
          values += "(?)";
          params.push_back (candidate.memory_id);
        }
      std::vector<std::any> expanded_params = params;
      expanded_params.push_back (static_cast<long long> (fanout));
      const auto expanded = tx.Execute (
          "WITH seed(id) AS (VALUES " + values + "), edge AS ("
          "  SELECT CASE WHEN a.source_memory_id = seed.id "
          "              THEN a.target_memory_id "
          "              ELSE a.source_memory_id END "
          "              AS memory_id,"
          "         MAX(COALESCE(a.weight, 0.0)) AS graph_weight "
          "  FROM seed "
          "  JOIN associations a "
          "    ON a.source_memory_id = seed.id "
          "    OR a.target_memory_id = seed.id "
          "  WHERE a.edge_type IN ('co_occurs', 'similar_to', "
          "                         'reinforces', 'causes', "
          "                         'derived_from', 'next_in_episode', "
          "                         'prev_in_episode', "
          "                         'within_same_event') "
          "  GROUP BY memory_id "
          "  ORDER BY graph_weight DESC, memory_id DESC "
          "  LIMIT ?"
          ") "
          "SELECT m.memory_id, m.embedding_id, m.start_ts, e.embedding, "
          "       m.source_id, edge.graph_weight "
          "FROM edge "
          "JOIN memories m ON m.memory_id = edge.memory_id "
          "JOIN embeddings e ON e.embedding_id = m.embedding_id "
          "WHERE m.kind IN ('LONG_TERM', 'ASSOCIATION') "
          "  AND COALESCE(m.start_ts, 0) < ?",
          [&] {
            std::vector<std::any> p = expanded_params;
            p.push_back (static_cast<long long> (exclusion_ts));
            return p;
          } ());

      for (const auto &row : expanded)
        {
          Candidate candidate;
          candidate.memory_id = AnyLongLong (row, "memory_id");
          candidate.embedding_id = AnyLongLong (row, "embedding_id");
          const long long start_ts = AnyLongLong (row, "start_ts");
          candidate.start_ts = start_ts;
          auto it_embedding = row.find ("embedding");
          if (candidate.memory_id <= 0 || candidate.embedding_id <= 0
              || it_embedding == row.end ()
              || !AnyToEmbedding (it_embedding->second, embedding_dim,
                                  candidate.embedding))
            {
              continue;
            }
          if (constructive_recall_enabled)
            {
              (void)RefreshCandidateToCurrentSurface (tx, candidate,
                                                      embedding_dim);
            }
          auto it_source = row.find ("source_id");
          if (it_source != row.end ()
              && it_source->second.type () == typeid (std::string))
            {
              candidate.source_id = std::any_cast<std::string> (
                  it_source->second);
            }
          candidate.graph_score = AnyDouble (row, "graph_weight");
          candidate.seed_score = core::Map01 (
              Cosine (signal.embedding, candidate.embedding));
          candidate.temporal_score = TemporalScore (signal.timestamp, start_ts,
                                                    F, S, T);
          candidate.score = seed_weight * candidate.seed_score
                            + graph_weight * candidate.graph_score
                            + temporal_weight * candidate.temporal_score;
          InsertOrBoost (candidates, std::move (candidate));
        }
      profile_timing ("GraphRetrieve.association_expansion", section_start);
    }

  const int source_radius = std::max (
      1, core::RetrievalDurableSourceMinTopK (F, T));
  const int source_expansion_limit = std::max (
      source_radius * 2,
      std::min (core::RetrievalDurableSourceLinkFanout (
                    F, S, T, static_cast<int> (seeded.size ())),
                output_limit * 4));
  int source_expansions = 0;

  if (source_seed_expansion_enabled)
    {
      start_profile_timing ();
      for (const auto &seed : source_expansion_seeds)
        {
          if (source_expansions >= source_expansion_limit)
            {
              break;
            }
          auto seed_it = p_ctx.retrieval_surface_index.find (seed.memory_id);
          if (seed_it == p_ctx.retrieval_surface_index.end ())
            {
              continue;
            }
          const auto &seed_entry
              = p_ctx.retrieval_surface_cache[seed_it->second];
          if (seed_entry.source_id.empty () || seed_entry.start_ts <= 0)
            {
              continue;
            }
          p_ctx.EnsureRetrievalSurfaceSourceIndexSorted (seed_entry.source_id);
          auto source_it
              = p_ctx.retrieval_surface_source_index.find (
                  seed_entry.source_id);
          if (source_it == p_ctx.retrieval_surface_source_index.end ())
            {
              continue;
            }
          const auto &source_entries = source_it->second;
          auto pos = std::find_if (
              source_entries.begin (), source_entries.end (),
              [&] (const auto &entry) {
                return entry.memory_id == seed_entry.memory_id;
              });
          if (pos == source_entries.end ())
            {
              continue;
            }
          const auto center = static_cast<int> (
              std::distance (source_entries.begin (), pos));
          const int begin = std::max (0, center - source_radius);
          const int end = std::min (
              static_cast<int> (source_entries.size ()),
              center + source_radius + 1);
          std::vector<int> neighbor_positions;
          neighbor_positions.reserve (static_cast<std::size_t> (end - begin));
          for (int i = begin; i < end; ++i)
            {
              if (i != center)
                {
                  neighbor_positions.push_back (i);
                }
            }
          bool natural_first_is_stale = false;
          bool has_fresh_signal_neighbor = false;
          if (!neighbor_positions.empty ())
            {
              const auto &natural_first = source_entries[
                  static_cast<std::size_t> (neighbor_positions.front ())];
              natural_first_is_stale
                  = std::llabs (natural_first.start_ts - seed_entry.start_ts)
                    > kStaleSourceNeighborGapMs;
              for (const int position : neighbor_positions)
                {
                  const auto &entry = source_entries[
                      static_cast<std::size_t> (position)];
                  if (std::llabs (
                          static_cast<long long> (signal.timestamp)
                          - entry.start_ts)
                      <= kFreshSourceNeighborToSignalGapMs)
                    {
                      has_fresh_signal_neighbor = true;
                      break;
                    }
                }
            }
          if (natural_first_is_stale
              && has_fresh_signal_neighbor
              && source_expansion_limit - source_expansions
                     < static_cast<int> (neighbor_positions.size ()))
            {
              std::sort (neighbor_positions.begin (), neighbor_positions.end (),
                         [&] (int a, int b) {
                const auto &left
                    = source_entries[static_cast<std::size_t> (a)];
                const auto &right
                    = source_entries[static_cast<std::size_t> (b)];
                const auto left_gap
                    = std::llabs (left.start_ts - seed_entry.start_ts);
                const auto right_gap
                    = std::llabs (right.start_ts - seed_entry.start_ts);
                if (left_gap != right_gap)
                  {
                    return left_gap < right_gap;
                  }
                if (left.start_ts != right.start_ts)
                  {
                    return left.start_ts < right.start_ts;
                  }
                return left.memory_id < right.memory_id;
              });
            }
          for (int i : neighbor_positions)
            {
              if (source_expansions >= source_expansion_limit)
                {
                  break;
                }
              const auto &neighbor_index
                  = source_entries[static_cast<std::size_t> (i)];
              if (neighbor_index.memory_id == seed_entry.memory_id
                  || neighbor_index.start_ts <= 0
                  || static_cast<std::uint64_t> (neighbor_index.start_ts)
                         >= exclusion_ts)
                {
                  continue;
                }
              auto neighbor_it = p_ctx.retrieval_surface_index.find (
                  neighbor_index.memory_id);
              if (neighbor_it == p_ctx.retrieval_surface_index.end ())
                {
                  continue;
                }
              const auto &neighbor
                  = p_ctx.retrieval_surface_cache[neighbor_it->second];
              if (neighbor.embedding.size () == 0
                  || neighbor.embedding.size () != signal.embedding.size ())
                {
                  continue;
                }

              Candidate candidate;
              candidate.memory_id = neighbor.memory_id;
              candidate.embedding_id = neighbor.embedding_id;
              candidate.source_id = neighbor.source_id;
              candidate.start_ts = neighbor.start_ts;
              candidate.embedding = neighbor.embedding;
              const int distance = std::max (1, std::abs (i - center));
              candidate.graph_score = 1.0 / static_cast<double> (distance);
              candidate.seed_score = core::Map01 (
                  Cosine (signal.embedding, candidate.embedding));
              candidate.temporal_score = TemporalScore (
                  signal.timestamp, neighbor.start_ts, F, S, T);
              candidate.score = seed_weight * candidate.seed_score
                                + graph_weight * candidate.graph_score
                                + temporal_weight * candidate.temporal_score;
              candidate.score = std::max (
                  candidate.score,
                  seed.score
                      * (0.95 / static_cast<double> (distance)));
              InsertOrBoost (candidates, std::move (candidate));
              ++source_expansions;
            }
        }
      profile_timing ("GraphRetrieve.source_cache_expansion", section_start);
    }

  if (source_seed_expansion_enabled)
    {
      start_profile_timing ();
      if (!p_ctx.retrieval_surface_cache.empty ())
        {
          for (const auto &seed : source_expansion_seeds)
            {
              if (source_expansions >= source_expansion_limit
                  || seed.memory_id <= 0)
                {
                  continue;
                }
              const auto source_rows = SourceExpansionRowsFromSurface (
                  p_ctx, seed, exclusion_ts, embedding_dim,
                  source_radius * 2);
              int source_rank = 0;
              for (const auto *entry : source_rows)
                {
                  if (source_expansions >= source_expansion_limit)
                    {
                      break;
                    }
                  Candidate candidate;
                  candidate.memory_id = entry->memory_id;
                  candidate.embedding_id = entry->embedding_id;
                  candidate.source_id = entry->source_id;
                  candidate.start_ts = entry->start_ts;
                  candidate.embedding = entry->embedding;
                  const double gap_ms = std::max (
                      1.0, std::abs (
                               static_cast<double> (candidate.start_ts)
                               - static_cast<double> (seed.start_ts)));
                  candidate.graph_score = 1.0 / (1.0 + gap_ms / 1000.0);
                  candidate.seed_score = core::Map01 (
                      Cosine (signal.embedding, candidate.embedding));
                  candidate.temporal_score = TemporalScore (
                      signal.timestamp, candidate.start_ts, F, S, T);
                  candidate.score = seed_weight * candidate.seed_score
                                    + graph_weight * candidate.graph_score
                                    + temporal_weight
                                          * candidate.temporal_score;
                  ++source_rank;
                  candidate.score = std::max (
                      candidate.score,
                      seed.score
                          * (0.95 / static_cast<double> (
                                        std::max (1, source_rank))));
                  InsertOrBoost (candidates, std::move (candidate));
                  ++source_expansions;
                }
            }
        }
      else
        {
          for (const auto &seed : source_expansion_seeds)
            {
              if (source_expansions >= source_expansion_limit
                  || seed.memory_id <= 0)
                {
                  continue;
                }
              const auto source_rows = tx.Execute (
                  "WITH anchor AS ("
                  "  SELECT source_id, COALESCE(start_ts, 0) AS start_ts "
                  "  FROM memories WHERE memory_id = ?"
                  ") "
                  "SELECT m.memory_id, m.embedding_id, m.source_id, "
                  "       m.start_ts, e.embedding "
                  "FROM memories m "
                  "JOIN anchor a ON a.source_id <> '' "
                  "             AND a.source_id = m.source_id "
                  "JOIN embeddings e ON e.embedding_id = m.embedding_id "
                  "WHERE m.memory_id != ? "
                  "  AND m.embedding_id IS NOT NULL "
                  "  AND COALESCE(m.source_id, '') <> '' "
                  "  AND m.kind IN ('LONG_TERM', 'ASSOCIATION') "
                  "  AND COALESCE(m.start_ts, 0) < ? "
                  "ORDER BY ABS(COALESCE(m.start_ts, 0) - a.start_ts), "
                  "         m.start_ts ASC, m.memory_id ASC "
                  "LIMIT ?",
                  { seed.memory_id, seed.memory_id,
                    static_cast<long long> (exclusion_ts),
                    static_cast<long long> (source_radius * 2) });
              int source_rank = 0;
              for (const auto &row : source_rows)
                {
                  if (source_expansions >= source_expansion_limit)
                    {
                      break;
                    }
                  Candidate candidate;
                  candidate.memory_id = AnyLongLong (row, "memory_id");
                  candidate.embedding_id = AnyLongLong (row, "embedding_id");
                  candidate.start_ts = AnyLongLong (row, "start_ts");
                  auto it_source = row.find ("source_id");
                  if (it_source != row.end ()
                      && it_source->second.type () == typeid (std::string))
                    {
                      candidate.source_id
                          = std::any_cast<std::string> (it_source->second);
                    }
                  auto it_embedding = row.find ("embedding");
                  if (candidate.memory_id <= 0 || candidate.embedding_id <= 0
                      || candidate.start_ts <= 0 || it_embedding == row.end ()
                      || !AnyToEmbedding (it_embedding->second, embedding_dim,
                                          candidate.embedding))
                    {
                      continue;
                    }
                  if (constructive_recall_enabled)
                    {
                      (void)RefreshCandidateToCurrentSurface (tx, candidate,
                                                              embedding_dim);
                    }
                  const double gap_ms = std::max (
                      1.0, std::abs (
                               static_cast<double> (candidate.start_ts)
                               - static_cast<double> (seed.start_ts)));
                  candidate.graph_score = 1.0 / (1.0 + gap_ms / 1000.0);
                  candidate.seed_score = core::Map01 (
                      Cosine (signal.embedding, candidate.embedding));
                  candidate.temporal_score = TemporalScore (
                      signal.timestamp, candidate.start_ts, F, S, T);
                  candidate.score = seed_weight * candidate.seed_score
                                    + graph_weight * candidate.graph_score
                                    + temporal_weight
                                          * candidate.temporal_score;
                  ++source_rank;
                  candidate.score = std::max (
                      candidate.score,
                      seed.score
                          * (0.95 / static_cast<double> (
                                        std::max (1, source_rank))));
                  InsertOrBoost (candidates, std::move (candidate));
                  ++source_expansions;
                }
            }
        }
      profile_timing ("GraphRetrieve.source_sql_expansion", section_start);
    }

  if (source_seed_expansion_enabled)
    {
      start_profile_timing ();
      for (const auto &seed : source_expansion_seeds)
        {
          if (source_expansions >= source_expansion_limit
              || seed.memory_id <= 0 || seed.source_id.empty ())
            {
              continue;
            }
          const auto parsed_source = ParseTurnSourceId (seed.source_id);
          if (!parsed_source)
            {
              continue;
            }
          for (int delta = -source_radius; delta <= source_radius; ++delta)
            {
              if (source_expansions >= source_expansion_limit)
                {
                  break;
                }
              if (delta == 0)
                {
                  continue;
                }
              const std::string turn_pattern = TurnSourceLikePattern (
                  *parsed_source, parsed_source->turn + delta);
              if (turn_pattern.empty ())
                {
                  continue;
                }
              const auto turn_rows = tx.Execute (
                  "SELECT m.memory_id, m.embedding_id, m.source_id, "
                  "       m.start_ts, e.embedding "
                  "FROM memories m "
                  "JOIN embeddings e ON e.embedding_id = m.embedding_id "
                  "WHERE m.memory_id != ? "
                  "  AND m.embedding_id IS NOT NULL "
                  "  AND m.kind IN ('LONG_TERM', 'ASSOCIATION') "
                  "  AND COALESCE(m.start_ts, 0) < ? "
                  "  AND COALESCE(m.source_id, '') LIKE ? ESCAPE '\\' "
                  "ORDER BY ABS(COALESCE(m.start_ts, 0) - ?), "
                  "         m.start_ts ASC, m.memory_id ASC "
                  "LIMIT ?",
                  { seed.memory_id, static_cast<long long> (exclusion_ts),
                    turn_pattern, seed.start_ts,
                    static_cast<long long> (source_radius * 2) });
              int turn_rank = 0;
              for (const auto &row : turn_rows)
                {
                  if (source_expansions >= source_expansion_limit)
                    {
                      break;
                    }
                  Candidate candidate;
                  candidate.memory_id = AnyLongLong (row, "memory_id");
                  candidate.embedding_id = AnyLongLong (row, "embedding_id");
                  candidate.start_ts = AnyLongLong (row, "start_ts");
                  auto it_source = row.find ("source_id");
                  if (it_source != row.end ()
                      && it_source->second.type () == typeid (std::string))
                    {
                      candidate.source_id
                          = std::any_cast<std::string> (it_source->second);
                    }
                  auto it_embedding = row.find ("embedding");
                  if (candidate.memory_id <= 0 || candidate.embedding_id <= 0
                      || candidate.start_ts <= 0 || it_embedding == row.end ()
                      || !AnyToEmbedding (it_embedding->second, embedding_dim,
                                          candidate.embedding))
                    {
                      continue;
                    }
                  if (constructive_recall_enabled)
                    {
                      (void)RefreshCandidateToCurrentSurface (tx, candidate,
                                                              embedding_dim);
                    }
                  const int turn_distance = std::max (1, std::abs (delta));
                  const double gap_ms = std::max (
                      1.0, std::abs (
                               static_cast<double> (candidate.start_ts)
                               - static_cast<double> (seed.start_ts)));
                  candidate.graph_score
                      = 1.0 / (static_cast<double> (turn_distance)
                               * (1.0 + gap_ms / 1000.0));
                  candidate.seed_score = core::Map01 (
                      Cosine (signal.embedding, candidate.embedding));
                  candidate.temporal_score = TemporalScore (
                      signal.timestamp, candidate.start_ts, F, S, T);
                  candidate.score = seed_weight * candidate.seed_score
                                    + graph_weight * candidate.graph_score
                                    + temporal_weight
                                          * candidate.temporal_score;
                  ++turn_rank;
                  candidate.score = std::max (
                      candidate.score,
                      seed.score
                          * (0.90
                             / static_cast<double> (
                                   std::max (1,
                                             turn_distance + turn_rank - 1))));
                  InsertOrBoost (candidates, std::move (candidate));
                  ++source_expansions;
                }
            }
        }
      profile_timing ("GraphRetrieve.turn_source_sql_expansion",
                      section_start);
    }

  start_profile_timing ();
  std::vector<Candidate> ranked;
  ranked.reserve (candidates.size ());
  for (auto &[id, candidate] : candidates)
    {
      (void)id;
      ranked.push_back (std::move (candidate));
    }
  std::sort (ranked.begin (), ranked.end (), [] (const Candidate &a,
                                                 const Candidate &b) {
    if (a.score != b.score)
      {
        return a.score > b.score;
      }
    return a.memory_id > b.memory_id;
  });
  std::vector<retrieval_trace::RejectedCandidate> rejected_trace;
  if (static_cast<int> (ranked.size ()) > output_limit)
    {
      const double cutoff_score = ranked[static_cast<std::size_t> (
                                         output_limit - 1)]
                                      .score;
      const std::size_t rejected_limit = std::min<std::size_t> (
          ranked.size (), static_cast<std::size_t> (output_limit) + 64);
      rejected_trace.reserve (
          rejected_limit - static_cast<std::size_t> (output_limit));
      for (std::size_t i = static_cast<std::size_t> (output_limit);
           i < rejected_limit; ++i)
        {
          retrieval_trace::RankedCandidate trace;
          trace.embedding_id = ranked[i].embedding_id;
          trace.memory_id = ranked[i].memory_id;
          trace.score = ranked[i].score;
          trace.relevance = ranked[i].seed_score;
          trace.temporal_score = ranked[i].temporal_score;
          trace.activation.base_level = ranked[i].seed_score;
          trace.activation.spreading_activation = ranked[i].graph_score;
          trace.activation.activation_total = ranked[i].score;
          retrieval_trace::RejectedCandidate rejected;
          rejected.candidate = trace;
          rejected.reason = "below_output_limit";
          rejected.stage = "selection";
          rejected.observed = ranked[i].score;
          rejected.threshold = cutoff_score;
          rejected_trace.push_back (std::move (rejected));
        }
    }
  if (static_cast<int> (ranked.size ()) > output_limit)
    {
      ranked.resize (static_cast<std::size_t> (output_limit));
    }
  profile_timing ("GraphRetrieve.final_rank", section_start);

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
    if (constructive_recall_enabled && signal.embedding.size () > 0)
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
                &timings);
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
