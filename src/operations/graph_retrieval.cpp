#include "cortext/operations/graph_retrieval.hpp"

#include "constructive_recall_internal.hpp"
#include "retrieval_trace_state.hpp"
#include "cortext/core/knobs.hpp"
#include "cortext/core/utils.hpp"
#include "cortext/processor/operation_context.hpp"
#include "cortext/store/store.hpp"
#include "cortext/store/utils.hpp"
#include "cortext/telemetry/telemetry.hpp"

#include <algorithm>
#include <any>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <map>
#include <string>
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
  Eigen::VectorXf embedding;
  double seed_score = 0.0;
  double graph_score = 0.0;
  double temporal_score = 0.0;
  double score = 0.0;
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
TemporalScore (std::uint64_t now_ts, long long start_ts, double F, double S,
               double T)
{
  if (start_ts <= 0 || now_ts <= static_cast<std::uint64_t> (start_ts))
    {
      return 0.0;
    }
  const auto age_ms = static_cast<std::uint64_t> (now_ts)
                      - static_cast<std::uint64_t> (start_ts);
  return core::RetrievalGraphExpandedRagTemporalRankScore (F, S, T, age_ms);
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
      if (it->second.embedding.size () == 0)
        {
          it->second.embedding = std::move (candidate.embedding);
          it->second.embedding_id = candidate.embedding_id;
        }
    }
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

  const auto rows = tx.Execute (
      "SELECT m.memory_id, m.embedding_id, m.start_ts, e.embedding "
      "FROM memories m "
      "JOIN embeddings e ON e.embedding_id = m.embedding_id "
      "WHERE m.embedding_id IS NOT NULL "
      "  AND m.kind IN ('LONG_TERM', 'ASSOCIATION') "
      "  AND COALESCE(m.start_ts, 0) < ? "
      "ORDER BY m.memory_id DESC LIMIT ?",
      { static_cast<long long> (exclusion_ts),
        static_cast<long long> (std::max (seed_limit * 8, output_limit * 4)) });

  std::vector<Candidate> seeded;
  seeded.reserve (rows.size ());
  std::unordered_map<long long, Candidate> candidates;
  candidates.reserve (rows.size ());
  for (const auto &row : rows)
    {
      Candidate candidate;
      candidate.memory_id = AnyLongLong (row, "memory_id");
      candidate.embedding_id = AnyLongLong (row, "embedding_id");
      const long long start_ts = AnyLongLong (row, "start_ts");
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
          if (auto current = constructive_recall::LoadCurrentEmbedding (
                  tx, candidate.memory_id, candidate.embedding_id,
                  embedding_dim))
            {
              candidate.embedding = std::move (*current);
            }
        }
      candidate.seed_score = core::Map01 (
          Cosine (signal.embedding, candidate.embedding));
      candidate.temporal_score = TemporalScore (signal.timestamp, start_ts, F,
                                                S, T);
      candidate.score = seed_weight * candidate.seed_score
                        + temporal_weight * candidate.temporal_score;
      seeded.push_back (candidate);
    }

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

  if (fanout > 0 && !seeded.empty ())
    {
      std::vector<std::any> params;
      params.reserve (seeded.size () + 1);
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
      params.push_back (static_cast<long long> (fanout));

      const auto expanded = tx.Execute (
          "WITH seed(id) AS (VALUES " + values + "), edge AS ("
          "  SELECT CASE WHEN a.source_memory_id = seed.id "
          "              THEN a.target_memory_id ELSE a.source_memory_id END "
          "              AS memory_id,"
          "         MAX(COALESCE(a.weight, 0.0)) AS graph_weight "
          "  FROM seed "
          "  JOIN associations a "
          "    ON a.source_memory_id = seed.id OR a.target_memory_id = seed.id "
          "  WHERE a.edge_type IN ('co_occurs', 'similar_to', 'reinforces', "
          "                         'causes', 'derived_from') "
          "  GROUP BY memory_id "
          "  ORDER BY graph_weight DESC, memory_id DESC "
          "  LIMIT ?"
          ") "
          "SELECT m.memory_id, m.embedding_id, m.start_ts, e.embedding, "
          "       edge.graph_weight "
          "FROM edge "
          "JOIN memories m ON m.memory_id = edge.memory_id "
          "JOIN embeddings e ON e.embedding_id = m.embedding_id "
          "WHERE m.kind IN ('LONG_TERM', 'ASSOCIATION') "
          "  AND COALESCE(m.start_ts, 0) < ?",
          [&] {
            std::vector<std::any> p = params;
            p.push_back (static_cast<long long> (exclusion_ts));
            return p;
          } ());

      for (const auto &row : expanded)
        {
          Candidate candidate;
          candidate.memory_id = AnyLongLong (row, "memory_id");
          candidate.embedding_id = AnyLongLong (row, "embedding_id");
          const long long start_ts = AnyLongLong (row, "start_ts");
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
              if (auto current = constructive_recall::LoadCurrentEmbedding (
                      tx, candidate.memory_id, candidate.embedding_id,
                      embedding_dim))
                {
                  candidate.embedding = std::move (*current);
                }
            }
          auto it_graph = row.find ("graph_weight");
          if (it_graph != row.end () && it_graph->second.type () == typeid (double))
            {
              candidate.graph_score = std::any_cast<double> (it_graph->second);
            }
          candidate.seed_score = core::Map01 (
              Cosine (signal.embedding, candidate.embedding));
          candidate.temporal_score = TemporalScore (signal.timestamp, start_ts,
                                                    F, S, T);
          candidate.score = seed_weight * candidate.seed_score
                            + graph_weight * candidate.graph_score
                            + temporal_weight * candidate.temporal_score;
          InsertOrBoost (candidates, std::move (candidate));
        }
    }

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
  if (static_cast<int> (ranked.size ()) > output_limit)
    {
      ranked.resize (static_cast<std::size_t> (output_limit));
    }

  std::unordered_map<long long, Eigen::VectorXf> out;
  out.reserve (ranked.size ());
  std::vector<long long> selected_order;
  selected_order.reserve (ranked.size ());
  std::vector<retrieval_trace::RankedCandidate> trace_ranked;
  trace_ranked.reserve (ranked.size ());
  for (const auto &candidate : ranked)
    {
      out.emplace (candidate.embedding_id, candidate.embedding);
      selected_order.push_back (candidate.embedding_id);

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
  retrieval_trace::SetLastRetrievalSummary ({});
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
        for (const auto &candidate : ranked)
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
            const double uncertainty = core::Clamp (
                reconstruction_policy.source_confidence_weight
                    * (1.0 - source_confidence)
                    + reconstruction_policy.candidate_score_weight
                          * (1.0 - candidate_score)
                    + reconstruction_policy.context_score_weight
                          * (1.0 - context_score),
                0.0, 1.0);
            const double blend = core::Clamp (
                reconstruction_policy.min_blend
                    + (reconstruction_policy.max_blend
                       - reconstruction_policy.min_blend)
                          * uncertainty,
                reconstruction_policy.min_blend,
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
  context.SetRetrievedMemoryEmbeddings (std::move (out));
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
