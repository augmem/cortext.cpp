#include "cortext/operations/reconsolidation.hpp"
#include "historical_surface_search_cache_internal.hpp"
#include "association_fanout_cache_internal.hpp"
#include "constructive_recall_internal.hpp"
#include "neuromodulator_internal.hpp"
#include "cortext/core/algorithms.hpp"
#include "cortext/core/knobs.hpp"
#include "cortext/core/utils.hpp"
#include "cortext/operations/constants.hpp"
#include "cortext/processor/operation_context.hpp"
#include "cortext/store/store.hpp"
#include "cortext/store/utils.hpp"
#include "cortext/telemetry/telemetry.hpp"
#include <Eigen/Dense>
#include <algorithm>
#include <any>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <optional>
#include <unordered_map>
#include <string>
#include <unordered_set>
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

/// @brief Normalizes a vector to unit length.
inline Eigen::VectorXf
Unit (const Eigen::VectorXf &v)
{
  const double n = v.norm ();
  if (n <= constants::kNormEpsilon)
    {
      return v;
    }
  return v / static_cast<float> (n);
}

inline std::vector<float>
ToFloatVector (const Eigen::VectorXf &v)
{
  std::vector<float> out;
  out.resize (static_cast<size_t> (v.size ()));
  for (int i = 0; i < v.size (); ++i)
    {
      out[static_cast<size_t> (i)] = v[i];
    }
  return out;
}

void
RefreshProcessorSurfaces (Transaction &tx, ProcessorContext *p_ctx,
                          long long memory_id, long long old_embedding_id,
                          long long new_embedding_id,
                          const Eigen::VectorXf &embedding,
                          bool current_surface_written)
{
  if (!p_ctx)
    {
      return;
    }

  auto cache_it = p_ctx->retrieval_surface_index.find (memory_id);
  if (cache_it != p_ctx->retrieval_surface_index.end ())
    {
      auto &entry = p_ctx->retrieval_surface_cache[cache_it->second];
      if (entry.embedding_id > 0)
        {
          p_ctx->retrieval_surface_embedding_index.erase (entry.embedding_id);
        }
      entry.embedding_id = new_embedding_id;
      entry.embedding = embedding;
      entry.embedding_norm = embedding.norm ();
      p_ctx->retrieval_surface_embedding_index[new_embedding_id]
          = cache_it->second;
    }

  if (old_embedding_id > 0 && old_embedding_id != new_embedding_id)
    {
      auto existing = p_ctx->retrieval_surface_embedding_index.find (
          old_embedding_id);
      if (existing == p_ctx->retrieval_surface_embedding_index.end ())
        {
          auto rows = tx.Execute (
              "SELECT memory_id FROM memories "
              "WHERE embedding_id = ? AND memory_id != ? "
              "ORDER BY memory_id ASC LIMIT 1",
              { old_embedding_id, memory_id });
          if (!rows.empty () && rows[0].count ("memory_id") == 1)
            {
              const long long sibling_memory_id = store::AnyToLongLong (
                  rows[0].at ("memory_id")).value_or (0);
              auto sibling_it = p_ctx->retrieval_surface_index.find (
                  sibling_memory_id);
              if (sibling_memory_id > 0
                  && sibling_it != p_ctx->retrieval_surface_index.end ())
                {
                  p_ctx->retrieval_surface_embedding_index[old_embedding_id]
                      = sibling_it->second;
                }
            }
        }
    }

  auto assoc_it = p_ctx->association_cache_index.find (memory_id);
  if (assoc_it != p_ctx->association_cache_index.end ())
    {
      auto &entry = p_ctx->association_cache[assoc_it->second];
      entry.embedding_id = new_embedding_id;
      entry.embedding = embedding;
      entry.embedding_norm = embedding.norm ();
    }
  if (current_surface_written)
    {
      historical_surface_search_cache_internal::UpsertCurrent (
          *p_ctx,
          { new_embedding_id, memory_id, 0, std::string (), std::string (),
            embedding });
    }
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
      const long long embedding_id = store::AnyToLongLong (
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

long long
WriteEmbeddingInPlaceOrFork (Transaction &tx, long long memory_id,
                             long long embedding_id,
                             const Eigen::VectorXf &embedding,
                             long long timestamp,
                             ProcessorContext *p_ctx = nullptr)
{
  if (p_ctx != nullptr)
    {
      // Replacing or forking a joined base row changes both the vector and its
      // join metadata. Fall back to authoritative SQL rather than risk a
      // partially updated pre-filter population.
      historical_surface_search_cache_internal::Erase (*p_ctx);
    }
  auto ref_rows = tx.Execute (
      "SELECT "
      "  (SELECT COUNT(*) FROM memories WHERE embedding_id = ?) "
      "+ (SELECT COUNT(*) FROM signals WHERE embedding_id = ?) "
      "+ (SELECT COUNT(*) FROM memory_reconstructions "
      "   WHERE embedding_id = ?) AS cnt",
      { embedding_id, embedding_id, embedding_id });
  long long ref_count = 0;
  if (!ref_rows.empty () && ref_rows[0].count ("cnt") == 1)
    {
      ref_count = store::AnyToLongLong (ref_rows[0].at ("cnt")).value_or (0);
    }
  if (ref_count <= 1)
    {
      tx.Execute ("DELETE FROM embeddings WHERE embedding_id = ?",
                  { embedding_id });
      tx.Execute ("INSERT INTO embeddings (embedding_id, embedding, created_at) "
                  "VALUES (?, ?, ?)",
                  { embedding_id, ToFloatVector (embedding), timestamp });
      tx.Execute (
          "DELETE FROM current_memory_embeddings WHERE memory_id = ?",
          { memory_id });
      tx.Execute (
          "INSERT INTO current_memory_embeddings("
          "memory_id, embedding, embedding_id, created_at"
          ") VALUES (?, ?, ?, ?)",
          { memory_id, ToFloatVector (embedding), embedding_id, timestamp });
      RefreshProcessorSurfaces (tx, p_ctx, memory_id, embedding_id,
                                embedding_id, embedding, true);
      return embedding_id;
    }

  tx.Execute ("INSERT INTO embeddings (embedding, created_at) VALUES (?, ?)",
              { ToFloatVector (embedding), timestamp });
  auto id_rows = tx.Execute ("SELECT last_insert_rowid() AS id", {});
  long long replacement_embedding_id = 0;
  if (!id_rows.empty () && id_rows[0].count ("id") == 1)
    {
      replacement_embedding_id
          = store::AnyToLongLong (id_rows[0].at ("id")).value_or (0);
    }
  if (replacement_embedding_id > 0)
    {
      tx.Execute ("UPDATE memories SET embedding_id = ? WHERE memory_id = ?",
                  { replacement_embedding_id, memory_id });
      tx.Execute (
          "DELETE FROM current_memory_embeddings WHERE memory_id = ?",
          { memory_id });
      tx.Execute (
          "INSERT INTO current_memory_embeddings("
          "memory_id, embedding, embedding_id, created_at"
          ") VALUES (?, ?, ?, ?)",
          { memory_id, ToFloatVector (embedding), replacement_embedding_id,
            timestamp });
      RefreshProcessorSurfaces (tx, p_ctx, memory_id, embedding_id,
                                replacement_embedding_id, embedding, true);
    }
  return replacement_embedding_id;
}

/// @brief Neighbor info returned from graph traversal.
struct NeighborInfo
{
  long long memory_id;
  long long embedding_id;
  int depth;
};

struct LabilityUpdate
{
  long long memory_id = 0;
  double lability = 0.0;
};

struct RetrievedReconCandidate
{
  long long memory_id = 0;
  long long embedding_id = 0;
  Eigen::VectorXf embedding;
};

constexpr long long kReconsolidationNeighborLimit = 128;
constexpr std::size_t kReconsolidationTraversalOvercollect
    = static_cast<std::size_t> (kReconsolidationNeighborLimit) * 2;
constexpr std::size_t kReconsolidationFanoutPerNode = 8;
constexpr std::size_t kReconsolidationLabilityBatchSize = 16;
constexpr std::size_t kReconsolidationLabilityOnlyLimit = 16;

/// @brief Queries graph neighbors of a given embedding via iterative BFS.
/// V2: Uses ASSOCIATIONS table with memory_ids instead of graph_edges.
/// Returns neighbors connected via 'co_occurs' or 'reinforces' edges
/// up to max_depth hops away.
inline std::vector<NeighborInfo>
QueryGraphNeighbors (Store *store, ProcessorContext &p_ctx,
                     long long seed_memory_id, long long embedding_id,
                     int max_depth)
{
  std::vector<NeighborInfo> neighbors;
  if (!store || max_depth < 1)
    {
      return neighbors;
    }

  if (seed_memory_id <= 0)
    {
      auto seed_rows = store->Execute (
          "SELECT memory_id FROM memories WHERE embedding_id = ? LIMIT 1",
          { embedding_id });
      if (!seed_rows.empty ())
        {
          auto it = seed_rows[0].find ("memory_id");
          if (it != seed_rows[0].end ())
            {
              seed_memory_id = store::AnyToLongLong (it->second).value_or (0);
            }
        }
    }
  if (seed_memory_id <= 0)
    {
      return neighbors;
    }

  auto *association_cache
      = association_fanout_cache::Ensure (store, p_ctx);
  if (association_cache && association_cache->valid)
    {
      std::unordered_map<long long, NeighborInfo> best;
      std::unordered_set<long long> visited;
      visited.insert (seed_memory_id);
      std::vector<long long> frontier { seed_memory_id };

      auto visit_edges =
          [&] (const auto &edges_by_memory, long long memory_id, int depth,
               std::vector<long long> &next_frontier) {
        auto edge_it = edges_by_memory.find (memory_id);
        if (edge_it == edges_by_memory.end ())
          {
            return false;
          }
        std::size_t considered = 0;
        for (const auto &edge : edge_it->second)
          {
            if (edge.memory_id <= 0
                || (edge.edge_type != "co_occurs"
                    && edge.edge_type != "reinforces"))
              {
                continue;
              }
            if (considered++ >= kReconsolidationFanoutPerNode)
              {
                break;
              }
            if (visited.insert (edge.memory_id).second)
              {
                if (next_frontier.size ()
                    < kReconsolidationTraversalOvercollect)
                  {
                    next_frontier.push_back (edge.memory_id);
                  }
              }
            if (edge.embedding_id <= 0)
              {
                continue;
              }
            auto best_it = best.find (edge.memory_id);
            if (best_it == best.end () || depth < best_it->second.depth)
              {
                best[edge.memory_id] = { edge.memory_id, edge.embedding_id,
                                         depth };
              }
            if (best.size () >= kReconsolidationTraversalOvercollect)
              {
                return true;
              }
          }
        return false;
      };

      for (int depth = 1; depth <= max_depth && !frontier.empty (); ++depth)
        {
          std::vector<long long> next_frontier;
          for (const long long memory_id : frontier)
            {
              if (visit_edges (association_cache->out_by_source, memory_id,
                               depth, next_frontier)
                  || visit_edges (association_cache->in_by_target, memory_id,
                                  depth, next_frontier))
                {
                  break;
                }
            }
          frontier.swap (next_frontier);
          if (best.size () >= kReconsolidationNeighborLimit)
            {
              break;
            }
        }

      neighbors.reserve (best.size ());
      for (const auto &[memory_id, info] : best)
        {
          if (memory_id != seed_memory_id && info.depth > 0)
            {
              neighbors.push_back (info);
            }
        }
      std::sort (neighbors.begin (), neighbors.end (),
                 [] (const auto &a, const auto &b) {
        if (a.depth != b.depth)
          {
            return a.depth < b.depth;
          }
        return a.memory_id < b.memory_id;
      });
      if (neighbors.size ()
          > static_cast<size_t> (kReconsolidationNeighborLimit))
        {
          neighbors.resize (
              static_cast<size_t> (kReconsolidationNeighborLimit));
        }
      return neighbors;
    }

  auto rows = store->Execute (
      "WITH RECURSIVE "
      "seed(memory_id) AS ("
      "  SELECT ? AS memory_id "
      "), "
      "expand(memory_id, depth) AS ("
      "  SELECT memory_id, 0 FROM seed "
      "  UNION "
      "  SELECT a.target_memory_id, expand.depth + 1 "
      "  FROM expand "
      "  JOIN associations a INDEXED BY idx_associations_source_edge_weight "
      "    ON a.source_memory_id = expand.memory_id "
      "   AND a.edge_type IN ('co_occurs', 'reinforces') "
      "  WHERE expand.depth < ? "
      "  UNION "
      "  SELECT a.source_memory_id, expand.depth + 1 "
      "  FROM expand "
      "  JOIN associations a INDEXED BY idx_associations_target_edge_weight "
      "    ON a.target_memory_id = expand.memory_id "
      "   AND a.edge_type IN ('co_occurs', 'reinforces') "
      "  WHERE expand.depth < ? "
      "), "
      "best(memory_id, depth) AS ("
      "  SELECT memory_id, MIN(depth) "
      "  FROM expand "
      "  WHERE depth > 0 "
      "    AND memory_id NOT IN (SELECT memory_id FROM seed) "
      "  GROUP BY memory_id"
      ") "
      "SELECT best.memory_id, m.embedding_id, best.depth "
      "FROM best "
      "JOIN memories m ON m.memory_id = best.memory_id "
      "WHERE m.embedding_id IS NOT NULL "
      "ORDER BY best.depth ASC, best.memory_id ASC "
      "LIMIT ?",
      { seed_memory_id, static_cast<long long> (max_depth),
        static_cast<long long> (max_depth),
        kReconsolidationNeighborLimit });

  neighbors.reserve (rows.size ());
  for (const auto &row : rows)
    {
      auto mem_it = row.find ("memory_id");
      auto emb_it = row.find ("embedding_id");
      auto depth_it = row.find ("depth");
      if (mem_it == row.end () || emb_it == row.end ()
          || depth_it == row.end ())
        {
          continue;
        }
      if (mem_it->second.type () != typeid (long long)
          || emb_it->second.type () != typeid (long long))
        {
          continue;
        }
      int depth = 0;
      if (depth_it->second.type () == typeid (long long))
        {
          depth = static_cast<int> (std::any_cast<long long> (
              depth_it->second));
        }
      else if (depth_it->second.type () == typeid (int))
        {
          depth = std::any_cast<int> (depth_it->second);
        }
      if (depth <= 0)
        {
          continue;
        }
      neighbors.push_back (
          { std::any_cast<long long> (mem_it->second),
            std::any_cast<long long> (emb_it->second), depth });
    }

  return neighbors;
}

inline void
WriteNeighborLability (Transaction &tx, long long memory_id, double lability,
                       long long timestamp)
{
  tx.Execute ("UPDATE memories "
              "SET lability_state = COALESCE(?, 0.0), lability_ts = ? "
              "WHERE memory_id = ?",
              { lability, timestamp, memory_id });
}

inline void
WriteNeighborLabilityBatch (Transaction &tx,
                            const std::vector<LabilityUpdate> &updates,
                            long long timestamp)
{
  for (size_t offset = 0; offset < updates.size ();
       offset += kReconsolidationLabilityBatchSize)
    {
      const size_t count = std::min (kReconsolidationLabilityBatchSize,
                                     updates.size () - offset);
      std::string sql
          = "UPDATE memories SET lability_state = CASE memory_id ";
      std::vector<std::any> params;
      params.reserve (count * 3 + 1);
      for (size_t i = 0; i < count; ++i)
        {
          const auto &update = updates[offset + i];
          sql += "WHEN ? THEN COALESCE(?, 0.0) ";
          params.push_back (update.memory_id);
          params.push_back (update.lability);
        }
      sql += "ELSE lability_state END, lability_ts = ? WHERE memory_id IN (";
      params.push_back (timestamp);
      for (size_t i = 0; i < count; ++i)
        {
          if (i > 0)
            {
              sql += ",";
            }
          sql += "?";
          params.push_back (updates[offset + i].memory_id);
        }
      sql += ")";
      tx.Execute (sql, params);
    }
}

/// @brief Writes ripple updates for a neighbor embedding (v2 schema).
inline void
WriteNeighborUpdates (Transaction &tx, long long embedding_id,
                      long long memory_id, double lability, long long timestamp,
                      const Eigen::VectorXf &blended,
                      constructive_recall::ReconstructionUpdatePolicy policy,
                      ProcessorContext *p_ctx)
{
  // Update lability_state and lability_ts in memories table (v2: merged from
  // memory_feedback)
  WriteNeighborLability (tx, memory_id, lability, timestamp);

  if (constructive_recall::Disabled ())
    {
      WriteEmbeddingInPlaceOrFork (tx, memory_id, embedding_id, blended,
                                   timestamp, p_ctx);
      return;
    }

  const auto blob_id = constructive_recall::LoadCurrentBlobId (tx, memory_id);
  constructive_recall::AppendReconstructionWithEmbedding (
      tx, memory_id, blended, blob_id, timestamp, std::min (1.0, lability),
      "reconsolidation", 1.0, 1.0, policy, p_ctx);
  if (auto current = constructive_recall::LoadCurrentEmbedding (
          tx, memory_id, embedding_id, static_cast<int> (blended.size ())))
    {
      const long long current_embedding_id = LoadCurrentSurfaceEmbeddingId (
          tx, memory_id, embedding_id);
      RefreshProcessorSurfaces (tx, p_ctx, memory_id, embedding_id,
                                current_embedding_id, *current,
                                policy.update_current_surface
                                    && !constructive_recall::
                                            CurrentSurfaceWritesDisabled ());
    }
}

} // namespace

void
ApplyReconsolidation::Execute (OperationContext &context, Transaction &tx) const
{
  auto &p_ctx = context.GetProcessorContext ();
  const auto &cfg = context.GetConfig ();
  Store *store = context.GetStore ();

  // Require a current context embedding to drift toward.
  if (p_ctx.recent_context_embeddings.empty ())
    {
      return;
    }
  const Eigen::VectorXf &x_cur = p_ctx.recent_context_embeddings.back ();
  const Eigen::VectorXf u_cur = Unit (x_cur);

  const auto &retrieved = context.GetRetrievedMemoryEmbeddings ();
  const auto &retrieved_records = context.GetRetrievedMemoryCandidates ();
  std::vector<RetrievedReconCandidate> retrieved_candidates;
  if (!retrieved_records.empty ())
    {
      retrieved_candidates.reserve (retrieved_records.size ());
      for (const auto &candidate : retrieved_records)
        {
          if (candidate.memory_id > 0 && candidate.embedding_id > 0
              && candidate.embedding.size () > 0)
            {
              retrieved_candidates.push_back (
                  { candidate.memory_id, candidate.embedding_id,
                    candidate.embedding });
            }
        }
    }
  else
    {
      retrieved_candidates.reserve (retrieved.size ());
      for (const auto &kv : retrieved)
        {
          if (kv.first > 0 && kv.second.size () > 0)
            {
              retrieved_candidates.push_back ({ 0, kv.first, kv.second });
            }
        }
    }
  if (retrieved_candidates.empty ())
    {
      return;
    }

  const double S = cfg.sensitivity;
  const double T = cfg.stability;
  const double F = cfg.focus;
  const double recon_gain = core::ReconsolidationGain (T);
  const double recon_mod_scale
      = neuromodulation::ReconsolidationScale (p_ctx.neuromod_ach);
  const double ripple_decay = core::RippleDecay (T);
  const int ripple_depth = core::RippleDepth (T);
  const double tau_labile = core::TauLabile (T);
  const double lability_susc = core::LabilitySusceptibility (S, T);
  const double drift_clamp = core::ReconsolidationDriftClamp (F, S, T);
  const double drift_skip_epsilon = core::ReconsolidationDriftSkipEpsilon (
      F, S, T);
  const double ripple_strength_min = core::RippleStrengthMin (F, S, T);
  const double ripple_drift_cap_factor = core::RippleDriftCapFactor (F, S, T);
  const int ripple_reconstruction_limit
      = core::ReconsolidationRippleReconstructionLimit (F, S, T);
  const int primary_reconstruction_limit
      = core::ReconsolidationPrimaryReconstructionLimit (F, S, T);
  const double uncertainty_relevance_weight
      = core::ReconsolidationUncertaintyRelevanceWeight (F, S, T);
  const constructive_recall::ReconstructionUpdatePolicy
      reconstruction_update_policy {
        core::ReconstructionHistoryLimit (F, S, T),
        core::ReconstructionPruneBatchLimit (F, S, T),
        core::ReconstructionMinUpdateIntervalMs (F, S, T),
        true
      };

  double max_drift = 0.0;
  const long long now_ts
      = static_cast<long long> (context.GetSignal ().timestamp);

  // Track already processed memories/embeddings (avoid duplicate ripple).
  std::unordered_set<long long> processed_ids;
  int primary_reconstruction_considered = 0;
  int ripple_reconstruction_attempts = 0;

  for (const auto &retrieved_candidate : retrieved_candidates)
    {
      const long long embedding_id = retrieved_candidate.embedding_id;
      const Eigen::VectorXf &m = retrieved_candidate.embedding;
      if (m.size () == 0 || m.size () != u_cur.size ())
        {
          continue;
        }

      long long memory_id = 0;
      double stored_lability = 0.0;
      long long lability_ts = 0;
      const auto metadata_start = SteadyClock::now ();
      auto mem_rows = retrieved_candidate.memory_id > 0
                          ? tx.Execute (
                                "SELECT memory_id, lability_state, "
                                "lability_ts "
                                "FROM memories WHERE memory_id = ?",
                                { retrieved_candidate.memory_id })
                          : tx.Execute (
                                "SELECT memory_id, lability_state, "
                                "lability_ts "
                                "FROM memories WHERE embedding_id = ?",
                                { embedding_id });
      context.AddOperationTiming ("Reconsolidation.metadata_sql",
                                  ElapsedMillis (metadata_start));
      if (!mem_rows.empty ())
        {
          auto it_mem = mem_rows[0].find ("memory_id");
          if (it_mem != mem_rows[0].end ())
            {
              if (it_mem->second.type () == typeid (long long))
                {
                  memory_id = std::any_cast<long long> (it_mem->second);
                }
              else if (it_mem->second.type () == typeid (int))
                {
                  memory_id = static_cast<long long> (
                      std::any_cast<int> (it_mem->second));
                }
            }
          auto state_it = mem_rows[0].find ("lability_state");
          if (state_it != mem_rows[0].end ())
            {
              if (state_it->second.type () == typeid (double))
                {
                  stored_lability = std::any_cast<double> (state_it->second);
                }
              else if (state_it->second.type () == typeid (float))
                {
                  stored_lability = static_cast<double> (
                      std::any_cast<float> (state_it->second));
                }
            }
          auto ts_it = mem_rows[0].find ("lability_ts");
          if (ts_it != mem_rows[0].end ())
            {
              if (ts_it->second.type () == typeid (long long))
                {
                  lability_ts = std::any_cast<long long> (ts_it->second);
                }
              else if (ts_it->second.type () == typeid (int))
                {
                  lability_ts = std::any_cast<int> (ts_it->second);
                }
            }
        }
      if (memory_id <= 0)
        {
          continue;
        }

      processed_ids.insert (memory_id);

      const Eigen::VectorXf u_m = Unit (m);
      // contextual_relevance ∈ [0,1]
      double contextual_relevance = core::CosineSimilarity (u_m, u_cur);
      if (contextual_relevance < constants::kNormalizedMin)
        contextual_relevance = constants::kNormalizedMin;
      if (contextual_relevance > constants::kNormalizedMax)
        contextual_relevance = constants::kNormalizedMax;

      double current_lability = lability_susc;
      if (lability_ts > 0 && now_ts > lability_ts)
        {
          const double dt_s
              = static_cast<double> (now_ts - lability_ts) / 1000.0;
          const double base
              = (stored_lability > 0.0) ? stored_lability : lability_susc;
          current_lability
              = base
                * std::exp (-dt_s
                            / std::max (tau_labile,
                                        constants::kNormEpsilon));
        }
      else if (stored_lability > 0.0)
        {
          current_lability = stored_lability;
        }

      const double drift_mag = core::ReconsolidationPrimaryDriftMagnitude (
          F, S, T, current_lability, contextual_relevance, recon_mod_scale);
      max_drift = std::max (max_drift, drift_mag);

      // Note: embeddings row already exists - no need to ensure it exists

      // v2: Update lability_state, original_centroid, and lability_ts in
      // memories table (merged from memory_feedback)
      const auto lability_update_start = SteadyClock::now ();
      tx.Execute (
          "UPDATE memories "
          "SET lability_state = COALESCE(?, 0.0), "
          "    original_centroid = COALESCE(original_centroid, ?), "
          "    lability_ts = ? "
          "WHERE memory_id = ?",
          { current_lability, ToFloatVector (u_m), now_ts, memory_id });
      context.AddOperationTiming (
          "Reconsolidation.primary_lability_update_sql",
          ElapsedMillis (lability_update_start));

      if (drift_mag < drift_skip_epsilon)
        {
          continue; // No embedding update when drift too small
        }

      // Blend toward current context and normalize.
      Eigen::VectorXf blended = static_cast<float> (1.0 - drift_mag) * u_m
                                + static_cast<float> (drift_mag) * u_cur;
      blended = Unit (blended);

      const bool persist_primary_reconstruction
          = primary_reconstruction_considered < primary_reconstruction_limit;
      if (persist_primary_reconstruction)
        {
          ++primary_reconstruction_considered;
        }
      if (persist_primary_reconstruction && constructive_recall::Disabled ())
        {
          const auto primary_append_start = SteadyClock::now ();
          WriteEmbeddingInPlaceOrFork (tx, memory_id, embedding_id, blended,
                                       now_ts, &p_ctx);
          context.AddOperationTiming ("Reconsolidation.primary_append",
                                      ElapsedMillis (primary_append_start));
        }
      else if (persist_primary_reconstruction)
        {
          const auto primary_append_start = SteadyClock::now ();
          const auto blob_id
              = constructive_recall::LoadCurrentBlobId (tx, memory_id);
          const double uncertainty
              = core::Clamp (current_lability
                                 * (1.0 - uncertainty_relevance_weight
                                                * contextual_relevance),
                             0.0, 1.0);
          constructive_recall::AppendReconstructionWithEmbedding (
              tx, memory_id, blended, blob_id, now_ts, uncertainty,
              "reconsolidation", 1.0, contextual_relevance,
              reconstruction_update_policy, &p_ctx);
          if (auto current = constructive_recall::LoadCurrentEmbedding (
                  tx, memory_id, embedding_id,
                  static_cast<int> (blended.size ())))
            {
              const long long current_embedding_id = LoadCurrentSurfaceEmbeddingId (
                  tx, memory_id, embedding_id);
              const bool current_surface_written
                  = reconstruction_update_policy.update_current_surface
                    && !constructive_recall::
                           CurrentSurfaceWritesDisabled ();
              RefreshProcessorSurfaces (tx, &p_ctx, memory_id, embedding_id,
                                        current_embedding_id, *current,
                                        current_surface_written);
            }
          context.AddOperationTiming ("Reconsolidation.primary_append",
                                      ElapsedMillis (primary_append_start));
        }

      // --- BEGIN RIPPLE PROPAGATION ---
      // Only propagate ripple if store is available and primary drift occurred
      if (!store)
        {
          continue;
        }

      const auto neighbor_query_start = SteadyClock::now ();
      auto neighbors = QueryGraphNeighbors (store, p_ctx, memory_id,
                                            embedding_id, ripple_depth);
      context.AddOperationTiming ("Reconsolidation.neighbor_query",
                                  ElapsedMillis (neighbor_query_start));

      std::vector<LabilityUpdate> pending_lability_updates;
      pending_lability_updates.reserve (neighbors.size ());
      for (const auto &neighbor : neighbors)
        {
          // Skip if already processed (either as primary or via earlier ripple)
          if (processed_ids.count (neighbor.memory_id))
            {
              continue;
            }
          processed_ids.insert (neighbor.memory_id);

          // Compute decayed lability for this neighbor
          // neighbor_lability = source_lability * pow(ripple_decay, depth)
          double neighbor_lability
              = current_lability
                * std::pow (ripple_decay, static_cast<double> (neighbor.depth));

          // Skip if lability too low
          if (neighbor_lability < ripple_strength_min)
            {
              continue;
            }

          const bool persist_neighbor_reconstruction
              = ripple_reconstruction_attempts < ripple_reconstruction_limit;
          if (!persist_neighbor_reconstruction)
            {
              if (pending_lability_updates.size ()
                  < kReconsolidationLabilityOnlyLimit)
                {
                  pending_lability_updates.push_back (
                      { neighbor.memory_id, neighbor_lability });
                }
              continue;
            }

          // Load neighbor embedding
          const auto neighbor_load_start = SteadyClock::now ();
          auto neighbor_emb = constructive_recall::LoadCurrentEmbedding (
              tx, neighbor.memory_id, neighbor.embedding_id,
              static_cast<int> (u_cur.size ()));
          context.AddOperationTiming ("Reconsolidation.neighbor_load",
                                      ElapsedMillis (neighbor_load_start));
          if (!neighbor_emb)
            {
              continue;
            }

          // Compute neighbor drift (proportional to decayed lability)
          double neighbor_drift = neighbor_lability * recon_gain;
          // Cap ripple drift at reduced factor compared to primary
          neighbor_drift = std::min (
              drift_clamp * ripple_drift_cap_factor,
              std::max (drift_skip_epsilon, neighbor_drift));

          if (neighbor_drift < drift_skip_epsilon)
            {
              continue;
            }

          // Blend neighbor toward current context
          Eigen::VectorXf u_neighbor = Unit (*neighbor_emb);
          Eigen::VectorXf neighbor_blended
              = static_cast<float> (1.0 - neighbor_drift) * u_neighbor
                + static_cast<float> (neighbor_drift) * u_cur;
          neighbor_blended = Unit (neighbor_blended);

          // Write updates for this neighbor
          const auto neighbor_write_start = SteadyClock::now ();
          WriteNeighborUpdates (tx, neighbor.embedding_id, neighbor.memory_id,
                                neighbor_lability, now_ts, neighbor_blended,
                                reconstruction_update_policy, &p_ctx);
          ++ripple_reconstruction_attempts;
          context.AddOperationTiming ("Reconsolidation.neighbor_write",
                                      ElapsedMillis (neighbor_write_start));
        }
      if (!pending_lability_updates.empty ())
        {
          const auto neighbor_write_start = SteadyClock::now ();
          WriteNeighborLabilityBatch (tx, pending_lability_updates, now_ts);
          context.AddOperationTiming ("Reconsolidation.neighbor_lability_write",
                                      ElapsedMillis (neighbor_write_start));
        }
      // --- END RIPPLE PROPAGATION ---
    }

  (void)max_drift;

  telemetry::LogDebug ("cortext.reconsolidation", {
    telemetry::Attribute::Double ("recon_gain", recon_gain),
    telemetry::Attribute::Double ("recon_mod_scale", recon_mod_scale),
    telemetry::Attribute::Double ("max_drift", max_drift)
  });
}

} // namespace cortext::operations
