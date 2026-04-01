#include "cortext/operations/graph_build.hpp"

#include "cortext/store/store.hpp"
#include "cortext/core/algorithms.hpp"
#include "cortext/core/knobs.hpp"
#include "cortext/core/utils.hpp"
#include "cortext/processor/operation_context.hpp"
#include "cortext/telemetry/telemetry.hpp"
#include <Eigen/Dense>
#include <any>
#include <cmath>
#include <map>
#include <string>
#include <vector>

namespace cortext::operations
{

namespace
{
/// @brief Executes a write query on the transaction.
void
Add (Transaction &tx, const std::string &q,
     const std::vector<std::any> &p = {})
{
  tx.Execute (q, p);
}

/// @brief Data structure holding memory info for graph edge construction.
struct MemoryData
{
  long long memory_id;
  long long embedding_id;
  long long created_at;
  double boundary_score;
  Eigen::VectorXf embedding;
};

/// @brief Load memories for cluster-based processing.
/// Groups by cluster_id for building edges within clusters.
std::map<int, std::vector<MemoryData>>
LoadClusterMemories (Store *store)
{
  std::map<int, std::vector<MemoryData>> clusters;

  // Query memories with cluster_id, ordered by timestamp
  auto rows = store->Execute (
      "SELECT m.memory_id, m.embedding_id, m.cluster_id, "
      "COALESCE(m.end_ts, m.start_ts, m.created_at, 0) AS created_at, "
      "COALESCE(m.boundary_score, 0.0) AS boundary_score, "
      "e.embedding "
      "FROM memories m "
      "JOIN embeddings e ON m.embedding_id = e.embedding_id "
      "WHERE m.cluster_id IS NOT NULL "
      "ORDER BY m.cluster_id, created_at",
      {});

  constexpr int kEmbeddingDim = 256;

  for (const auto &row : rows)
    {
      auto it_mem_id = row.find ("memory_id");
      auto it_emb_id = row.find ("embedding_id");
      auto it_cluster = row.find ("cluster_id");
      auto it_ts = row.find ("created_at");
      auto it_boundary = row.find ("boundary_score");
      auto it_emb = row.find ("embedding");

      if (it_mem_id == row.end () || it_emb_id == row.end ()
          || it_cluster == row.end () || it_emb == row.end ())
        {
          continue;
        }

      MemoryData data;

      // Parse cluster_id
      int cluster_id = 0;
      if (it_cluster->second.type () == typeid (long long))
        {
          cluster_id = static_cast<int> (
              std::any_cast<long long> (it_cluster->second));
        }
      else if (it_cluster->second.type () == typeid (int))
        {
          cluster_id = std::any_cast<int> (it_cluster->second);
        }
      else
        {
          continue;
        }

      // Parse memory_id
      if (it_mem_id->second.type () == typeid (long long))
        {
          data.memory_id = std::any_cast<long long> (it_mem_id->second);
        }
      else
        {
          continue;
        }

      // Parse embedding_id
      if (it_emb_id->second.type () == typeid (long long))
        {
          data.embedding_id = std::any_cast<long long> (it_emb_id->second);
        }
      else
        {
          continue;
        }

      // Parse timestamp
      if (it_ts != row.end () && it_ts->second.type () == typeid (long long))
        {
          data.created_at = std::any_cast<long long> (it_ts->second);
        }
      else
        {
          data.created_at = 0;
        }

      if (it_boundary != row.end () && it_boundary->second.type () == typeid (double))
        {
          data.boundary_score = std::any_cast<double> (it_boundary->second);
        }
      else
        {
          data.boundary_score = 0.0;
        }

      // Decode embedding
      if (!core::DecodeFloatBlob (it_emb->second, kEmbeddingDim, data.embedding))
        {
          continue;
        }

      clusters[cluster_id].push_back (std::move (data));
    }

  return clusters;
}

/// @brief Build co-occurrence edges within clusters.
/// Creates 'co_occurs' edges in ASSOCIATIONS between memories
/// with cosine similarity above threshold.
void
BuildCoOccurrenceEdges (Transaction &tx,
                        const std::map<int, std::vector<MemoryData>> &clusters,
                        double threshold, double similar_threshold,
                        long long now_ts)
{
  for (const auto &[cluster_id, memories] : clusters)
    {
      // Need at least 2 memories for co-occurrence
      if (memories.size () < 2)
        {
          continue;
        }

      // Compare all pairs within cluster
      for (size_t i = 0; i < memories.size (); ++i)
        {
          for (size_t j = i + 1; j < memories.size (); ++j)
            {
              double sim = core::CosineSimilarity (memories[i].embedding,
                                                   memories[j].embedding);

              const double weight01 = core::Map01 (sim);
              // Create co_occurs edge (use smaller id as source)
              if (sim > threshold)
                {
                  long long id1 = std::min (memories[i].memory_id,
                                            memories[j].memory_id);
                  long long id2 = std::max (memories[i].memory_id,
                                            memories[j].memory_id);

                  Add (tx,
                       "INSERT OR REPLACE INTO associations "
                       "(source_memory_id, target_memory_id, edge_type, weight, "
                       "last_reinforced) "
                       "VALUES (?, ?, 'co_occurs', ?, ?)",
                       { id1, id2, weight01, now_ts });
                }

              // Create similar_to edge for very high similarity
              if (sim > similar_threshold)
                {
                  long long id1 = std::min (memories[i].memory_id,
                                            memories[j].memory_id);
                  long long id2 = std::max (memories[i].memory_id,
                                            memories[j].memory_id);

                  Add (tx,
                       "INSERT OR REPLACE INTO associations "
                       "(source_memory_id, target_memory_id, edge_type, weight, "
                       "last_reinforced) "
                       "VALUES (?, ?, 'similar_to', ?, ?)",
                       { id1, id2, weight01, now_ts });
                }
            }
        }
    }
}

/// @brief Build causal/temporal edges within clusters.
/// Creates 'causes' edges in ASSOCIATIONS between temporally adjacent
/// memories when drift magnitude exceeds threshold.
void
BuildCausalEdges (Transaction &tx,
                  const std::map<int, std::vector<MemoryData>> &clusters,
                  double drift_threshold, double implies_threshold,
                  long long now_ts)
{
  for (const auto &[cluster_id, memories] : clusters)
    {
      // Need at least 2 memories for causal edges
      if (memories.size () < 2)
        {
          continue;
        }

      // Memories are already ordered by timestamp from the query
      for (size_t i = 0; i + 1 < memories.size (); ++i)
        {
          // Compute drift vector
          Eigen::VectorXf drift
              = memories[i + 1].embedding - memories[i].embedding;
          double drift_mag = drift.norm ();

          const double weight01
              = core::Clamp (drift_mag / 2.0, 0.0, 1.0);
          if (drift_mag > drift_threshold)
            {
              // Create causes edge (directional: earlier -> later)
              Add (tx,
                   "INSERT OR REPLACE INTO associations "
                   "(source_memory_id, target_memory_id, edge_type, weight, "
                   "last_reinforced) "
                   "VALUES (?, ?, 'causes', ?, ?)",
                   { memories[i].memory_id, memories[i + 1].memory_id,
                     weight01, now_ts });
            }
          else if (drift_mag > implies_threshold)
            {
              // Create implies edge for softer directional drift
              Add (tx,
                   "INSERT OR REPLACE INTO associations "
                   "(source_memory_id, target_memory_id, edge_type, weight, "
                   "last_reinforced) "
                   "VALUES (?, ?, 'implies', ?, ?)",
                   { memories[i].memory_id, memories[i + 1].memory_id,
                     weight01, now_ts });
            }
        }
    }
}

/// @brief Build sequential edges within clusters.
void
BuildSequentialEdges (Transaction &tx,
                      const std::map<int, std::vector<MemoryData>> &clusters,
                      double tau_s, long long now_ts)
{
  for (const auto &[cluster_id, memories] : clusters)
    {
      (void)cluster_id;
      if (memories.size () < 2)
        {
          continue;
        }
      for (size_t i = 0; i + 1 < memories.size (); ++i)
        {
          const auto &m_i = memories[i];
          const auto &m_j = memories[i + 1];
          const double gap_s = std::max (
              0.0, static_cast<double> (m_j.created_at - m_i.created_at) / 1000.0);
          const double w_seq = core::Clamp (
              std::exp (-gap_s / std::max (tau_s, 1e-6))
                  * (1.0 - core::Clamp (m_j.boundary_score, 0.0, 1.0)),
              0.0, 1.0);
          Add (tx,
               "INSERT OR REPLACE INTO associations "
               "(source_memory_id, target_memory_id, edge_type, weight, "
               "last_reinforced) "
               "VALUES (?, ?, 'next_in_episode', ?, ?)",
               { m_i.memory_id, m_j.memory_id, w_seq, now_ts });
          Add (tx,
               "INSERT OR REPLACE INTO associations "
               "(source_memory_id, target_memory_id, edge_type, weight, "
               "last_reinforced) "
               "VALUES (?, ?, 'prev_in_episode', ?, ?)",
               { m_j.memory_id, m_i.memory_id, w_seq, now_ts });
          if (m_j.boundary_score < 0.3)
            {
              Add (tx,
                   "INSERT OR REPLACE INTO associations "
                   "(source_memory_id, target_memory_id, edge_type, weight, "
                   "last_reinforced) "
                   "VALUES (?, ?, 'within_same_event', ?, ?)",
                   { m_i.memory_id, m_j.memory_id, w_seq, now_ts });
            }
        }
    }
}

/// @brief Build contradiction edges within clusters.
/// Creates 'contradicts' edges in ASSOCIATIONS between memories
/// with cosine similarity below threshold (negative similarity).
void
BuildContradictionEdges (Transaction &tx,
                         const std::map<int, std::vector<MemoryData>> &clusters,
                         double threshold, long long now_ts)
{
  for (const auto &[cluster_id, memories] : clusters)
    {
      if (memories.size () < 2)
        {
          continue;
        }

      for (size_t i = 0; i < memories.size (); ++i)
        {
          for (size_t j = i + 1; j < memories.size (); ++j)
            {
              double sim = core::CosineSimilarity (memories[i].embedding,
                                                   memories[j].embedding);

              if (sim < threshold)
                {
                  const double weight01 = core::Map01 (sim);
                  // Create contradicts edge with normalized weight
                  long long id1 = std::min (memories[i].memory_id,
                                            memories[j].memory_id);
                  long long id2 = std::max (memories[i].memory_id,
                                            memories[j].memory_id);

                  Add (tx,
                       "INSERT OR REPLACE INTO associations "
                       "(source_memory_id, target_memory_id, edge_type, weight, "
                       "last_reinforced) "
                       "VALUES (?, ?, 'contradicts', ?, ?)",
                       { id1, id2, weight01, now_ts });
                }
            }
        }
    }
}

/// @brief Apply decay to reinforcement edges.
/// Decays edge weights by decay_rate factor.
void
DecayReinforcementEdges (Transaction &tx, double decay_rate)
{
  Add (tx,
       "UPDATE associations SET weight = weight * ?1 "
       "WHERE edge_type = 'reinforces'",
       { decay_rate });

  // Remove edges that have decayed below minimum threshold
  Add (tx,
       "DELETE FROM associations "
       "WHERE edge_type = 'reinforces' AND weight < 0.1",
       {});
}

} // namespace

void
BuildGraphFromConsolidation::Execute (OperationContext &context, Transaction &tx) const
{
  if (!context.GetConsolidationShouldStart ())
    {
      return;
    }
  Store *store = context.GetStore ();
  const auto &cfg = context.GetConfig ();
  const long long now_ts
      = static_cast<long long> (context.GetSignal ().timestamp);

  // Derive knob-based thresholds
  const double co_occurrence_threshold = core::CoOccurrenceThreshold (cfg.focus);
  const double similar_threshold = core::SimilarToThreshold (cfg.focus);
  const double causal_drift_threshold = core::CausalDriftThreshold (cfg.stability);
  const double implies_drift_threshold = core::ImpliesDriftThreshold (cfg.stability);
  const double contradiction_threshold = core::ContradictionThreshold ();
  const double reinforcement_decay = core::ReinforcementDecay (cfg.stability);
  const double tau_seq = core::Lerp (10.0, 60.0, cfg.stability);

  // V2 schema: Nodes are MEMORIES, edges are ASSOCIATIONS
  // No need to create graph_nodes or graph_edges tables - using V2 tables

  // 1) Build edges from MEMORIES with cluster_id (set by ConsolidationSummarize)
  if (store)
    {
      auto clusters = LoadClusterMemories (store);

      // 2) Co-occurrence edges: memory <-> memory within clusters
      // Creates 'co_occurs' edges for highly similar memories
      BuildCoOccurrenceEdges (tx, clusters, co_occurrence_threshold,
                              similar_threshold, now_ts);

      // 3) Causal edges: memory -> memory based on temporal drift
      // Creates 'causes' edges for significant semantic drift over time
      BuildCausalEdges (tx, clusters, causal_drift_threshold,
                        implies_drift_threshold, now_ts);

      // 4) Contradiction edges: memory <-> memory for opposing semantics
      // Creates 'contradicts' edges for strong negative similarity
      BuildContradictionEdges (tx, clusters, contradiction_threshold, now_ts);

      // 5) Sequential edges: preserve episode order
      if (cfg.sequential_edges_enabled)
        {
          BuildSequentialEdges (tx, clusters, tau_seq, now_ts);
        }
    }

  // 6) Decay reinforcement edges (created during retrieval)
  if (cfg.reinforcement_enabled)
    {
      DecayReinforcementEdges (tx, reinforcement_decay);
    }

  // Count associations for logging
  long long edges_count = 0;

  if (store)
    {
      auto edge_rows = store->Execute (
          "SELECT COUNT(*) AS cnt FROM associations", {});
      if (!edge_rows.empty () && edge_rows[0].count ("cnt") == 1)
        {
          const auto &v = edge_rows[0].at ("cnt");
          if (v.type () == typeid (long long))
            edges_count = std::any_cast<long long> (v);
        }
    }

  telemetry::LogDebug ("cortext.graph_build", {
      telemetry::Attribute::Int64 ("associations_count", edges_count)
  });
}

} // namespace cortext::operations
