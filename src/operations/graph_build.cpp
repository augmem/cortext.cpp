#include "cortext/operations/graph_build.hpp"

#include "cortext/buffered_write_instruction.hpp"
#include "cortext/core/algorithms.hpp"
#include "cortext/core/knobs.hpp"
#include "cortext/core/utils.hpp"
#include "cortext/processor/operation_context.hpp"
#include "cortext/store/store.hpp"
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
/// @brief Adds a buffered write instruction to the context.
void
Add (OperationContext &ctx, const std::string &q,
     const std::vector<std::any> &p = {})
{
  BufferedWriteInstruction op;
  op.query = q;
  op.params = p;
  ctx.AddWriteInstruction (std::move (op));
}

/// @brief Data structure holding embedding info for graph edge construction.
struct EmbeddingData
{
  long long embedding_id;
  long long created_at;
  Eigen::VectorXf embedding;
};

/// @brief Load embeddings for all source memories in consolidation_sources.
/// Groups by summary_id for cluster-based processing.
std::map<std::string, std::vector<EmbeddingData>>
LoadClusterSourceEmbeddings (Store *store)
{
  std::map<std::string, std::vector<EmbeddingData>> clusters;

  // Query sources with timestamps and embeddings
  auto rows = store->Execute (
      "SELECT cs.summary_id, cs.source_embedding_id, "
      "COALESCE(m.timestamp, e.created_at, 0) AS created_at, "
      "e.embedding "
      "FROM consolidation_sources cs "
      "JOIN embeddings e ON cs.source_embedding_id = e.embedding_id "
      "LEFT JOIN memories m ON cs.source_embedding_id = m.embedding_id "
      "WHERE cs.source_embedding_id IS NOT NULL "
      "ORDER BY cs.summary_id, created_at",
      {});

  constexpr int kEmbeddingDim = 256;

  for (const auto &row : rows)
    {
      auto it_summary = row.find ("summary_id");
      auto it_id = row.find ("source_embedding_id");
      auto it_ts = row.find ("created_at");
      auto it_emb = row.find ("embedding");

      if (it_summary == row.end () || it_id == row.end ()
          || it_emb == row.end ())
        {
          continue;
        }

      EmbeddingData data;

      // Parse summary_id
      if (it_summary->second.type () == typeid (std::string))
        {
          std::string summary_id
              = std::any_cast<std::string> (it_summary->second);

          // Parse embedding_id
          if (it_id->second.type () == typeid (long long))
            {
              data.embedding_id = std::any_cast<long long> (it_id->second);
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

          // Decode embedding
          if (!core::DecodeFloatBlob (it_emb->second, kEmbeddingDim,
                                      data.embedding))
            {
              continue;
            }

          clusters[summary_id].push_back (std::move (data));
        }
    }

  return clusters;
}

/// @brief Build co-occurrence edges within clusters.
/// Creates 'co_occurs_with' edges between source memories with
/// cosine similarity above MergeThreshold(F).
void
BuildCoOccurrenceEdges (OperationContext &ctx,
                        const std::map<std::string, std::vector<EmbeddingData>> &clusters,
                        double threshold, long long now_ts)
{
  for (const auto &[summary_id, sources] : clusters)
    {
      // Need at least 2 sources for co-occurrence
      if (sources.size () < 2)
        {
          continue;
        }

      // Compare all pairs within cluster
      for (size_t i = 0; i < sources.size (); ++i)
        {
          for (size_t j = i + 1; j < sources.size (); ++j)
            {
              double sim = core::CosineSimilarity (sources[i].embedding,
                                                   sources[j].embedding);

              if (sim > threshold)
                {
                  // Create co_occurs_with edge (bidirectional via smaller id first)
                  long long id1 = std::min (sources[i].embedding_id,
                                            sources[j].embedding_id);
                  long long id2 = std::max (sources[i].embedding_id,
                                            sources[j].embedding_id);

                  Add (ctx,
                       "INSERT OR REPLACE INTO graph_edges"
                       "(source_id, target_id, edge_type, weight, last_reinforced) "
                       "VALUES ('emb:' || ?1, 'emb:' || ?2, 'co_occurs_with', ?3, ?4)",
                       { id1, id2, sim, now_ts });
                }
            }
        }
    }
}

/// @brief Build causal/temporal edges within clusters.
/// Creates 'causes' edges between temporally adjacent source memories
/// when drift magnitude exceeds CausalDriftThreshold(T).
void
BuildCausalEdges (OperationContext &ctx,
                  const std::map<std::string, std::vector<EmbeddingData>> &clusters,
                  double drift_threshold, long long now_ts)
{
  for (const auto &[summary_id, sources] : clusters)
    {
      // Need at least 2 sources for causal edges
      if (sources.size () < 2)
        {
          continue;
        }

      // Sources are already ordered by timestamp from the query
      for (size_t i = 0; i + 1 < sources.size (); ++i)
        {
          // Compute drift vector: m_j.embedding - m_i.embedding
          Eigen::VectorXf drift = sources[i + 1].embedding - sources[i].embedding;
          double drift_mag = drift.norm ();

          if (drift_mag > drift_threshold)
            {
              // Create causes edge (directional: earlier -> later)
              Add (ctx,
                   "INSERT OR REPLACE INTO graph_edges"
                   "(source_id, target_id, edge_type, weight, last_reinforced) "
                   "VALUES ('emb:' || ?1, 'emb:' || ?2, 'causes', ?3, ?4)",
                   { sources[i].embedding_id, sources[i + 1].embedding_id,
                     drift_mag, now_ts });
            }
        }
    }
}

/// @brief Build contradiction edges within clusters.
/// Creates 'contradicts' edges between source memories with
/// cosine similarity below ContradictionThreshold (-0.5).
void
BuildContradictionEdges (OperationContext &ctx,
                         const std::map<std::string, std::vector<EmbeddingData>> &clusters,
                         double threshold, long long now_ts)
{
  for (const auto &[summary_id, sources] : clusters)
    {
      if (sources.size () < 2)
        {
          continue;
        }

      for (size_t i = 0; i < sources.size (); ++i)
        {
          for (size_t j = i + 1; j < sources.size (); ++j)
            {
              double sim = core::CosineSimilarity (sources[i].embedding,
                                                   sources[j].embedding);

              if (sim < threshold)
                {
                  // Create contradicts edge with weight = |similarity|
                  long long id1 = std::min (sources[i].embedding_id,
                                            sources[j].embedding_id);
                  long long id2 = std::max (sources[i].embedding_id,
                                            sources[j].embedding_id);

                  Add (ctx,
                       "INSERT OR REPLACE INTO graph_edges"
                       "(source_id, target_id, edge_type, weight, last_reinforced) "
                       "VALUES ('emb:' || ?1, 'emb:' || ?2, 'contradicts', ?3, ?4)",
                       { id1, id2, std::abs (sim), now_ts });
                }
            }
        }
    }
}

/// @brief Apply decay to reinforcement edges.
/// Decays edge weights by ReinforcementDecay(T) factor.
void
DecayReinforcementEdges (OperationContext &ctx, double decay_rate)
{
  Add (ctx,
       "UPDATE graph_edges SET weight = weight * ?1 "
       "WHERE edge_type = 'reinforces'",
       { decay_rate });

  // Remove edges that have decayed below minimum threshold
  Add (ctx,
       "DELETE FROM graph_edges "
       "WHERE edge_type = 'reinforces' AND weight < 0.1",
       {});
}

} // namespace

void
BuildGraphFromConsolidation::Execute (OperationContext &context) const
{
  Store *store = context.GetStore ();
  const auto &cfg = context.GetConfig ();
  const long long now_ts
      = static_cast<long long> (context.GetSignal ().timestamp);

  // Derive knob-based thresholds
  const double co_occurrence_threshold = core::CoOccurrenceThreshold (cfg.focus);
  const double causal_drift_threshold = core::CausalDriftThreshold (cfg.stability);
  const double contradiction_threshold = core::ContradictionThreshold ();
  const double reinforcement_decay = core::ReinforcementDecay (cfg.stability);

  // Ensure required tables exist. (Safe to emit repeatedly.)
  Add (context,
       "CREATE TABLE IF NOT EXISTS graph_nodes ("
       "  node_id TEXT PRIMARY KEY,"
       "  type TEXT NOT NULL,"
       "  label TEXT,"
       "  embedding_id INTEGER,"
       "  metadata TEXT,"
       "  created_at INTEGER"
       ");");
  Add (context,
       "CREATE TABLE IF NOT EXISTS graph_edges ("
       "  source_id TEXT NOT NULL,"
       "  target_id TEXT NOT NULL,"
       "  edge_type TEXT NOT NULL,"
       "  weight REAL NOT NULL DEFAULT 1.0,"
       "  decay_rate REAL,"
       "  last_reinforced INTEGER,"
       "  PRIMARY KEY (source_id, target_id, edge_type)"
       ");");
  Add (context,
       "CREATE TABLE IF NOT EXISTS entity_index ("
       "  name TEXT PRIMARY KEY,"
       "  node_id TEXT NOT NULL"
       ");");

  // 1) Create nodes for all summaries.
  //    Node id: summary:<summary_id>
  Add (context,
       "INSERT OR IGNORE INTO graph_nodes(node_id, type, label, created_at) "
       "SELECT 'summary:' || summary_id, 'summary', summary_id, ?1 "
       "FROM consolidation_summaries;",
       { now_ts });

  // 2) Ensure entity nodes exist for extraction_entities.
  Add (context,
       "INSERT OR IGNORE INTO graph_nodes(node_id, type, label, embedding_id, "
       "created_at) "
       "SELECT 'entity:' || name, 'entity', name, embedding_id, ?1 "
       "FROM extraction_entities;",
       { now_ts });

  // 2b) Maintain mapping table (name -> node_id).
  Add (context,
       "INSERT OR IGNORE INTO entity_index(name, node_id) "
       "SELECT name, 'entity:' || name FROM extraction_entities;");

  // 3) Mentions edges: summary -> entity, weight=salience
  Add (context,
       "INSERT OR REPLACE INTO graph_edges(source_id, target_id, edge_type, weight, "
       "last_reinforced) "
       "SELECT 'summary:' || e.summary_id, 'entity:' || e.name, 'mentions', "
       "COALESCE(e.salience, 1.0), ?1 "
       "FROM extraction_entities e;",
       { now_ts });

  // 4) Relation edges: entity(subject) -> entity(object), edge_type=predicate
  // Ensure both endpoints exist.
  Add (context,
       "INSERT OR IGNORE INTO graph_nodes(node_id, type, label, created_at) "
       "SELECT 'entity:' || subject, 'entity', subject, ?1 "
       "FROM extraction_relations;",
       { now_ts });
  Add (context,
       "INSERT OR IGNORE INTO graph_nodes(node_id, type, label, created_at) "
       "SELECT 'entity:' || object, 'entity', object, ?1 "
       "FROM extraction_relations;",
       { now_ts });

  // 4a) Non-implication relation edges: preserve original predicate as edge_type
  Add (context,
       "INSERT OR REPLACE INTO graph_edges(source_id, target_id, edge_type, weight, "
       "last_reinforced) "
       "SELECT 'entity:' || subject, 'entity:' || object, predicate, "
       "COALESCE(confidence, 1.0), ?1 "
       "FROM extraction_relations "
       "WHERE lower(predicate) NOT LIKE '%implies%' "
       "AND lower(predicate) NOT LIKE '%suggests%' "
       "AND lower(predicate) NOT LIKE '%indicates%' "
       "AND lower(predicate) NOT LIKE '%means%' "
       "AND lower(predicate) NOT LIKE '%entails%';",
       { now_ts });

  // 4b) Implication relation edges: normalize to 'implies' edge type
  // Per Section 9.5: implies captures directional correlation distinct from
  // causes (which uses temporal drift). Implication keywords are normalized.
  Add (context,
       "INSERT OR REPLACE INTO graph_edges(source_id, target_id, edge_type, weight, "
       "last_reinforced) "
       "SELECT 'entity:' || subject, 'entity:' || object, 'implies', "
       "COALESCE(confidence, 1.0), ?1 "
       "FROM extraction_relations "
       "WHERE lower(predicate) LIKE '%implies%' "
       "OR lower(predicate) LIKE '%suggests%' "
       "OR lower(predicate) LIKE '%indicates%' "
       "OR lower(predicate) LIKE '%means%' "
       "OR lower(predicate) LIKE '%entails%';",
       { now_ts });

  // 5) Derived-from edges: summary -> embedded memory id (if available).
  Add (context,
       "INSERT OR IGNORE INTO graph_nodes(node_id, type, label, embedding_id, "
       "created_at) "
       "SELECT 'emb:' || source_embedding_id, 'memory', NULL, source_embedding_id, "
       "?1 "
       "FROM consolidation_sources "
       "WHERE source_embedding_id IS NOT NULL;",
       { now_ts });

  Add (context,
       "INSERT OR REPLACE INTO graph_edges(source_id, target_id, edge_type, weight, "
       "last_reinforced) "
       "SELECT 'summary:' || summary_id, 'emb:' || source_embedding_id, "
       "'derived_from', 1.0, ?1 "
       "FROM consolidation_sources "
       "WHERE source_embedding_id IS NOT NULL;",
       { now_ts });

  // --- Phase 4: Knowledge Graph Enhancement ---

  // 6) Load cluster source embeddings for new edge types
  if (store)
    {
      auto clusters = LoadClusterSourceEmbeddings (store);

      // 7) Co-occurrence edges: memory <-> memory within clusters
      // Creates 'co_occurs_with' edges for highly similar memories
      BuildCoOccurrenceEdges (context, clusters, co_occurrence_threshold, now_ts);

      // 8) Causal edges: memory -> memory based on temporal drift
      // Creates 'causes' edges for significant semantic drift over time
      BuildCausalEdges (context, clusters, causal_drift_threshold, now_ts);

      // 9) Contradiction edges: memory <-> memory for opposing semantics
      // Creates 'contradicts' edges for strong negative similarity
      BuildContradictionEdges (context, clusters, contradiction_threshold, now_ts);
    }

  // 10) Decay reinforcement edges (created during retrieval)
  DecayReinforcementEdges (context, reinforcement_decay);
}

} // namespace cortext::operations
