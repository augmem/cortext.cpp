#include "cortext/operations/graph_retrieval.hpp"

#include "cortext/store/store.hpp"
#include "cortext/core/algorithms.hpp"
#include "cortext/core/knobs.hpp"
#include "cortext/core/utils.hpp"
#include "cortext/operations/constants.hpp"
#include "cortext/processor/operation_context.hpp"
#include "cortext/store/store.hpp"
#include "cortext/telemetry/telemetry.hpp"
#include <algorithm>
#include <any>
#include <cmath>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace cortext::operations
{

namespace
{
/// @brief Constructs a graph node ID from an embedding ID.
std::string
EmbNodeId (long long embedding_id)
{
  return std::string ("emb:") + std::to_string (embedding_id);
}

/// @brief Creates reinforcement edges for co-retrieved memories.
/// Upserts 'reinforces' edges between all pairs of retrieved memories,
/// incrementing weight on existing edges.
void
CreateReinforcementEdges (Transaction &tx,
                          const std::vector<long long> &retrieved_ids,
                          long long now_ts)
{
  // Need at least 2 retrievals for co-retrieval edges
  if (retrieved_ids.size () < 2)
    {
      return;
    }

  // Create reinforcement edges for all pairs
  for (size_t i = 0; i < retrieved_ids.size (); ++i)
    {
      for (size_t j = i + 1; j < retrieved_ids.size (); ++j)
        {
          // Order IDs consistently (smaller first) for consistent edge direction
          long long id1 = std::min (retrieved_ids[i], retrieved_ids[j]);
          long long id2 = std::max (retrieved_ids[i], retrieved_ids[j]);

          tx.Execute (
              "INSERT INTO graph_edges "
              "(source_id, target_id, edge_type, weight, last_reinforced) "
              "VALUES ('emb:' || ?1, 'emb:' || ?2, 'reinforces', 1.0, ?3) "
              "ON CONFLICT (source_id, target_id, edge_type) DO UPDATE "
              "SET weight = weight + 1.0, last_reinforced = excluded.last_reinforced",
              { id1, id2, now_ts });
        }
    }
}
} // namespace

void
GraphAugmentedRetrieveCandidates::Execute (OperationContext &context, Transaction &tx) const
{
  // Check streaming pacing gate - skip retrieval if not triggered
  if (!context.GetShouldCheckRetrieval ())
    {
      return;
    }

  Store *store = context.GetStore ();
  if (!store)
    {
      return;
    }

  const auto &cfg = context.GetConfig ();
  auto &p_ctx = context.GetProcessorContext ();
  const auto &signal = context.GetSignal ();

  // Section 7.6: Use memory centroid (μ_acc) as query vector for both initial
  // retrieval and re-ranking. This ensures retrieved memories are ranked by
  // relevance to overall context rather than momentary signal fluctuations.
  const Eigen::VectorXf *q_ptr = nullptr;
  auto acc_it = p_ctx.accumulator_states.find (signal.source_id);
  if (acc_it != p_ctx.accumulator_states.end () && acc_it->second.n_signals > 0
      && acc_it->second.mu_acc.size () > 0)
    {
      q_ptr = &acc_it->second.mu_acc;
    }
  else
    {
      // Fallback to signal embedding if no accumulator state exists
      q_ptr = &signal.embedding;
    }
  const Eigen::VectorXf &q = *q_ptr;

  if (q.size () == 0)
    {
      return;
    }

  const int k = std::max (1, core::MaxResults (cfg.focus));
  const int depth = std::max (1, core::GraphDepth (cfg.focus));

  // Seed vector retrieval via sqlite-vec KNN query.
  struct Scored
  {
    long long id;
    double score;
    Eigen::VectorXf vec;
  };
  std::vector<Scored> seeds;
  seeds.reserve (static_cast<size_t> (k));

  // Convert query vector to std::vector<float> for parameter binding.
  std::vector<float> q_vec (q.data (), q.data () + q.size ());

  auto rows = store->Execute (
      "SELECT embedding_id, embedding, distance "
      "FROM embeddings "
      "WHERE embedding MATCH ? "
      "ORDER BY distance "
      "LIMIT ?",
      { q_vec, static_cast<long long> (k) });

  for (const auto &row : rows)
    {
      auto it_id = row.find ("embedding_id");
      auto it_emb = row.find ("embedding");
      auto it_dist = row.find ("distance");
      if (it_id == row.end () || it_emb == row.end ())
        continue;
      if (it_id->second.type () != typeid (long long))
        continue;

      const long long id = std::any_cast<long long> (it_id->second);
      Eigen::VectorXf v;
      if (!core::DecodeFloatBlob (it_emb->second, q.size (), v))
        continue;

      // sqlite-vec returns L2 distance; convert to similarity score.
      double dist = 0.0;
      if (it_dist != row.end () && it_dist->second.type () == typeid (double))
        dist = std::any_cast<double> (it_dist->second);
      const double sim = 1.0 / (1.0 + dist);

      seeds.push_back (Scored{ id, sim, v });
    }

  if (seeds.empty ())
    {
      return;
    }

  // Update last retrieval timestamp for consolidation idle-gating.
  p_ctx.last_retrieval_ts = signal.timestamp;

  // Graph expansion from seed embedding node IDs.
  std::vector<std::string> seed_nodes;
  seed_nodes.reserve (seeds.size ());
  for (const auto &s : seeds)
    {
      seed_nodes.push_back (EmbNodeId (s.id));
    }

  std::unordered_set<std::string> expanded_nodes;
  for (const auto &s : seed_nodes)
    {
      expanded_nodes.insert (s);
    }

  try
    {
      // Build SQL with a VALUES list for seeds.
      std::string sql = "WITH RECURSIVE seed(id) AS (VALUES ";
      std::vector<std::any> params;
      params.reserve (seed_nodes.size () + 2);
      for (size_t i = 0; i < seed_nodes.size (); ++i)
        {
          if (i > 0)
            sql += ", ";
          sql += "(?)";
          params.push_back (seed_nodes[i]);
        }
      sql += "), expand(id, depth) AS ("
             "  SELECT id, 0 FROM seed "
             "  UNION "
             "  SELECT ge.target_id, expand.depth+1 "
             "  FROM graph_edges ge JOIN expand ON ge.source_id = expand.id "
             "  WHERE expand.depth < ? "
             "  UNION "
             "  SELECT ge.source_id, expand.depth+1 "
             "  FROM graph_edges ge JOIN expand ON ge.target_id = expand.id "
             "  WHERE expand.depth < ? "
             ") "
             "SELECT DISTINCT id FROM expand;";
      params.push_back (static_cast<long long> (depth));
      params.push_back (static_cast<long long> (depth));

      auto rows = store->Execute (sql, params);
      for (const auto &r : rows)
        {
          auto it = r.find ("id");
          if (it != r.end () && it->second.type () == typeid (std::string))
            {
              expanded_nodes.insert (std::any_cast<std::string> (it->second));
            }
        }
    }
  catch (...)
    {
      // No graph tables or query failed; use seed-only behavior.
    }

  // Map expanded graph node ids to embedding_ids via graph_nodes (and by emb: prefix).
  std::unordered_set<long long> expanded_embedding_ids;
  expanded_embedding_ids.reserve (expanded_nodes.size ());
  for (const auto &n : expanded_nodes)
    {
      if (core::StartsWith (n, "emb:"))
        {
          const char *p = n.c_str () + 4;
          char *end = nullptr;
          const long long id = std::strtoll (p, &end, 10);
          if (end != p)
            expanded_embedding_ids.insert (id);
        }
    }

  // Also include any nodes that have embedding_id set.
  if (!expanded_nodes.empty ())
    {
      try
        {
          std::string sql = "SELECT embedding_id FROM graph_nodes WHERE node_id IN (";
          std::vector<std::any> params;
          params.reserve (expanded_nodes.size ());
          bool first = true;
          for (const auto &nid : expanded_nodes)
            {
              if (!first)
                sql += ",";
              first = false;
              sql += "?";
              params.push_back (nid);
            }
          sql += ") AND embedding_id IS NOT NULL;";

          auto rows = store->Execute (sql, params);
          for (const auto &r : rows)
            {
              auto it = r.find ("embedding_id");
              if (it != r.end () && it->second.type () == typeid (long long))
                {
                  expanded_embedding_ids.insert (
                      std::any_cast<long long> (it->second));
                }
            }
        }
      catch (...)
        {
        }
    }

  // Fetch embeddings for expanded ids and re-rank.
  std::unordered_map<long long, Eigen::VectorXf> out;
  std::vector<Scored> scored;
  scored.reserve (expanded_embedding_ids.size ());

  // If we cannot fetch, at least return seeds.
  bool fetched_any = false;
  if (!expanded_embedding_ids.empty ())
    {
      try
        {
          std::string sql = "SELECT embedding_id, embedding FROM embeddings WHERE embedding_id IN (";
          std::vector<std::any> params;
          params.reserve (expanded_embedding_ids.size ());
          bool first = true;
          for (const long long id : expanded_embedding_ids)
            {
              if (!first)
                sql += ",";
              first = false;
              sql += "?";
              params.push_back (id);
            }
          sql += ");";
          auto rows = store->Execute (sql, params);
          for (const auto &row : rows)
            {
              auto it_id = row.find ("embedding_id");
              auto it_emb = row.find ("embedding");
              if (it_id == row.end () || it_emb == row.end ())
                continue;
              if (it_id->second.type () != typeid (long long))
                continue;
              const long long id = std::any_cast<long long> (it_id->second);
              Eigen::VectorXf v;
              if (!core::DecodeFloatBlob (it_emb->second, q.size (), v))
                continue;
              const double sim = core::CosineSimilarity (q, v);
              scored.push_back (Scored{ id, sim, v });
              fetched_any = true;
            }
        }
      catch (...)
        {
        }
    }

  if (!fetched_any)
    {
      for (const auto &s : seeds)
        {
          out.emplace (s.id, s.vec);
        }
      context.SetRetrievedMemoryEmbeddings (std::move (out));

      // Create reinforcement edges for co-retrieved seeds
      {
        std::vector<long long> seed_ids;
        seed_ids.reserve (seeds.size ());
        for (const auto &s : seeds)
          {
            seed_ids.push_back (s.id);
          }
        CreateReinforcementEdges (
            tx, seed_ids,
            static_cast<long long> (signal.timestamp));
      }

      // Cache seeds for implicit feedback detection
      const uint64_t now_ts = signal.timestamp;
      auto &cache = p_ctx.recent_retrievals_cache;
      for (const auto &s : seeds)
        {
          cache.push_back ({ s.id, s.vec, now_ts });
        }
      while (cache.size () > ProcessorContext::kMaxRetrievalCacheSize)
        {
          cache.pop_front ();
        }
      return;
    }

  std::sort (scored.begin (), scored.end (),
             [] (const Scored &a, const Scored &b) {
               return a.score > b.score;
             });
  if (static_cast<int> (scored.size ()) > k)
    {
      scored.resize (static_cast<size_t> (k));
    }
  for (const auto &s : scored)
    {
      out.emplace (s.id, s.vec);
    }

  context.SetRetrievedMemoryEmbeddings (std::move (out));

  // Create reinforcement edges for co-retrieved memories
  // This strengthens connections between memories that are frequently
  // retrieved together.
  {
    std::vector<long long> retrieved_ids;
    retrieved_ids.reserve (scored.size ());
    for (const auto &s : scored)
      {
        retrieved_ids.push_back (s.id);
      }
    CreateReinforcementEdges (
        tx, retrieved_ids,
        static_cast<long long> (signal.timestamp));
  }

  // Cache retrieved embeddings for implicit feedback detection
  // (DetectMemoryUsage will compare future signals against these cached
  // retrievals)
  const uint64_t now_ts = signal.timestamp;
  auto &cache = p_ctx.recent_retrievals_cache;
  for (const auto &s : scored)
    {
      cache.push_back (
          { s.id, s.vec, now_ts });
    }
  // Trim cache to max size (FIFO eviction)
  while (cache.size () > ProcessorContext::kMaxRetrievalCacheSize)
    {
      cache.pop_front ();
    }

  // Debug logging
  telemetry::LogDebug ("cortext.graph_retrieval", {
    telemetry::Attribute::Int64 ("seed_count", static_cast<int64_t> (seeds.size ())),
    telemetry::Attribute::Int64 ("expansion_depth", static_cast<int64_t> (depth)),
    telemetry::Attribute::Int64 ("final_candidate_count", static_cast<int64_t> (scored.size ()))
  });
}

} // namespace cortext::operations

