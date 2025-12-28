#include "cortext/operations/graph_retrieval.hpp"

#include "cortext/store/store.hpp"
#include "cortext/core/algorithms.hpp"
#include "cortext/core/knobs.hpp"
#include "cortext/core/utils.hpp"
#include "cortext/core/sparse.hpp"
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

  // Create reinforcement edges for all pairs using ASSOCIATIONS table
  for (size_t i = 0; i < retrieved_ids.size (); ++i)
    {
      for (size_t j = i + 1; j < retrieved_ids.size (); ++j)
        {
          constexpr double kReinforcementStep = 0.1;
          // Order IDs consistently (smaller first) for consistent edge direction
          long long id1 = std::min (retrieved_ids[i], retrieved_ids[j]);
          long long id2 = std::max (retrieved_ids[i], retrieved_ids[j]);

          tx.Execute (
              "INSERT INTO associations "
              "(source_memory_id, target_memory_id, edge_type, weight, last_reinforced) "
              "VALUES (?1, ?2, 'reinforces', ?3, ?4) "
              "ON CONFLICT (source_memory_id, target_memory_id, edge_type) DO UPDATE "
              "SET weight = MIN(weight + excluded.weight, 1.0), "
              "    last_reinforced = excluded.last_reinforced",
              { id1, id2, kReinforcementStep, now_ts });
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

  if (p_ctx.memory_stream.empty ())
    {
      return;
    }

  // Section 7.6: Use the current accumulator centroid (μ_acc) as query.
  // Cache prior to any accumulator reset to keep retrieval grounded in the
  // current unit's evolving context.
  auto acc_it = p_ctx.accumulator_states.find (signal.source_id);
  if (acc_it == p_ctx.accumulator_states.end ()
      || acc_it->second.mu_acc.size () == 0)
    {
      return;
    }
  Eigen::VectorXf q = acc_it->second.mu_acc;
  Eigen::VectorXf q_ctx
      = (acc_it->second.c_t.size () > 0) ? acc_it->second.c_t : q;

  const double ctx_mix = core::RetrievalContextMix (cfg.focus);
  if (ctx_mix > 0.0 && !p_ctx.recent_context_embeddings.empty ()
      && q.size () > 0)
    {
      Eigen::VectorXf mean_ctx = Eigen::VectorXf::Zero (q.size ());
      int count = 0;
      for (const auto &v : p_ctx.recent_context_embeddings)
        {
          if (v.size () == q.size ())
            {
              mean_ctx += v;
              ++count;
            }
        }
      if (count > 0)
        {
          mean_ctx /= static_cast<float> (count);
          q = q * static_cast<float> (1.0 - ctx_mix)
              + mean_ctx * static_cast<float> (ctx_mix);
        }
    }

  if (q.size () == 0)
    {
      return;
    }
  const float q_norm = q.norm ();
  if (q_norm > 0.0f)
    {
      q /= q_norm;
    }
  if (q_ctx.size () > 0)
    {
      const float ctx_norm = q_ctx.norm ();
      if (ctx_norm > 1e-9f)
        {
          q_ctx /= ctx_norm;
        }
    }

  const int k = std::max (1, core::MaxResults (cfg.focus));
  const int depth = std::max (1, core::GraphDepth (cfg.stability));
  const double min_edge_weight = core::MinEdgeWeight (cfg.focus);
  const int k_key = core::SparseKeySize (cfg.focus);
  const std::string sparse_key = core::SparseKey (q, k_key);

  // Seed vector retrieval via sqlite-vec KNN query.
  struct Scored
  {
    long long embedding_id;
    long long memory_id;
    long long created_at;
    double score;
    double ctx_score;
    double source_confidence;
    double proc_score;
    int source_contradictions;
    Eigen::VectorXf vec;
    Eigen::VectorXf ctx;
  };
  std::vector<Scored> seeds;
  seeds.reserve (static_cast<size_t> (k));

  // Convert query vector to std::vector<float> for parameter binding.
  std::vector<float> q_vec (q.data (), q.data () + q.size ());

  // V2: Query embeddings via KNN first, then join with memories for memory_id
  // sqlite-vec requires simple queries for MATCH, so we do KNN first
  auto rows = store->Execute (
      "SELECT embedding_id, embedding, distance, created_at "
      "FROM embeddings "
      "WHERE embedding MATCH ? AND k = ?",
      { q_vec, static_cast<long long> (k) });

  for (const auto &row : rows)
    {
      auto it_emb_id = row.find ("embedding_id");
      auto it_emb = row.find ("embedding");
      auto it_dist = row.find ("distance");
      auto it_created = row.find ("created_at");
      if (it_emb_id == row.end () || it_emb == row.end ())
        continue;
      if (it_emb_id->second.type () != typeid (long long))
        continue;

      const long long emb_id = std::any_cast<long long> (it_emb_id->second);
      Eigen::VectorXf v;
      if (!core::DecodeFloatBlob (it_emb->second, q.size (), v))
        continue;

      // Prefer cosine similarity for scoring to align with manuscript logic.
      double sim = 0.0;
      if (v.size () == q.size () && v.size () > 0)
        {
          sim = core::CosineSimilarity (q, v);
        }
      else if (it_dist != row.end () && it_dist->second.type () == typeid (double))
        {
          const double dist = std::any_cast<double> (it_dist->second);
          sim = 1.0 / (1.0 + dist);
        }

      // V2: Look up memory_id from memories table
      long long mem_id = 0;
      auto mem_rows = store->Execute (
          "SELECT memory_id FROM memories WHERE embedding_id = ?", { emb_id });
      if (!mem_rows.empty ())
        {
          auto it_mem = mem_rows[0].find ("memory_id");
          if (it_mem != mem_rows[0].end ()
              && it_mem->second.type () == typeid (long long))
            {
              mem_id = std::any_cast<long long> (it_mem->second);
            }
        }

      long long created_at = 0;
      if (it_created != row.end () && it_created->second.type () == typeid (long long))
        {
          created_at = std::any_cast<long long> (it_created->second);
        }
      seeds.push_back (Scored{ emb_id, mem_id, created_at, sim, 0.0, 1.0,
                               0.0, 0, v, Eigen::VectorXf () });
    }

  if (seeds.empty ())
    {
      return;
    }

  // Update last retrieval timestamp for consolidation idle-gating.
  p_ctx.last_retrieval_ts = signal.timestamp;

  // V2: Graph expansion from seed memory IDs using ASSOCIATIONS table.
  std::unordered_set<long long> expanded_memory_ids;
  for (const auto &s : seeds)
    {
      if (s.memory_id > 0)
        {
          expanded_memory_ids.insert (s.memory_id);
        }
    }

  // Add index-store seeds for pattern completion.
  if (!sparse_key.empty ())
    {
      auto it = p_ctx.index_store.find (sparse_key);
      if (it != p_ctx.index_store.end ())
        {
          for (const auto mem_id : it->second)
            {
              if (mem_id > 0)
                expanded_memory_ids.insert (mem_id);
            }
        }
    }

  // Expand via ASSOCIATIONS graph traversal
  if (!expanded_memory_ids.empty ())
    {
      try
        {
          // Build SQL with a VALUES list for seeds.
          std::string sql = "WITH RECURSIVE seed(id) AS (VALUES ";
          std::vector<std::any> params;
          params.reserve (expanded_memory_ids.size () + 2);
          bool first = true;
          for (long long mem_id : expanded_memory_ids)
            {
              if (!first)
                sql += ", ";
              first = false;
              sql += "(?)";
              params.push_back (mem_id);
            }
          sql += "), expand(id, depth) AS ("
                 "  SELECT id, 0 FROM seed "
                 "  UNION "
                 "  SELECT a.target_memory_id, expand.depth+1 "
                 "  FROM associations a JOIN expand ON a.source_memory_id = expand.id "
                 "  WHERE expand.depth < ? AND a.weight >= ? "
                 "  UNION "
                 "  SELECT a.source_memory_id, expand.depth+1 "
                 "  FROM associations a JOIN expand ON a.target_memory_id = expand.id "
                 "  WHERE expand.depth < ? AND a.weight >= ? "
                 ") "
                 "SELECT DISTINCT id FROM expand;";
          params.push_back (static_cast<long long> (depth));
          params.push_back (min_edge_weight);
          params.push_back (static_cast<long long> (depth));
          params.push_back (min_edge_weight);

          auto exp_rows = store->Execute (sql, params);
          for (const auto &r : exp_rows)
            {
              auto it = r.find ("id");
              if (it != r.end () && it->second.type () == typeid (long long))
                {
                  expanded_memory_ids.insert (std::any_cast<long long> (it->second));
                }
            }
        }
      catch (...)
        {
          // No associations or query failed; use seed-only behavior.
        }
    }

  // Fetch embeddings for expanded memory_ids via MEMORIES table
  std::unordered_map<long long, Eigen::VectorXf> out;
  std::vector<Scored> scored;
  scored.reserve (expanded_memory_ids.size ());

  bool fetched_any = false;
  if (!expanded_memory_ids.empty ())
    {
      try
        {
          std::string sql = "SELECT m.memory_id, m.embedding_id, e.embedding, "
                            "COALESCE(m.created_at, e.created_at, 0) AS created_at, "
                            "m.context, m.source_origin, m.source_reliability, "
                            "m.source_contradiction_count "
                            "FROM memories m "
                            "JOIN embeddings e ON m.embedding_id = e.embedding_id "
                            "WHERE m.memory_id IN (";
          std::vector<std::any> params;
          params.reserve (expanded_memory_ids.size ());
          bool first = true;
          for (const long long id : expanded_memory_ids)
            {
              if (!first)
                sql += ",";
              first = false;
              sql += "?";
              params.push_back (id);
            }
          sql += ");";
          auto fetch_rows = store->Execute (sql, params);
          for (const auto &row : fetch_rows)
            {
              auto it_mem_id = row.find ("memory_id");
              auto it_emb_id = row.find ("embedding_id");
              auto it_emb = row.find ("embedding");
              auto it_created = row.find ("created_at");
              auto it_ctx = row.find ("context");
              auto it_origin = row.find ("source_origin");
              auto it_rel = row.find ("source_reliability");
              auto it_contra = row.find ("source_contradiction_count");
              if (it_mem_id == row.end () || it_emb_id == row.end () || it_emb == row.end ())
                continue;
              if (it_mem_id->second.type () != typeid (long long))
                continue;
              if (it_emb_id->second.type () != typeid (long long))
                continue;
              const long long mem_id = std::any_cast<long long> (it_mem_id->second);
              const long long emb_id = std::any_cast<long long> (it_emb_id->second);
              Eigen::VectorXf v;
              if (!core::DecodeFloatBlob (it_emb->second, q.size (), v))
                continue;
              const double sim = core::CosineSimilarity (q, v);
              long long created_at = 0;
              if (it_created != row.end () && it_created->second.type () == typeid (long long))
                {
                  created_at = std::any_cast<long long> (it_created->second);
                }
              Eigen::VectorXf ctx_vec;
              if (it_ctx != row.end () && it_ctx->second.has_value ())
                {
                  core::DecodeFloatBlob (it_ctx->second, q_ctx.size (), ctx_vec);
                }
              const double ctx_sim
                  = (ctx_vec.size () == q_ctx.size () && q_ctx.size () > 0)
                        ? core::CosineSimilarity (q_ctx, ctx_vec)
                        : 0.0;

              std::string origin;
              if (it_origin != row.end () && it_origin->second.type () == typeid (std::string))
                {
                  origin = std::any_cast<std::string> (it_origin->second);
                }

              double base_rel = 0.7;
              if (it_rel != row.end () && it_rel->second.type () == typeid (double))
                {
                  base_rel = std::any_cast<double> (it_rel->second);
                }
              if (origin == "user")
                base_rel = std::max (base_rel, 0.8);
              else if (origin == "assistant")
                base_rel = std::max (base_rel, 0.6);
              else if (origin == "system")
                base_rel = std::max (base_rel, 0.9);

              int contradiction_count = 0;
              if (it_contra != row.end () && it_contra->second.type () == typeid (long long))
                {
                  contradiction_count
                      = static_cast<int> (std::any_cast<long long> (it_contra->second));
                }

              double age_s = 0.0;
              if (created_at > 0
                  && signal.timestamp > static_cast<uint64_t> (created_at))
                {
                  age_s = static_cast<double> (signal.timestamp - created_at) / 1000.0;
                }
              const double freshness = std::exp (-age_s / 3600.0);
              const double source_conf
                  = core::Clamp (base_rel * (1.0 - 0.15 * contradiction_count)
                                     * (0.7 + 0.3 * freshness),
                                 0.0, 1.0);

              double proc_score = 0.0;
              if (!sparse_key.empty ())
                {
                  auto pit = p_ctx.procedural_store.find (sparse_key);
                  if (pit != p_ctx.procedural_store.end ())
                    {
                      auto mit = pit->second.find (mem_id);
                      if (mit != pit->second.end ())
                        {
                          proc_score = mit->second;
                        }
                    }
                }

              scored.push_back (Scored{ emb_id, mem_id, created_at, sim,
                                        ctx_sim, source_conf, proc_score,
                                        contradiction_count, v, ctx_vec });
              fetched_any = true;
            }
        }
      catch (...)
        {
        }
    }

  auto is_overlap_wm = [&p_ctx] (const Eigen::VectorXf &v,
                                 double dup_thresh) -> bool {
    if (v.size () == 0)
      {
        return false;
      }
    for (const auto &slot : p_ctx.wm_slots)
      {
        if (slot.embedding.size () == v.size () && slot.embedding.size () > 0)
          {
            const double sim = core::CosineSimilarity (slot.embedding, v);
            if (sim >= dup_thresh)
              {
                return true;
              }
          }
      }
    return false;
  };

  const double dup_thresh = core::DupThresh (cfg.focus, cfg.stability);
  const std::uint64_t write_exclusion_ts
      = context.GetWriteExclusionTs ().value_or (signal.timestamp);
  const auto stored_id = context.GetStoredEmbeddingId ();
  const auto weights
      = core::RetrievalDiversificationWeights (cfg.focus, cfg.sensitivity,
                                               cfg.stability);
  const double w_rel = weights.first;
  const double w_div = weights.second;
  const double f_eff = core::FocusBias (cfg.focus);
  const double s_eff = core::SensitivityBias (cfg.sensitivity);
  const double w_ctx = core::Lerp (0.15, 0.35, f_eff)
                       * core::Lerp (1.0, 0.85, s_eff);
  const double w_proc = core::Lerp (0.10, 0.25, s_eff);

  const double source_thresh = core::Lerp (0.15, 0.45, cfg.stability);
  auto filter_candidates = [&] (const std::vector<Scored> &candidates) {
    std::vector<Scored> eligible;
    eligible.reserve (candidates.size ());
    for (const auto &s : candidates)
      {
        if (stored_id.has_value () && s.embedding_id == *stored_id)
          {
            continue;
          }
        if (s.created_at > 0
            && static_cast<std::uint64_t> (s.created_at) >= write_exclusion_ts)
          {
            continue;
          }
        if (is_overlap_wm (s.vec, dup_thresh))
          {
            continue;
          }
        if (s.source_confidence < source_thresh)
          {
            continue;
          }
        eligible.push_back (s);
      }
    return eligible;
  };

  auto relax_filters = [&] (const std::vector<Scored> &candidates) {
    std::vector<Scored> eligible;
    eligible.reserve (candidates.size ());
    for (const auto &s : candidates)
      {
        if (stored_id.has_value () && s.embedding_id == *stored_id)
          {
            continue;
          }
        if (is_overlap_wm (s.vec, dup_thresh))
          {
            continue;
          }
        eligible.push_back (s);
      }
    return eligible;
  };

  auto select_diversified = [&] (const std::vector<Scored> &candidates, int k) {
    std::vector<Scored> selected;
    if (candidates.empty () || k <= 0)
      {
        return selected;
      }
    const size_t n = candidates.size ();
    std::vector<bool> used (n, false);
    selected.reserve (std::min (n, static_cast<size_t> (k)));

    for (int iter = 0; iter < k; ++iter)
      {
        double best_score = -1e9;
        int best_idx = -1;
        for (size_t i = 0; i < n; ++i)
          {
            if (used[i])
              {
                continue;
              }
            const double relevance = std::max (0.0, candidates[i].score);
            const double ctx_sim = std::max (0.0, candidates[i].ctx_score);
            const double proc_sim = std::max (0.0, candidates[i].proc_score);
            double max_redundancy = 0.0;
            for (const auto &sel : selected)
              {
                const double sim
                    = core::CosineSimilarity (candidates[i].vec, sel.vec);
                max_redundancy = std::max (max_redundancy, std::max (0.0, sim));
              }
            const double mmr
                = w_rel * relevance + w_ctx * ctx_sim + w_proc * proc_sim
                  - w_div * max_redundancy;
            if (mmr > best_score)
              {
                best_score = mmr;
                best_idx = static_cast<int> (i);
              }
          }
        if (best_idx < 0)
          {
            break;
          }
        used[best_idx] = true;
        selected.push_back (candidates[best_idx]);
      }
    return selected;
  };

  auto reinstate_context = [&] (const std::vector<Scored> &selected) {
    if (selected.empty ())
      {
        return;
      }
    Eigen::VectorXf mean_ctx;
    int ctx_count = 0;
    for (const auto &s : selected)
      {
        if (s.ctx.size () == q_ctx.size () && q_ctx.size () > 0)
          {
            if (mean_ctx.size () == 0)
              {
                mean_ctx = s.ctx;
              }
            else
              {
                mean_ctx += s.ctx;
              }
            ++ctx_count;
          }
      }
    if (ctx_count > 0)
      {
        mean_ctx /= static_cast<float> (ctx_count);
        const float norm = mean_ctx.norm ();
        if (norm > 1e-9f)
          {
            mean_ctx /= norm;
          }
        if (acc_it != p_ctx.accumulator_states.end ())
          {
            auto &acc = acc_it->second;
            const double alpha = core::Lerp (0.20, 0.05, cfg.stability);
            if (acc.c_t.size () != mean_ctx.size ())
              {
                acc.c_t = mean_ctx;
              }
            else
              {
                acc.c_t = acc.c_t * static_cast<float> (1.0 - alpha)
                          + mean_ctx * static_cast<float> (alpha);
              }
            const float acc_norm = acc.c_t.norm ();
            if (acc_norm > 1e-9f)
              {
                acc.c_t /= acc_norm;
              }
          }
      }
  };

  if (!fetched_any)
    {
      for (auto &s : seeds)
        {
          if (s.vec.size () == q.size () && s.vec.size () > 0)
            {
              s.score = core::CosineSimilarity (q, s.vec);
            }
        }
      auto eligible = filter_candidates (seeds);
      if (eligible.empty ())
        {
          eligible = relax_filters (seeds);
        }
      auto selected = select_diversified (eligible, k);
      for (const auto &s : selected)
        {
          out.emplace (s.embedding_id, s.vec);
        }
      context.SetRetrievedMemoryEmbeddings (std::move (out));
      reinstate_context (selected);

      // Create reinforcement edges for co-retrieved seeds
      {
        std::vector<long long> seed_mem_ids;
        seed_mem_ids.reserve (selected.size ());
        for (const auto &s : selected)
          {
            if (s.memory_id > 0)
              {
                seed_mem_ids.push_back (s.memory_id);
              }
          }
        CreateReinforcementEdges (
            tx, seed_mem_ids,
            static_cast<long long> (signal.timestamp));
      }

      return;
    }

  auto eligible = filter_candidates (scored);
  if (eligible.empty ())
    {
      eligible = relax_filters (scored);
    }
  auto selected = select_diversified (eligible, k);
  for (const auto &s : selected)
    {
      out.emplace (s.embedding_id, s.vec);
    }

  context.SetRetrievedMemoryEmbeddings (std::move (out));
  reinstate_context (selected);

  // Create reinforcement edges for co-retrieved memories
  // This strengthens connections between memories that are frequently
  // retrieved together.
  {
    std::vector<long long> retrieved_mem_ids;
    retrieved_mem_ids.reserve (selected.size ());
    for (const auto &s : selected)
      {
        if (s.memory_id > 0)
          {
            retrieved_mem_ids.push_back (s.memory_id);
          }
      }
    CreateReinforcementEdges (
        tx, retrieved_mem_ids,
        static_cast<long long> (signal.timestamp));
  }

  // Debug logging
  telemetry::LogDebug ("cortext.graph_retrieval", {
    telemetry::Attribute::Double ("dup_thresh", dup_thresh),
    telemetry::Attribute::Int64 ("write_exclusion_ts", static_cast<int64_t> (write_exclusion_ts)),
    telemetry::Attribute::Bool ("stored_id_present", stored_id.has_value ()),
    telemetry::Attribute::Int64 ("selected_count", static_cast<int64_t> (selected.size ())),
    telemetry::Attribute::Int64 ("seed_count", static_cast<int64_t> (seeds.size ())),
    telemetry::Attribute::Int64 ("expansion_depth", static_cast<int64_t> (depth)),
    telemetry::Attribute::Int64 ("final_candidate_count", static_cast<int64_t> (scored.size ())),
    telemetry::Attribute::Double ("retrieval_w_rel", w_rel),
    telemetry::Attribute::Double ("retrieval_w_div", w_div),
    telemetry::Attribute::Double ("retrieval_w_ctx", w_ctx),
    telemetry::Attribute::Double ("retrieval_w_proc", w_proc)
  });
}

} // namespace cortext::operations
