#include "cortext/operations/graph_retrieval.hpp"

#include "cortext/store/store.hpp"
#include "cortext/store/utils.hpp"
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
#include <chrono>
#include <cmath>
#include <functional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include <cctype>
#include <cstdlib>

namespace cortext::operations
{

namespace
{
bool
EnvFlag (const char *name)
{
  const char *value = std::getenv (name);
  if (!value)
    {
      return false;
    }
  std::string s (value);
  std::transform (s.begin (), s.end (), s.begin (),
                  [] (unsigned char c) { return static_cast<char> (std::tolower (c)); });
  return s == "1" || s == "true" || s == "yes" || s == "on";
}
} // namespace

void
GraphAugmentedRetrieveCandidates::Execute (OperationContext &context, Transaction &tx) const
{
  (void)tx;
  auto elapsed_ms = [] (const std::chrono::steady_clock::time_point &start,
                        const std::chrono::steady_clock::time_point &end) {
    return std::chrono::duration_cast<
        std::chrono::duration<double, std::milli> > (end - start)
        .count ();
  };

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

  const double f_eff = core::FocusBias (cfg.focus);
  const double s_eff = core::SensitivityBias (cfg.sensitivity);
  const int k = std::max (1, core::MaxResults (cfg.focus));
  const int depth = std::max (1, core::GraphDepth (cfg.stability));
  const double min_edge_weight = core::MinEdgeWeight (cfg.focus);
  const int k_key = core::SparseKeySize (cfg.focus);
  const std::string sparse_key = core::SparseKey (q, k_key);
  const double s_eff_seed = s_eff;

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
    double emotion_intensity;
    double arousal_avg;
    bool is_association;
    bool is_label;
    Eigen::VectorXf vec;
    Eigen::VectorXf ctx;
  };
  std::vector<Scored> seeds;
  seeds.reserve (static_cast<size_t> (k));

  // Convert query vector to std::vector<float> for parameter binding.
  std::vector<float> q_vec (q.data (), q.data () + q.size ());

  std::unordered_set<long long> seen_seed_embeddings;

  auto append_seeds = [&] (const std::string &sql,
                           const std::vector<std::any> &params,
                           const char *timing_key) {
    const auto t_start = std::chrono::steady_clock::now ();
    auto rows = store->Execute (sql, params);
    const auto t_end = std::chrono::steady_clock::now ();
    if (timing_key)
      {
        context.AddOperationTiming (timing_key, elapsed_ms (t_start, t_end));
      }
    for (const auto &row : rows)
      {
        auto it_emb_id = row.find ("embedding_id");
        auto it_dist = row.find ("distance");
        auto it_created = row.find ("created_at");
        auto it_mem_id = row.find ("memory_id");
        auto it_kind = row.find ("kind");
        if (it_emb_id == row.end ())
          continue;
        if (it_emb_id->second.type () != typeid (long long))
          continue;

        const long long emb_id = std::any_cast<long long> (it_emb_id->second);
        if (!seen_seed_embeddings.insert (emb_id).second)
          {
            continue;
          }

        // Prefer cosine similarity for scoring to align with manuscript logic.
        double sim = 0.0;
        if (it_dist != row.end ()
                 && it_dist->second.type () == typeid (double))
          {
            const double dist = std::any_cast<double> (it_dist->second);
            sim = 1.0 / (1.0 + dist);
          }

        long long mem_id = 0;
        if (it_mem_id != row.end ())
          {
            mem_id = store::AnyToLongLong (it_mem_id->second).value_or (0);
          }
        if (mem_id == 0)
          {
            auto sig_rows = store->Execute (
                "SELECT memory_id FROM signals WHERE embedding_id = ? LIMIT 1",
                { emb_id });
            if (!sig_rows.empty ())
              {
                auto it_sig = sig_rows[0].find ("memory_id");
                if (it_sig != sig_rows[0].end ())
                  {
                    mem_id = store::AnyToLongLong (it_sig->second).value_or (0);
                  }
              }
          }

        long long created_at = 0;
        if (it_created != row.end ()
            && it_created->second.type () == typeid (long long))
          {
            created_at = std::any_cast<long long> (it_created->second);
          }

        bool is_association = false;
        bool is_label = false;
        if (it_kind != row.end () && it_kind->second.type () == typeid (std::string))
          {
            const std::string kind
                = std::any_cast<std::string> (it_kind->second);
            is_association = (kind == "ASSOCIATION");
            is_label = (kind == "LABEL");
          }

        seeds.push_back (Scored{ emb_id, mem_id, created_at, sim, 0.0, 1.0,
                                 0.0, 0, 0.0, 0.0, is_association, is_label,
                                 Eigen::VectorXf (), Eigen::VectorXf () });
      }
  };

  const std::string seed_sql =
      "SELECT e.embedding_id, e.distance, "
      "COALESCE(m.created_at, e.created_at, 0) AS created_at, "
      "m.memory_id, m.kind "
      "FROM embeddings e "
      "JOIN memories m ON m.embedding_id = e.embedding_id "
      "WHERE e.embedding MATCH ? AND k = ? "
      "AND m.kind != 'WORKING'";
  append_seeds (seed_sql, { q_vec, static_cast<long long> (k) },
                "GraphRetrieve.seed_sql");

  const double k_summary_raw
      = core::Lerp (2.0, 6.0, s_eff_seed) * core::Lerp (1.0, 0.75, f_eff)
        * core::Lerp (1.0, 0.85, cfg.stability);
  const int k_summary
      = std::max (1, static_cast<int> (std::round (k_summary_raw)));
  const auto &summary_cache = p_ctx.summary_cache;
  if (k_summary > 0 && !summary_cache.empty ())
    {
      const auto t_start = std::chrono::steady_clock::now ();
      std::vector<std::pair<double, size_t>> ranked;
      ranked.reserve (summary_cache.size ());
      for (size_t i = 0; i < summary_cache.size (); ++i)
        {
          const auto &entry = summary_cache[i];
          if (entry.embedding.size () != q.size ())
            {
              continue;
            }
          if (entry.embedding_norm <= 1e-9f)
            {
              continue;
            }
          const double sim
              = static_cast<double> (entry.embedding.dot (q))
                / static_cast<double> (entry.embedding_norm);
          ranked.emplace_back (sim, i);
        }
      const int take
          = std::min (k_summary, static_cast<int> (ranked.size ()));
      if (take > 0)
        {
          auto mid = ranked.begin () + take;
          std::nth_element (
              ranked.begin (), mid, ranked.end (),
              [] (const auto &a, const auto &b) { return a.first > b.first; });
          std::sort (ranked.begin (), mid,
                     [] (const auto &a, const auto &b) { return a.first > b.first; });
          for (int i = 0; i < take; ++i)
            {
              const auto &entry = summary_cache[ranked[i].second];
              if (!seen_seed_embeddings.insert (entry.embedding_id).second)
                {
                  continue;
                }
              seeds.push_back (Scored{
                  entry.embedding_id,
                  entry.memory_id,
                  0,
                  ranked[i].first,
                  0.0,
                  1.0,
                  0.0,
                  0,
                  0.0,
                  0.0,
                  entry.is_association,
                  entry.is_label,
                  entry.embedding,
                  Eigen::VectorXf () });
            }
        }
      const auto t_end = std::chrono::steady_clock::now ();
      context.AddOperationTiming ("GraphRetrieve.summary_cache",
                                  elapsed_ms (t_start, t_end));
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
  auto get_double = [] (const std::any &v, double def) -> double {
    if (v.type () == typeid (double))
      return std::any_cast<double> (v);
    if (v.type () == typeid (float))
      return static_cast<double> (std::any_cast<float> (v));
    if (v.type () == typeid (int))
      return static_cast<double> (std::any_cast<int> (v));
    if (v.type () == typeid (long long))
      return static_cast<double> (std::any_cast<long long> (v));
    return def;
  };

  if (!expanded_memory_ids.empty ())
    {
      try
        {
          const auto t_start = std::chrono::steady_clock::now ();
          // Build SQL with a VALUES list for seeds.
          const char *edge_filter
              = cfg.reinforcement_enabled ? "" : " AND a.edge_type != 'reinforces' ";
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
                 "  WHERE expand.depth < ? AND a.weight >= ? ";
          sql += edge_filter;
          sql += "  UNION "
                 "  SELECT a.source_memory_id, expand.depth+1 "
                 "  FROM associations a JOIN expand ON a.target_memory_id = expand.id "
                 "  WHERE expand.depth < ? AND a.weight >= ? ";
          sql += edge_filter;
          sql += ") "
                 "SELECT DISTINCT id FROM expand;";
          params.push_back (static_cast<long long> (depth));
          params.push_back (min_edge_weight);
          params.push_back (static_cast<long long> (depth));
          params.push_back (min_edge_weight);

          auto exp_rows = store->Execute (sql, params);
          const auto t_end = std::chrono::steady_clock::now ();
          context.AddOperationTiming ("GraphRetrieve.expand_sql",
                                      elapsed_ms (t_start, t_end));
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

  const double min_assoc_raw
      = core::Lerp (3.0, 1.0, f_eff) * core::Lerp (1.0, 0.85, cfg.stability);
  const double min_label_raw
      = core::Lerp (0.0, 3.0, s_eff) * core::Lerp (1.0, 0.85, f_eff);
  int min_assoc = static_cast<int> (std::round (min_assoc_raw));
  int min_label = static_cast<int> (std::round (min_label_raw));
  if (min_assoc < 0)
    min_assoc = 0;
  if (min_label < 0)
    min_label = 0;

  if (!expanded_memory_ids.empty () && (min_assoc > 0 || min_label > 0))
    {
      std::unordered_set<long long> summary_ids;
      if (min_assoc > 0)
        {
          try
            {
              const auto t_start = std::chrono::steady_clock::now ();
              std::string sql =
                  "SELECT DISTINCT a.source_memory_id AS id "
                  "FROM associations a "
                  "JOIN memories m ON m.memory_id = a.source_memory_id "
                  "WHERE a.edge_type = 'derived_from' "
                  "AND a.target_memory_id IN (";
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
              sql += ") AND m.kind = 'ASSOCIATION'";
              auto rows = store->Execute (sql, params);
              const auto t_end = std::chrono::steady_clock::now ();
              context.AddOperationTiming ("GraphRetrieve.summary_assoc_sql",
                                          elapsed_ms (t_start, t_end));
              for (const auto &r : rows)
                {
                  auto it = r.find ("id");
                  if (it != r.end ()
                      && it->second.type () == typeid (long long))
                    {
                      const long long mem_id
                          = std::any_cast<long long> (it->second);
                      if (mem_id > 0)
                        {
                          summary_ids.insert (mem_id);
                          expanded_memory_ids.insert (mem_id);
                        }
                    }
                }
            }
          catch (...)
            {
            }
        }

      if (min_label > 0 && !summary_ids.empty ())
        {
          try
            {
              const auto t_start = std::chrono::steady_clock::now ();
              std::string sql =
                  "SELECT DISTINCT a.target_memory_id AS id "
                  "FROM associations a "
                  "JOIN memories m ON m.memory_id = a.target_memory_id "
                  "WHERE a.edge_type = 'has_label' "
                  "AND a.source_memory_id IN (";
              std::vector<std::any> params;
              params.reserve (summary_ids.size ());
              bool first = true;
              for (const long long id : summary_ids)
                {
                  if (!first)
                    sql += ",";
                  first = false;
                  sql += "?";
                  params.push_back (id);
                }
              sql += ") AND m.kind = 'LABEL'";
              auto rows = store->Execute (sql, params);
              const auto t_end = std::chrono::steady_clock::now ();
              context.AddOperationTiming ("GraphRetrieve.summary_label_sql",
                                          elapsed_ms (t_start, t_end));
              for (const auto &r : rows)
                {
                  auto it = r.find ("id");
                  if (it != r.end ()
                      && it->second.type () == typeid (long long))
                    {
                      const long long mem_id
                          = std::any_cast<long long> (it->second);
                      if (mem_id > 0)
                        {
                          expanded_memory_ids.insert (mem_id);
                        }
                    }
                }
            }
          catch (...)
            {
            }
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
          const auto t_start = std::chrono::steady_clock::now ();
          std::string sql = "SELECT m.memory_id, m.embedding_id, e.embedding, "
                            "COALESCE(m.created_at, e.created_at, 0) AS created_at, "
                            "m.context, m.source_origin, m.source_reliability, "
                            "m.source_contradiction_count, m.emotional_intensity, "
                            "m.s_arousal_avg, m.kind "
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
          const auto t_end = std::chrono::steady_clock::now ();
          context.AddOperationTiming ("GraphRetrieve.fetch_embeddings_sql",
                                      elapsed_ms (t_start, t_end));
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
              auto it_emotion = row.find ("emotional_intensity");
              auto it_arousal = row.find ("s_arousal_avg");
              auto it_kind = row.find ("kind");
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

              double memory_emotion = 0.0;
              if (it_emotion != row.end () && it_emotion->second.has_value ())
                {
                  memory_emotion
                      = core::Clamp (get_double (it_emotion->second, 0.0), 0.0, 1.0);
                }
              double memory_arousal = 0.0;
              if (it_arousal != row.end () && it_arousal->second.has_value ())
                {
                  memory_arousal
                      = core::Clamp (get_double (it_arousal->second, 0.0), 0.0, 1.0);
                }
              bool is_association = false;
              bool is_label = false;
              if (it_kind != row.end () && it_kind->second.type () == typeid (std::string))
                {
                  const std::string kind
                      = std::any_cast<std::string> (it_kind->second);
                  is_association = (kind == "ASSOCIATION");
                  is_label = (kind == "LABEL");
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
              if (cfg.procedural_enabled && !sparse_key.empty ())
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
                                        contradiction_count, memory_emotion,
                                        memory_arousal, is_association,
                                        is_label, v, ctx_vec });
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
  const double dup_thresh_summary = core::Clamp (
      dup_thresh * core::Lerp (1.35, 1.10, f_eff), 0.0, 0.99);
  const bool bypass_summary_overlap = (f_eff < 0.75);
  const std::uint64_t write_exclusion_ts
      = context.GetWriteExclusionTs ().value_or (signal.timestamp);
  const auto stored_id = context.GetStoredEmbeddingId ();
  const auto weights
      = core::RetrievalDiversificationWeights (cfg.focus, cfg.sensitivity,
                                               cfg.stability);
  const double w_rel = weights.first;
  const double w_div = weights.second;
  const double w_ctx = core::Lerp (0.15, 0.35, f_eff)
                       * core::Lerp (1.0, 0.85, s_eff);
  const double w_proc = core::Lerp (0.10, 0.25, s_eff);
  const double w_emotion
      = cfg.affect_retrieval
            ? core::RetrievalEmotionWeight (cfg.sensitivity)
            : 0.0;
  const double assoc_boost
      = core::AssociationBoost (cfg.focus, cfg.sensitivity, cfg.stability);
  const double label_boost = core::Lerp (0.05, 0.18, s_eff);

  const double salience = core::Clamp (
      context.GetMetric (operations::Metric::salience).value_or (0.0), 0.0, 1.0);
  const double emotion_intensity
      = core::Clamp (context.GetEmotionIntensity (), 0.0, 1.0);
  const double arousal = core::Clamp (context.GetArousal (), 0.0, 1.0);
  const double s_affect = core::AffectSensitivityBias (cfg.sensitivity);
  const double w_arousal_raw = core::Lerp (0.30, 0.55, s_affect);
  const double w_emotion_raw = core::Lerp (0.30, 0.55, s_affect);
  const double w_salience_raw = core::Lerp (0.10, 0.20, s_affect);
  const double w_aff_sum = std::max (constants::kNormEpsilon,
                                     w_arousal_raw + w_emotion_raw
                                       + w_salience_raw);
  const double w_aff_arousal = w_arousal_raw / w_aff_sum;
  const double w_aff_emotion = w_emotion_raw / w_aff_sum;
  const double w_aff_salience = w_salience_raw / w_aff_sum;
  const double affect_gain = core::AffectGain (cfg.sensitivity);
  const double affect_drive
      = core::Clamp (affect_gain
                         * (w_aff_arousal * arousal
                            + w_aff_emotion * emotion_intensity
                            + w_aff_salience * salience),
                     0.0, 1.0);
  const double w_mem_emotion_raw = core::Lerp (0.60, 0.80, s_eff);
  const double w_mem_arousal_raw = core::Lerp (0.20, 0.40, s_eff);
  const double w_mem_sum = std::max (constants::kNormEpsilon,
                                     w_mem_emotion_raw + w_mem_arousal_raw);
  const double w_mem_emotion = w_mem_emotion_raw / w_mem_sum;
  const double w_mem_arousal = w_mem_arousal_raw / w_mem_sum;

  const double source_thresh = core::Lerp (0.15, 0.45, cfg.stability);
  static const bool disable_source_conf
      = EnvFlag ("CORTEXT_DISABLE_SOURCE_CONF");
  struct FilterStats
  {
    int64_t total = 0;
    int64_t skipped_stored = 0;
    int64_t skipped_write_exclusion = 0;
    int64_t skipped_overlap = 0;
    int64_t skipped_source_conf = 0;
    int64_t kept = 0;
    int64_t kept_assoc = 0;
    int64_t kept_label = 0;
    int64_t kept_other = 0;
  };

  auto filter_candidates = [&] (const std::vector<Scored> &candidates,
                                 bool enforce_write_exclusion,
                                 bool enforce_source_conf,
                                 FilterStats *stats) {
    std::vector<Scored> eligible;
    eligible.reserve (candidates.size ());
    for (const auto &s : candidates)
      {
        if (stats)
          {
            stats->total++;
          }
        if (stored_id.has_value () && s.embedding_id == *stored_id)
          {
            if (stats)
              {
                stats->skipped_stored++;
              }
            continue;
          }
        const bool summary_like = (s.is_association || s.is_label);
        if (enforce_write_exclusion && !summary_like && s.created_at > 0
            && static_cast<std::uint64_t> (s.created_at) >= write_exclusion_ts)
          {
            if (stats)
              {
                stats->skipped_write_exclusion++;
              }
            continue;
          }
        const double cand_dup_thresh
            = summary_like ? dup_thresh_summary : dup_thresh;
        if (!(summary_like && bypass_summary_overlap)
            && is_overlap_wm (s.vec, cand_dup_thresh))
          {
            if (stats)
              {
                stats->skipped_overlap++;
              }
            continue;
          }
        if (!disable_source_conf && enforce_source_conf
            && s.source_confidence < source_thresh)
          {
            if (stats)
              {
                stats->skipped_source_conf++;
              }
            continue;
          }
        eligible.push_back (s);
        if (stats)
          {
            stats->kept++;
            if (s.is_association)
              {
                stats->kept_assoc++;
              }
            else if (s.is_label)
              {
                stats->kept_label++;
              }
            else
              {
                stats->kept_other++;
              }
          }
      }
    return eligible;
  };

  auto base_score = [&] (const Scored &s) {
    const double relevance = std::max (0.0, s.score);
    const double ctx_sim = std::max (0.0, s.ctx_score);
    const double proc_sim = std::max (0.0, s.proc_score);
    const double memory_affect
        = w_mem_emotion * s.emotion_intensity + w_mem_arousal * s.arousal_avg;
    const double emotion_bonus = affect_drive * memory_affect;
    const double association_bonus = s.is_association ? assoc_boost : 0.0;
    const double label_bonus = s.is_label ? label_boost : 0.0;
    return w_rel * relevance + w_ctx * ctx_sim + w_proc * proc_sim
           + w_emotion * emotion_bonus + association_bonus + label_bonus;
  };

  auto select_diversified = [&] (const std::vector<Scored> &candidates, int k,
                                 const std::vector<Scored> &initial) {
    std::vector<Scored> selected;
    if (candidates.empty () || k <= 0)
      {
        return selected;
      }
    const size_t n = candidates.size ();
    std::vector<bool> used (n, false);
    selected.reserve (std::min (n, static_cast<size_t> (k)));

    std::unordered_map<long long, size_t> candidate_index;
    candidate_index.reserve (n);
    for (size_t i = 0; i < n; ++i)
      {
        candidate_index.emplace (candidates[i].embedding_id, i);
      }

    for (const auto &seed : initial)
      {
        auto it = candidate_index.find (seed.embedding_id);
        if (it == candidate_index.end ())
          {
            continue;
          }
        const size_t idx = it->second;
        if (used[idx])
          {
            continue;
          }
        used[idx] = true;
        selected.push_back (candidates[idx]);
      }

    for (int iter = static_cast<int> (selected.size ()); iter < k; ++iter)
      {
        double best_score = -1e9;
        int best_idx = -1;
        for (size_t i = 0; i < n; ++i)
          {
            if (used[i])
              {
                continue;
              }
            const double core_score = base_score (candidates[i]);
            double max_redundancy = 0.0;
            for (const auto &sel : selected)
              {
                const double sim
                    = core::CosineSimilarity (candidates[i].vec, sel.vec);
                max_redundancy = std::max (max_redundancy, std::max (0.0, sim));
              }
            const double mmr = core_score - w_div * max_redundancy;
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

  int64_t reinforcement_candidate_count = 0;
  if (!fetched_any)
    {
      if (!seeds.empty ())
        {
          try
            {
              const auto t_start = std::chrono::steady_clock::now ();
              std::string sql = "SELECT embedding_id, embedding FROM embeddings "
                                "WHERE embedding_id IN (";
              std::vector<std::any> params;
              params.reserve (seeds.size ());
              bool first = true;
              for (const auto &s : seeds)
                {
                  if (!first)
                    sql += ",";
                  first = false;
                  sql += "?";
                  params.push_back (s.embedding_id);
                }
              sql += ");";
              auto rows = store->Execute (sql, params);
              const auto t_end = std::chrono::steady_clock::now ();
              context.AddOperationTiming ("GraphRetrieve.seed_fetch_sql",
                                          elapsed_ms (t_start, t_end));
              std::unordered_map<long long, Eigen::VectorXf> seed_vecs;
              seed_vecs.reserve (rows.size ());
              for (const auto &row : rows)
                {
                  auto it_emb_id = row.find ("embedding_id");
                  auto it_emb = row.find ("embedding");
                  if (it_emb_id == row.end () || it_emb == row.end ())
                    continue;
                  if (it_emb_id->second.type () != typeid (long long))
                    continue;
                  Eigen::VectorXf v;
                  if (!core::DecodeFloatBlob (it_emb->second, q.size (), v))
                    continue;
                  seed_vecs.emplace (std::any_cast<long long> (it_emb_id->second),
                                     std::move (v));
                }
              for (auto &s : seeds)
                {
                  auto it = seed_vecs.find (s.embedding_id);
                  if (it == seed_vecs.end ())
                    continue;
                  s.vec = it->second;
                  if (s.vec.size () == q.size () && s.vec.size () > 0)
                    {
                      s.score = core::CosineSimilarity (q, s.vec);
                    }
                }
            }
          catch (...)
            {
            }
        }
      for (auto &s : seeds)
        {
          if (s.vec.size () == q.size () && s.vec.size () > 0)
            {
              s.score = core::CosineSimilarity (q, s.vec);
            }
        }
      const auto t_score_start = std::chrono::steady_clock::now ();
      FilterStats strict_stats;
      auto eligible = filter_candidates (seeds, true, true, &strict_stats);
      FilterStats relax_stats;
      if (eligible.empty ())
        {
          eligible = filter_candidates (seeds, false, false, &relax_stats);
        }
      auto selected = select_diversified (eligible, k, {});
      const auto t_score_end = std::chrono::steady_clock::now ();
      context.AddOperationTiming ("GraphRetrieve.score", elapsed_ms (t_score_start, t_score_end));
      for (const auto &s : selected)
        {
          out.emplace (s.embedding_id, s.vec);
        }
      context.SetRetrievedMemoryEmbeddings (std::move (out));
      reinstate_context (selected);

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
        reinforcement_candidate_count
            = static_cast<int64_t> (seed_mem_ids.size ());
      }

      return;
    }

  FilterStats strict_stats;
  const auto t_score_start = std::chrono::steady_clock::now ();
  auto eligible = filter_candidates (scored, true, true, &strict_stats);
  FilterStats relax_stats;
  if (eligible.empty ())
    {
      eligible = filter_candidates (scored, false, false, &relax_stats);
    }
  if (min_assoc > k)
    min_assoc = k;
  if (min_label > k - min_assoc)
    min_label = k - min_assoc;

  auto pick_top = [&] (const std::vector<Scored> &candidates,
                       int count,
                       const std::function<bool(const Scored &)> &pred) {
    std::vector<std::pair<double, Scored>> ranked;
    ranked.reserve (candidates.size ());
    for (const auto &cand : candidates)
      {
        if (!pred (cand))
          {
            continue;
          }
        ranked.emplace_back (base_score (cand), cand);
      }
    std::sort (ranked.begin (), ranked.end (),
               [] (const auto &a, const auto &b) { return a.first > b.first; });
    std::vector<Scored> out;
    for (int i = 0; i < count && i < static_cast<int> (ranked.size ()); ++i)
      {
        out.push_back (ranked[i].second);
      }
    return out;
  };

  std::vector<Scored> initial;
  initial.reserve (static_cast<size_t> (min_assoc + min_label));
  std::unordered_set<long long> initial_ids;
  if (min_assoc > 0)
    {
      auto assoc_seeds = pick_top (
          eligible, min_assoc,
          [] (const Scored &s) { return s.is_association; });
      for (const auto &seed : assoc_seeds)
        {
          if (initial_ids.insert (seed.embedding_id).second)
            {
              initial.push_back (seed);
            }
        }
    }
  if (min_label > 0)
    {
      auto label_seeds = pick_top (
          eligible, min_label,
          [] (const Scored &s) { return s.is_label; });
      for (const auto &seed : label_seeds)
        {
          if (initial_ids.insert (seed.embedding_id).second)
            {
              initial.push_back (seed);
            }
        }
    }

  auto selected = select_diversified (eligible, k, initial);
  const auto t_score_end = std::chrono::steady_clock::now ();
  context.AddOperationTiming ("GraphRetrieve.score", elapsed_ms (t_score_start, t_score_end));
  for (const auto &s : selected)
    {
      out.emplace (s.embedding_id, s.vec);
    }

  context.SetRetrievedMemoryEmbeddings (std::move (out));
  reinstate_context (selected);

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
    reinforcement_candidate_count
        = static_cast<int64_t> (retrieved_mem_ids.size ());
  }

  // Debug logging
  auto count_kinds = [] (const std::vector<Scored> &items,
                         int64_t &assoc, int64_t &label, int64_t &other) {
    assoc = 0;
    label = 0;
    other = 0;
    for (const auto &s : items)
      {
        if (s.is_association)
          {
            assoc++;
          }
        else if (s.is_label)
          {
            label++;
          }
        else
          {
            other++;
          }
      }
  };

  int64_t scored_assoc = 0, scored_label = 0, scored_other = 0;
  int64_t eligible_assoc = 0, eligible_label = 0, eligible_other = 0;
  int64_t selected_assoc = 0, selected_label = 0, selected_other = 0;
  int64_t initial_assoc = 0, initial_label = 0, initial_other = 0;
  count_kinds (scored, scored_assoc, scored_label, scored_other);
  count_kinds (eligible, eligible_assoc, eligible_label, eligible_other);
  count_kinds (selected, selected_assoc, selected_label, selected_other);
  count_kinds (initial, initial_assoc, initial_label, initial_other);

  telemetry::LogDebug ("cortext.graph_retrieval", {
    telemetry::Attribute::Double ("dup_thresh", dup_thresh),
    telemetry::Attribute::Int64 ("write_exclusion_ts", static_cast<int64_t> (write_exclusion_ts)),
    telemetry::Attribute::Bool ("stored_id_present", stored_id.has_value ()),
    telemetry::Attribute::Int64 ("selected_count", static_cast<int64_t> (selected.size ())),
    telemetry::Attribute::Int64 ("seed_count", static_cast<int64_t> (seeds.size ())),
    telemetry::Attribute::Int64 ("expansion_depth", static_cast<int64_t> (depth)),
    telemetry::Attribute::Int64 ("final_candidate_count", static_cast<int64_t> (scored.size ())),
    telemetry::Attribute::Int64 ("scored_assoc_count", scored_assoc),
    telemetry::Attribute::Int64 ("scored_label_count", scored_label),
    telemetry::Attribute::Int64 ("eligible_assoc_count", eligible_assoc),
    telemetry::Attribute::Int64 ("eligible_label_count", eligible_label),
    telemetry::Attribute::Int64 ("selected_assoc_count", selected_assoc),
    telemetry::Attribute::Int64 ("selected_label_count", selected_label),
    telemetry::Attribute::Int64 ("initial_assoc_count", initial_assoc),
    telemetry::Attribute::Int64 ("initial_label_count", initial_label),
    telemetry::Attribute::Int64 ("filtered_total", strict_stats.total),
    telemetry::Attribute::Int64 ("filtered_stored", strict_stats.skipped_stored),
    telemetry::Attribute::Int64 ("filtered_write_exclusion", strict_stats.skipped_write_exclusion),
    telemetry::Attribute::Int64 ("filtered_overlap", strict_stats.skipped_overlap),
    telemetry::Attribute::Int64 ("filtered_source_conf", strict_stats.skipped_source_conf),
    telemetry::Attribute::Int64 ("filtered_kept", strict_stats.kept),
    telemetry::Attribute::Double ("retrieval_w_rel", w_rel),
    telemetry::Attribute::Double ("retrieval_w_div", w_div),
    telemetry::Attribute::Double ("retrieval_w_ctx", w_ctx),
    telemetry::Attribute::Double ("retrieval_w_proc", w_proc),
    telemetry::Attribute::Int64 ("reinforcement_candidate_count",
                                 reinforcement_candidate_count),
    telemetry::Attribute::Bool ("reinforcement_enabled",
                                cfg.reinforcement_enabled)
  });
}

} // namespace cortext::operations
