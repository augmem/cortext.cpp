#include "cortext/operations/short_term_memory_shadow.hpp"

#include "cortext/core/algorithms.hpp"
#include "cortext/core/knobs.hpp"
#include "cortext/core/utils.hpp"
#include "cortext/processor/operation_context.hpp"
#include "cortext/processor/processor_context.hpp"
#include "cortext/store/store.hpp"

#include <algorithm>
#include <any>
#include <chrono>
#include <cstdlib>
#include <deque>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace cortext::operations
{
namespace
{

bool
EnvBool (const char *name, bool fallback)
{
  const char *value = std::getenv (name);
  if (value == nullptr)
    {
      return fallback;
    }
  return std::string (value) == "1" || std::string (value) == "true"
         || std::string (value) == "on" || std::string (value) == "yes";
}

bool
ShadowStmEnabled ()
{
  const char *value = std::getenv ("CORTEXT_STM_SHADOW_ENABLE");
  if (value == nullptr)
    {
      return !EnvBool ("CORTEXT_STM_SHADOW_DISABLE", false);
    }
  return std::string (value) == "1" || std::string (value) == "true"
         || std::string (value) == "on" || std::string (value) == "yes";
}

std::string
LoadLabelText (Transaction &tx, long long label_memory_id)
{
  auto rows = tx.Execute (
      "SELECT COALESCE(label, '') AS label "
      "FROM memories WHERE memory_id = ? AND kind = 'LABEL'",
      { label_memory_id });
  if (rows.empty ())
    {
      return {};
    }
  auto it = rows[0].find ("label");
  if (it == rows[0].end () || it->second.type () != typeid (std::string))
    {
      return {};
    }
  return std::any_cast<std::string> (it->second);
}

bool
LabelBankAttached (Transaction &tx)
{
  try
    {
      auto rows = tx.Execute ("PRAGMA database_list", {});
      for (const auto &row : rows)
        {
          auto it = row.find ("name");
          if (it != row.end () && it->second.type () == typeid (std::string)
              && std::any_cast<std::string> (it->second)
                     == "cortext_label_bank")
            {
              return true;
            }
        }
    }
  catch (...)
    {
    }
  return false;
}

std::vector<ProcessorContext::ShadowLabelEdge>
SelectAttachedVectorLabels (OperationContext &context, Transaction &tx,
                            int step_index)
{
  const auto &signal = context.GetSignal ();
  const auto &cfg = context.GetConfig ();
  const auto &q = signal.embedding;
  if (q.size () == 0 || !LabelBankAttached (tx))
    {
      return {};
    }

  const int cluster_take = std::max (
      1, core::RetrievalClusterLabelM (cfg.focus, cfg.stability));
  const int labels_per_cluster = std::max (
      1, core::RetrievalClusterLabelN (cfg.focus, cfg.sensitivity,
                                       cfg.stability));
  const int top_k = std::max (1, cluster_take * labels_per_cluster);
  const double min_score = core::STMLabelMinScore (
      cfg.focus, cfg.sensitivity, cfg.stability);
  std::vector<float> q_vec (q.data (), q.data () + q.size ());

  std::vector<ProcessorContext::ShadowLabelEdge> edges;
  try
    {
      auto rows = tx.Execute (
          "SELECT label_id, key, label, distance "
          "FROM cortext_label_bank.label_bank_vec "
          "WHERE embedding MATCH ? AND k = ? "
          "ORDER BY distance",
          { q_vec, static_cast<long long> (top_k) });
      edges.reserve (rows.size ());
      for (const auto &row : rows)
        {
          auto label_it = row.find ("label");
          if (label_it == row.end ()
              || label_it->second.type () != typeid (std::string))
            {
              continue;
            }
          const double distance
              = row.count ("distance")
                        && row.at ("distance").type () == typeid (double)
                    ? std::any_cast<double> (row.at ("distance"))
                    : 0.0;
          const double score = core::RetrievalVectorDistanceScore (
              distance, cfg.focus, cfg.sensitivity, cfg.stability);
          if (score < min_score)
            {
              continue;
            }
          long long label_id = 0;
          auto id_it = row.find ("label_id");
          if (id_it != row.end () && id_it->second.type () == typeid (long long))
            {
              label_id = std::any_cast<long long> (id_it->second);
            }
          ProcessorContext::ShadowLabelEdge edge;
          edge.source_id = signal.source_id;
          edge.timestamp = signal.timestamp;
          edge.step_index = step_index;
          edge.label_memory_id = label_id > 0 ? -label_id : 0;
          edge.label = std::any_cast<std::string> (label_it->second);
          edge.weight = score;
          edge.signal_embedding = signal.embedding;
          edges.push_back (std::move (edge));
        }
    }
  catch (...)
    {
    }
  return edges;
}

std::vector<ProcessorContext::ShadowLabelEdge>
SelectClusteredLabels (OperationContext &context, Transaction &tx,
                       int step_index)
{
  auto &processor = context.GetProcessorContext ();
  const auto &signal = context.GetSignal ();
  const auto &cfg = context.GetConfig ();
  const auto &q = signal.embedding;
  if (q.size () == 0)
    {
      return {};
    }
  auto attached_edges = SelectAttachedVectorLabels (context, tx, step_index);
  if (!attached_edges.empty ())
    {
      return attached_edges;
    }
  if (processor.summary_cache.empty ())
    {
      return {};
    }

  const int requested_clusters = std::max (
      1, core::RetrievalClusterLabelK (cfg.focus, cfg.stability));
  const int cluster_take = std::max (
      1, core::RetrievalClusterLabelM (cfg.focus, cfg.stability));
  const int labels_per_cluster = std::max (
      1, core::RetrievalClusterLabelN (cfg.focus, cfg.sensitivity,
                                       cfg.stability));
  const int router_top_k = std::max (
      1, core::STMLabelRouterTopK (cfg.focus, cfg.stability));
  const double min_score = core::STMLabelMinScore (
      cfg.focus, cfg.sensitivity, cfg.stability);

  auto &cluster_cache = processor.label_cluster_cache;
  const bool cache_matches = cluster_cache.valid
                             && cluster_cache.requested_clusters
                                    == requested_clusters
                             && cluster_cache.embedding_dim == q.size ()
                             && cluster_cache.summary_cache_size
                                    == processor.summary_cache.size ();
  if (!cache_matches)
    {
      std::vector<size_t> label_indices;
      label_indices.reserve (processor.summary_cache.size ());
      for (size_t i = 0; i < processor.summary_cache.size (); ++i)
        {
          const auto &entry = processor.summary_cache[i];
          if (!entry.is_label || entry.embedding.size () != q.size ()
              || entry.embedding_norm <= 1e-9f)
            {
              continue;
            }
          label_indices.push_back (i);
        }
      const int label_count = static_cast<int> (label_indices.size ());
      if (label_count == 0)
        {
          cluster_cache.valid = true;
          cluster_cache.requested_clusters = requested_clusters;
          cluster_cache.embedding_dim = static_cast<int> (q.size ());
          cluster_cache.summary_cache_size = processor.summary_cache.size ();
          cluster_cache.centroids.clear ();
          cluster_cache.members.clear ();
          return {};
        }

      const int cluster_count = std::min (requested_clusters, label_count);
      std::vector<Eigen::VectorXf> centroids;
      centroids.reserve (static_cast<size_t> (cluster_count));
      for (int c = 0; c < cluster_count; ++c)
        {
          const int idx = (c * label_count) / cluster_count;
          centroids.push_back (
              processor.summary_cache[label_indices[static_cast<size_t> (idx)]]
                  .embedding);
        }

      std::vector<int> assignments (static_cast<size_t> (label_count), 0);
      const int cluster_iterations = std::max (
          1, core::STMLabelClusterIterations (cfg.focus, cfg.sensitivity,
                                              cfg.stability));
      for (int iter = 0; iter < cluster_iterations; ++iter)
        {
          bool changed = false;
          for (int i = 0; i < label_count; ++i)
            {
              const auto &v = processor
                                  .summary_cache[label_indices[static_cast<size_t> (i)]]
                                  .embedding;
              int best_cluster = 0;
              double best_dist = std::numeric_limits<double>::max ();
              for (int c = 0; c < cluster_count; ++c)
                {
                  const double dist = static_cast<double> (
                      (v - centroids[static_cast<size_t> (c)]).squaredNorm ());
                  if (dist < best_dist)
                    {
                      best_dist = dist;
                      best_cluster = c;
                    }
                }
              if (assignments[static_cast<size_t> (i)] != best_cluster)
                {
                  assignments[static_cast<size_t> (i)] = best_cluster;
                  changed = true;
                }
            }

          std::vector<Eigen::VectorXf> next;
          std::vector<int> counts (static_cast<size_t> (cluster_count), 0);
          next.reserve (static_cast<size_t> (cluster_count));
          for (int c = 0; c < cluster_count; ++c)
            {
              next.push_back (Eigen::VectorXf::Zero (q.size ()));
            }
          for (int i = 0; i < label_count; ++i)
            {
              const int c = assignments[static_cast<size_t> (i)];
              next[static_cast<size_t> (c)]
                  += processor
                         .summary_cache[label_indices[static_cast<size_t> (i)]]
                         .embedding;
              counts[static_cast<size_t> (c)]++;
            }
          for (int c = 0; c < cluster_count; ++c)
            {
              if (counts[static_cast<size_t> (c)] > 0)
                {
                  next[static_cast<size_t> (c)]
                      /= static_cast<float> (counts[static_cast<size_t> (c)]);
                }
              else
                {
                  next[static_cast<size_t> (c)]
                      = centroids[static_cast<size_t> (c)];
                }
            }
          centroids = std::move (next);
          if (!changed)
            {
              break;
            }
        }

      cluster_cache.valid = true;
      cluster_cache.requested_clusters = requested_clusters;
      cluster_cache.embedding_dim = static_cast<int> (q.size ());
      cluster_cache.summary_cache_size = processor.summary_cache.size ();
      cluster_cache.centroids = std::move (centroids);
      cluster_cache.members.assign (static_cast<size_t> (cluster_count), {});
      for (int i = 0; i < label_count; ++i)
        {
          const int c = assignments[static_cast<size_t> (i)];
          cluster_cache.members[static_cast<size_t> (c)].push_back (
              label_indices[static_cast<size_t> (i)]);
        }
    }

  if (cluster_cache.centroids.empty () || cluster_cache.members.empty ())
    {
      return {};
    }

  std::vector<int> cache_cluster (processor.summary_cache.size (), -1);
  std::vector<std::pair<double, size_t>> ranked_global_labels;
  ranked_global_labels.reserve (processor.summary_cache.size ());
  for (size_t cluster_idx = 0; cluster_idx < cluster_cache.members.size ();
       ++cluster_idx)
    {
      for (const size_t cache_idx : cluster_cache.members[cluster_idx])
        {
          if (cache_idx >= processor.summary_cache.size ())
            {
              continue;
            }
          cache_cluster[cache_idx] = static_cast<int> (cluster_idx);
          const auto &entry = processor.summary_cache[cache_idx];
          const double sim = static_cast<double> (entry.embedding.dot (q))
                             / static_cast<double> (entry.embedding_norm);
          const double score = core::Clamp (std::max (0.0, sim), 0.0, 1.0);
          if (score >= min_score)
            {
              ranked_global_labels.emplace_back (score, cache_idx);
            }
        }
    }
  std::sort (ranked_global_labels.begin (), ranked_global_labels.end (),
             [] (const auto &a, const auto &b) {
               if (a.first != b.first)
                 {
                   return a.first > b.first;
                 }
               return a.second < b.second;
             });

  std::vector<int> routed_clusters;
  routed_clusters.reserve (static_cast<size_t> (cluster_take));
  const int take_router_labels
      = std::min (router_top_k, static_cast<int> (ranked_global_labels.size ()));
  for (int rank = 0; rank < take_router_labels; ++rank)
    {
      const size_t routed_cache_idx
          = ranked_global_labels[static_cast<size_t> (rank)].second;
      int cluster_id = -1;
      if (routed_cache_idx < cache_cluster.size ())
        {
          cluster_id = cache_cluster[routed_cache_idx];
        }
      if (cluster_id < 0
          || std::find (routed_clusters.begin (), routed_clusters.end (),
                        cluster_id)
                 != routed_clusters.end ())
        {
          continue;
        }
      routed_clusters.push_back (cluster_id);
      if (static_cast<int> (routed_clusters.size ()) >= cluster_take)
        {
          break;
        }
    }

  std::vector<ProcessorContext::ShadowLabelEdge> edges;
  for (const int cluster_id : routed_clusters)
    {
      std::vector<std::pair<double, size_t>> ranked_labels;
      if (cluster_id < 0
          || cluster_id >= static_cast<int> (cluster_cache.members.size ()))
        {
          continue;
        }
      for (const size_t cache_idx :
           cluster_cache.members[static_cast<size_t> (cluster_id)])
        {
          if (cache_idx >= processor.summary_cache.size ())
            continue;
          const auto &entry = processor.summary_cache[cache_idx];
          const double sim = static_cast<double> (entry.embedding.dot (q))
                             / static_cast<double> (entry.embedding_norm);
          const double score = core::Clamp (std::max (0.0, sim), 0.0, 1.0);
          if (score >= min_score)
            {
              ranked_labels.emplace_back (score, cache_idx);
            }
        }
      std::sort (ranked_labels.begin (), ranked_labels.end (),
                 [] (const auto &a, const auto &b) {
                   return a.first > b.first;
                 });
      const int take_labels
          = std::min (labels_per_cluster,
                      static_cast<int> (ranked_labels.size ()));
      for (int i = 0; i < take_labels; ++i)
        {
          const auto &[score, cache_idx] = ranked_labels[static_cast<size_t> (i)];
          const auto &entry = processor.summary_cache[cache_idx];
          const std::string label = LoadLabelText (tx, entry.memory_id);
          if (label.empty ())
            {
              continue;
            }
          ProcessorContext::ShadowLabelEdge edge;
          edge.source_id = signal.source_id;
          edge.timestamp = signal.timestamp;
          edge.step_index = step_index;
          edge.label_memory_id = entry.memory_id;
          edge.label = label;
          edge.weight = score;
          edge.signal_embedding = signal.embedding;
          edges.push_back (std::move (edge));
        }
    }
  return edges;
}

} // namespace

void
UpdateShortTermMemoryShadow::Execute (OperationContext &context,
                                      Transaction &tx) const
{
  (void)tx;
  auto &processor = context.GetProcessorContext ();
  if (!ShadowStmEnabled ())
    {
      processor.shadow_stm_enabled = false;
      return;
    }

  const auto started = std::chrono::steady_clock::now ();
  const auto &signal = context.GetSignal ();
  const auto &cfg = context.GetConfig ();
  auto &short_term_graph = processor.short_term_graphs[signal.source_id];
  auto &buffer = short_term_graph.items;
  const int ttl_steps = std::max (
      1, core::STMShadowTTLSteps (cfg.stability));
  const int capacity = std::max (
      1, core::STMShadowCapacity (cfg.stability));
  const int step_index = processor.signals_processed;
  const double boundary_score = context.GetBoundaryScore ().value_or (0.0);
  const bool boundary_decay_enabled = !EnvBool (
      "CORTEXT_STM_SHADOW_DISABLE_BOUNDARY_DECAY", false);
  const double hard_boundary_threshold
      = core::STMShadowHardBoundaryThreshold (cfg.focus, cfg.sensitivity,
                                              cfg.stability);
  const int hard_boundary_retain_steps = std::max (
      1, core::STMShadowHardBoundaryRetainSteps (cfg.focus, cfg.sensitivity,
                                                 cfg.stability));
  const bool hard_boundary = boundary_decay_enabled
                             && (boundary_score >= hard_boundary_threshold
                                 || signal.force_boundary);

  std::deque<ProcessorContext::ShadowSTMItem> retained;
  for (const auto &item : buffer)
    {
      const int age = step_index - item.step_index;
      if (age <= 0)
        {
          continue;
        }
      const bool keep = hard_boundary ? age <= hard_boundary_retain_steps
                                      : age <= ttl_steps;
      if (keep)
        {
          retained.push_back (item);
        }
    }

  if (static_cast<int> (retained.size ()) > capacity - 1)
    {
      const int remove_count = static_cast<int> (retained.size ())
                               - (capacity - 1);
      for (int i = 0; i < remove_count && !retained.empty (); ++i)
        {
          retained.pop_front ();
        }
      processor.shadow_stm_compaction_count += 1;
    }

  ProcessorContext::ShadowSTMItem item;
  item.source_id = signal.source_id;
  item.timestamp = signal.timestamp;
  item.step_index = step_index;
  item.embedding = signal.embedding;
  item.boundary_score = boundary_score;
  item.boundary_diagnostics = context.GetBoundaryDiagnostics ().value_or (
      BoundaryDiagnostics{});
  item.boundary_type = context.GetBoundaryType ();
  retained.push_back (std::move (item));
  buffer = std::move (retained);

  auto &label_edges = short_term_graph.label_edges;
  std::deque<ProcessorContext::ShadowLabelEdge> retained_edges;
  for (const auto &edge : label_edges)
    {
      const int age = step_index - edge.step_index;
      if (age <= 0)
        {
          continue;
        }
      const bool keep = hard_boundary ? age <= hard_boundary_retain_steps
                                      : age <= ttl_steps;
      if (keep)
        {
          retained_edges.push_back (edge);
        }
    }
  auto selected_edges = SelectClusteredLabels (context, tx, step_index);
  for (auto &edge : selected_edges)
    {
      retained_edges.push_back (std::move (edge));
    }
  const int edge_capacity = std::max (
      capacity,
      core::STMLabelEdgeCapacity (cfg.focus, cfg.sensitivity, cfg.stability));
  while (static_cast<int> (retained_edges.size ()) > edge_capacity)
    {
      retained_edges.pop_front ();
    }
  label_edges = std::move (retained_edges);

  const auto elapsed = std::chrono::duration<double, std::micro> (
      std::chrono::steady_clock::now () - started);
  processor.shadow_stm_enabled = true;
  processor.shadow_stm_last_size = static_cast<int> (buffer.size ());
  processor.shadow_stm_max_size = std::max (processor.shadow_stm_max_size,
                                            processor.shadow_stm_last_size);
  processor.shadow_stm_update_count += 1;
  processor.shadow_stm_last_update_us = elapsed.count ();
  processor.shadow_stm_total_update_us += elapsed.count ();
  processor.shadow_stm_recent_update_us.push_back (elapsed.count ());
  while (processor.shadow_stm_recent_update_us.size () > 512)
    {
      processor.shadow_stm_recent_update_us.pop_front ();
    }
}

} // namespace cortext::operations
