#include "cortext/operations/consolidation_cluster.hpp"

#include "cortext/core/algorithms.hpp"
#include "cortext/core/knobs.hpp"
#include "cortext/core/utils.hpp"
#include "cortext/processor/operation_context.hpp"
#include "cortext/store/store.hpp"
#include "cortext/telemetry/telemetry.hpp"
#include <algorithm>
#include <any>
#include <cmath>
#include <unordered_map>
#include <vector>

namespace cortext::operations
{

ConsolidationClusterParams
ConsolidationClusterParams::FromKnobs (double F, double /*S*/, double /*T*/)
{
  ConsolidationClusterParams p;
  p.merge_threshold = core::MergeThreshold (F);
  p.min_cluster_size = core::MinClusterSize (F);
  p.max_clusters = 100; // Reasonable cap
  return p;
}

void
ConsolidationCluster::Execute (OperationContext &context, Transaction &tx) const
{
  (void)tx;
  (void)tx;
  if (!context.GetConsolidationShouldStart ())
    {
      return;
    }

  Store *store = context.GetStore ();
  if (!store)
    {
      return;
    }

  const auto &cfg = context.GetConfig ();
  auto params = ConsolidationClusterParams::FromKnobs (cfg.focus, cfg.sensitivity,
                                                       cfg.stability);
  const bool force_consolidation
      = context.GetSignal ().consolidation_mode.has_value ();

  // v2: Load candidates from context (in-memory passing)
  const auto &input_candidates = context.GetConsolidationCandidates ();
  if (input_candidates.empty ())
    {
      return;
    }

  // 2. Prepare candidates for clustering.
  struct Candidate
  {
    long long embedding_id;
    double score;
    Eigen::VectorXf embedding;
    bool assigned = false;
  };
  std::vector<Candidate> items;
  items.reserve (input_candidates.size ());

  for (const auto &ic : input_candidates)
    {
      Candidate c;
      c.embedding_id = ic.embedding_id;
      c.score = ic.score;
      c.embedding = ic.embedding;
      items.push_back (std::move (c));
    }

  if (items.empty ())
    {
      return;
    }

  // 3. Density-based clustering (DBSCAN-style).
  const int min_pts = std::max (1, params.min_cluster_size);
  const double eps = params.merge_threshold;
  const int n = static_cast<int> (items.size ());

  std::vector<int> cluster_labels (n, -1);
  std::vector<bool> visited (n, false);
  int next_cluster_id = 0;
  {
    auto rows = tx.Execute (
        "SELECT COALESCE(MAX(cluster_id), -1) + 1 AS next_cluster_id "
        "FROM memories WHERE cluster_id IS NOT NULL",
        {});
    if (!rows.empty ())
      {
        auto it = rows[0].find ("next_cluster_id");
        if (it != rows[0].end ())
          {
            if (it->second.type () == typeid (long long))
              {
                next_cluster_id
                    = static_cast<int> (std::any_cast<long long> (it->second));
              }
            else if (it->second.type () == typeid (int))
              {
                next_cluster_id = std::any_cast<int> (it->second);
              }
          }
      }
  }
  std::vector<ClusterInfo> clusters;

  auto neighbors_of = [&](int idx) {
    std::vector<int> neighbors;
    neighbors.reserve (static_cast<size_t> (n));
    const auto &base = items[static_cast<size_t> (idx)].embedding;
    for (int j = 0; j < n; ++j)
      {
        const auto &cand = items[static_cast<size_t> (j)].embedding;
        if (cand.size () == 0 || cand.size () != base.size ())
          {
            continue;
          }
        const double sim = core::CosineSimilarity (base, cand);
        if (sim >= eps)
          {
            neighbors.push_back (j);
          }
      }
    return neighbors;
  };

  for (int i = 0; i < n; ++i)
    {
      if (visited[static_cast<size_t> (i)])
        {
          continue;
        }
      visited[static_cast<size_t> (i)] = true;
      std::vector<int> neighbors = neighbors_of (i);
      if (static_cast<int> (neighbors.size ()) < min_pts)
        {
          continue; // noise
        }

      const int cluster_id = next_cluster_id++;
      cluster_labels[static_cast<size_t> (i)] = cluster_id;

      // Expand cluster
      for (size_t k = 0; k < neighbors.size (); ++k)
        {
          const int nb = neighbors[k];
          if (!visited[static_cast<size_t> (nb)])
            {
              visited[static_cast<size_t> (nb)] = true;
              std::vector<int> nb_neighbors = neighbors_of (nb);
              if (static_cast<int> (nb_neighbors.size ()) >= min_pts)
                {
                  neighbors.insert (neighbors.end (),
                                    nb_neighbors.begin (),
                                    nb_neighbors.end ());
                }
            }
          if (cluster_labels[static_cast<size_t> (nb)] < 0)
            {
              cluster_labels[static_cast<size_t> (nb)] = cluster_id;
            }
        }

      if (static_cast<int> (clusters.size ()) >= params.max_clusters)
        {
          break;
        }
    }

  // Build clusters from labels.
  std::unordered_map<int, std::vector<int>> cluster_members;
  for (int i = 0; i < n; ++i)
    {
      const int label = cluster_labels[static_cast<size_t> (i)];
      if (label < 0)
        {
          continue;
        }
      cluster_members[label].push_back (i);
    }

  for (const auto &kv : cluster_members)
    {
      const auto &members = kv.second;
      if (static_cast<int> (members.size ()) < min_pts)
        {
          continue;
        }

      ClusterInfo cluster;
      cluster.cluster_id = kv.first;
      cluster.avg_score = 0.0;
      Eigen::VectorXf centroid;
      bool centroid_init = false;

      for (int idx : members)
        {
          const auto &item = items[static_cast<size_t> (idx)];
          cluster.embedding_ids.push_back (item.embedding_id);
          cluster.avg_score += item.score;
          if (!centroid_init)
            {
              centroid = item.embedding;
              centroid_init = true;
            }
          else if (centroid.size () == item.embedding.size ())
            {
              centroid += item.embedding;
            }
        }

      if (!members.empty ())
        {
          cluster.avg_score /= static_cast<double> (members.size ());
        }
      if (centroid_init && !members.empty ())
        {
          centroid
              = centroid
                / static_cast<float> (members.size ());
        }

      cluster.centroid.resize (static_cast<size_t> (centroid.size ()));
      for (int k = 0; k < centroid.size (); ++k)
        {
          cluster.centroid[static_cast<size_t> (k)] = centroid (k);
        }

      clusters.push_back (std::move (cluster));
      if (static_cast<int> (clusters.size ()) >= params.max_clusters)
        {
          break;
        }
    }

  if (clusters.empty () && force_consolidation && n >= min_pts)
    {
      // Forced fallback: pick a coherent mini-cluster around the lowest-score item.
      int anchor_idx = 0;
      double best_score = items[0].score;
      for (int i = 1; i < n; ++i)
        {
          if (items[static_cast<size_t> (i)].score < best_score)
            {
              best_score = items[static_cast<size_t> (i)].score;
              anchor_idx = i;
            }
        }

      std::vector<std::pair<double, int>> sims;
      sims.reserve (static_cast<size_t> (n));
      const auto &anchor = items[static_cast<size_t> (anchor_idx)].embedding;
      for (int i = 0; i < n; ++i)
        {
          const auto &cand = items[static_cast<size_t> (i)].embedding;
          if (cand.size () == 0 || cand.size () != anchor.size ())
            {
              continue;
            }
          const double sim = core::CosineSimilarity (anchor, cand);
          sims.emplace_back (sim, i);
        }
      std::sort (sims.begin (), sims.end (),
                 [] (const auto &a, const auto &b) { return a.first > b.first; });

      const int k = std::min (min_pts, static_cast<int> (sims.size ()));
      ClusterInfo cluster;
      cluster.cluster_id = next_cluster_id++;
      cluster.avg_score = 0.0;
      Eigen::VectorXf centroid;
      bool centroid_init = false;

      for (int i = 0; i < k; ++i)
        {
          const auto &item = items[static_cast<size_t> (sims[i].second)];
          cluster.embedding_ids.push_back (item.embedding_id);
          cluster.avg_score += item.score;
          if (!centroid_init)
            {
              centroid = item.embedding;
              centroid_init = true;
            }
          else if (centroid.size () == item.embedding.size ())
            {
              centroid += item.embedding;
            }
        }

      if (k > 0)
        {
          cluster.avg_score /= static_cast<double> (k);
        }
      if (centroid_init && k > 0)
        {
          centroid = centroid / static_cast<float> (k);
        }
      cluster.centroid.resize (static_cast<size_t> (centroid.size ()));
      for (int i = 0; i < centroid.size (); ++i)
        {
          cluster.centroid[static_cast<size_t> (i)] = centroid (i);
        }
      clusters.push_back (std::move (cluster));
    }

  // 4. Pass clusters to next operation via context.
  const int cluster_count = static_cast<int>(clusters.size());
  context.SetConsolidationClusters (std::move (clusters));

  telemetry::LogDebug("cortext.consolidation_cluster", {
    telemetry::Attribute::Int64("cluster_count", cluster_count)
  });
}

} // namespace cortext::operations
