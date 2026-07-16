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
ConsolidationClusterParams::FromKnobs (double F, double S, double T)
{
  ConsolidationClusterParams p;
  p.merge_threshold = core::MergeThreshold (F);
  p.min_cluster_size = core::MinClusterSize (F);
  p.max_clusters = core::ConsolidationMaxClusters (F, S, T);
  return p;
}

void
ConsolidationCluster::Execute (OperationContext &context, Transaction &tx) const
{
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

  // v2: Load candidates from context (in-memory passing)
  const auto &input_candidates = context.GetConsolidationCandidates ();
  if (input_candidates.empty ())
    {
      return;
    }

  // 2. Prepare candidates for clustering.
  struct Candidate
  {
    long long memory_id;
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
      c.memory_id = ic.memory_id;
      c.embedding_id = ic.embedding_id;
      c.score = ic.score;
      c.embedding = ic.embedding;
      items.push_back (std::move (c));
    }
  std::sort (items.begin (), items.end (),
             [] (const Candidate &a, const Candidate &b) {
               if (a.score != b.score)
                 {
                   return a.score < b.score;
                 }
               return a.memory_id < b.memory_id;
             });

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
                  for (int candidate : nb_neighbors)
                    {
                      neighbors.push_back (candidate);
                    }
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

  std::vector<int> cluster_ids;
  cluster_ids.reserve (cluster_members.size ());
  for (const auto &kv : cluster_members)
    {
      cluster_ids.push_back (kv.first);
    }
  std::sort (cluster_ids.begin (), cluster_ids.end ());

  for (const int cluster_id : cluster_ids)
    {
      const auto &members = cluster_members[cluster_id];
      if (static_cast<int> (members.size ()) < min_pts)
        {
          continue;
        }

      ClusterInfo cluster;
      cluster.cluster_id = cluster_id;
      cluster.avg_score = 0.0;
      Eigen::VectorXf centroid;
      bool centroid_init = false;
      int accepted_embedding_count = 0;

      for (int idx : members)
        {
          const auto &item = items[static_cast<size_t> (idx)];
          cluster.memory_ids.push_back (item.memory_id);
          cluster.embedding_ids.push_back (item.embedding_id);
          cluster.avg_score += item.score;
          if (!centroid_init)
            {
              centroid = item.embedding;
              centroid_init = true;
              ++accepted_embedding_count;
            }
          else if (centroid.size () == item.embedding.size ())
            {
              centroid += item.embedding;
              ++accepted_embedding_count;
            }
        }

      if (!members.empty ())
        {
          cluster.avg_score /= static_cast<double> (members.size ());
        }
      if (centroid_init && accepted_embedding_count > 0)
        {
          centroid
              = centroid / static_cast<float> (accepted_embedding_count);
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

  // 4. Pass clusters to next operation via context.
  const int cluster_count = static_cast<int>(clusters.size());
  context.SetConsolidationClusters (std::move (clusters));

  telemetry::LogDebug("cortext.consolidation_cluster", {
    telemetry::Attribute::Int64("cluster_count", cluster_count)
  });
}

} // namespace cortext::operations
